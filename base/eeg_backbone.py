import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import math
import logging
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from omegaconf import ListConfig
import numpy as np
import torch


def _freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def _freeze_parameter(param):
    param.requires_grad = False

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGProjectLayer(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class BaseModel(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, embedding_dim=1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x


class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )


class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1400)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )


class EEGnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1248)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )


class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )





class EEGProjectLayerLiteAdapterSafeV32(nn.Module):

 
    def __init__(
        self,
        z_dim,
        c_num,
        timesteps,
        core_drop_proj=0.2,
        adapter_drop_proj=0.1,
        adapter_ratio=8,
        gate_hidden_ratio=4,
        delta_scale_init=0.10,
        feat_scale_init=0.03,
        temp_scale_init=0.14,
        delta_alpha_scale_init=0.05,
        use_input_calibration=True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.t_len = self.timesteps[1] - self.timesteps[0]
        self.input_dim = self.c_num * self.t_len
        proj_dim = z_dim
        self.use_input_calibration = use_input_calibration

        
        self.channel_scale = nn.Parameter(torch.ones(1, self.c_num, 1))
        time_hidden = max(8, self.t_len // 8)
        self.temporal_gate = nn.Sequential(
            nn.Linear(self.t_len, time_hidden),
            nn.GELU(),
            nn.Linear(time_hidden, self.t_len)
        )
        nn.init.zeros_(self.temporal_gate[-1].weight)
        nn.init.zeros_(self.temporal_gate[-1].bias)
        self.temp_scale = nn.Parameter(torch.tensor(float(temp_scale_init)))

        
        self.core = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(core_drop_proj),
            )),
            nn.LayerNorm(proj_dim)
        )

        
        adapter_hidden = max(32, proj_dim // adapter_ratio)
        self.adapter = nn.Sequential(
            nn.Linear(proj_dim, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, proj_dim),
            nn.Dropout(adapter_drop_proj)
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init)))

        
        alpha_hidden = max(16, proj_dim // 8)
        self.delta_alpha_head = nn.Sequential(
            nn.Linear(proj_dim, alpha_hidden),
            nn.GELU(),
            nn.Linear(alpha_hidden, 1)
        )
        nn.init.zeros_(self.delta_alpha_head[-1].weight)
        nn.init.zeros_(self.delta_alpha_head[-1].bias)
        self.delta_alpha_scale = nn.Parameter(torch.tensor(float(delta_alpha_scale_init)))
        
        gate_hidden = max(16, proj_dim // gate_hidden_ratio)
        self.feature_gate = nn.Sequential(
            nn.Linear(proj_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, proj_dim)
        )
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)
        self.feat_scale = nn.Parameter(torch.tensor(float(feat_scale_init)))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x, subject_id=None):
        if x.dim() != 3:
            raise RuntimeError(f"EEG input must be 3D, got {x.shape}")
        if x.shape[1] == self.c_num and x.shape[2] == self.t_len:
            pass
        elif x.shape[1] == self.t_len and x.shape[2] == self.c_num:
            x = x.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                f"Unexpected EEG shape {x.shape}, expected [B,{self.c_num},{self.t_len}] "
                f"or [B,{self.t_len},{self.c_num}]"
            )

        
        if self.use_input_calibration:
            x = x * self.channel_scale
            t_context = x.mean(dim=1)  # [B,T]
            t_delta = torch.tanh(self.temporal_gate(t_context))
            x = x * (1.0 + self.temp_scale * t_delta.unsqueeze(1))
        
        x = x.view(x.shape[0], self.input_dim)
        base = self.core(x)
        
        delta = torch.tanh(self.adapter(base))  # bounded residual
        alpha = 1.0 + self.delta_alpha_scale * torch.tanh(self.delta_alpha_head(base))  # [B,1]
        x = base + alpha * self.delta_scale * delta
        
        g = torch.tanh(self.feature_gate(x))
        x = x * (1.0 + self.feat_scale * g)
        return x

