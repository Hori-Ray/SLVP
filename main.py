import argparse, os
from base.data_eegcvpr40 import load_eegcvpr40_data
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import math
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

##import user lib
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config, ClipLoss, instantiate_from_config, get_device

device = get_device('auto')

def load_model(config, train_loader, test_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader, test_loader)
    return pl_model

class AWMBContextualVisualAdapter(nn.Module):
    
    def __init__(
        self,
        z_dim=1024,
        stat_dim=6,
        hidden_dim=1024,
        dropout=0.15,
        adapter_lambda_init=0.05,
        adapter_lambda_max=0.25,
        eps=1e-6,
        use_stat=True,
        zero_init=True,
    ):
        super().__init__()

        self.z_dim = int(z_dim)
        self.stat_dim = int(stat_dim)
        self.hidden_dim = int(hidden_dim)
        self.adapter_lambda_max = float(adapter_lambda_max)
        self.eps = float(eps)
        self.use_stat = bool(use_stat)

        if self.use_stat:
            in_dim = self.z_dim * 4 + self.stat_dim
        else:
            in_dim = self.z_dim * 4

        self.input_norm = nn.LayerNorm(in_dim)

        self.adapter = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.z_dim),
        )

        if zero_init:
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        adapter_lambda_init = float(adapter_lambda_init)
        adapter_lambda_init = max(
            min(adapter_lambda_init, self.adapter_lambda_max - 1e-6),
            1e-6
        )
        ratio = adapter_lambda_init / self.adapter_lambda_max
        lambda_logit_init = math.log(ratio / (1.0 - ratio))

        self.adapter_lambda_logit = nn.Parameter(
            torch.tensor(lambda_logit_init, dtype=torch.float32)
        )

    def get_adapter_lambda(self):
        return self.adapter_lambda_max * torch.sigmoid(self.adapter_lambda_logit)

    def build_context_feature(self, ctx_features, stats):
        """
        ctx_features: [B, 3, D]
        stats[:, 3:6] = [low_mean, mid_mean, high_mean]
        """
        if ctx_features.dim() != 3 or ctx_features.shape[1] != 3:
            raise RuntimeError(
                f"ctx_features should be [B,3,D], got {ctx_features.shape}"
            )

        ctx_features = F.normalize(ctx_features.float(), dim=-1, eps=self.eps)

        weights = stats[:, 3:6].float()
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        z_ctx = torch.sum(
            ctx_features * weights.unsqueeze(-1),
            dim=1
        )
        z_ctx = F.normalize(z_ctx, dim=-1, eps=self.eps)

        return z_ctx

    def forward(
        self,
        fused_feat,
        original_feat,
        stats,
        ctx_features,
        return_debug=False,
    ):
        z_f = F.normalize(fused_feat.float(), dim=-1, eps=self.eps)
        z_o = F.normalize(original_feat.float(), dim=-1, eps=self.eps).detach()

        stats = stats.float()
        if stats.dim() == 1:
            stats = stats.unsqueeze(0)

        z_ctx = self.build_context_feature(ctx_features, stats)
        z_ctx = z_ctx.detach()

        ctx_delta = z_ctx - z_f
        ori_delta = z_o - z_f

        if self.use_stat:
            adapter_input = torch.cat(
                [
                    z_f,
                    z_ctx,
                    ctx_delta,
                    ori_delta,
                    stats,
                ],
                dim=-1
            )
        else:
            adapter_input = torch.cat(
                [
                    z_f,
                    z_ctx,
                    ctx_delta,
                    ori_delta,
                ],
                dim=-1
            )

        adapter_input = self.input_norm(adapter_input)

        delta = self.adapter(adapter_input)

        lam = self.get_adapter_lambda().to(z_f.device)

        out = z_f + lam * delta
        out = F.normalize(out, dim=-1, eps=self.eps)

        if return_debug:
            with torch.no_grad():
                debug = {
                    "adapter_lambda": float(lam.detach().cpu()),
                    "ctx_sim": float(torch.sum(z_f * z_ctx, dim=-1).mean().detach().cpu()),
                    "ori_sim": float(torch.sum(z_f * z_o, dim=-1).mean().detach().cpu()),
                    "delta_norm": float(delta.norm(dim=-1).mean().detach().cpu()),
                }
            return out, debug

        return out


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader, model_type='RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']
        
        vc_cfg = self.config.get('visual_calibration', {})
        self.use_visual_calibration = bool(vc_cfg.get('enabled', False))
        self.visual_calibration_mode = str(vc_cfg.get('mode', 'acvc'))
        self.semantic_keep_weight = float(vc_cfg.get('semantic_keep_weight', 0.0))
        self.vc_lr_mult = float(vc_cfg.get('lr_mult', 1.0))

        if self.use_visual_calibration:
            if self.visual_calibration_mode == 'contextual_adapter':
                self.visual_calibration = AWMBContextualVisualAdapter(
                    z_dim=int(vc_cfg.get('z_dim', self.z_dim)),
                    stat_dim=int(vc_cfg.get('stat_dim', 6)),
                    hidden_dim=int(vc_cfg.get('hidden_dim', 1024)),
                    dropout=float(vc_cfg.get('dropout', 0.15)),
                    adapter_lambda_init=float(vc_cfg.get('adapter_lambda_init', 0.05)),
                    adapter_lambda_max=float(vc_cfg.get('adapter_lambda_max', 0.25)),
                    use_stat=bool(vc_cfg.get('use_stat', True)),
                    zero_init=bool(vc_cfg.get('zero_init', True)),
                )

            else:
                self.visual_calibration = AttentionConditionedVisualCalibration(
                    z_dim=int(vc_cfg.get('z_dim', self.z_dim)),
                    stat_dim=int(vc_cfg.get('stat_dim', 6)),
                    stat_hidden_dim=int(vc_cfg.get('stat_hidden_dim', 32)),
                    lambda_init=float(vc_cfg.get('lambda_init', 0.05)),
                    lambda_max=float(vc_cfg.get('lambda_max', 0.20)),
                    gate_logit_init=float(vc_cfg.get('gate_logit_init', -2.0)),
                    use_extra_adapter=bool(vc_cfg.get('use_extra_adapter', False)),
                    adapter_hidden_dim=int(vc_cfg.get('adapter_hidden_dim', 512)),
                    adapter_dropout=float(vc_cfg.get('adapter_dropout', 0.10)),
                    adapter_lambda_init=float(vc_cfg.get('adapter_lambda_init', 0.02)),
                    adapter_lambda_max=float(vc_cfg.get('adapter_lambda_max', 0.10)),
                )
        else:
            self.visual_calibration = None

        ctx_cfg = self.config.get('context_teacher', {})
        self.use_context_teacher = bool(ctx_cfg.get('enabled', False))
        self.context_teacher_weight = float(ctx_cfg.get('weight', 0.0))
        self.context_teacher_detach = bool(ctx_cfg.get('detach', True))

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3

        self.mAP_total = 0.0
        self.match_similarities = []

        self.is_eegcvpr40 = (self.config.get("dataset", "") == "eegcvpr40")
        self.eegcvpr40_task = self.config.get("eegcvpr40", {}).get("task", "retrieval")

        if self.is_eegcvpr40 and self.eegcvpr40_task == "prototype":
            proto_labels, proto_bank = self._build_eegcvpr40_class_prototypes(train_loader.dataset)

            self.register_buffer("eegcvpr40_proto_labels", proto_labels)

            self.register_buffer("eegcvpr40_proto_fused", proto_bank["fused"])
            self.register_buffer("eegcvpr40_proto_original", proto_bank["original"])
            self.register_buffer("eegcvpr40_proto_stats", proto_bank["stats"])
            self.register_buffer("eegcvpr40_proto_ctx", proto_bank["ctx"])

            self.prototype_label_to_index = {
                int(lb): i for i, lb in enumerate(proto_labels.tolist())
            }

            print(f"[EEGCVPR40] built {len(self.prototype_label_to_index)} class prototypes.")
        else:
            self.prototype_label_to_index = {}

    def forward(self, batch, sample_posterior=False):

        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']

        raw_img_z = batch['img_features']
        if isinstance(raw_img_z, torch.Tensor):
            raw_img_z = raw_img_z.to(eeg.device)

        img_stats = batch.get('img_stats', None)
        if img_stats is not None and isinstance(img_stats, torch.Tensor):
            img_stats = img_stats.to(eeg.device)

        ctx_features = batch.get('ctx_features', None)
        if ctx_features is not None and isinstance(ctx_features, torch.Tensor):
            ctx_features = ctx_features.to(eeg.device)

        eeg_z = self.brain(eeg)

        original_img_z = None

        if self.use_visual_calibration:

            if raw_img_z.dim() != 3 or raw_img_z.shape[1] < 2:
                raise RuntimeError(
                    "visual_calibration=True requires img_features with shape [B,2,D]. "
                    "Please set return_original_feature_for_calibration=True and rebuild image cache."
                )

            fused_img_z = raw_img_z[:, 0, :]
            original_img_z = raw_img_z[:, 1, :]

            if img_stats is None:
                raise RuntimeError(
                    "visual_calibration=True requires batch['img_stats']. "
                    "Please modify Dataset __getitem__ and rebuild cache."
                )

            if self.visual_calibration_mode == 'contextual_adapter':
                if ctx_features is None:
                    raise RuntimeError(
                        "contextual_adapter requires batch['ctx_features']. "
                        "Please set return_awmb_context_for_teacher=True and rebuild image cache."
                    )

                img_z = self.visual_calibration(
                    fused_img_z,
                    original_img_z,
                    img_stats,
                    ctx_features
                )

            else:
                img_z = self.visual_calibration(
                    fused_img_z,
                    original_img_z,
                    img_stats
                )

        else:
            if raw_img_z.dim() == 3:
                raw_img_z = raw_img_z[:, 0, :]

            img_z = raw_img_z / raw_img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)

        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if (
                self.use_visual_calibration
                and self.semantic_keep_weight > 0
                and original_img_z is not None
        ):
            original_img_z_norm = F.normalize(original_img_z.float(), dim=-1).detach()
            sem_loss = 1.0 - torch.sum(img_z * original_img_z_norm, dim=-1)
            sem_loss = sem_loss.mean()
            total_loss = total_loss + self.semantic_keep_weight * sem_loss

        if (
                self.use_context_teacher
                and self.context_teacher_weight > 0
                and ctx_features is not None
                and img_stats is not None
        ):
            if ctx_features.dim() != 3 or ctx_features.shape[1] != 3:
                raise RuntimeError(
                    f"context_teacher requires ctx_features shape [B,3,D], got {ctx_features.shape}"
                )

            ctx_weights = img_stats[:, 3:6].float()
            ctx_weights = torch.clamp(ctx_weights, min=0.0)
            ctx_weights = ctx_weights / (ctx_weights.sum(dim=-1, keepdim=True) + 1e-6)

            ctx_features_norm = F.normalize(ctx_features.float(), dim=-1)

            ctx_z = torch.sum(
                ctx_features_norm * ctx_weights.unsqueeze(-1),
                dim=1
            )
            ctx_z = F.normalize(ctx_z, dim=-1)

            if self.context_teacher_detach:
                ctx_z = ctx_z.detach()

            ctx_loss = 1.0 - torch.sum(img_z * ctx_z, dim=-1)
            ctx_loss = ctx_loss.mean()

            total_loss = total_loss + self.context_teacher_weight * ctx_loss

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label

            loss = total_loss
        else:
            loss = total_loss

        return eeg_z, img_z, loss

    def _reset_metric_cache(self):
        self.all_predicted_classes = []
        self.all_true_labels = []

        self.eval_eeg_z = []
        self.eval_img_z = []
        self.eval_img_paths = []
        self.eval_labels = []

    def _is_eegcvpr40_retrieval(self):
        return (
                self.config.get("dataset", "") == "eegcvpr40"
                and self.config.get("eegcvpr40", {}).get("task", "retrieval") == "retrieval"
        )

    def _compute_eegcvpr40_retrieval_metrics(self):
        
        if len(self.eval_eeg_z) == 0 or len(self.eval_img_z) == 0:
            return {
                "top1": 0.0,
                "top5": 0.0,
                "mAP": 0.0,
                "similarity": 0.0,
            }

        eeg_z = torch.cat(self.eval_eeg_z, dim=0)
        img_z = torch.cat(self.eval_img_z, dim=0)

        eeg_z = F.normalize(eeg_z.float(), dim=-1)
        img_z = F.normalize(img_z.float(), dim=-1)

        similarity = eeg_z @ img_z.t()  # [N, N]

        retrieval_target = self.config.get("eegcvpr40", {}).get("retrieval_target", "image")

        if retrieval_target == "label":
            query_keys = [int(x) for x in self.eval_labels]
            gallery_keys = [int(x) for x in self.eval_labels]
        else:
            query_keys = [str(x) for x in self.eval_img_paths]
            gallery_keys = [str(x) for x in self.eval_img_paths]

        n = similarity.shape[0]

        top1_correct = 0
        top5_correct = 0
        ap_sum = 0.0
        gt_sims = []

        topk = min(5, n)

        for i in range(n):
            sims = similarity[i]
            sorted_indices = torch.argsort(-sims)

            positive_indices = [
                j for j, key in enumerate(gallery_keys)
                if key == query_keys[i]
            ]

            if len(positive_indices) == 0:
                continue

            positive_set = set(positive_indices)

            top1_idx = int(sorted_indices[0].item())
            topk_indices = [int(x.item()) for x in sorted_indices[:topk]]

            if top1_idx in positive_set:
                top1_correct += 1

            if any(idx in positive_set for idx in topk_indices):
                top5_correct += 1

            hit_count = 0
            precision_sum = 0.0

            for rank_zero_based, idx_tensor in enumerate(sorted_indices):
                idx = int(idx_tensor.item())

                if idx in positive_set:
                    hit_count += 1
                    precision_sum += hit_count / float(rank_zero_based + 1)

                    if hit_count == len(positive_set):
                        break

            ap = precision_sum / float(len(positive_set))
            ap_sum += ap

            pos_sims = similarity[i, positive_indices]
            gt_sims.append(float(pos_sims.max().detach().cpu().item()))

        top1 = top1_correct / float(n)
        top5 = top5_correct / float(n)
        mAP = ap_sum / float(n)
        mean_similarity = float(np.mean(gt_sims)) if len(gt_sims) > 0 else 0.0

        return {
            "top1": float(top1),
            "top5": float(top5),
            "mAP": float(mAP),
            "similarity": mean_similarity,
        }

    def on_train_epoch_start(self):
        self._reset_metric_cache()

    def on_validation_epoch_start(self):
        self._reset_metric_cache()

    def on_test_epoch_start(self):
        self._reset_metric_cache()
        self.mAP_total = 0.0
        self.match_similarities = []

    def _get_eval_similarity_and_targets(self, eeg_z, img_z, batch):

        eeg_z = F.normalize(eeg_z.float(), dim=-1)

        if img_z is None:
            raise RuntimeError(
                "Retrieval evaluation requires img_z, but got None."
            )

        img_z = F.normalize(img_z.float(), dim=-1)

        similarity = eeg_z @ img_z.t()

        targets = torch.arange(
            eeg_z.shape[0],
            device=eeg_z.device,
            dtype=torch.long
        )

        return similarity, targets

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch, sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            counter = Counter(self.match_label)
            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]

        eeg_z, img_z, loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

        if self._is_eegcvpr40_retrieval():
            self.eval_eeg_z.append(eeg_z.detach().cpu())
            self.eval_img_z.append(img_z.detach().cpu())

            self.eval_img_paths.extend([str(p) for p in batch["img_path"]])
            self.eval_labels.extend([int(x) for x in batch["label"].detach().cpu().tolist()])

            return loss

        similarity, label = self._get_eval_similarity_and_targets(eeg_z, img_z, batch)

        topk = min(5, similarity.shape[1])
        _, top_k_indices = similarity.topk(topk, dim=-1)

        self.all_predicted_classes.append(top_k_indices.detach().cpu().numpy())
        self.all_true_labels.extend(label.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        if self._is_eegcvpr40_retrieval():
            metrics = self._compute_eegcvpr40_retrieval_metrics()

            self.log('val_top1_acc', metrics["top1"], on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)
            self.log('val_top5_acc', metrics["top5"], on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=True)

            self.eval_eeg_z = []
            self.eval_img_z = []
            self.eval_img_paths = []
            self.eval_labels = []
            self.all_predicted_classes = []
            self.all_true_labels = []
            return

        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = float(np.mean(top_1_correct))

        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = float(np.mean(top_k_correct))

        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=batch_size)

        if self._is_eegcvpr40_retrieval():
            self.eval_eeg_z.append(eeg_z.detach().cpu())
            self.eval_img_z.append(img_z.detach().cpu())

            self.eval_img_paths.extend([str(p) for p in batch["img_path"]])
            self.eval_labels.extend([int(x) for x in batch["label"].detach().cpu().tolist()])

            return loss

        similarity, label = self._get_eval_similarity_and_targets(eeg_z, img_z, batch)

        topk = min(5, similarity.shape[1])
        _, top_k_indices = similarity.topk(topk, dim=-1)

        self.all_predicted_classes.append(top_k_indices.detach().cpu().numpy())
        self.all_true_labels.extend(label.detach().cpu().numpy())

        gt_similarity = similarity.gather(1, label.unsqueeze(1)).squeeze(1)
        self.match_similarities.extend(gt_similarity.detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == label[i]).nonzero(as_tuple=False)[0, 0] + 1
            self.mAP_total += 1.0 / float(rank)

        return loss

    def on_test_epoch_end(self):
        if self._is_eegcvpr40_retrieval():
            metrics = self._compute_eegcvpr40_retrieval_metrics()

            self.log('test_top1_acc', metrics["top1"], sync_dist=True)
            self.log('test_top5_acc', metrics["top5"], sync_dist=True)
            self.log('mAP', metrics["mAP"], sync_dist=True)
            self.log('similarity', metrics["similarity"], sync_dist=True)

            self.eval_eeg_z = []
            self.eval_img_z = []
            self.eval_img_paths = []
            self.eval_labels = []
            self.all_predicted_classes = []
            self.all_true_labels = []

            avg_test_loss = self.trainer.callback_metrics['test_loss']

            return {
                'test_loss': avg_test_loss.item(),
                'test_top1_acc': metrics["top1"],
                'test_top5_acc': metrics["top5"],
                'mAP': metrics["mAP"],
                'similarity': metrics["similarity"],
            }

        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = float(np.mean(top_1_correct))

        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = float(np.mean(top_k_correct))

        self.mAP = float(self.mAP_total / len(all_true_labels))
        self.match_similarities = float(np.mean(self.match_similarities)) if self.match_similarities else 0.0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {
            'test_loss': avg_test_loss.item(),
            'test_top1_acc': top_1_accuracy,
            'test_top5_acc': top_k_accuracy,
            'mAP': self.mAP,
            'similarity': self.match_similarities
        }

    def configure_optimizers(self):
        lr = float(self.config['train']['lr'])
        optimizer_name = self.config['train'].get('optimizer', 'AdamW')
        weight_decay = float(self.config['train'].get('weight_decay', 0.01))

        visual_lr_mult = 1.0
        if hasattr(self, "vc_lr_mult"):
            visual_lr_mult = float(self.vc_lr_mult)

        spatial_lr_mult = float(self.config['train'].get('spatial_lr_mult', 1.0))

        if optimizer_name == 'AdamW':
            optim_cls = torch.optim.AdamW
        elif optimizer_name == 'Adam':
            optim_cls = torch.optim.Adam
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        base_params = []
        visual_params = []
        spatial_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if name.startswith("visual_calibration."):
                visual_params.append(p)
            elif "spatial_mixer" in name:
                spatial_params.append(p)
            else:
                base_params.append(p)

        param_groups = []

        if len(base_params) > 0:
            param_groups.append(
                {
                    "params": base_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

        if len(visual_params) > 0:
            param_groups.append(
                {
                    "params": visual_params,
                    "lr": lr * visual_lr_mult,
                    "weight_decay": weight_decay,
                }
            )

        if len(spatial_params) > 0:
            param_groups.append(
                {
                    "params": spatial_params,
                    "lr": lr * spatial_lr_mult,
                    "weight_decay": weight_decay,
                }
            )

        optimizer = optim_cls(param_groups)
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg","eegcvpr40"],
        help="Choose dataset: 'eeg' , 'meg' or 'eegcvpr40'",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'], exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name=config['name'],
                               version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}")
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))

    if config["dataset"] == "eeg":
        train_loader, val_loader, test_loader = load_eeg_data(config)
    elif config["dataset"] == "meg":
        train_loader, val_loader, test_loader = load_meg_data(config)
    elif config["dataset"] == "eegcvpr40":
        train_loader, val_loader, test_loader = load_eegcvpr40_data(config)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    print(
        f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='max'
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode='min'
        )

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False),
                      callbacks=[early_stop_callback, checkpoint_callback], max_epochs=config['train']['epoch'],
                      devices=[device], accelerator='cuda', logger=logger)
    print(trainer.logger.log_dir)

    ckpt_path = 'last'  # None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()