import torch,os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import pickle

import cv2
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from scipy.optimize import fsolve


class DirectT:
    def __init__(self):
        pass
    def __call__(self,x,U=None):
        return x
    
class UniformBlur:
    def __init__(self,blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)
    
class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h,w), np.float32)
        
        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1-c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]
        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]
        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1,distance/max_distance)
                y0 = fun_degrade(x0,**kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
        return blended
    
    def __call__(self, img, blur_kernel_size=None): 
        if blur_kernel_size ==None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blured = cv2.GaussianBlur(img, (blur_kernel_size,blur_kernel_size), 0)
        blended = self.alphaBlend(img, blured, 1- self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)
    
    def linear(self,x,**kwargs):
        return 1-x
    
    def exp(self,x,**kwargs):
        system_g = kwargs.get('system_g', 4)
        return  np.exp(-system_g * x)
    
    def quadratic(self,x,**kwargs):
        return  1 - x**2
    
    def log(self,x,**kwargs):
        b = 1/(np.e-1)
        a = np.log(b) + 1
        return  a - np.log(x + b)
    
    def brachistochrone(self,x,**kwargs):
        
        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return  y0



class AttentionWeightedMultiBlur:
  

    def __init__(
        self,
        h: int,
        w: int,
        blur_kernel_size: int,
        min_kernel_size: int = 3,
        mid_kernel_ratio: float = 0.45,   
        attn_smooth_ksize: int = 9,        
        grad_gamma: float = 0.8,           
        center_prior: float = 0.25,        
        center_sigma_ratio: float = 0.45,  
        softness: float = 0.15,           
        *args,
        **kwargs
    ):
        self.h = int(h)
        self.w = int(w)

        self.min_kernel_size = int(min_kernel_size)
        self.mid_kernel_ratio = float(mid_kernel_ratio)

        self.attn_smooth_ksize = int(attn_smooth_ksize)
        if self.attn_smooth_ksize % 2 == 0:
            self.attn_smooth_ksize += 1

        self.grad_gamma = float(grad_gamma)
        self.center_prior = float(center_prior)
        self.center_sigma_ratio = float(center_sigma_ratio)
        self.softness = float(softness)

        self.blur_kernel_size = self._make_odd(int(blur_kernel_size))

        self._center_prior_map = self._make_center_prior() if self.center_prior > 0 else None

    @staticmethod
    def _make_odd(k: int) -> int:
        k = max(1, int(k))
        return k if (k % 2 == 1) else (k + 1)

    def _make_center_prior(self) -> np.ndarray:
       
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        cy, cx = (self.h - 1) * 0.5, (self.w - 1) * 0.5
        sigma = self.center_sigma_ratio * min(self.h, self.w)
        sigma = max(1.0, float(sigma))

        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        return g

    def _compute_attention(self, img_bgr: np.ndarray) -> np.ndarray:
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        if self.grad_gamma != 1.0:
            mag = np.power(mag, self.grad_gamma).astype(np.float32)

        if self.attn_smooth_ksize >= 3:
            mag = cv2.GaussianBlur(mag, (self.attn_smooth_ksize, self.attn_smooth_ksize), 0)

        if self._center_prior_map is not None:
            A = (1.0 - self.center_prior) * mag + self.center_prior * self._center_prior_map
        else:
            A = mag

        A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        return A.astype(np.float32)

    def _tri_weights(self, A: np.ndarray):
        
        c_low, c_mid, c_high = 0.85, 0.50, 0.15
        s = max(1e-4, self.softness)

        def bump_remap(x, c):
            return np.exp(-((x - c) ** 2) / (2.0 * s * s)).astype(np.float32)

        w_low = bump_remap(A, c_low)
        w_mid = bump_remap(A, c_mid)
        w_high = bump_remap(A, c_high)

        w_sum = w_low + w_mid + w_high + 1e-6
        w_low /= w_sum
        w_mid /= w_sum
        w_high /= w_sum
        return w_low, w_mid, w_high

    def __call__(self, img: Image.Image, blur_kernel_size: int = None) -> Image.Image:
        
        if blur_kernel_size is None:
            k_max = self.blur_kernel_size
        else:
            k_max = self._make_odd(int(blur_kernel_size))

        k_min = self._make_odd(self.min_kernel_size)
        k_mid = int(k_min + self.mid_kernel_ratio * (k_max - k_min))
        k_mid = self._make_odd(k_mid)

        img_np = np.array(img)
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            img_np = np.stack([img_np] * 3, axis=-1)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        A = self._compute_attention(img_bgr)  # (H,W) float32

        blur_low = cv2.GaussianBlur(img_bgr, (k_min, k_min), 0)
        blur_mid = cv2.GaussianBlur(img_bgr, (k_mid, k_mid), 0)
        blur_high = cv2.GaussianBlur(img_bgr, (k_max, k_max), 0)

        w_low, w_mid, w_high = self._tri_weights(A)
        w_low3 = w_low[..., None]
        w_mid3 = w_mid[..., None]
        w_high3 = w_high[..., None]

        out = (blur_low.astype(np.float32) * w_low3 +
               blur_mid.astype(np.float32) * w_mid3 +
               blur_high.astype(np.float32) * w_high3)

        out = np.clip(out, 0, 255).astype(np.uint8)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)



class AttentionWeightedMultiBlurWithStats:

    def __init__(
        self,
        h=224,
        w=224,
        blur_kernel_size=41,
        min_kernel_size=3,
        mid_kernel_ratio=0.35,
        attn_smooth_ksize=9,
        grad_gamma=0.8,
        center_prior=0.15,
        center_sigma_ratio=0.45,
        softness=0.15,
        *args,
        **kwargs
    ):
        import numpy as np

        self.h = int(h)
        self.w = int(w)

        self.blur_kernel_size = self._make_odd(int(blur_kernel_size))
        self.min_kernel_size = self._make_odd(int(min_kernel_size))

        mid_k = int(round(self.blur_kernel_size * float(mid_kernel_ratio)))
        self.mid_kernel_size = self._make_odd(max(self.min_kernel_size, mid_k))

        self.attn_smooth_ksize = self._make_odd(int(attn_smooth_ksize))
        self.grad_gamma = float(grad_gamma)
        self.center_prior = float(center_prior)
        self.center_sigma_ratio = float(center_sigma_ratio)
        self.softness = float(softness)

        self._center_prior_map = self._build_center_prior_map(self.h, self.w)

    @staticmethod
    def _make_odd(k):
        k = int(k)
        if k <= 1:
            return 1
        return k if k % 2 == 1 else k + 1

    def _build_center_prior_map(self, h, w):
        import numpy as np

        yy, xx = np.mgrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        sigma = self.center_sigma_ratio * min(h, w)

        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        prior = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-6)
        return prior

    def _compute_attention(self, img_bgr):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        mag = np.sqrt(gx * gx + gy * gy)
        mag = mag / (mag.max() + 1e-6)
        mag = np.power(mag, self.grad_gamma).astype(np.float32)

        if self.attn_smooth_ksize > 1:
            mag = cv2.GaussianBlur(
                mag,
                (self.attn_smooth_ksize, self.attn_smooth_ksize),
                0
            )

        if self.center_prior > 0:
            A = (1.0 - self.center_prior) * mag + self.center_prior * self._center_prior_map
        else:
            A = mag

        A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        return A.astype(np.float32)

    def _compute_weights(self, attention):
        
        import numpy as np

        A = attention.astype(np.float32)

        centers = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        sigma = max(self.softness, 1e-4)

        weights = []
        for c in centers:
            w = np.exp(-((A - c) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
            weights.append(w)

        weights = np.stack(weights, axis=0)  # [3,H,W]
        weights = weights / (weights.sum(axis=0, keepdims=True) + 1e-6)

        return weights.astype(np.float32)

    def _compute_stats(self, attention, weights):
        import numpy as np

        A = attention.astype(np.float32)
        W = weights.astype(np.float32)

        attn_mean = float(A.mean())
        attn_std = float(A.std())

        entropy = -np.sum(W * np.log(W + 1e-6), axis=0)
        weight_entropy = float(entropy.mean())

        low_mean = float(W[0].mean())
        mid_mean = float(W[1].mean())
        high_mean = float(W[2].mean())

        stats = np.array(
            [attn_mean, attn_std, weight_entropy, low_mean, mid_mean, high_mean],
            dtype=np.float32
        )
        return stats

    @staticmethod
    def _bgr_to_pil(img_bgr):
        import cv2
        import numpy as np
        from PIL import Image

        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def _build_all(self, img):
        import cv2
        import numpy as np
        from PIL import Image

        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")

        img = img.convert("RGB")
        img_np = np.array(img)

        if img_np.ndim != 3 or img_np.shape[2] != 3:
            img_np = np.stack([img_np] * 3, axis=-1)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]
        if h != self.h or w != self.w:
            self.h, self.w = h, w
            self._center_prior_map = self._build_center_prior_map(h, w)

        attention = self._compute_attention(img_bgr)
        weights = self._compute_weights(attention)

        low_img = cv2.GaussianBlur(
            img_bgr,
            (self.min_kernel_size, self.min_kernel_size),
            0
        ).astype(np.float32)

        mid_img = cv2.GaussianBlur(
            img_bgr,
            (self.mid_kernel_size, self.mid_kernel_size),
            0
        ).astype(np.float32)

        high_img = cv2.GaussianBlur(
            img_bgr,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        ).astype(np.float32)

        bank = np.stack([low_img, mid_img, high_img], axis=0)  # [3,H,W,3]
        fused = (bank * weights[..., None]).sum(axis=0)

        stats = self._compute_stats(attention, weights)

        fused_pil = self._bgr_to_pil(fused)
        low_pil = self._bgr_to_pil(low_img)
        mid_pil = self._bgr_to_pil(mid_img)
        high_pil = self._bgr_to_pil(high_img)

        return fused_pil, stats, [low_pil, mid_pil, high_pil]

    def get_fused_and_stats(self, img):
        fused_pil, stats, _ = self._build_all(img)
        return fused_pil, stats

    def get_fused_stats_context_images(self, img):
        return self._build_all(img)

    def __call__(self, img, blur_kernel_size=None):
        fused_pil, _, _ = self._build_all(img)
        return fused_pil