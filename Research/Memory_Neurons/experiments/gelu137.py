"""
GELU137 — Second-order z-score gate (surprise about the surprise level).

gelu80 measures: how much does token x deviate from EMA_mean per channel?

This variant asks a meta-question: is this amount of surprise itself unusual?

A channel that is NORMALLY highly variable (high |z_d| on average) is less
interesting when it shows another high-deviation token.  A channel that
normally sits quietly but suddenly spikes IS interesting.

Implementation:
    z_d        = (x_d - ema_mean_d) / ema_std_d          — 1st-order z-score
    z1_d       = |z_d|                                     — raw surprise
    ema_z1_d   = EMA(|z_d|)                               — expected surprise level
    ema_z1sq_d = EMA(z1_d^2)                              — expected squared surprise
    std_z1_d   = sqrt(ema_z1sq_d - ema_z1_d^2 + eps)     — std of surprise

    meta_z_d   = (z1_d - ema_z1_d) / (std_z1_d + eps)    — 2nd-order z-score
    surp       = mean_d(relu(meta_z_d))                   — token-level meta-surprise
                                                               (only + excursions matter)
    gate = 1 + alpha * tanh(sigma * surp)

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,),      ← for 1st-order z-score
        _ema_z1 (D,), _ema_z1sq (D,)       ← for 2nd-order (surprise about surprise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU137(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean",  torch.zeros(d_ff))
        self.register_buffer("_ema_sq",    torch.ones(d_ff))
        self.register_buffer("_ema_z1",    torch.ones(d_ff))   # expected |z|, init ≈ 1
        self.register_buffer("_ema_z1sq",  torch.ones(d_ff) * 2.0)
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._ema_z1.fill_(1.0)
        self._ema_z1sq.fill_(2.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            self._ema_mean.copy_(x.detach().mean(dim=(0, 1)))
            self._ema_sq.copy_((x.detach() ** 2).mean(dim=(0, 1)))
            self._warmup = False
            return base

        # 1st-order z-score
        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        z1 = (x.detach() - self._ema_mean).abs() / ema_std   # (B, T, D), no grad

        # 2nd-order z-score — meta-surprise
        z1_std = (self._ema_z1sq - self._ema_z1 ** 2).clamp(min=1e-6).sqrt()
        meta_z = (z1 - self._ema_z1) / (z1_std + 1e-6)       # (B, T, D)
        surp   = meta_z.clamp(min=0.0).mean(dim=-1, keepdim=True)  # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)

        # recompute z1 WITH grad for gate computation
        z1_grad = (x - self._ema_mean).abs() / ema_std
        gate  = 1.0 + alpha * torch.tanh(sigma * (z1_grad.mean(-1, keepdim=True) + surp))
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            bm  = x.detach().mean(dim=(0, 1))
            bsq = (x.detach() ** 2).mean(dim=(0, 1))
            bz1 = z1.mean(dim=(0, 1))
            bz1sq = (z1 ** 2).mean(dim=(0, 1))
            self._ema_mean.mul_(d).add_(bm  * (1-d))
            self._ema_sq.mul_(d).add_(bsq * (1-d))
            self._ema_z1.mul_(d).add_(bz1 * (1-d))
            self._ema_z1sq.mul_(d).add_(bz1sq * (1-d))

        return out
