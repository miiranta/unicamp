"""
GELU133 — Robust per-channel z-score via EMA of absolute deviations (MAD).

Standard EMA-variance can be skewed by extreme outliers that inflate sigma and
under-gate future deviations.  This variant tracks EMA_mean and EMA_MAD
(mean absolute deviation from EMA_mean) for a more robust spread estimate.

    MAD_d  = EMA[ |x_d - EMA_mean_d| ]
    z_d    = (x_d - EMA_mean_d) / (1.4826 * MAD_d + eps)   # robust z-score
    gate   = 1 + alpha * tanh(sigma * |z_d|)
    out    = gelu(x) * gate

The constant 1.4826 makes MAD comparable to std for Gaussians.

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_mad (D,)
"""

import torch
import torch.nn as nn


CONSISTENCY_FACTOR = 1.4826   # makes MAD ≈ std for Gaussian data


class GELU133(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean", torch.zeros(d_ff))
        self.register_buffer("_ema_mad",  torch.ones(d_ff))  # init to 1 → std≈1.48
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_mad.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            bm = x.detach().mean(dim=(0, 1))
            self._ema_mean.copy_(bm)
            self._ema_mad.copy_((x.detach() - bm).abs().mean(dim=(0, 1)))
            self._warmup = False
            return base

        robust_std = CONSISTENCY_FACTOR * self._ema_mad + 1e-6
        z = (x - self._ema_mean) / robust_std        # (B, T, D)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * z.abs())
        out   = base * gate

        with torch.no_grad():
            d  = self._decay
            xd = x.detach()
            bm = xd.mean(dim=(0, 1))
            bm_abs = (xd - self._ema_mean).abs().mean(dim=(0, 1))
            self._ema_mean.mul_(d).add_(bm * (1 - d))
            self._ema_mad.mul_(d).add_(bm_abs * (1 - d))

        return out
