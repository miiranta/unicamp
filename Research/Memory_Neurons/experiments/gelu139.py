"""
GELU139 — Concentrated surprise gate via log1p(mean z²).

gelu80 uses: mean_d(|z_d|)  under tanh  → average absolute z-score

The average of |z| is insensitive to whether surprise is concentrated
(one channel at |z|=10) vs distributed (10 channels at |z|=1):
    mean(|z|) is the same for both cases.

log1p(mean(z²)) has a very different profile:
    - z² penalizes large deviations quadratically
    - mean(z²) = average squared deviation (like MSE)
    - log1p compresses the scale so tanh isn't needed (already bounded growth)
    - Concentrated surprise (z²=100) dominates over distributed (10×z²=1)
        mean_z²(concentrated) = 100/D  >>  mean_z²(distributed) = 10/D

Gate:
    surp = log1p(sigma * mean_d(z_d^2))        — (B, T, 1)
    gate = 1 + alpha * surp                    — no tanh since log1p already grows slowly

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn


class GELU139(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))   # small init: gate near 1 at start
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))   # start with small sigma
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean", torch.zeros(d_ff))
        self.register_buffer("_ema_sq",   torch.ones(d_ff))
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            self._ema_mean.copy_(x.detach().mean(dim=(0, 1)))
            self._ema_sq.copy_((x.detach() ** 2).mean(dim=(0, 1)))
            self._warmup = False
            return base

        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        # z² per channel (with gradient for gate)
        z_sq = ((x - self._ema_mean) ** 2) / ema_var         # (B, T, D)
        mean_z_sq = z_sq.mean(dim=-1, keepdim=True)          # (B, T, 1)

        sigma = torch.exp(self.log_sigma)
        alpha = torch.exp(self.log_alpha)
        surp  = torch.log1p(sigma * mean_z_sq)               # (B, T, 1)
        gate  = 1.0 + alpha * surp
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
