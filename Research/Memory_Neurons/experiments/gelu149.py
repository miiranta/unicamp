"""
GELU149 — Channel-group z-score gate (pooled statistics per group).

gelu80: per-channel z-score — compute one z-score per channel (D=1024 scores)
gelu103: top-K z-score — only top 16 of 1024 channels

This variant divides the D channels into G groups of K channels each and
computes a POOLED (group-level) z-score:

    group_g_activation = mean(x_{g*K : (g+1)*K})   per token, per group
    group_g_z = (group_activation - ema_group_g_mean) / ema_group_g_std

    gate = 1 + alpha * tanh(sigma * mean_G(|group_z_g|))

WHY THIS MIGHT WORK BETTER THAN PER-CHANNEL (gelu80):
1. Each group tracks channels that interact together (the group acts as a "feature detector")
2. Pooling reduces noise: group mean is more stable than individual channel
3. G=32 statistics are simpler to track well (less overfitting to rare channels)
4. Captures inter-channel structure within each group

Group assignment is fixed (contiguous blocks), so groups get to specialize:
early channels in D_FF tend to represent different features from late channels.

WHY THIS MIGHT WORK BETTER THAN SCALAR (gelu80):
- Still has G=32 independent scores → per-group novelty discrimination
- Captures which FEATURE GROUPS are activating unusually

G=32 groups of K=32 channels each (requires D_FF divisible by G).

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_group_mean (G,), _ema_group_sq (G,)
"""

import torch
import torch.nn as nn


class GELU149(nn.Module):
    G = 32   # number of groups

    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.g = self.G
        self.k = d_ff // self.g      # channels per group
        assert d_ff % self.g == 0, f"D_FF={d_ff} must be divisible by G={self.g}"

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean", torch.zeros(self.g))
        self.register_buffer("_ema_sq",   torch.ones(self.g))
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        base = self._gelu(x)

        # reshape to (B, T, G, K) then mean over K
        xg = x.reshape(B, T, self.g, self.k)
        group_mean = xg.mean(dim=-1)    # (B, T, G)

        if self._warmup:
            bm = group_mean.detach().mean(dim=(0, 1))   # (G,)
            bs = (group_mean.detach() ** 2).mean(dim=(0, 1))
            self._ema_mean.copy_(bm)
            self._ema_sq.copy_(bs)
            self._warmup = False
            return base

        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()                          # (G,)
        z_group = (group_mean - self._ema_mean) / ema_std  # (B, T, G)
        surp    = z_group.abs().mean(dim=-1, keepdim=True)  # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp)
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            bm = group_mean.detach().mean(dim=(0, 1))
            bs = (group_mean.detach() ** 2).mean(dim=(0, 1))
            self._ema_mean.mul_(d).add_(bm * (1-d))
            self._ema_sq.mul_(d).add_(bs * (1-d))

        return out
