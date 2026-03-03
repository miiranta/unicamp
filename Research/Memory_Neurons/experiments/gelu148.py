"""
GELU148 — EMA velocity gate (how fast is the concept space changing?).

gelu80 measures surprise relative to a stable EMA mean.
gelu109 measured 2nd-order drift (how much is EMA itself changing).

This variant takes a different angle: it tracks the VELOCITY of the EMA
(its first-order time derivative) and uses that as the gate signal.

Fast-moving EMA → the model is in an unstable / changing regime → tokens need
stronger processing to handle the shifting context.
Slow-moving EMA → stable learned regime → familiar → suppress.

    velocity_d = |ema_mean_d(t) - ema_mean_d(t-1)|    — per-channel change magnitude
    gate = 1 + alpha * tanh(sigma * mean_d(velocity_d / ema_std_d))

Unlike gelu109 which used the acceleration, this uses the raw velocity (current
EMA minus its value at the PREVIOUS step), and normalizes it per-channel by the
running standard deviation for scale-invariance.

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,), _prev_mean (D,) — prev step EMA mean
"""

import torch
import torch.nn as nn


class GELU148(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean",  torch.zeros(d_ff))
        self.register_buffer("_ema_sq",    torch.ones(d_ff))
        self.register_buffer("_prev_mean", torch.zeros(d_ff))
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._prev_mean.zero_()
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            bm = x.detach().mean(dim=(0, 1))
            bs = (x.detach() ** 2).mean(dim=(0, 1))
            self._prev_mean.copy_(bm)    # prev_mean = current on first call
            self._ema_mean.copy_(bm)
            self._ema_sq.copy_(bs)
            self._warmup = False
            return base

        # per-channel velocity (current EMA vs previous EMA step)
        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        velocity = (self._ema_mean - self._prev_mean).abs() / (ema_std + 1e-6)   # (D,)

        # gate ~ mean of normalized velocity across channels
        surp = velocity.mean()   # scalar — how much is the concept space moving?

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp)  # broadcast (B, T, D)
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            bm = x.detach().mean(dim=(0, 1))
            bs = (x.detach() ** 2).mean(dim=(0, 1))
            self._prev_mean.copy_(self._ema_mean)
            self._ema_mean.mul_(d).add_(bm * (1-d))
            self._ema_sq.mul_(d).add_(bs * (1-d))

        return out
