"""
GELU142 — Count-based surprise: fraction of simultaneously surprised channels.

gelu80 measures average surprise across channels.  If 10% of channels have
z=10 and 90% have z=0, the mean |z| is 1.0 — same as 100% of channels at z=1.

But these are qualitatively different: the first is a SPECIFIC unusual pattern
(few specific channels activated); the second is diffuse noise.

This variant instead counts the FRACTION of channels that exceed a threshold:
    p_surp = mean_d(sigmoid(beta * (|z_d| - threshold)))  ∈ (0, 1)
    gate   = 1 + alpha * p_surp

where threshold is a global learnable scalar and beta controls softness.

Extra bonus: when p_surp is large (many channels surprised together), that's
a coherent novel signal.  When p_surp is small (few channels unusual), it's
either a specific pattern or noise.  The model learns to distinguish.

Alternative reading: p_surp is a probabilistic "vote" — each channel casts a
soft binary vote for "I'm surprised", and the gate = 1 + (fraction voting yes).

Params: log_alpha (scalar), log_beta (scalar), threshold_raw (scalar)
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU142(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha     = nn.Parameter(torch.tensor(0.0))
        self.log_beta      = nn.Parameter(torch.tensor(0.0))   # sharpness of threshold
        self.threshold_raw = nn.Parameter(torch.tensor(0.0))  # threshold (unconstrained)
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
        ema_std = ema_var.sqrt()
        z_abs   = (x - self._ema_mean).abs() / ema_std          # (B, T, D)

        alpha     = torch.exp(self.log_alpha)
        beta      = torch.exp(self.log_beta)
        threshold = F.softplus(self.threshold_raw)               # threshold > 0

        # soft fraction of channels above threshold
        p_surp = torch.sigmoid(beta * (z_abs - threshold)).mean(dim=-1, keepdim=True)  # (B, T, 1)
        gate   = 1.0 + alpha * p_surp
        out    = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
