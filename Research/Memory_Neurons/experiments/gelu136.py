"""
GELU136 — Learned channel-weighting for z-score aggregation.

gelu80 uses: gate = 1 + alpha * tanh(sigma * mean_d(|z_d|))
            = equal weighting of all D channel deviations

This variant learns which channels reliably signal novelty:
    gate = 1 + alpha * tanh(sigma * sum_d(softmax(w)_d * |z_d|))

softmax(w) is a learned D-dim probability vector — a distribution over
channels that assigns higher weight to channels whose surprise is
most informative about token novelty.  Over training, the model learns
to focus on the most predictive channels rather than treating them equally.

Params: log_alpha (scalar), log_sigma (scalar), channel_logits (D,)
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU136(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha      = nn.Parameter(torch.tensor(0.0))
        self.log_sigma      = nn.Parameter(torch.tensor(0.0))
        self.channel_logits = nn.Parameter(torch.zeros(d_ff))   # → softmax = uniform init
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
        ema_std = ema_var.sqrt()          # (D,)
        z_abs   = (x - self._ema_mean).abs() / ema_std     # (B, T, D)

        # learned weighted average of |z| across channels
        chan_w = torch.softmax(self.channel_logits, dim=0)  # (D,) sums to 1
        surp   = (chan_w * z_abs).sum(dim=-1, keepdim=True) # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp)     # (B, T, 1) scalar per token
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
