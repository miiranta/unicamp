"""
GELU135 — Signed z-score gate (direction-aware novelty).

gelu80 uses |z| so it amplifies any large deviation regardless of sign.
This variant uses the signed z-score with separate learnable responses for
positive and negative deviations:

    gate = 1 + alpha_pos * tanh(sigma * max(z, 0))
               + alpha_neg * tanh(sigma * max(-z, 0))

This lets channels independently learn whether to amplify positive excursions,
negative excursions, or both.  Setting alpha_pos=alpha_neg recovers the gelu80
behaviour.

Params: log_alpha_pos (scalar), log_alpha_neg (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn


class GELU135(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha_pos = nn.Parameter(torch.tensor(0.0))
        self.log_alpha_neg = nn.Parameter(torch.tensor(0.0))
        self.log_sigma     = nn.Parameter(torch.tensor(0.0))
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
        z = (x - self._ema_mean) / ema_std           # (B, T, D)

        a_pos = torch.exp(self.log_alpha_pos)
        a_neg = torch.exp(self.log_alpha_neg)
        sigma = torch.exp(self.log_sigma)

        pos_term = a_pos * torch.tanh(sigma * torch.clamp(z,  min=0.0))
        neg_term = a_neg * torch.tanh(sigma * torch.clamp(-z, min=0.0))
        gate = 1.0 + pos_term + neg_term
        out  = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
