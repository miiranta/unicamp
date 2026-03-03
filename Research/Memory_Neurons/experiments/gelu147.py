"""
GELU147 — Sparse-threshold residual gate.

Insight from sparse coding theory: a signal can be decomposed into
a sparse (predictable/familiar) part and a non-sparse (novel) residual.

Soft thresholding at an EMA-adaptive threshold creates a sparse code:
    sparse_x_d = sign(x_d) * max(|x_d| - threshold_d, 0)

The proportion of energy NOT captured by the sparse code is the novel part:
    residual = x - sparse_x                          (B, T, D)
    residual_frac = ||residual||² / (||x||² + eps)  — fraction of energy unexplained
    gate = 1 + alpha * tanh(sigma * residual_frac)

The threshold is set to EMA_mean(|x_d|) per channel — on average half the
signal is sparsified; tokens with stronger/different patterns have higher residuals.

Key insight: familiar tokens "compress well" (small residual); novel tokens
have large residual energy that doesn't fit the learned sparse code basis.

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_abs_mean (D,) — per-channel EMA of |x_d| → adaptive threshold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU147(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

        self.register_buffer("_ema_abs", torch.ones(d_ff))  # EMA(|x_d|)
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_abs.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            self._ema_abs.copy_(x.detach().abs().mean(dim=(0, 1)))
            self._warmup = False
            return base

        # soft threshold at EMA(|x_d|)  — familiar channels pass, unusual residual exits
        threshold = self._ema_abs                                   # (D,) detached
        x_abs     = x.abs()
        sparse_x  = x.sign() * F.relu(x_abs - threshold)           # (B, T, D)
        residual  = x - sparse_x                                    # novel component

        # fraction of token energy in residual (scalar per token)
        res_energy = (residual ** 2).sum(dim=-1, keepdim=True)      # (B, T, 1)
        x_energy   = (x ** 2).sum(dim=-1, keepdim=True) + 1e-8
        res_frac   = res_energy / x_energy                          # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * res_frac)
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_abs.mul_(d).add_(x.detach().abs().mean(dim=(0,1)) * (1-d))

        return out
