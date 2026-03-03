"""
GELU144 — Soft top-K z-score gate (sparse surprise aggregation).

gelu80 averages |z_d| over ALL D channels.  With D=1024, a genuine surprise
signal in K=16 channels is diluted by 1024-16=1008 channels with |z|≈0.

    mean_all(|z|) ≈ K/D * z_rare + (1-K/D) * z_common
                 ≈ (16/1024) * 10 + (1008/1024) * 0 ≈ 0.16

Compare to concentrating on the top-K surprising channels:
    mean_topK(|z|) ≈ z_rare ≈ 10

This variant selects the top-K channel z-scores per token and aggregates only those.
This is especially powerful when novelty is LOCALIZED to a small subset of channels
(a specific semantic pattern activating a specific feature detector).

Gate:
    z_abs = |z_d|                    (B, T, D) per-channel z-score magnitudes
    top_k = topk(z_abs, k=K, dim=-1).values   (B, T, K) — differentiable!
    surp  = mean(top_k, dim=-1)      (B, T)
    gate  = 1 + alpha * tanh(sigma * surp)

torch.topk is differentiable w.r.t. its input (straight-through), so gradient
flows from top_k → z_abs → x → the transformer parameters.

K is a fixed hyperparameter (K=32).

Params: log_alpha (scalar), log_sigma (scalar)
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn


class GELU144(nn.Module):
    K = 32

    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
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
        z_abs   = (x - self._ema_mean).abs() / ema_std    # (B, T, D)

        # top-K z-scores per token — differentiable
        k = min(self.K, self.d_ff)
        top_vals = torch.topk(z_abs, k=k, dim=-1).values   # (B, T, K)
        surp     = top_vals.mean(dim=-1, keepdim=True)      # (B, T, 1)

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp)
        out   = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
