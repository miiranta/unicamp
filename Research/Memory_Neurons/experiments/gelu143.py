"""
GELU143 — Dual-timescale z-score with learnable blend.

gelu80 uses one EMA timescale (decay=0.99 ≈ 100-step memory).
Different types of novelty are detectable at different timescales:
    - Fast novelty  (decay=0.9,   ≈10 steps): sentence-level unusual patterns
    - Slow novelty  (decay=0.999, ≈1000 steps): topic/domain-level rarities

This variant maintains BOTH timescales and learns how to blend them:
    surp_fast = mean_d(|z_d_fast|)     — fast EMA z-score
    surp_slow = mean_d(|z_d_slow|)     — slow EMA z-score
    blend = sigmoid(logit_blend)       — learnable [0,1] mixture weight
    surp  = blend * surp_fast + (1-blend) * surp_slow
    gate  = 1 + alpha * tanh(sigma * surp)

If the fast timescale (sentence) dominates → model learns to track local context.
If the slow timescale (corpus) dominates → model learns to track globally rare patterns.
The blend itself becomes a learned inductive bias about what kind of novelty matters.

Params: log_alpha, log_sigma, logit_blend = 3 scalars
State:  _ema_mean_f (D,), _ema_sq_f (D,), _ema_mean_s (D,), _ema_sq_s (D,)
"""

import torch
import torch.nn as nn


class GELU143(nn.Module):
    DECAY_FAST = 0.9
    DECAY_SLOW = 0.999

    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma   = nn.Parameter(torch.tensor(0.0))
        self.logit_blend = nn.Parameter(torch.tensor(0.0))  # blend=0.5 initially
        self._gelu = nn.GELU()

        for suf in ("_f", "_s"):
            self.register_buffer(f"_ema_mean{suf}", torch.zeros(d_ff))
            self.register_buffer(f"_ema_sq{suf}",   torch.ones(d_ff))
        self._warmup = True

    def reset_state(self):
        for suf in ("_f", "_s"):
            getattr(self, f"_ema_mean{suf}").zero_()
            getattr(self, f"_ema_sq{suf}").fill_(1.0)
        self._warmup = True

    def _zscore(self, x, suf):
        mean = getattr(self, f"_ema_mean{suf}")
        sq   = getattr(self, f"_ema_sq{suf}")
        var  = (sq - mean ** 2).clamp(min=1e-6)
        std  = var.sqrt()
        return (x - mean).abs() / std                        # (B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            bm = x.detach().mean(dim=(0, 1))
            bs = (x.detach() ** 2).mean(dim=(0, 1))
            self._ema_mean_f.copy_(bm);  self._ema_sq_f.copy_(bs)
            self._ema_mean_s.copy_(bm);  self._ema_sq_s.copy_(bs)
            self._warmup = False
            return base

        surp_f = self._zscore(x, "_f").mean(dim=-1, keepdim=True)   # (B, T, 1)
        surp_s = self._zscore(x, "_s").mean(dim=-1, keepdim=True)

        blend = torch.sigmoid(self.logit_blend)
        surp  = blend * surp_f + (1.0 - blend) * surp_s

        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * surp)
        out   = base * gate

        with torch.no_grad():
            bm = x.detach().mean(dim=(0, 1))
            bs = (x.detach() ** 2).mean(dim=(0, 1))
            df, ds = self.DECAY_FAST, self.DECAY_SLOW
            self._ema_mean_f.mul_(df).add_(bm * (1-df))
            self._ema_sq_f.mul_(df).add_(bs * (1-df))
            self._ema_mean_s.mul_(ds).add_(bm * (1-ds))
            self._ema_sq_s.mul_(ds).add_(bs * (1-ds))

        return out
