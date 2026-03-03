"""GELU162 – Within-Sequence Position Diversity Gate (Stateless).

THE KEY INSIGHT:
    All batch-level novelty measures (gelu80, gelu138, etc.) ask:
    "Is this token different from other tokens in the batch?"

    But there's a complementary per-sequence question:
    "Is this sequence INTERNALLY diverse — do its positions cover different parts
    of activation space?"

    A REPETITIVE sequence (e.g., "the the the the") has low position diversity:
    all T positions land near the same point in D-dimensional space.

    A COMPLEX sequence (varied vocabulary, topic transitions) has high position
    diversity: consecutive positions cover very different regions.

    Measurement: For each batch item b, compute the mean pairwise distance between
    all T position vectors. Like an intra-sequence "spread."

    Because we want a per-batch scalar gate (not per-token), we average over B.

IMPLEMENTATION (stateless):
    Mean pairwise L2 distance (approximated via variance):
        Var_T(x[b, :, d]) = mean_t(x[b,t,d]²) - (mean_t(x[b,t,d]))²  per (b,d)
        pos_div_b = sum_d(Var_T(x[b, :, d])) / D — total variance across positions

    This equals (1/T²) * sum_{s<t} ||x_s - x_t||² / D (a variance decomposition).
    High pos_div → positions are spread out; low pos_div → positions are similar.

    Normalize by EMA of pos_div to get relative novelty:
        surp = pos_div.mean(B) / (ema_pos_div + eps)  — >1 when this batch is more
                                                         diverse than typical

    gate = 1 + alpha * tanh(sigma * relu(surp - 1.0))   — only boost when above history

STATE: EMA of pos_div to normalize (so the gate doesn't just fire on scale differences).

CAUSALITY: EMA updated after forward. ✓
GRADIENT: All computation under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_div (scalar EMA of typical position diversity).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9


class GELU162(nn.Module):
    """Within-sequence position diversity gate: spread of T positions in D-space = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_div: torch.Tensor = None   # scalar EMA of typical position diversity
        self._ready = False

    def reset_state(self):
        self._ema_div = None
        self._ready   = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        with torch.no_grad():
            x_d   = x.detach()
            # Variance across T positions, per (B, D)
            mu_t  = x_d.mean(1)                             # (B, D)
            sq_t  = x_d.pow(2).mean(1)                     # (B, D)
            var_t = (sq_t - mu_t.pow(2)).clamp(min=0)      # (B, D)
            # Mean diversity per sequence = total variance across D / D
            pos_div = var_t.mean(-1)                         # (B,) scalar per seq
            pos_div_mean = pos_div.mean()                   # scalar

        if not self._ready:
            with torch.no_grad():
                self._ema_div = pos_div_mean.clone()
                self._ready   = True
            return out

        with torch.no_grad():
            # Relative diversity (signed): >1 = more diverse than typical, <1 = more clustered
            # surp > 0 when diverse (boost), surp < 0 when clustered (suppress)
            surp = pos_div_mean / (self._ema_div + 1e-8) - 1.0   # signed scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        # EMA update after forward
        with torch.no_grad():
            self._ema_div = EMA_DECAY * self._ema_div + (1 - EMA_DECAY) * pos_div_mean

        return output
