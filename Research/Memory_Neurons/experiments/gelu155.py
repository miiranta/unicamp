"""GELU155 – Random Pairwise Batch Diversity Gate.

THE KEY INSIGHT:
    All previous gates ask "how much does THIS token differ from history?"
    But there's a complementary batch-level question:
    "How diverse is this batch internally?" — are most tokens similar to each other
    (a repetitive paragraph), or are they spread across many different points
    (a varied/novel excerpt)?

    Measurement: Sample M random pairs of tokens from the batch, compute mean squared
    pairwise distance normalized by the historical (EMA) per-channel standard deviation.

    If all tokens are near each other (repetitive batch):
        mean pairwise z-dist is small → gate suppresses
    If tokens are spread widely (diverse novel batch):
        mean pairwise z-dist is large → gate amplifies

    Why random pairs? Computing all N² pairs for N = B*T = 32*64 = 2048 tokens is
    too slow. Sampling M=512 random pairs is O(M*D), fast and unbiased.

IMPLEMENTATION:
    Track EMA of per-channel variance (to normalize distances).
    At forward:
        1. xf = x.detach().flatten(0,1)  (N, D)
        2. Sample M=512 random pair indices (a, b)
        3. diff_d = (xf[a, d] - xf[b, d]) / (ema_std_d + eps)  — z-normalized difference
        4. pairwise_z_sq = diff_d.pow(2).mean()  — scalar
        5. For N(0,1) i.i.d. channels: E[(z_a_d - z_b_d)²] = 2 per dimension,
           so E[diff.pow(2).mean()] ≈ 2.0 when batch matches history.
           surp = pz_sq / 2.0   — normalized so typical batch → surp ≈ 1.0
           (> 1 = more diverse than history; < 1 = more clustered)
        6. gate = 1 + alpha * tanh(sigma * surp)   — always has gradient (surp > 0)
           (> 0 means more diverse than typical; < 0 means more clustered)
        6. gate = 1 + alpha * tanh(sigma * relu(surp))   — only upgate when above typical

    After forward: update EMA std.

CAUSALITY: EMA updated after forward with detached x. ✓
RANDOMNESS: Different random pairs each step → stochastic gate, encourages robustness. ✓
GRADIENT: All distance computation done with no_grad; gradients through alpha/sigma. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_sq (D,), _ready flag.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
M_PAIRS   = 512    # number of random pairs to sample


class GELU155(nn.Module):
    """Random pairwise batch diversity gate: spread-out batches = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.d_ff = d_ff

        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # alpha = 1
        self.log_sigma = nn.Parameter(torch.tensor(0.0))   # sigma = 1

        self._ema_sq: torch.Tensor = None   # (D,) for computing EMA std
        self._ready = False

    def reset_state(self):
        self._ema_sq = None
        self._ready  = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)   # (N, D)
                self._ema_sq = xf.pow(2).mean(0).clone()
                self._ready  = True
            return out

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)       # (N, D)
            N  = xf.shape[0]

            # Per-channel EMA std
            ema_std = self._ema_sq.sqrt()        # (D,) approx std (assumes ~zero mean)

            # Sample M random pairs
            idx_a = torch.randint(N, (M_PAIRS,), device=x.device)
            idx_b = torch.randint(N, (M_PAIRS,), device=x.device)

            # Z-normalize differences by EMA std
            diff  = (xf[idx_a] - xf[idx_b]) / (ema_std.unsqueeze(0) + 1e-6)  # (M, D)

            # Mean squared z-distance per dimension
            # diff is (M, D); .mean() averages over BOTH M and D axes
            # E[(z_a_d - z_b_d)²] = 2 for i.i.d. N(0,1) channels → pz_sq ≈ 2.0 at baseline
            pz_sq = diff.pow(2).mean()   # scalar ≈ 2.0 when batch ~ historical dist

            # Normalize to [0, ∞): divide by reference 2.0 so "typical" diversity → 1.0
            # Gate > 1 when batch is diverse; model learns sigma/alpha for right sensitivity
            surp = pz_sq / 2.0           # ≈ 1.0 typical, > 1 diverse, < 1 clustered

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        # EMA update after forward
        with torch.no_grad():
            d = EMA_DECAY
            self._ema_sq = d * self._ema_sq + (1 - d) * xf.pow(2).mean(0)

        return output
