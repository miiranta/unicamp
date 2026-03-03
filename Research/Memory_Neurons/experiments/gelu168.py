"""GELU168 – Per-Channel Batch Gini Coefficient Gate (Stateless).

THE KEY INSIGHT:
    Per-channel z-score (gelu80) asks: "how large is the current activation
    relative to the typical activation for this channel?"

    The GINI COEFFICIENT asks a different question: "how UNEQUALLY distributed
    are the activations across the batch for this channel?"

    A channel with Gini ≈ 0: all tokens activate it similarly, evenly distributed.
        → Familiar: this channel fires uniformly — it's not discriminating.
    A channel with Gini ≈ 1: a tiny fraction of tokens dominates activation.
        → Novel: only rare, highly specific tokens trigger large activations here.

    The Gini coefficient captures SELECTIVITY — a highly selective channel
    (only fires for rare tokens) carries more information than a non-selective one.

    This is inspired by the neuroscience finding that highly selective neurons
    (face neurons, place cells) fire for novel/specific stimuli.

    Key difference from IDF/sparsity approaches (gelu92, gelu115):
    - Those measure cross-channel sparsity: which channels fire for this token?
    - This measures within-channel sparsity: which tokens fire strongly for this channel?

FORMULA:
    For each channel d (N token activations, ascending sorted absolute values):
        gini_d = (2 * Σ_n(n * |x_{sorted_n,d}|) / (N * Σ_n(|x_{sorted_n,d}|))) - (N+1)/N

    Range: 0 (perfect equality) to (N-1)/N ≈ 1 (maximum inequality).

    surp = mean_d(gini_d)   — average channel selectivity
    gate = 1 + alpha * tanh(sigma * surp)

IMPLEMENTATION (stateless):
    abs_xf = xf.abs()                            (N, D)
    sorted_abs = abs_xf.sort(0).values           (N, D) ascending
    rnk = arange(1, N+1, dtype=float, device)    (N,)
    gini_d = (2 * (rnk * sorted_abs).sum(0)) / (N * sorted_abs.sum(0).clamp(1e-8)) - (N+1)/N

CAUSALITY: Fully within-batch. ✓
STATELESS: No EMA. ✓
GRADIENT: Gini computation under no_grad; alpha/sigma get gradients.

Params: log_alpha, log_sigma (2 scalars).
State: None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU168(nn.Module):
    """Per-channel batch Gini coefficient gate: selective channels = novel stimulus."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def reset_state(self):
        pass   # fully stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        with torch.no_grad():
            xf   = x.detach().flatten(0, 1)        # (N, D)
            N    = xf.shape[0]

            abs_xf     = xf.abs()                  # (N, D)
            sorted_abs = abs_xf.sort(0).values     # (N, D) ascending, dim=0

            rnk = torch.arange(1, N + 1, dtype=sorted_abs.dtype, device=x.device).unsqueeze(1)  # (N, 1)

            col_sum = sorted_abs.sum(0).clamp(min=1e-8)              # (D,)
            gini_d  = (2.0 * (rnk * sorted_abs).sum(0)) / (N * col_sum) - (N + 1.0) / N  # (D,)
            surp    = gini_d.mean()   # scalar ∈ [0, (N-1)/N ≈ 1]

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
