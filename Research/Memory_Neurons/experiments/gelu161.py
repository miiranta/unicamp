"""GELU161 – Batch Empirical Rank Extremeness Gate (Non-Parametric, Stateless).

THE KEY INSIGHT:
    Z-score approaches (gelu80 and variants) implicitly assume the activation
    distribution is approximately Gaussian. This means they characterize "extreme"
    values by σ distance from the mean. But neural network activations are
    often heavy-tailed, bimodal, or skewed — far from Gaussian.

    The empirical RANK is model-free. For each channel d:
        rank(x[n, d]) = position of x[n, d] when all N values in the batch
                        are sorted, normalized to [0, 1].

    A value near rank=0.5 is median → familiar, expected.
    A value near rank≈0 or rank≈1 is an outlier → novel, rare.

    Extremeness: e[n, d] = 2 * |rank[n, d] - 0.5|    ∈ [0, 1]
    = 0 when at the median;  = 1 when at min or max of the column.

    Mean extremeness across channels: surp[n] = mean_d e[n, d]
    Gate is based on the mean batch-level extremeness.

    This is distribution-free: works equally well for Gaussian, Laplace,
    bimodal, or any other distribution.

IMPLEMENTATION (stateless):
    xf = x.detach().flatten(0, 1)       (N, D)
    ranks = xf.argsort(0).argsort(0).float() / (N - 1)  (N, D) ∈ [0, 1]
    extremeness = 2 * (ranks - 0.5).abs()               (N, D) ∈ [0, 1]
    surp = extremeness.mean()                            scalar ∈ [0, 1]

    Baseline: for N uniform samples, mean extremeness = 0.5.
    gate = 1 + alpha * tanh(sigma * surp)

NOTE: Double-argsort on GPU is expensive for large N×D. Here N=B*T=2048, D=1024.
Two argsort calls on a (2048×1024) matrix is around 40M comparisons — acceptable on GPU.

CAUSALITY: Fully within-batch. ✓
STATELESS: No EMA state. ✓
GRADIENT: Double-argsort done with no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU161(nn.Module):
    """Non-parametric rank extremeness gate: outlier tokens in any distribution = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def reset_state(self):
        pass

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)        # (N, D)
            N   = xf.shape[0]
            # Double argsort → ranks ∈ {0, 1, ..., N-1}, normalized to [0, 1]
            ranks      = xf.argsort(0).argsort(0).float() / max(N - 1, 1)  # (N, D) ∈ [0,1]
            extremeness = 2.0 * (ranks - 0.5).abs()   # (N, D) ∈ [0, 1]
            surp       = extremeness.mean()            # scalar ∈ [0, 1]; baseline ≈ 0.5

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
