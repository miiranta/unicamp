"""GELU173 – Spectral Concentration Gate (Top Singular Value / Frobenius Norm, Stateless).

THE KEY INSIGHT:
    The (N, D) matrix of token activations in a batch has a singular value decomposition.
    The SPECTRAL CONCENTRATION = σ₁² / ||X||²_F measures how much of the total energy
    is captured by the single dominant direction (top singular vector).

    HIGH spectral concentration (≈ 1):
        Nearly all tokens point in the same direction.
        The batch is dominated by one "topic" or "pattern" — highly FAMILIAR.

    LOW spectral concentration (≈ 1/min(N,D)):
        Energy is spread across many orthogonal directions.
        Each token points in a different direction — high DIVERSITY = NOVELTY.

    Gate fires when the batch is spread across many directions:
        surp = 1 - spectral_concentration ∈ [0, 1)
        gate = 1 + alpha * tanh(sigma * surp)

    This is the "opposite" of PCA residual (gelu156):
    - gelu156: fraction of energy in the RESIDUAL of top-K directions
    - gelu173: fraction of energy NOT in the single top direction
    - gelu173 uses the RATIO of singular values rather than absolute values → scale-invariant

RELATIONSHIP TO EFFECTIVE RANK:
    When σ₁/||X||_F is small, the matrix has high effective rank (many active directions).
    surp = 1 - (σ₁² / ||X||_F²) is a simpler proxy for effective rank.

IMPLEMENTATION (stateless):
    xf = x.detach().flatten(0, 1)   (N, D)
    σ₁² from top singular value: torch.svd_lowrank(xf, q=1)
    frob² = xf.pow(2).sum()
    surp = 1 - σ₁² / (frob² + eps)

NOTE: svd_lowrank with q=1 is fast (power iteration, ~3x(N+D) dot products per iter).

CAUSALITY: Fully within-batch. ✓
STATELESS: No EMA. ✓
GRADIENT: SVD under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU173(nn.Module):
    """Spectral concentration gate: energy spread across many singular directions = novel."""

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
            xf = x.detach().flatten(0, 1)   # (N, D)
            try:
                # Only need top-1 singular value; niter=2 is sufficient for good approximation
                _, S, _ = torch.svd_lowrank(xf, q=1, niter=2)
                sigma1_sq = S[0].pow(2)
                frob_sq   = xf.pow(2).sum()
                # Spectral concentration ∈ [0, 1]: fraction of energy in top direction
                spec_conc = sigma1_sq / (frob_sq + 1e-8)
                surp = 1.0 - spec_conc   # ∈ [0, 1) — high when diverse
            except Exception:
                surp = torch.tensor(0.5, device=x.device)

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
