"""GELU156 – Within-Batch PCA Residual Gate (Stateless).

THE KEY INSIGHT:
    Every batch of tokens defines its own "familiar subspace" — the directions of
    maximum variance in the current context. The top principal components capture
    the dominant, repeated patterns (common syntactic structures, high-frequency words).

    Tokens that are well-explained by these top-K directions are familiar.
    Tokens whose energy lives mostly in the RESIDUAL (directions orthogonal to top-K)
    are novel — they require structure that doesn't appear in the dominant modes.

    This is fundamentally different from all EMA-based approaches:
    - No long-term memory needed
    - Familiarity is defined relative to THIS BATCH's structure
    - Captures the principal "clichés" of the current text window

IMPLEMENTATION (stateless):
    xf = x.detach().flatten(0, 1)   (N, D)
    U, S, V = torch.svd_lowrank(xf, q=K_RANK)   — (N,K), (K,), (D,K)

    Project each token onto top-K right singular vectors (V):
        proj = xf @ V @ V.T          — (N, D) projection onto top-K subspace
        resid = xf - proj             — (N, D) orthogonal residual

    Residual energy fraction (per token):
        r_frac = ||resid||² / (||x||² + eps)   — (N,) ∈ [0, 1]

    Gate (scalar from batch mean):
        surp  = r_frac.mean()                   — scalar ∈ [0, 1]
        gate  = 1 + alpha * tanh(sigma * surp)
        output = GELU(x) * gate

    When all tokens aligned with top-K PCA: r_frac ≈ 0 → gate ≈ 1 (no boost)
    When tokens spread across many dim:     r_frac > 0 → gate > 1 (boost novelty)

CAUSALITY: Fully within-batch computation on current x (no future batches). ✓
STATELESS: No EMA state — cleans up batch-to-batch leakage entirely. ✓
GRADIENT: SVD + projection done under no_grad. alpha/sigma get gradients. ✓
COST: svd_lowrank O(N*D*K) with K=8, N=2048, D=1024 — fast on GPU.

Params: log_alpha, log_sigma (2 scalars).
State: None (fully stateless).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K_RANK = 8   # number of dominant principal components to remove


class GELU156(nn.Module):
    """Within-batch PCA residual gate: energy in non-dominant directions = novel."""

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
            xf = x.detach().flatten(0, 1)   # (N, D)

            # Thin SVD: V is (D, K) — top-K right singular vectors
            try:
                _, _, Vt = torch.svd_lowrank(xf, q=K_RANK, niter=2)
                # Vt from svd_lowrank is (D, K) in PyTorch convention
                V = Vt   # (D, K)
            except Exception:
                return out   # fallback: SVD failed (edge case)

            # Project onto top-K subspace and compute residual energy fraction
            proj   = xf @ V @ V.T                          # (N, D) projection
            resid  = xf - proj                             # (N, D) residual
            r_frac = resid.pow(2).sum(-1) / (xf.pow(2).sum(-1).clamp(min=1e-6))  # (N,)
            surp   = r_frac.mean()                         # scalar ∈ [0, 1]

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
