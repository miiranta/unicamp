"""GELU160 – Random Fourier Feature (RBF Kernel) Novelty Gate.

THE KEY INSIGHT:
    Linear distances (like L2 or cosine) in the D=1024 dimensional space have a
    known weakness: they are dominated by the largest-variance components. Even
    per-channel z-scores (gelu80) treat channels independently.

    A NONLINEAR distance measure can capture complex interactions between channels.
    The RBF (Radial Basis Function) kernel is k(x, y) = exp(-||x-y||²/(2σ²)), which
    equals 1 when x=y and decays to 0 for distant points.

    Bochner's theorem: the RBF kernel can be approximated by Random Fourier Features:
        phi(x) = sqrt(2/R) * cos(W^T x + b)   where W ~ N(0, I/bandwidth²), b ~ U[0, 2π]

    This maps each token to an R-dimensional feature vector phi(x[b,t]) ∈ R^R such that:
        <phi(x), phi(y)> ≈ k(x, y) = RBF similarity

    The batch mean feature vector phi_mean ≈ E[phi(x)] = the "familiar" kernel centroid.
    Tokens far from the centroid in kernel space: phi(x) is dissimilar to phi_mean.

    Nonlinearity of cosine features means cross-channel interactions ARE captured.
    A token that activates an unusual COMBINATION of channels will be novel even if
    each individual channel is near its mean.

    Novelty: 1 - <phi(x[b,t]), phi_mean> / (||phi_mean|| + eps)

IMPLEMENTATION:
    Fixed random W (D, R) and b (R,) — not gradient-tracked, NOT updated.
    EMA of phi_mean across batches — tracks "familiar" kernel centroid.

    Per forward:
        phi_b = cos(x @ W / bw + b) / sqrt(R)      — (B, T, R)
        nov_bt = 1 - (phi_b * ema_phi).sum(-1)      — (B, T) ∈ [-1, 1+]
        surp   = F.relu(nov_bt).mean()              — scalar
        gate   = 1 + alpha * tanh(sigma * surp)
        output = GELU(x) * gate

    After forward: ema_phi updated with batch mean phi.

BANDWIDTH:
    Bandwidth (bw) controls the length scale of the RBF kernel.
    bw = sqrt(D) ≈ 32 — "unit sphere" scaling appropriate for D=1024 with typical ||x|| ~ 10.

CAUSALITY: EMA updated after forward. ✓
GRADIENT: Feature computation under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_phi (R,), _W (D, R) [buffer, not param], _b (R,) [buffer].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

R_FEATURES = 64      # number of random Fourier features
BANDWIDTH  = 32.0    # RBF bandwidth ≈ sqrt(D)
EMA_DECAY  = 0.95


class GELU160(nn.Module):
    """Random Fourier Feature RBF novelty gate: nonlinear kernel distance from EMA centroid."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.d_ff = d_ff

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        # Fixed random projection (not trainable, not EMA-updated)
        W = torch.randn(d_ff, R_FEATURES) / BANDWIDTH
        b = torch.rand(R_FEATURES) * (2 * math.pi)
        self.register_buffer('W', W)       # (D, R)
        self.register_buffer('b', b)       # (R,)

        self._ema_phi: torch.Tensor = None   # (R,) EMA of phi_mean across batches
        self._ready = False

    def reset_state(self):
        self._ema_phi = None
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
            # Compute RFF features for all tokens: (B, T, R)
            # Use W and b from buffers (moved to correct device automatically)
            phi_raw = torch.cos(x.detach() @ self.W + self.b.view(1, 1, R_FEATURES))  # (B, T, R)
            phi_bt  = phi_raw / math.sqrt(R_FEATURES)                                  # normalize

            phi_mean_b = phi_bt.mean(dim=(0, 1))   # (R,) mean over batch + sequence

        if not self._ready:
            with torch.no_grad():
                self._ema_phi = phi_mean_b.clone()
                self._ready   = True
            return out

        with torch.no_grad():
            # EMA phi: the "familiar" kernel centroid
            ema_phi_n = F.normalize(self._ema_phi, dim=0)    # (R,) unit vector

            # Per-token kernel similarity to EMA centroid
            # phi_bt normalized per-token for cosine
            phi_n  = F.normalize(phi_bt.flatten(0, 1), dim=-1)   # (N, R)
            cos_bt = (phi_n * ema_phi_n.unsqueeze(0)).sum(-1)     # (N,) ∈ [-1, 1]

            # Novelty = 1 - cosine sim (relu: only count distances > 0)
            nov    = F.relu(1.0 - cos_bt).mean()                  # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * nov)
        output = out * gate

        # EMA update after forward
        with torch.no_grad():
            self._ema_phi = EMA_DECAY * self._ema_phi + (1 - EMA_DECAY) * phi_mean_b

        return output
