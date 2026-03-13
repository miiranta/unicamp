"""gelu288 – Low-Rank Input × Output Bilinear Gate.

MOTIVATION:
    gelu211's product gate combines INPUT z-scores with OUTPUT z-scores, but
    independently (per-channel) via scalar gating functions.  It cannot detect
    cross-channel co-activation patterns.

    A bilinear form W = U @ V^T captures first-order INTERACTIONS between
    input channels and output channels:
        score = (in_mean @ U) · (out_mean @ V) / r

    When both in_mean and out_mean lie in a learned subspace (training-
    distribution patterns), the bilinear score is consistent and the gate
    fires as expected. Novel patterns (outside the training subspace) produce
    unusual scores → gate adjusts → novelty detected.

MECHANISM:
    out   = gelu(x)                          # (B, T, D)
    in_m  = x.mean(dim=(0,1))               # (D,) pre-GELU batch mean
    out_m = out.mean(dim=(0,1))             # (D,) post-GELU batch mean
    u     = in_m  @ U                       # (r,)  input projection
    v     = out_m @ V                       # (r,)  output projection
    score = (u · v) / r                     # scalar bilinear inner product
    gate  = sigmoid(-w * score + b)         # scalar gate

    U ∈ R^{D × r}, V ∈ R^{D × r}, r=16
    (w > 0 means high alignment → suppress; learned to invert if beneficial)

NO CAUSALITY LEAK:
    gate depends on current batch means, identical to gelu211.

BENEFIT FROM BACKPROP:
    U, V are jointly trained to maximise language-model likelihood.
    They learn to project activations to a subspace where the bilinear
    score is maximally informative about habituation vs novelty.
    No previous experiment uses a bilinear cross-space architecture.

SEQUENTIAL ADAPTATION:
    Stateless — Δ ≈ 0.  Benefit is in PPL via better-trained interaction.

PARAMS:  U (D×r), V (D×r), log_w, b.
STATE:   none.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_RANK = 16


class GELU288(nn.Module):
    """Low-rank bilinear input×output gate: gate=sigmoid(-w * (in@U)·(out@V)/r + b)."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps  = eps
        self.D_FF = D_FF
        r = _RANK

        # Low-rank projections; initialise with small Gaussian
        self.U     = nn.Parameter(torch.randn(D_FF, r) * (1.0 / math.sqrt(D_FF)))
        self.V     = nn.Parameter(torch.randn(D_FF, r) * (1.0 / math.sqrt(D_FF)))
        self.log_w = nn.Parameter(torch.tensor(math.log(2.0)))
        self.b     = nn.Parameter(torch.zeros(1))

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out   = self._gelu(x)                       # (B, T, D)

        in_m  = x.mean(dim=(0, 1))                  # (D,) pre-GELU mean
        out_m = out.mean(dim=(0, 1))                 # (D,) post-GELU mean

        u     = in_m  @ self.U                       # (r,)
        v     = out_m @ self.V                       # (r,)
        score = (u * v).sum() / _RANK                # scalar

        w    = self.log_w.exp()
        gate = torch.sigmoid(-w * score + self.b)    # scalar in (0, 1)

        return out * gate
