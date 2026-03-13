"""GELU324 – Diagonal GLU Gate (Gated Linear Unit Variant).

GLU (Gated Linear Units) — Dauphin et al. 2017 — showed that learnable element-wise
gating significantly improves LM performance. The standard GLU is:
    GLU(x, W, V, b, c) = (xW + b) ⊙ σ(xV + c)

DIAGONAL VERSION: restrict W to diagonal (per-channel weights) to keep params small.
    gate_d = sigmoid(w_d * x_d + b_d)      — learned per-channel gate on input
    output = GELU(x) ⊙ 2 * gate_d          — scale so gate=1 at w=0, b=0 (σ(0)=0.5 → 2*0.5=1)

KEY PROPERTIES:
    - Fully differentiable: w_d and b_d get gradient through gate_d through output
    - Causally safe: gate is based on the current x_d only (no cross-position info)
    - Simple: only 2D params (w (D,) and b (D,))
    - Can learn "only activate channel d when x_d is positive/large/small"
    - Reduces to GELU when all w=0, b=0 (σ(0)=0.5, 2*0.5=1) — good init
    - No EMA state needed

CREATIVE TWIST: add a per-channel BASELINE bias b₀ (D,) to the output:
    output = GELU(x) * (2 * sigmoid(w * x + b)) + b₀  ? — no, keep simple

PARAMS: w (D,), b (D,)  — 2*D = 2048 for D=1024
STATE:  none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GELU324(nn.Module):
    """Diagonal GLU: GELU(x) * 2σ(w_d*x_d + b_d). Fully differentiable, no state."""

    def __init__(self, D_FF: int = 1024):
        super().__init__()
        D = D_FF
        # Init w≈0, b≈0 → gate ≈ 2*σ(0)=1 → near-identity at start
        self.w = nn.Parameter(torch.zeros(D))
        self.b = nn.Parameter(torch.zeros(D))

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self._gelu(x)
        # Per-channel learned gate on input: differentiable, no state
        gate = 2.0 * torch.sigmoid(self.w.view(1, 1, -1) * x + self.b.view(1, 1, -1))
        return out * gate
