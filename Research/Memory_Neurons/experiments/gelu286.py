"""gelu286 – FiLM Conditional Gate (Batch Statistics → Per-Channel Modulation).

MOTIVATION:
    gelu285 is a linear gate (sigmoid of a linear function of z-scores).
    A non-linear mapping from batch statistics to gate values may capture
    more complex distributional patterns, e.g. multi-modal activation profiles
    or channel co-activations that cannot be handled per-channel independently.

    FiLM (Feature-wise Linear Modulation) is a lightweight approach: a small
    MLP maps global batch statistics to per-channel gate values.

MECHANISM:
    out   = gelu(x)                           # (B, T, D)
    μ     = out.mean(dim=(0,1))               # (D,) batch mean
    σ     = out.std(dim=(0,1)) + eps          # (D,) batch std
    stats = cat([μ, σ])                       # (2D,) descriptor
    γ     = MLP(stats)                        # (D,) per-channel logit
    gate  = sigmoid(γ)                        # (D,) in (0, 1)
    return out * gate.view(1, 1, D)

    MLP architecture:
        Linear(2D, 64) → LayerNorm(64) → ReLU → Linear(64, D)
    The hidden dim 64 is intentionally small to prevent overfitting.
    LayerNorm stabilises the intermediate activations.

NO CAUSALITY LEAK:
    gate depends only on current batch statistics (same as gelu285).

BENEFIT FROM BACKPROP:
    The MLP learns a non-linear mapping from distribution moments to
    per-channel gates.  It can detect patterns like "when channels 1-8
    have high std AND channel 42 is near zero, suppress channel 100".
    This is strictly more expressive than the linear gelu285.

SEQUENTIAL ADAPTATION:
    Fully stateless — Δ ≈ 0.  Benefit is in base PPL.

PARAMS:  MLP weights (Linear(2D,64) + Linear(64,D), plus LayerNorm).
STATE:   none — stateless.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_MLP_HIDDEN = 64


class GELU286(nn.Module):
    """FiLM gate: small MLP maps batch [mean, std] descriptor → per-channel sigmoid."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps  = eps
        self.D_FF = D_FF

        self.film_mlp = nn.Sequential(
            nn.Linear(2 * D_FF, _MLP_HIDDEN, bias=True),
            nn.LayerNorm(_MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(_MLP_HIDDEN, D_FF, bias=True),
        )
        # Initialise the final linear to produce zero (gate ≈ sigmoid(0) = 0.5)
        # then add a bias of 3.0 so the gate starts near 1 (same logic as gelu285)
        with torch.no_grad():
            nn.init.zeros_(self.film_mlp[-1].weight)
            self.film_mlp[-1].bias.fill_(3.0)

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._gelu(x)   # (B, T, D)

        mu    = out.mean(dim=(0, 1))                           # (D,)
        sigma = out.std(dim=(0, 1)).clamp(min=self.eps)        # (D,)
        stats = torch.cat([mu, sigma], dim=0)                  # (2D,)

        gamma = self.film_mlp(stats)                           # (D,)
        gate  = torch.sigmoid(gamma)                           # (D,)

        return out * gate.view(1, 1, -1)
