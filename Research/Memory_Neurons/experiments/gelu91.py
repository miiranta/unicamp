"""GELU91 – Stateless Input-Output Coherence Gate.

CORE INSIGHT (COMPLETELY STATELESS — NO EMA NEEDED):
    When is GELU "doing something interesting"?
    When its output direction DIFFERS from its input direction.

    GELU is nearly linear for large positive x (GELU(x) ≈ x).
    GELU introduces nonlinearity near zero (GELU(0) = 0, GELU(-large) ≈ 0).

    High input-output cosine similarity: GELU is passing input nearly unchanged
    → "expected", linear, familiar-like activation.

    Low input-output cosine similarity: GELU is transforming input significantly
    → "unexpected", nonlinear, novel-like activation.

    We can use input-output coherence as an INSTANTANEOUS, STATELESS novelty signal!

MECHANISM:
    out  = GELU(x)                                           (B, T, D)
    cos_io = cosine(x, out)                                  (B, T)  ← per-token coherence

    coherence_novelty = 1 - cos_io ∈ [0, 2]                 (B, T)
    ← 0 when x and GELU(x) perfectly aligned (linear regime)
    ← 1 when orthogonal (GELU is doing something novel)
    ← 2 when anti-correlated (GELU completely reversed direction — extreme nonlinearity)

    scale = 1 + w × coherence_novelty                        (B, T, 1)
    result = out × scale

PROPERTIES:
    - ZERO state: no EMA, no prototypes, no stored tensors
    - Zero overhead: only requires computing 1 cosine per token
    - Differentiable: gradients flow through cos_io to x and out
    - Self-normalizing: when GELU passes linear input, no change; when nonlinear, amplified

WHEN DOES THIS HELP?
    Novel tokens tend to have unusual input patterns:
    → unusual inputs → GELU in nonlinear regime → low cos_io → amplified
    
    Familiar tokens tend to have large, nearly-linear inputs (repeated patterns get large weights):
    → large familiar inputs → GELU nearly linear → high cos_io → minimal scaling

STABILITY:
    scale ≥ 1: output is never suppressed, only amplified (or unchanged)
    This MIGHT seem problematic (no suppression?) but actually:
    - Familiar tokens (high cos_io): scale ≈ 1 → same as standard GELU
    - Novel tokens (low cos_io): scale > 1 → amplified
    The relative contrast between familiar and novel is increased.
    Standard GELU loss will push w to be small if amplification causes instability.

    Alternative: use scale = 1/(1 + w × cos_io) to suppress familiar instead:
        scale = 1 / (1 + w × relu(cos_io))   ← familiar suppressed, novel = 1
    We try the AMPLIFICATION form here (novel boosted), since suppression proved harder.

Params: log_w_raw = 1 scalar. State: none!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU91(nn.Module):
    """Stateless input-output coherence gate: amplify when GELU is nonlinear."""

    def __init__(self):
        super().__init__()
        self.log_w_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w   = F.softplus(self.log_w_raw)
        out = self._gelu(x)                                            # (B, T, D)

        # ── Input-output cosine similarity ──────────────────────────────
        # Both x and out might have same-sign channels → high cosine expected for large x
        x_norm   = F.normalize(x, dim=-1)                             # (B, T, D)
        out_norm = F.normalize(out.detach(), dim=-1)                   # detach out for gate only
        cos_io   = (x_norm * out_norm).sum(dim=-1)                    # (B, T) ∈ [-1, 1]

        # ── Coherence novelty: 1 - cos_io (0 = linear, 2 = anti-correlated) ──
        coherence_novelty = 1.0 - cos_io                              # (B, T) ∈ [0, 2]
        scale = 1.0 + w * coherence_novelty                           # (B, T) ≥ 1

        return out * scale.unsqueeze(-1)
