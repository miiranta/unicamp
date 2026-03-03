"""GELU122 – Gradient-Trained Prototype Dictionary.

THE CORE IDEA — GRADIENT-OPTIMIZED FAMILIARITY REPRESENTATION:
    Prior prototype methods (gelu100, gelu12, gelu13) use EMA to update prototypes:
        proto[k] = d * proto[k] + (1-d) * x_mean  (momentum update, no gradient)
    
    EMA has a fundamental limitation: it's a WEIGHTED AVERAGE that converges to
    the mean of the input distribution. It can't learn that "the combination of
    channels [1, 5, 7] firing together = familiar" — only individual channel means.
    
    GRADIENT TRAINING solves this: the loss gradient directly tells each prototype
    "move toward the patterns that reduce training loss."
    
    Prototypes learn to position themselves to MAXIMIZE the loss reduction from
    the novelty suppression — they become principal representatives of the data,
    positioned by task performance, not simple statistics.

THE MECHANISM:
    K = 16 learnable prototype vectors P_k ∈ R^D (nn.Parameter, gradient-trained).
    
    For each input x:
        x_norm   = normalize(x, dim=-1)         (B, T, D)
        P_norm   = normalize(P, dim=-1)         (K, D)
        sim[b,t,k] = x_norm[b,t] · P_norm[k]  (B, T, K)
        max_sim  = max_k(sim)                  (B, T) ∈ [-1, 1]
        familiarity = (max_sim + 1) / 2        (B, T) ∈ [0, 1]
        novelty   = 1 - familiarity            (B, T)
        gate = 1 + alpha * novelty             (B, T, 1)
        output = GELU(x * gate)

WHY GRADIENT TRAINING OUTPERFORMS EMA:
    EMA updates: proto ← mean(x) — converges to cluster center of data
    Gradient: proto ← loss gradient — finds positions that MAXIMIZE novelty signal
    
    The gradient pushes prototypes toward directions that:
    1. ARE actually familiar (to maximize suppression of familiar patterns)
    2. HELP the language model predict (by identifying which patterns are uninformative)
    
    The model discovers that "common syntactic patterns" should be suppressed
    (their direction → prototypes) while "semantic content words" should be amplified.

WHY K = 16:
    K=4 (gelu100): too few to cover the prototypical distribution
    K=16: richer coverage of familiar-pattern space
    K=32+: diminishing returns as prototypes start overlapping
    
    The 16 × D parameters are small relative to model size,
    but provide K-dimensional "familiarity atlas."

PARAMETER COUNT:
    K × D = 16 × 1024 = 16,384 extra params per activation module.
    With 4 transformer layers = 65,536 total extra params (~1.6% overhead).

STABILITY:
    gate ∈ [1, 1+alpha]. alpha = softplus(log_alpha). Init log_alpha=0, alpha≈0.69.
    cos ∈ [-1, 1] → familiarity ∈ [0, 1] → novelty ∈ [0, 1] → gate ∈ [1, 1.69].
    Gradient flows through sim = x · P → P updated by task loss.
    No numerical issues (all operations bounded).

Params: K×D prototype matrix + log_alpha = K*D + 1 scalars.
State: none (fully stateless — no EMA).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU122(nn.Module):
    """Gradient-trained K=16 prototype dictionary: novelty = distance to nearest prototype."""

    def __init__(self, d_ff: int = 1024, n_proto: int = 16):
        super().__init__()
        # Learnable prototypes: K × D, initialized with small random values
        self.proto = nn.Parameter(torch.randn(n_proto, d_ff) * 0.02)
        # Amplification strength
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x
            * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * x.pow(3))
            ))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        alpha = F.softplus(self.log_alpha)  # > 0

        x_norm    = F.normalize(x,          dim=-1)  # (B, T, D)
        proto_norm = F.normalize(self.proto, dim=-1)  # (K, D)

        # Cosine similarity to each prototype: (B, T, K)
        sim = torch.einsum('btd,kd->btk', x_norm, proto_norm)

        # Maximum similarity across prototypes: (B, T)
        max_sim, _ = sim.max(dim=-1)

        # Familiarity ∈ [0, 1], novelty ∈ [0, 1]
        familiarity = (max_sim + 1.0) / 2.0
        novelty     = 1.0 - familiarity

        # Gate: amplify novel tokens (gate ≥ 1 always)
        gate = 1.0 + alpha * novelty.unsqueeze(-1)  # (B, T, 1)

        return self._gelu(x * gate)
