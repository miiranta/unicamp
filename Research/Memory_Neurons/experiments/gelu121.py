"""GELU121 – Intra-Sequence Token Uniqueness Gate (Stateless).

THE CORE IDEA — CONTEXTUAL ISOLATION:
    All prior gates measure how a token deviates from HISTORY (cross-batch EMA).
    But "novel" could mean something much more immediate: this token is UNIQUE
    within its own sequence context — no other token in this sequence looks like it.
    
    A word that appears repeatedly in a passage (e.g. "the", "is") has HIGH
    similarity to itself at other positions → familiar within THIS context.
    
    A rare, unusual word that doesn't match any other token → LOW similarity
    to all others → uniquely novel → deserves amplification.

THE MECHANISM:
    For each token x[b,t], compute its maximum cosine similarity to any OTHER
    token in the same sequence:
    
        x_norm          = normalize(x, dim=-1)               (B, T, D)
        sim             = x_norm @ x_norm.T                  (B, T, T)
        sim.diag = -1   (zero out self-comparison)
        max_sim[b,t]    = max_s≠t  cos(x[b,t], x[b,s])     (B, T) ∈ [-1, 1]
        
        familiarity     = (max_sim + 1) / 2                  ∈ [0, 1]
        novelty         = 1 - familiarity                    ∈ [0, 1]
        gate            = 1 + alpha × novelty                ∈ [1, 1+alpha]
        output          = GELU(x × gate)

    UNIQUENESS = max cosine similarity → unique token → novel → amplify.
    
    For a sequence like ["the", "cat", "the", "mat"]:
    - "the" at pos 0: max cos = cos(emb_the, emb_the@pos2) ≈ 1.0 → familiar
    - "cat": max cos = max over {the, the, mat} ≈ low → novel
    - "mat": max cos = max over {the, cat, the} ≈ low → novel
    
    This is SEQUENCE-ADAPTIVE: context determines which tokens are novel.
    Rare words in a repetitive context get boosted. Common words in a novel
    context DON'T get boosted by being common cross-batch.

WHY THIS IS COMPLEMENTARY TO EMA-BASED GATES:
    EMA gates (gelu80) use CROSS-BATCH statistics: "has this pattern appeared before?"
    Uniqueness gate uses WITHIN-SEQUENCE contrast: "does this token match anything here?"
    
    These are orthogonal signals:
    - "the" (very common cross-batch, but seen 3 times in this sequence) → double suppression
    - Rare technical term (rare cross-batch, unique in this sequence) → double amplification
    - Common word appearing only once in a rare sequence → cross-batch familiar but within-seq novel

COMPUTATIONAL COST:
    O(B × T² × D) → for B=32, T=64, D=1024: ~134M ops per forward pass per layer.
    Reduced to O(B × T²) after normalization: 32 × 64² = 131K multiplications.
    Manageable — T is small (64).

STABILITY:
    gate ∈ [1, 1+alpha]. alpha = softplus(log_alpha) ≥ 0.
    Init log_alpha = 0 → alpha = softplus(0) ≈ 0.693. Max gate ≈ 1.69 initially.
    No EMA state: fully stateless, no reset_state needed.
    Self-similarity correctly masked to -1 (not max).

Params: log_alpha = 1 scalar.
State: none (fully stateless).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU121(nn.Module):
    """Intra-sequence token uniqueness gate: amplify tokens unique within their sequence context."""

    def __init__(self):
        super().__init__()
        # alpha = softplus(log_alpha) > 0: amplification strength
        # init 0 → alpha ≈ 0.693, gate in [1, 1.69]
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
        B, T, D = x.shape
        alpha = F.softplus(self.log_alpha)  # > 0

        # Normalize along feature dimension for cosine similarity
        x_norm = F.normalize(x, dim=-1)                         # (B, T, D)

        # Pairwise cosine similarity matrix
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))         # (B, T, T)

        # Causal mask: token t can only compare to PAST tokens t' < t.
        # Upper triangle (t' >= t) is masked out — no future information.
        # Also masks diagonal (self-comparison).
        # causal_mask[i, j] = True means sim[i,j] should be masked (j >= i)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0
        ).unsqueeze(0)                                           # (1, T, T)
        sim = sim.masked_fill(causal_mask, -2.0)                 # mask self + future

        # For position 0 (no past tokens), max over all-masked row gives -2.0.
        # Familiarity = (-2+1)/2 = -0.5 → novelty = 1.5 → clamp to 1 → gate = 1+alpha.
        # This is correct: position 0 is always maximally novel (no prior context).
        max_sim, _ = sim.max(dim=-1)                             # (B, T), in [-2, 1]
        max_sim = max_sim.clamp(min=-1.0)                        # clamp -2 sentinel to -1

        # Familiarity: 1 = identical to some other token, 0 = maximally different
        familiarity = (max_sim + 1.0) / 2.0                      # (B, T), in [0, 1]
        novelty     = 1.0 - familiarity                          # (B, T), in [0, 1]

        # Gate: no suppression (gate ≥ 1), only amplification for novel tokens
        gate = 1.0 + alpha * novelty.unsqueeze(-1)               # (B, T, 1)

        return self._gelu(x * gate)
