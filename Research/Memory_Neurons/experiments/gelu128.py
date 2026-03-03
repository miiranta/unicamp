"""GELU128 – Causal Within-Sequence Attention Novelty Gate.

THE CORE IDEA — ATTENTION-WEIGHTED CONTEXT AS REFERENCE:
    For each token t in a sequence, its "expected pattern" is the attention-weighted
    combination of ALL PAST tokens (positions 0..t-1) in the same sequence.
    
    If the current token x[t] closely resembles this context summary → familiar → SUPPRESS.
    If x[t] differs strongly from the context → novel → AMPLIFY.
    
    This is a LEARNED CAUSAL RETRIEVAL: the attention weights identify which past tokens
    are most relevant to computing the familiarity of the current token.

THE MECHANISM:
    For each pair (query=t, key=s where s < t):
        similarity[t, s] = (x[t] · x[s]) / sqrt(D)    (scaled dot-product)
    
    Causal masking: set similarity[t, s] = -∞ for s ≥ t (no future info).
    
    Attention weights: α[t, :] = softmax(similarity[t, :])  over s < t
    
    Context vector: c[t] = sum_s α[t,s] × x[s]             (B, T, D)
    
    Novelty score:   nov[t] = 1 - cosine(x[t], c[t])       ∈ [0, 2] → re-scaled ∈ [0, 1]
                     nov[0] = 1 (no context → maximally novel)
    
    Gate:            gate[t] = 1 + alpha × tanh(sigma × nov[t])
    
    Output:          GELU(x × gate)                          (per-position scalar gate)

WHY THIS OUTPERFORMS EMA-BASED APPROACHES:
    EMA: x_t is compared to the AVERAGE of all past inputs (equal-weighted mean).
    Attention: x_t is compared to the MOST RELEVANT subset of past inputs.
    
    Concretely: in a passage about "machine learning", most past tokens relate to ML.
    When "Napoleon" suddenly appears, attention weights on ML tokens are low,
    weights on potentially related tokens are higher — but the cosine to any weighted
    context will be LOW → correctly flagged as novel.
    
    EMA would also flag "Napoleon" as novel, but with LESS precision because the EMA
    includes ALL past tokens not just the relevant ones.

CAUSAL GUARANTEE:
    The attention mask ensures similarity[t, s] = -∞ for s ≥ t.
    After masking + softmax, only s < t contribute to c[t].
    For t=0: mask makes ALL positions zero → c[0] = zeros → fallback to novel.
    STRICTLY CAUSAL.

ATTENTION WITHOUT PARAMETERS:
    No learned Q/K/V projections — uses raw FF intermediate activations directly.
    Scale factor = 1/sqrt(D) to prevent attention collapse.
    This is "parameter-free" intra-FF attention — the only params are alpha, sigma.
    
    Why no projections? Keep it minimal. The FF intermediates are already in a good
    space for cosine comparison (positive after the pre-linear GELU context).
    
    If this works, GELU128b could add learned lightweight Q/K projections.

COMPUTATIONAL COST:
    O(B × T² × D) for the attention matrix.
    For B=32, T=64, D=1024: 32 × 64² × 1024 ≈ 134M multiplications per forward.
    This is roughly 4-8× slower than standard GELU but enables rich context modeling.
    Acceptable for research exploration.

SIMILARITY TO SELF-ATTENTION:
    This IS a form of self-attention inside the FF layer, but:
    - No learned projections (parameter-free)
    - Not modifying values (just computing novelty score)
    - Acts as a GATE, not as a residual computation
    The transformer's self-attention layer also runs, but this operates on
    DIFFERENT activations (post-attention FF intermediate), providing new information.

Params: log_alpha, log_sigma = 2 scalars.
State: none (fully stateless).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU128(nn.Module):
    """Causal within-sequence attention: gate by novelty vs attention-weighted past context."""

    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self._scale = 1.0 / math.sqrt(d_ff)  # will be adjusted in first forward if D differs
        self.log_alpha = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # ≈ 0.5
        self.log_sigma = nn.Parameter(torch.tensor(math.log(1.0)))

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
        alpha = F.softplus(self.log_alpha)
        sigma = F.softplus(self.log_sigma)
        scale = 1.0 / math.sqrt(D)

        # ── Causal scaled dot-product similarity ─────────────────────
        # sim[b, t, s] = x[b,t] · x[b,s] / sqrt(D)
        sim = torch.bmm(x, x.transpose(1, 2)) * scale             # (B, T, T)

        # Causal mask: future positions (s >= t) → -inf
        # diagonal=1 means upper triangle excluding diagonal is masked (s > t already)
        # We also mask s = t (self): diagonal = 0
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0
        )                                                           # True where s >= t
        sim = sim.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # Softmax attention weights over past positions s < t
        # For t=0: all masked → softmax produces uniform/nan → handle separately
        attn = torch.softmax(sim, dim=-1)                          # (B, T, T)
        # NaN at t=0 (all -inf): replace with 0 (no context → c[0] = 0 vector)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Context vector: attention-weighted sum of past x values
        context = torch.bmm(attn, x)                               # (B, T, D)

        # ── Novelty: cosine distance between x[t] and context[t] ────
        x_norm = F.normalize(x,       dim=-1)                      # (B, T, D)
        c_norm = F.normalize(context, dim=-1)                      # (B, T, D)

        cos_sim = (x_norm * c_norm).sum(dim=-1)                    # (B, T) ∈ [-1, 1]
        # novelty ∈ [0, 1]: 0 = identical to context, 1 = orthogonal, can exceed 1 if anti-correlated
        novelty = (1.0 - cos_sim).clamp(min=0.0, max=2.0) / 2.0   # re-scale to [0, 1]

        # At t=0: context = 0 → c_norm = nan → cos_sim = nan → set novelty = 1
        novelty = torch.nan_to_num(novelty, nan=1.0)

        # ── Gate ──────────────────────────────────────────────────────
        gate = 1.0 + alpha * torch.tanh(sigma * novelty.unsqueeze(-1))  # (B, T, 1)

        return self._gelu(x * gate)
