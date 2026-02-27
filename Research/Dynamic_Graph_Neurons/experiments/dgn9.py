"""
DGN-9  –  Adaptive Token Gate (Self-Regulated DGN)
══════════════════════════════════════════════════════════════════════════════
Key idea: not every token needs the same amount of graph-mediated update.
A token that already carries sufficient context (e.g., the end of a long
coherent phrase) may benefit from a SMALL delta; a token in a novel or
ambiguous position may need a LARGE update.

We implement this as a PER-TOKEN learned gate:

    gate  = sigmoid( x @ w_gate + b_gate )    ∈ (0,1)  shape (B, T, 1)

The gate is applied to SCALE the final delta produced by the standard DGN1
mechanism:

    A       = causal_topk( x, K=8 )
    msg     = unweighted_mean( x, A )
    blended = mix * x + (1-mix) * msg
    delta_raw = GELU( blended * gain + bias ) * scale
    delta   = gate * delta_raw                    ← adaptive scaling

The gate parameters (w_gate, b_gate) are a single (D → 1) linear projection
— very cheap (D+1 extra params) but rich enough to condition on the full
hidden state.

Motivation:
    • A token whose hidden state is already "informative" (large norm, clear
      direction) gets a low gate and preserves its representation.
    • A token with an uncertain or low-energy state gets a high gate and
      receives a stronger update from its graph neighbourhood.

    This is related to mixture-of-experts gating and highway networks, but
    applied to the GRAPH update magnitude rather than layer connectivity.

    Unlike fixed scale (DGN1's scalar log_scale), the gate is both per-token
    AND input-dependent, allowing dynamic allocation of graph bandwidth.

Connectivity criterion  : causal top-K cosine similarity (binary, K=8)
Learnable params        : w_gate (D,), b_gate scalar, gain (D,), bias (D,),
                          log_mix, log_scale
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import causal_topk_adjacency, unweighted_aggregate

K_DEFAULT = 8


class DGN9(nn.Module):
    """
    DGN with a learned per-token gate that adaptively scales the graph delta.

    Gate is a single linear D→1 projection conditioned on the token's own state.
    High gate → token solicits strong neighbourhood update.
    Low gate  → token is self-sufficient, suppresses incoming update.
    """

    def __init__(self, d_model: int, K: int = K_DEFAULT):
        super().__init__()
        self.K = K

        # ── Adaptive gate (input-dependent, per-token scalar) ──────────────
        # w_gate: (D,) vector — dot with x gives raw gate logit
        self.w_gate = nn.Parameter(torch.zeros(d_model))
        self.b_gate = nn.Parameter(torch.tensor(0.0))        # scalar bias

        # ── Internal neuron parameters (same as DGN1) ─────────────────────
        self.gain      = nn.Parameter(torch.ones(d_model))
        self.bias      = nn.Parameter(torch.zeros(d_model))
        self.log_mix   = nn.Parameter(torch.tensor(0.0))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        returns: (B, T, D) delta
        """
        mix   = torch.sigmoid(self.log_mix)
        scale = F.softplus(self.log_scale) + 0.01

        # ── 1. Per-token adaptive gate ─────────────────────────────────────
        # gate: (B, T, 1)  — how strongly should graph update be applied?
        gate_logit = (x * self.w_gate).sum(dim=-1, keepdim=True) + self.b_gate
        gate       = torch.sigmoid(gate_logit)               # (B, T, 1)

        # ── 2. Binary adjacency (top-K cosine, causal) ─────────────────────
        A   = causal_topk_adjacency(x, self.K)               # (B, T, T) binary

        # ── 3. Unweighted message aggregation ──────────────────────────────
        msg = unweighted_aggregate(x, A)                     # (B, T, D)

        # ── 4. Standard DGN1-style update ──────────────────────────────────
        blended   = mix * x + (1.0 - mix) * msg
        delta_raw = F.gelu(blended * self.gain + self.bias) * scale  # (B, T, D)

        # ── 5. Gate-modulated output ────────────────────────────────────────
        # Each token scales its own delta independently
        delta = gate * delta_raw                             # (B, T, D)

        return delta
