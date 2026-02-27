"""
DGN-6  –  Progressive Neighbourhood Expansion (Growing K)
══════════════════════════════════════════════════════════════════════════════
Key idea: combine DGN3-style looped message passing with a geometrically
GROWING neighbourhood at each round.

    Round 0: K₀ = 4   → narrow local neighbourhood (close semantic neighbours)
    Round 1: K₁ = 8   → medium context
    Round 2: K₂ = 16  → broad context

Each round recomputes the adjacency from the current hidden state H and uses
a progressively larger K.  The topological expansion is strictly hierarchical:
early rounds select the most-similar neighbours; later rounds extend farther.

This contrasts with DGN3 (fixed K everywhere) by allowing:
    • Round 0 to sharpen local structure (small K = selective).
    • Round 2 to integrate long-range context (large K = inclusive).

Intuition: akin to receptive-field growth in convolutional networks — first
detect fine edges, then aggregate larger patterns.

Architecture:
    h = x
    for r in range(R):
        A_r   = causal_topk( h, K_r )       K_r grows with r
        msg   = unweighted_mean( h, A_r )
        h_new = GELU( (mix_r * h + (1-mix_r) * msg) * gain_r  + bias_r )
        h     = momentum * h + (1 - momentum) * h_new
    delta = (h - x) * scale

Connectivity criterion  : causal top-K cosine similarity  (K varies per round)
Edge weights            : NONE
Learnable params (per-round): gain_r (D,), bias_r (D,), log_mix_r
Global learnable         : log_momentum, log_scale
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import causal_topk_adjacency, unweighted_aggregate

# K grows geometrically: 4 → 8 → 16
K_SCHEDULE = [4, 8, 16]
R_DEFAULT  = len(K_SCHEDULE)


class DGN6(nn.Module):
    """
    Progressive Neighbourhood Expansion DGN.

    R looped rounds where K doubles each round.
    Topology is recomputed from the evolving hidden state at each round
    (same as DGN3), but the neighbourhood size grows progressively.
    """

    def __init__(self, d_model: int, k_schedule: list = None):
        super().__init__()
        if k_schedule is None:
            k_schedule = K_SCHEDULE
        self.k_schedule = k_schedule
        self.R          = len(k_schedule)

        # ── Per-round internal parameters ──────────────────────────────────
        self.gain    = nn.ParameterList([
            nn.Parameter(torch.ones(d_model))   for _ in range(self.R)])
        self.bias    = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model))  for _ in range(self.R)])
        self.log_mix = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))     for _ in range(self.R)])

        # ── Global parameters ───────────────────────────────────────────────
        # Momentum: how strongly the previous state is preserved each round.
        # Initialised high (~0.73) for stable propagation across rounds.
        self.log_momentum = nn.Parameter(torch.tensor(1.0))
        self.log_scale    = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        returns: (B, T, D) delta
        """
        momentum = torch.sigmoid(self.log_momentum)
        scale    = F.softplus(self.log_scale) + 0.01

        h = x.clone()   # running hidden state

        for r, K_r in enumerate(self.k_schedule):
            mix = torch.sigmoid(self.log_mix[r])

            # ── Recompute topology from CURRENT state with larger K ─────────
            A   = causal_topk_adjacency(h, K_r)          # (B, T, T) binary
            msg = unweighted_aggregate(h, A)              # (B, T, D)

            # ── Round-specific neuron update ───────────────────────────────
            blended = mix * h + (1.0 - mix) * msg
            h_new   = F.gelu(blended * self.gain[r] + self.bias[r])

            # ── Momentum blend ─────────────────────────────────────────────
            h = momentum * h + (1.0 - momentum) * h_new

        # Return the net change introduced by the expanding propagation
        return (h - x) * scale
