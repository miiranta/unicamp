"""
DGN-3  –  Recurrent Looped Message Passing
══════════════════════════════════════════════════════════════════════════════
Key idea: information circulates through the graph in MULTIPLE ROUNDS (R=3).
After each round, the hidden states H change → the NEXT round's connectivity
is recomputed from the new states.

This creates genuine LOOPS in the information flow:
  round 0: X  → build A(X)  → msg → H_1
  round 1: H_1→ build A(H_1)→ msg → H_2   (different topology!)
  round 2: H_2→ build A(H_2)→ msg → H_3

Each round has its OWN set of internal neuron parameters (gain, bias, mix),
so the processing function evolves across rounds. An EMA "momentum" term
blends the current state with the update — preventing runaway amplification.

Innovation over DGN1: the graph is not static during processing; it re-forms
based on the current state of the neurons, enabling information to propagate
across transitive connections (A→B→C can reach C from A in 2 rounds even if
A and C are not directly connected).

Inspired by: message-passing GNNs, Hopfield networks, and biological
recurrent circuits where activity shapes connectivity.

Connectivity criterion  : causal top-K cosine similarity (recomputed each round)
Edge weights            : NONE
Learnable params (per-round, internal): gain_r (D,), bias_r (D,), log_mix_r
Global learnable        : log_momentum (blend between rounds)
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import causal_topk_adjacency, unweighted_aggregate

K_DEFAULT = 8
R_DEFAULT = 3   # number of message-passing rounds (loop depth)


class DGN3(nn.Module):
    def __init__(self, d_model: int, K: int = K_DEFAULT, R: int = R_DEFAULT):
        super().__init__()
        self.K = K
        self.R = R

        # ── Per-round internal neuron parameters ───────────────────────────
        # Each round gets its own gain / bias / mix — they learn to specialise.
        # e.g., round 0 might aggregate raw content; round 2 refines refined reps.
        self.gain    = nn.ParameterList([
            nn.Parameter(torch.ones(d_model))   for _ in range(R)])
        self.bias    = nn.ParameterList([
            nn.Parameter(torch.zeros(d_model))  for _ in range(R)])
        self.log_mix = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))     for _ in range(R)])

        # ── Global momentum: how much of the old state to keep each round ──
        # High momentum → smooth, stable propagation
        # Low momentum  → rapid replacement (aggressive updates)
        self.log_momentum = nn.Parameter(torch.tensor(1.0))   # sigmoid → ~0.73 init

        # ── Output scale ───────────────────────────────────────────────────
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, D)
        returns: (B, T, D) delta (to be added as residual)
        """
        momentum = torch.sigmoid(self.log_momentum)
        scale    = F.softplus(self.log_scale) + 0.01

        h = x.clone()   # running hidden state, updated each round

        for r in range(self.R):
            mix = torch.sigmoid(self.log_mix[r])

            # ── Recompute connectivity from CURRENT state ──────────────────
            # This is the "loop": topology changes based on evolving activations
            A   = causal_topk_adjacency(h, self.K)      # (B, T, T) binary — new each round
            msg = unweighted_aggregate(h, A)             # (B, T, D)

            # ── Neuron update (round-specific params) ─────────────────────
            blended = mix * h + (1.0 - mix) * msg                   # (B, T, D)
            h_new   = F.gelu(blended * self.gain[r] + self.bias[r]) # (B, T, D)

            # ── Momentum blend: stable integration across rounds ──────────
            h = momentum * h + (1.0 - momentum) * h_new

        # Return the CHANGE introduced by the looped processing
        return (h - x) * scale
