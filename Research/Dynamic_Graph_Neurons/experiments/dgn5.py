"""
DGN-5  –  EdgeDrop: Stochastic Binary Graph (Graph Dropout Regularisation)
══════════════════════════════════════════════════════════════════════════════
Key idea: compute the standard top-K binary adjacency (as DGN1), but during
TRAINING randomly zero out each edge independently with probability p_drop.
At EVALUATION the full deterministic top-K adjacency is used unchanged.

This is the graph analogue of DropConnect / DropEdge:
    A_full  = causal_topk( x, K=8 )               binary  (B, T, T)
    mask    ~ Bernoulli(1 - p_drop)                per-edge  (B, T, T)
    A_drop  = A_full * mask              (during training only)

Why it might help:
    • Prevents the model from over-relying on a fixed set of K neighbours.
    • Acts as data augmentation over graph structures.
    • Reduces co-adaptation between neurons connected through the same edges.
    • Forces each neuron to function robustly with partial context, improving
      generalisation similarly to how Dropout improves feedforward networks.

Hyperparameters:
    K       = 8     (same as DGN1 for fair comparison)
    p_drop  = 0.5   (edge keep probability = 0.5 during training)

Connectivity criterion  : causal top-K cosine similarity  (+ Bernoulli drop)
Edge weights            : NONE
Learnable params        : gain (D,), bias (D,), log_mix, log_scale
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import causal_topk_adjacency, unweighted_aggregate

K_DEFAULT    = 8
P_DROP_TRAIN = 0.5   # probability of DROPPING an edge during training


class DGN5(nn.Module):
    """
    EdgeDrop DGN: top-K binary adjacency with stochastic edge masking at
    train time (deterministic at eval).

    Neurons learn robust processing under partial connectivity.
    """

    def __init__(self, d_model: int, K: int = K_DEFAULT, p_drop: float = P_DROP_TRAIN):
        super().__init__()
        self.K      = K
        self.p_drop = p_drop

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

        # ── 1. Base binary adjacency (top-K cosine, causal) ────────────────
        A = causal_topk_adjacency(x, self.K)                 # (B, T, T) binary

        # ── 2. EdgeDrop: stochastic masking during training ─────────────────
        if self.training and self.p_drop > 0:
            # Bernoulli mask: each edge kept with prob (1 - p_drop)
            # Only applied where A=1 (no harm masking already-0 positions)
            keep_mask = torch.bernoulli(
                torch.full_like(A, 1.0 - self.p_drop)
            )                                                  # (B, T, T)
            A = A * keep_mask                                  # stochastic A

            # Rescale to compensate for dropped edges (like inverted dropout)
            # Prevents the magnitude of the message from collapsing.
            A = A / (1.0 - self.p_drop + 1e-8)

        # ── 3. Unweighted message aggregation ──────────────────────────────
        msg = unweighted_aggregate(x, A)                     # (B, T, D)

        # ── 4. Neuron internal processing ──────────────────────────────────
        blended = mix * x + (1.0 - mix) * msg
        delta   = F.gelu(blended * self.gain + self.bias) * scale

        return delta
