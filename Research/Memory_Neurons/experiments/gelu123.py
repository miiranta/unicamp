"""GELU123 – Tiny Learned Gating MLP (Gradient-Trained Per-Channel Modulation).

THE CORE IDEA — LETTING THE OPTIMIZER DECIDE WHAT'S "NOVEL":
    All prior experiments use HAND-CRAFTED novelty metrics:
        - Cosine similarity to EMA prototype
        - Per-channel z-score vs running statistics
        - Contrastive within-sequence comparison
    
    These define what "novel" means a priori, based on our assumptions.
    
    What if we let the MODEL DECIDE? A tiny neural network can learn ANY gate
    function from the input signal that the loss gradient finds useful.
    
    The network doesn't have to implement "novelty suppression" — it might learn
    something fundamentally different: sharpening relevant directions, amplifying
    syntactic vs semantic features differently, etc.

THE MECHANISM:
    A bottleneck MLP: D → k → D with sigmoid output.
    
        h = ReLU(W_1 x + b_1)       (B, T, k)     — projection to bottleneck
        gate = sigmoid(W_2 h + b_2)  (B, T, D)     — per-channel gate ∈ (0, 1)
        
        scale = 1 + alpha × gate     (B, T, D)     — gate in [1, 1+alpha]
        output = GELU(x × scale)
    
    The bottleneck forces the gate to be a low-rank function of x:
        "The gate for all D channels must be explained by k latent factors"
    
    This is rank-k: W_2 W_1 is a D×D matrix of effective rank k.
    With k=8: 8 latent "novelty dimensions" determine all D channel gates.
    
    WHY NOT FULL D×D?: Rank-D would have D² params — 1M per layer. Too large.
    WHY BOTTLENECK k=8?: D×k×2 = 16K per layer, 64K total — small overhead.

UNIQUENESS vs ALL PRIOR WORK:
    NO EMA STATE: This is purely feedforward, stateless.
    NO COSINE/Z-SCORE: No explicit statistical comparison.
    TASK-ADAPTIVE: The gate function directly minimizes cross-entropy loss.
    CAN AMPLIFY & SUPPRESS: Unlike EMA novelty (gate usually near 1),
        the MLP can learn asymmetric, non-linear patterns.
    
    Concretely: the MLP might learn that x activations in [2, 5] range should be
    boosted, while activations near -1 should pass unchanged — something no EMA
    could discover because EMA doesn't know about the downstream loss.

BOTTLENECK k = 8:
    8 latent factors to explain D=1024 gate dimensions.
    This is extremely compressed: 8 factors capture the gate signal.
    If the gate needs more expressivity, gradient will saturate and push naturally.
    k=8 was chosen to be minimal: if this works, could try k=16, k=32.

INITIALIZATION:
    W_1, W_2 initialized near zero → gate ≈ sigmoid(0) = 0.5 initially.
    log_alpha initialized to -1 → alpha ≈ 0.27 → initial scale ≈ 1.135.
    Small initialization → starts near identity transformation.

STABILITY:
    gate = sigmoid(·) ∈ (0, 1). scale = 1 + alpha*gate ∈ (1, 1+alpha).
    alpha = sigmoid(log_alpha) × max_alpha. Max amplification = 2× (max_alpha=1).
    Gradients flow cleanly through linear+ReLU+linear+sigmoid.

Params: D×k + k + k×D + D + log_alpha ≈ 2Dk + k + D + 1.
For D=1024, k=8: 2×1024×8 + 8 + 1024 + 1 = 17,425 per layer × 4 layers = 69,700.
State: none.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU123(nn.Module):
    """Tiny learned bottleneck gate D→k→D: gradient-trained per-channel modulation."""

    def __init__(self, d_ff: int = 1024, bottleneck: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(d_ff, bottleneck, bias=True)
        self.fc2 = nn.Linear(bottleneck, d_ff, bias=True)
        # alpha = sigmoid(log_alpha) × max_alpha ∈ (0, 1)
        # max possible scale = 1 + 1 = 2×
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))  # init small

        # Initialize weights near zero for stable startup
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

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
        alpha = torch.sigmoid(self.log_alpha)  # ∈ (0, 1), max amplification = 2×

        # Bottleneck gate: D → bottleneck → D, sigmoid output ∈ (0, 1)
        h    = F.relu(self.fc1(x))                 # (B, T, k)
        gate = torch.sigmoid(self.fc2(h))           # (B, T, D)

        # Scale: 1 + alpha × gate ∈ [1, 1+alpha]
        scale = 1.0 + alpha * gate                  # (B, T, D)

        return self._gelu(x * scale)
