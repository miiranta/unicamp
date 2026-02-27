"""GELU94 – Softmax-Temperature Channel Competition (Learned Sparsity via Temperature).

MOTIVATION:
    All previous gate mechanisms use pointwise operations on individual channels.
    What if we use CHANNEL COMPETITION — channels compete for the right to activate?

    In competitive learning (Rumelhart & Zipser, 1985; Kohonen SOM), units compete
    and the most active unit "wins" — others are suppressed. This creates:
    - Sparse, distributed representations
    - Automatic basis discovery
    - Competitive inhibition of redundant features

THE MECHANISM:
    out = GELU(x)                                          (B, T, D)

    # Normalize to unit sphere (remove magnitude, keep direction)
    out_norm = out / (rms(out) + eps)                      (B, T, D) per-token RMS norm

    # Learnable temperature softmax: determines how competitive the channels are
    # temperature τ_comp controls sparsity:
    #   τ_comp → 0: winner-take-all (only top channel is 1, rest ~0)
    #   τ_comp → ∞: uniform (all channels equal = identity)
    softweights = softmax(out / τ_comp, dim=-1)            (B, T, D)  sums to 1

    # "Competition weight": how much did channel d "win"?
    # Scale so that uniform baseline maps to 1: weights × D
    comp = softweights × D                                 (B, T, D): uniform = 1, winner >> 1

    # Blend toward identity
    gate_d = α × comp + (1 - α)                           (B, T, D)
    gate_d = gate_d / mean(gate_d)                        ← normalize energy

    result = out × gate_d

WHY THIS IS NOVEL:
    This is the ONLY gate where channel competition is based on CURRENT ACTIVATIONS
    rather than historical statistics. No EMA, no z-scores.

    The gate is a soft version of winner-take-all:
    - Channels with large GELU activations WIN the competition → amplified further
    - Channels with small GELU activations LOSE → suppressed
    - Creates HIGHLY SPARSE output representations

    But GELU already does this (suppresses negative inputs)!
    The difference: softmax operates over POSITIVE channels too.
    Among the channels that GELU activates, only the MOST ACTIVE ones dominate.
    This is an additional layer of competition WITHIN the active channels.

RELATIONSHIP TO SPARSE CODING:
    This is soft "K-winner take all" (KWTA). For large D=1024 with τ_comp=1:
    - A token that has 10 strongly-active channels: those 10 get high softmax weight
    - 1014 weakly-active channels: near-zero softmax weight
    - After × D: 10 channels get weight ≈ 100, 1014 channels get weight ≈ 0
    - Blend with identity: 10 channels get gate ≈ (α×100 + 1-α), others ≈ (0+1-α) ≈ 1-α

    This creates EXTREME SPARSIFICATION of the representation! Only the top-activated
    channels "survive" with significant weight. The rest are nearly zeroed out.

    With α=0.1 and energy preservation (renorm): gentle version that maintains diversity.

TEMPERATURE LEARNING:
    τ_comp is learnable. If large: uniform (no competition).
    If small: very sparse (winner-take-all).
    The model learns the optimal sparsity level.

Params: log_temp_raw, log_alpha_raw = 2 scalars. State: none!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU94(nn.Module):
    """Softmax-temperature channel competition: KWTA-style amplification of dominant channels."""

    def __init__(self, init_temp: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Temperature τ_comp > 0: softplus ensures positivity
        self.log_temp_raw  = nn.Parameter(torch.tensor(math.log(math.exp(init_temp) - 1.0)))
        # Alpha: blend strength ∈ (0, 1)
        self.log_alpha_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 init

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        tau_comp = F.softplus(self.log_temp_raw)           # > 0, in graph for gradient
        alpha    = torch.sigmoid(self.log_alpha_raw)       # ∈ (0,1), in graph

        out = self._gelu(x)   # (B, T, D)

        # ── Per-token RMS normalize for fair competition ─────────────────
        rms    = out.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        out_n  = out / rms                                  # (B, T, D) unit 2-norm per token

        # ── Softmax competition at learned temperature ───────────────────
        # Large τ → uniform weights; small τ → winner-take-all
        softw  = torch.softmax(out_n / tau_comp, dim=-1)   # (B, T, D) ∈ [0, 1], sum=1
        # Scale: uniform weight = 1/D → ×D = 1; competitive winner > 1
        comp   = softw * D                                  # (B, T, D): uniform=1

        # ── Blend toward identity and renormalize ─────────────────────────
        gate_raw = alpha * comp + (1.0 - alpha)             # (B, T, D)
        gate     = gate_raw / (gate_raw.mean(dim=-1, keepdim=True) + self.eps)  # energy preserve

        return out * gate
