"""GELU126 – Causal Within-Sequence EMA Z-Score Gate.

THE CORE IDEA — WITHIN-CONTEXT NOVELTY:
    All prior experiments compute novelty relative to CROSS-BATCH history:
        "Has this pattern appeared in PREVIOUS batches?"
    
    GELU126 computes novelty relative to the CURRENT SEQUENCE context:
        "Is this token surprising given what came BEFORE in THIS sequence?"
    
    This is a purely within-sequence causal scan — no persistent EMA state,
    completely stateless across batches.

THE MECHANISM — CAUSAL LINEAR SCAN:
    For each batch, scan positions t = 0 ... T-1 in order:
    
        For t=0: gate = 1 (no history)
        For t>0:
            z[b,t,d] = (x[b,t,d] - ema_mean[b,d]) / (ema_std[b,d] + eps)
            gate[b,t,d] = 1 + α × tanh(σ × |z[b,t,d]|)
            ema_mean[b,d] = d × ema_mean[b,d] + (1-d) × x[b,t,d]
            ema_sq[b,d]   = d × ema_sq[b,d]   + (1-d) × x[b,t,d]²

    The scan maintains a per-sequence EMA that grows as we move along T.
    Earlier positions contribute more to the "expected pattern" than later.

WHAT THIS CAPTURES:
    Any token that looks DIFFERENT from the earlier tokens in its sequence gets amplified.
    
    Example: A sequence about "cats and dogs" → suddenly switches to quantum physics.
    - "cat": familiar within sequence after seeing "dog" several times
    - "quantum": very different pattern from cat/dog → high z-score → amplified
    
    This is truly sequence-level context awareness, distinct from token-level cross-batch stats.

WHY THIS IS STRICTLY CAUSAL:
    At position t, ema_mean contains ONLY information from positions 0..t-1.
    The gate for position t is computed BEFORE updating the EMA with x[t].
    Position t is fully independent of positions t+1, ..., T-1.

COMPLEMENTARITY WITH CROSS-BATCH EMA:
    Cross-batch EMA (gelu80): "Is this token type rare across all of training?"
    Within-sequence scan (gelu126): "Is this token different from THIS passage?"
    
    Together: a token can be:
    - Rare globally + novel in context → double amplification
    - Common globally + novel in context → context-novelty amplification only
    - Rare globally + fits the context → global-novelty amplification only
    - Common globally + fits context → both suppressed

COMPUTATIONAL COST:
    O(B × T × D) element-wise operations across T iterations.
    For B=32, T=64, D=1024: 64 × (32 × 1024) = 2.1M ops per layer.
    Sequential scan, no parallelism — slightly slower than batch ops but manageable.

FULLY STATELESS:
    No parameters that change between batches (no EMA state to reset).
    All state is local to the current forward() call.
    Safe to use without reset_state() calls.

Params: logit_decay, log_alpha, log_sigma = 3 scalars.
State: none (per-call scan only).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU126(nn.Module):
    """Causal within-sequence EMA z-score gate: per-channel novelty within current context."""

    def __init__(self, d_ff: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        self.logit_decay = nn.Parameter(
            torch.tensor(math.log(ema_decay / (1.0 - ema_decay)))
        )
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
        d     = torch.sigmoid(self.logit_decay).item()  # EMA decay ∈ (0, 1)
        alpha = F.softplus(self.log_alpha)               # amplification strength
        sigma = F.softplus(self.log_sigma)               # z-score sensitivity

        # Running statistics (per-sample in batch, per-channel)
        ema_mean = torch.zeros(B, D, device=x.device)
        ema_sq   = torch.zeros(B, D, device=x.device)

        gates = []

        for t in range(T):
            xt = x[:, t, :].detach()  # (B, D) — detach to avoid scan-through-time gradients

            if t == 0:
                # No history at position 0 → neutral gate
                gate_t = torch.ones(B, D, device=x.device)
                # Initialize EMA from first position
                ema_mean = xt.clone()
                ema_sq   = xt.pow(2)
            else:
                # Compute z-score against running EMA (causal: uses positions 0..t-1)
                var    = (ema_sq - ema_mean.pow(2)).clamp(min=self.eps_var)  # (B, D)
                std    = var.sqrt()
                z      = (xt - ema_mean) / (std + self.eps)                  # (B, D)
                gate_t = 1.0 + alpha * torch.tanh(sigma * z.abs())           # (B, D)

                # Update EMA AFTER computing gate (causal)
                ema_mean = d * ema_mean + (1.0 - d) * xt
                ema_sq   = d * ema_sq   + (1.0 - d) * xt.pow(2)

            gates.append(gate_t)

        gate = torch.stack(gates, dim=1)  # (B, T, D)
        return self._gelu(x * gate)
