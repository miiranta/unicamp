"""
GELU138 — Intra-batch variance gate (stateless cross-sample novelty).

Observation: tokens where the batch samples DISAGREE with each other are
inherently context-specific — they represent positions where different
preceding contexts lead to meaningfully different activation patterns.
High variance across the batch at a given position → the model is in an
unusual or highly context-dependent regime.

This is orthogonal to EMA-based methods (no state!) and captures novelty
defined relative to the current batch's diversity rather than historical
activations.

    batch_var_d[t] = Var_B(x[b, t, d])       (B, T, D) → (T, D) after var(dim=0)
    channel_z[t,d] = batch_var_d[t] / (mean_d(batch_var_d[t]) + eps)
                   = relative per-channel variance at each position
    gate = 1 + alpha * tanh(sigma * channel_z)      # (1, T, D) broadcasts to (B,T,D)

No EMA state.  Each forward pass is independent.  Gradient flows through alpha, sigma.

Params: log_alpha (scalar), log_sigma (scalar)
"""

import torch
import torch.nn as nn


class GELU138(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._gelu = nn.GELU()

    def reset_state(self):
        pass   # stateless

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        base = self._gelu(x)

        if x.size(0) < 2:
            # can't compute batch variance with a single sample
            return base

        # variance over batch dimension (unbiased)
        # detach the variance computation to keep gate as a modifier, not a normalizer
        with torch.no_grad():
            bvar = x.var(dim=0)            # (T, D)
            mean_bvar = bvar.mean(dim=-1, keepdim=True).clamp(min=1e-8)  # (T, 1)
            rel_var = bvar / mean_bvar     # (T, D)  — normalized channel variance

        # use the detached variance but allow grad through alpha/sigma
        alpha = torch.exp(self.log_alpha)
        sigma = torch.exp(self.log_sigma)
        gate  = 1.0 + alpha * torch.tanh(sigma * rel_var.unsqueeze(0))  # (1, T, D)
        return base * gate
