"""GELU313 – Fully Differentiable Batch-Statistics Gate (No EMA State).

MOTIVATION: Every previous experiment uses EMA buffers to track running statistics,
which requires `torch.no_grad()` for buffer updates and disconnects the normalization
statistics from backpropagation.

RADICAL SIMPLIFICATION: Use CURRENT BATCH statistics for z-score normalisation.
    mu_in  = x.mean(dim=(0,1))          — mean over B×T (same as EMA target)
    std_in = x.std(dim=(0,1))           — std  over B×T
    z_in   = (x - mu_in) / (std_in + ε)  — fully differentiable normalisation

ADVANTAGES:
    - FULLY DIFFERENTIABLE: every parameter (β_up, β_dn, γ, β_out, γ_out) gets
      gradient through both the gate shape AND the normalisation statistics.
    - NO STATE: no EMA buffers, no reset_state() needed.
    - SIMPLER CODE: no warmup step, no EMA update block.
    - SAME CAUSALITY LEVEL as EMA experiments: batch-level stats include all B×T
      positions, which is equivalent to traditional batch normalisation in the FFN.

GATE: identical to gelu303 (gelu211 without cosine) but with fully differentiable stats.
    gate_in  = asym(z_in_batch)
    gate_out = sym(z_out_batch)
    output   = out × gate_in × gate_out

PARAMS: log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out  (5 scalars — no decay!)
STATE:  none
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU313(nn.Module):
    """Fully differentiable batch-stats z-score gate — no EMA, no state, gradient flows everywhere."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        # Fully differentiable z-scores from current batch statistics
        # mean/std over the B×T positions — no futures leaked beyond batch-level
        mu_in   = x.mean(dim=(0, 1), keepdim=True)                        # (1, 1, D)
        std_in  = x.std(dim=(0, 1), keepdim=True).clamp(min=self.eps)     # (1, 1, D)
        mu_out  = out.mean(dim=(0, 1), keepdim=True)
        std_out = out.std(dim=(0, 1), keepdim=True).clamp(min=self.eps)

        z_in  = (x   - mu_in)  / (std_in  + self.eps)   # (B, T, D) — fully differentiable
        z_out = (out - mu_out) / (std_out + self.eps)    # (B, T, D) — fully differentiable

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        return out * gate_in * gate_out
