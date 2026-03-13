"""GELU323 – Fully Differentiable gelu211 (Batch Stats Everywhere, No EMA).

gelu211 uses EMA buffers for z-score normalisation AND a separate EMA direction
for the cosine gate. Both are updated inside `torch.no_grad()`, so all three
gate components (gate_in, gate_out, gate_cos) use statistics that the
optimiser cannot adjust.

THIS EXPERIMENT: replace EVERY statistic with current-batch equivalents:
    mu_in   = x.mean((0,1))                — batch input mean
    std_in  = x.std((0,1))                 — batch input std
    mu_out  = out.mean((0,1))              — batch output mean
    std_out = out.std((0,1))               — batch output std
    dir_out = normalize(out.mean((0,1)))   — batch output direction

All fully differentiable → β_up, β_dn, γ, β_out, γ_out, τ all get
gradients through the ACTUAL normalisation statistics, not just through
the gate multiplication.

No state, no warmup, no EMA updates. Stateless but fully expressive.

PARAMS: log_tau, log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out  (6)
STATE:  none
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU323(nn.Module):
    """Fully differentiable gelu211 using current-batch statistics everywhere."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

    def reset_state(self):
        pass  # stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        # Fully differentiable z-scores: batch mean/std, gradient flows to β, γ
        mu_in   = x.mean(dim=(0, 1), keepdim=True)
        std_in  = x.std(dim=(0, 1), keepdim=True).clamp(min=self.eps)
        mu_out  = out.mean(dim=(0, 1), keepdim=True)
        std_out = out.std(dim=(0, 1), keepdim=True).clamp(min=self.eps)

        z_in  = (x   - mu_in)  / (std_in  + self.eps)
        z_out = (out - mu_out) / (std_out + self.eps)

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # Fully differentiable cosine gate with batch mean direction
        batch_dir = F.normalize(out.mean(dim=(0, 1)).detach(), dim=0)  # detach dir, not out
        cos_sim   = (F.normalize(out, dim=-1) * batch_dir.view(1, 1, -1)).sum(-1, keepdim=True).clamp(-1, 1)
        gate_cos  = torch.exp(-tau * cos_sim)                          # τ gets gradient

        return out * gate_in * gate_out * gate_cos
