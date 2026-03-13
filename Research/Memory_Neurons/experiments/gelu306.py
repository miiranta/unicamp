"""GELU306 – Per-Channel Learnable Static Normalization (No EMA State).

MOTIVATION: gelu211 uses online EMA buffers to normalize x and out.
This requires stateful adaptation. An alternative: learn fixed per-channel
mean and log-std parameters μ_d and log_σ_d directly via backprop, capturing
the full-dataset statistics at training time.

ADVANTAGES:
    - Fully differentiable: both z_in and z_out reference only nn.Parameter tensors
    - No state to reset between eval passes (no EMA at all)
    - Simpler eval: behaviour is the same in train and test
    - 2*D + 7 params total (2058 params for D=1024) but fully trained

TRADEOFF: z-scores are now STATIC (not adaptive). The EMA in gelu211 helps
the gate adapt dynamically during eval. Without it, z-scores use training
distribution statistics. This tests whether static normalization is sufficient.

GATE:
    z_in  = (x   - μ_in)  / (σ_in  + ε)    [fully differentiable]
    z_out = (out - μ_out) / (σ_out + ε)    [fully differentiable]
    gate_in  = asym(z_in)
    gate_out = sym(z_out)
    output   = out × gate_in × gate_out

NO cosine gate (simplification over gelu211).

PARAMS: mu_in (D,), log_std_in (D,), mu_out (D,), log_std_out (D,),
        log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out  (4D + 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU306(nn.Module):
    """Fully-differentiable static per-channel normalization + asymmetric×symmetric gate."""

    def __init__(self, D_FF: int = 1024, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        D = D_FF

        # Learnable per-channel normalization (4D params)
        self.mu_in      = nn.Parameter(torch.zeros(D))
        self.log_std_in = nn.Parameter(torch.zeros(D))   # log(σ_in); 0 → σ=1
        self.mu_out     = nn.Parameter(torch.zeros(D))
        self.log_std_out= nn.Parameter(torch.zeros(D))

        # Gate shape scalars
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

    def reset_state(self):
        pass  # no state

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

        # Fully differentiable z-scores via learned parameters
        std_in  = F.softplus(self.log_std_in).view(1, 1, D)    # > 0
        std_out = F.softplus(self.log_std_out).view(1, 1, D)   # > 0
        mu_in   = self.mu_in.view(1, 1, D)
        mu_out  = self.mu_out.view(1, 1, D)

        z_in  = (x   - mu_in)  / (std_in  + self.eps)   # (B, T, D) — differentiable
        z_out = (out - mu_out) / (std_out + self.eps)    # (B, T, D) — differentiable

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        return out * gate_in * gate_out
