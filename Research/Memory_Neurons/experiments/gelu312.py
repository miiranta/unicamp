"""GELU312 – Hebbian Co-Activation Gate (Creative: Channel-Pair Correlation).

CREATIVE APPROACH: Implement a low-rank Hebbian tracking of channel co-activations.
Gate a channel based on whether its current activation is CONSISTENT WITH
the channels it typically co-activates with (familiarity) or not (novelty).

MECHANISM:
    Track a low-rank outer product EMA:  R ≈ EMA of (x̄ × x̄ᵀ) via a rank-r approximation.
    Since full D×D is too large, use a projection: V ∈ R^{D×r}, r=8.
    
    context_d = x @ V @ Vᵀ → back to D-space (the "familiar context" of x)
    agreement_d = x_d × context_d  (positive → x agrees with expected context, negative → novel)
    z_hebbian_d = agreement_d / (ema_power_d + ε)  (normalised agreement)
    
    gate_hebbian = clamp(1 + β_h * tanh(γ_h * z_hebbian), 0.1, 5.0)
    output = GELU(x) × gate_hebbian × gate_cos

WHY CAUSALLY SAFE:
    - x @ V @ Vᵀ is a LINEAR projection of x, not of future tokens
    - V is updated via batch-level EMA → no per-position leakage
    - The "Hebbian" part refers to which channels historically co-activate together

PARAMS: logit_decay, log_tau, log_beta_h, log_gamma_h  (4 scalars) + V (D×r = 8192) + v_ema_power (D,)
STATE: _ema_V (r, D rank-r factors), _ema_power_d (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU312(nn.Module):
    """Low-rank Hebbian co-activation gate: novelty = disagreement with learned context."""

    RANK = 8

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        D = D_FF
        r = self.RANK

        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_h  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_h = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Learnable low-rank projection V: D → r; updated mix of EMA + gradient
        self.V = nn.Parameter(torch.randn(D, r) * 0.02)

        self._ema_power:   torch.Tensor = None
        self._ema_out_dir: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_power   = None
        self._ema_out_dir = None
        self._ready       = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_h  = F.softplus(self.log_beta_h)
        gamma_h = F.softplus(self.log_gamma_h)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                of = out.detach().flatten(0, 1)
                xf = x.detach().flatten(0, 1)
                self._ema_power   = xf.pow(2).mean(0).clone()      # (D,)
                self._ema_out_dir = F.normalize(of.mean(0), dim=0).clone()
                self._ready       = True
            return out

        # ── Hebbian context: low-rank projection x → r → D (differentiable in V) ──
        # x_proj: (B, T, r) = x @ V
        x_proj   = x.detach() @ self.V                              # (B, T, r) — detached x, grad to V
        context  = x_proj @ self.V.t()                              # (B, T, D) — reconstruction
        # Agreement: element-wise product of x and context
        agreement = x.detach() * context                            # (B, T, D)
        # Normalise by running power baseline
        with torch.no_grad():
            power_baseline = self._ema_power.view(1, 1, D).clamp(min=self.eps_var)
        z_hebbian = agreement / power_baseline                       # (B, T, D) — grad flows to V

        gate_hebbian = (1.0 + beta_h * torch.tanh(gamma_h * z_hebbian)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_hebbian * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_power   = d_val * self._ema_power   + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_dir = d_val * self._ema_out_dir + (1 - d_val) * F.normalize(of.mean(0), dim=0)

        return output
