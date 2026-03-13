"""GELU320 – Exponential-Shape Gate (exp(β·tanh(γ·z)) vs 1+β·tanh(γ·z)).

gelu211 uses a LINEARLY-OFFSET tanh gate shape:
    gate = 1 + β * tanh(γ * z)    — range (1-β, 1+β); centred at 1

ALTERNATIVE: MULTIPLICATIVE exponential shape:
    gate = exp(β * tanh(γ * z))   — range (exp(-β), exp(β)); centred at 1

KEY DIFFERENCES:
    1. At z=0: both give gate=1 (neutral) ✓
    2. For large |z|: exp gate saturates at exp(±β) vs 1±β
       exp(β=2) ≈ 7.4 vs 1+2 = 3 → much more amplification for same β
       exp(-β) > 0 always → never suppresses to 0 (naturally bounded)
    3. GRADIENT: d/dβ of exp(β*tanh(γz)) = tanh(γz)*exp(β*tanh(γz))
       gradient SCALES WITH gate value — larger gates get larger gradient updates
       vs d/dβ of (1+β*tanh(γz)) = tanh(γz) — gradient independent of current gate value

    The scaling gradient of exp-gate creates a fundamentally different optimisation landscape.

COMBINED GATE:
    gate_in  = exp(β_up * ReLU(tanh(γ*z_in)) - β_dn * ReLU(tanh(-γ*z_in)))
             = exp(β_up * ReLU(tanh(γ*z_in))) × exp(-β_dn * ReLU(tanh(-γ*z_in)))
    gate_out = exp(β_out * tanh(γ_out * z_out))
    output   = out × gate_in × gate_out × gate_cos

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_beta_out, log_gamma_out  (7)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU320(nn.Module):
    """Exponential gate shape: exp(β·tanh(γ·z)) — scaling gradient, natural boundedness."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach()   - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

        # Exponential gate shape — log_gate is naturally additive, exp gives final gate
        log_gate_in  = ( beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in)))   # (B, T, D)
        log_gate_out =   beta_out * torch.tanh(gamma_out * z_out)        # (B, T, D)

        gate_in  = torch.exp(log_gate_in.clamp(-2.2, 2.2))    # ≈ bounds (0.11, 9.0)
        gate_out = torch.exp(log_gate_out.clamp(-1.6, 1.6))   # ≈ bounds (0.20, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
