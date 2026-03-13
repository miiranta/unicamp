"""GELU321 – Sigmoid Gate Shape (2·σ(β·z) vs 1+β·tanh(γ·z)).

gelu211 uses: gate = 1 + β·tanh(γ·z)   centred at 1, range (1-β, 1+β)

ALTERNATIVE GATE SHAPE: 2·sigmoid(β·z)
    - At z=0:       2·σ(0) = 2·0.5 = 1.0    ← neutral ✓
    - As z → +∞:   2·σ(∞) → 2.0            ← maximum amplification = 2
    - As z → -∞:   2·σ(-∞) → 0.0           ← complete suppression
    - Range: (0, 2) — naturally bounded without explicit clamp

COMPARISON TO TANH GATE:
    tanh gate:    1 + β·tanh(γ·z) — bounded by β, requires clamp, gradient dies at extremes
    sigmoid gate: 2·σ(β·z)        — smoothly bounded (0,2), gradient is 2·σ·(1-σ)·β
                  → never zero gradient (β·σ·(1-σ) always > 0 for finite z)
                  → no clamp needed
                  → simpler optimisation landscape

ASYMMETRIC VERSION: use separate β for z>0 (amplification) and z<0 (suppression):
    gate_in = 2 · σ(β_up · ReLU(z_in) − β_dn · ReLU(−z_in))
    When z>0: gate = 2σ(β_up · z)  — amplification strength
    When z<0: gate = 2σ(−β_dn · |z|) — suppression strength

COMBINED WITH output gate + cosine (same as gelu211):
    gate_out = 2 · σ(β_out_up · ReLU(z_out) − β_out_dn · ReLU(−z_out))

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_beta_out_up, log_beta_out_dn  (6)
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq, _ema_out_dir
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU321(nn.Module):
    """Sigmoid gate shape: 2σ(β·z) — smooth bounds (0,2), no clamp needed, natural gradient."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay    = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # init larger for sigmoid
        self.log_beta_dn    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out_up= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_out_dn= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

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

        d_val       = torch.sigmoid(self.logit_decay).detach().item()
        tau         = self.log_tau.exp()
        beta_up     = F.softplus(self.log_beta_up)
        beta_dn     = F.softplus(self.log_beta_dn)
        beta_out_up = F.softplus(self.log_beta_out_up)
        beta_out_dn = F.softplus(self.log_beta_out_dn)

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

        # Sigmoid gate shape: 2·σ(logit) — smoothly bounded in (0, 2), no clamp
        logit_in  =  beta_up * F.relu( z_in) - beta_dn * F.relu(-z_in)      # (B, T, D)
        logit_out =  beta_out_up * F.relu( z_out) - beta_out_dn * F.relu(-z_out)

        gate_in  = 2.0 * torch.sigmoid(logit_in)    # (B, T, D) ∈ (0, 2)
        gate_out = 2.0 * torch.sigmoid(logit_out)   # (B, T, D) ∈ (0, 2)

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
