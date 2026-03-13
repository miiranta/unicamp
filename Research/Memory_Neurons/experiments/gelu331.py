"""GELU331 – Squared Z-Score Gate (Deviation Magnitude Gate).

MOTIVATION: gelu211 uses asymmetric gate: amplify positively novel (z > 0), suppress
familiar (z < 0). But positional: a channel below its mean is "suppressed" even if
its deviation is large.

HYPOTHESIS: Gate on MAGNITUDE of deviation, not sign. A channel that deviates strongly
from its mean in EITHER direction is "interesting" — amplify it. Only channels near
their mean are gated down toward 1.

KEY CHANGE:
    gate_in = 1 + β * tanh(γ * z²)

    - z = 0  → gate = 1.0          (mean activation: identity)
    - |z| → ∞ → gate → 1 + β      (max amplification)
    - SYMMETRIC: z=+2 and z=-2 give same gate
    - Never suppresses (gate ≥ 1 always, since tanh(z²) ≥ 0 for all z)
    - Differentiable everywhere (no asymmetric split into ReLU branches)

ASYMMETRIC OUTPUT GATE: Keep output gate directional (z_out sign matters for suppression).
    gate_out = clamp(1 + β_out * tanh(γ_out * z_out), 0.1, 5.0)

This combined design:
    - Input gate = "how unusual is this activation?" (magnitude)
    - Output gate = "does this output match expectations?" (direction)
    - Cosine gate = "is this output novel vs historical direction?" (same as gelu211)

PARAMS: 7 scalars: logit_decay, log_tau, log_beta_sq, log_gamma_sq, log_beta_out,
        log_gamma_out  (6 scalars — 1 fewer than gelu211 — simpler)
STATE: same 5 EMA buffers as gelu211
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU331(nn.Module):
    """Squared z-score gate: 1 + β*tanh(γ*z²) — responds to deviation magnitude, not sign."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_sq   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # β for squared gate
        self.log_gamma_sq  = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # γ for z²
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = None
        self._ema_out_mean = self._ema_out_sq = self._ema_out_dir = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _z(self, val, mean, sq):
        var = (sq - mean.pow(2)).clamp(min=self.eps_var)
        return (val - mean) / (var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_sq   = F.softplus(self.log_beta_sq)
        gamma_sq  = F.softplus(self.log_gamma_sq)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
                bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
                self._ema_mean     = bm_x.clone(); self._ema_sq     = bsq_x.clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = bsq_o.clone()
                self._ema_out_dir  = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        # SQUARED z-score gate: response based on deviation MAGNITUDE
        # gamma_sq * z_in^2 is always >= 0, so tanh(...) ∈ [0,1), gate ∈ [1.0, 1+β_sq)
        gate_in  = (1.0 + beta_sq * torch.tanh(gamma_sq * z_in.pow(2))).clamp(1.0, 6.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0);  bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0);  bsq_o = of.pow(2).mean(0)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * bm_x
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * bsq_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(bm_o, dim=0)

        return output
