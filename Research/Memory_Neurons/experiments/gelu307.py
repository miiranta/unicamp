"""GELU307 – gelu211 with Cross-Product (Trilinear) Gate.

MOTIVATION: gelu211 multiplies gate_in × gate_out independently.
This misses CONJUNCTIVE information: the case where z_in AND z_out are
simultaneously unusual in related ways.

TRILINEAR EXTENSION:
    gate_in    = asym(z_in)                        [same as gelu211]
    gate_out   = sym(z_out)                        [same as gelu211]
    gate_cross = clamp(1 + β_x * tanh(γ_x * z_in * z_out), 0.1, 5.0)  [NEW]

    z_in * z_out > 0  when both are high (or both low) → amplify
    z_in * z_out < 0  when one high, one low → suppress
    This captures channel-level input-output concordance.

    gate_final = gate_in × gate_out × gate_cross × gate_cos

WHY THIS COULD BEAT gelu211:
    - The cross-term adds a non-linear conjunction without much extra cost
    - Only 2 new params (β_x, γ_x) — 9 total
    - Better separates "routine activations" from "truly novel" ones

CAUSALITY: z_in and z_out both use batch-level EMA statistics → safe.
NO torch.no_grad on gate computation.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma,
        log_beta_out, log_gamma_out, log_beta_cross, log_gamma_cross  (9)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU307(nn.Module):
    """gelu211 + cross-product gate: captures z_in × z_out conjunction."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay    = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma      = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out   = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out  = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Cross-product gate (new)
        self.log_beta_cross = nn.Parameter(torch.tensor(math.log(math.exp(0.2) - 1.0)))
        self.log_gamma_cross= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

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

        d_val      = torch.sigmoid(self.logit_decay).detach().item()
        tau        = self.log_tau.exp()
        beta_up    = F.softplus(self.log_beta_up)
        beta_dn    = F.softplus(self.log_beta_dn)
        gamma      = F.softplus(self.log_gamma)
        beta_out   = F.softplus(self.log_beta_out)
        gamma_out  = F.softplus(self.log_gamma_out)
        beta_cross = F.softplus(self.log_beta_cross)
        gamma_cross= F.softplus(self.log_gamma_cross)

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

        gate_in    = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                         - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out   = (1.0 + beta_out   * torch.tanh(gamma_out   * z_out)).clamp(0.1, 5.0)
        gate_cross = (1.0 + beta_cross * torch.tanh(gamma_cross * z_in * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cross * gate_cos

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
