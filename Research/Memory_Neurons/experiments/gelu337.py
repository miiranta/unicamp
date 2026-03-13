"""GELU337 – Sigmoid Gate with Neural Depletion (8 params).

MOTIVATION: gelu321 (sigmoid gate) shows strong sequential adaptation (+2.83 PPL
improvement pass 1→3). But single-pass performance (159.68) is slightly below gelu211
(159.35). Adding a DEPLETION mechanism (like gelu291) may push single-pass PPL lower
while preserving gelu321's adaptation advantage.

DEPLETION MECHANISM (borrowed from gelu291):
    When a channel's gate is consistently > 1 (it's being amplified frequently),
    the depletion accumulates and attenuates future activations of that channel.
    This prevents any single channel from monopolizing the gate across the sequence.

    excess    = relu(gate_in * gate_out - 1.0)    — how much above baseline?
    _depl    ← d_depl * _depl + (1 - d_depl) * excess.mean(dim=(0,1))  [EMA in no_grad]
    gate_depl = exp(-U * _depl)                   — attenuates over-active channels

COMBINED GATE:
    gate_in   = 2σ(β_up * relu(z_in)  - β_dn * relu(-z_in))   [gelu321 asymmetric sigmoid]
    gate_out  = 2σ(β_out_up * relu(z) - β_out_dn * relu(-z))  [same]
    gate_cos  = exp(-τ * cos(out, ema_dir))
    gate_depl = exp(-U * depl)
    output    = GELU(x) × gate_in × gate_out × gate_cos × gate_depl

INIT: _depl = zeros(D) — starts at zero → gate_depl = 1 initially

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_beta_out_up, log_beta_out_dn,
        logit_d_depl, log_U  (8 scalars)
STATE:  5 EMA buffers from gelu321 + _depl (D,) = 6 buffers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU337(nn.Module):
    """gelu321 sigmoid gate + per-channel depletion to prevent channel dominance."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        # gelu321 params
        self.logit_decay    = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau        = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_dn    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out_up= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_out_dn= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        # depletion params
        self.logit_d_depl   = nn.Parameter(torch.tensor(math.log(19.0)))   # d_depl ≈ 0.95
        self.log_U          = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # U ≈ 0.5

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._depl:         torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = None
        self._ema_out_mean = self._ema_out_sq = self._ema_out_dir = None
        self._depl = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _z(self, val, mean, sq):
        var = (sq - mean.pow(2)).clamp(min=self.eps_var)
        return (val - mean) / (var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        d_depl   = torch.sigmoid(self.logit_d_depl).detach().item()
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        beta_oup = F.softplus(self.log_beta_out_up)
        beta_odn = F.softplus(self.log_beta_out_dn)
        U        = F.softplus(self.log_U)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
                bm_x = xf.mean(0); bm_o = of.mean(0)
                self._ema_mean = bm_x.clone(); self._ema_sq = xf.pow(2).mean(0).clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = of.pow(2).mean(0).clone()
                self._ema_out_dir = F.normalize(bm_o, dim=0).clone()
                self._depl = torch.zeros(D, device=x.device)
                self._ready = True
            return out

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        logit_in  = beta_up * F.relu(z_in)  - beta_dn * F.relu(-z_in)
        logit_out = beta_oup * F.relu(z_out) - beta_odn * F.relu(-z_out)
        gate_in   = 2.0 * torch.sigmoid(logit_in)   # ∈ (0, 2)
        gate_out  = 2.0 * torch.sigmoid(logit_out)  # ∈ (0, 2)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)
            # Depletion gate: attenuate over-active channels
            gate_depl_vec = torch.exp(-U * self._depl).view(1, 1, D)

        output = out * gate_in * gate_out * gate_cos * gate_depl_vec

        with torch.no_grad():
            xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * bm_x
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * bsq_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(bm_o, dim=0)
            # Update depletion: excess above baseline gate=1
            excess = F.relu(gate_in.detach() * gate_out.detach() - 1.0).mean(dim=(0, 1))
            self._depl = d_depl * self._depl + (1-d_depl) * excess

        return output
