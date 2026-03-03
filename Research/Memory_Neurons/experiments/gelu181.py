"""GELU181 – Per-Channel Signed Z-Score Vector Gate.

THE CENTRAL UNRESOLVED LIMITATION OF GELU80:
    gelu80 computes per-channel z-scores z_d = (x_d - μ_d) / σ_d   — D values
    Then immediately collapses: surp = tanh(σ × mean_d |z_d|)       — 1 scalar
    Then applies: output = GELU(x) × gate_scalar                    — scalar × D

    The per-channel information is DISCARDED at the aggregation step.
    After 180 experiments, this scalar gate is the highest-impact untested change.

THE NEW IDEA: Per-Channel Gate Vector
    Keep z_d at full D-dimensional resolution:
        gate_d = clamp(1 + β × tanh(γ × z_d),  lo=0.1, hi=5.0)    — (B, T, D)

    Where:
        z_d > 0 → channel is above its historical mean → gate_d > 1 (AMPLIFY)
        z_d < 0 → channel is below its historical mean → gate_d < 1 (SUPPRESS)
        z_d ≈ 0 → familiar channel → gate_d ≈ 1 (PASS THROUGH)

    Output: GELU(x_d) × gate_d × cos_gate_scalar       — per-channel modulation

WHY SIGNED (NOT ABSOLUTE):
    |z_d| discards whether the channel is unusually HIGH or LOW.
    Using z_d (signed) allows the gate to amplify unusual-high AND suppress unusual-low.
    Channels ON when they should be OFF (or vice versa) are modulated independently.

GEOMETRIC INTERPRETATION:
    gelu80 scalar gate: scales the entire GELU(x) vector uniformly
    gelu181 vector gate: rotates AND scales GELU(x) in channel-specific directions
    The output vector MOVES toward high-z channels and away from low-z channels.

PARAMS: logit_decay, log_tau, log_beta, log_gamma = 4 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU181(nn.Module):
    """Per-channel signed z-score vector gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            var      = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std      = var.sqrt().view(1, 1, D)
            mu_      = self._ema_mean.view(1, 1, D)
            z        = (x.detach() - mu_) / (std + self.eps)              # (B, T, D) signed
            gate_vec = (1.0 + beta * torch.tanh(gamma * z)).clamp(0.1, 5.0)  # (B, T, D)

            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)             # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
