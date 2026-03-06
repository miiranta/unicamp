"""GELU214 – Asymmetric Gate with Learned Dead-Zone (Soft Threshold).

MOTIVATION — FIX THE "NEAR-ZERO Z-SCORE" NOISE PROBLEM:
    In gelu190 (best PPL, 160.54), every channel contributes to the gate even
    when z_d ≈ 0 (channel is near its mean). tanh(γ × 0) = 0, so gate_d = 1.
    But in practice, small non-zero z-scores still introduce small gate deviations
    away from 1, adding noise rather than signal.

    SOFT THRESHOLD GATE:
        Instead of ReLU(tanh(γ × z_d)), use a SOFT THRESHOLD:
            f(z_d) = ReLU(|z_d| − θ) / (|z_d| − θ + ε)   (for z_d > θ: gate grows)
            
        Or equivalently, a DEADBAND version:
            f_up(z_d) = ReLU(tanh(γ × (z_d − θ)))          — fires when z_d > +θ
            f_dn(z_d) = ReLU(tanh(γ × (−z_d − θ)))         — fires when z_d < −θ

        gate_d = 1 + β_up × f_up(z_d) − β_dn × f_dn(z_d)

    The threshold θ acts as a "dead zone" around z=0. Channels within
    [−θ, +θ] standard deviations of their mean are left STRICTLY at gate_d = 1.
    Only channels with |z_d| > θ receive any modulation.

    This should REDUCE noise from channels that are "slightly familiar" without
    being truly familiar or truly novel, improving the signal-to-noise ratio of
    the gate.

LEARNABLE THRESHOLD:
    θ = softplus(log_theta) init = 0.5 (one standard deviation deadband)
    All other params from gelu190: β_up, β_dn, γ, τ, decay

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma, log_theta
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU214(nn.Module):
    """Asymmetric per-channel gate with soft dead-zone: modulate only channels with |z| > threshold."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Dead-zone threshold: init θ ≈ 0.5 std
        self.log_theta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

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

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)
        theta   = F.softplus(self.log_theta)     # dead-zone half-width

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
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)
            mu_ = self._ema_mean.view(1, 1, D)
            z   = (x.detach() - mu_) / (std + self.eps)          # (B, T, D)

        # ── Dead-zone asymmetric gate ──────────────────────────────────
        # Only fire when |z_d| exceeds threshold θ
        up_arm   = beta_up * F.relu(torch.tanh(gamma * (z   - theta)))   # z > +θ
        dn_arm   = beta_dn * F.relu(torch.tanh(gamma * (-z  - theta)))   # z < -θ
        gate_vec = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)              # (B, T, D)

        # ── Cosine output EMA gate ─────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
