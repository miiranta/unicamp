"""GELU222 – gelu211 with Per-Channel Asymmetric Output Arm.

UPGRADE FROM gelu211 (PPL 159.35, current best):
    gelu211 uses:
        gate_in_d   = 1 + β_up*ReLU(tanh(γ*z_in_d)) − β_dn*ReLU(tanh(−γ*z_in_d))
        gate_out_d  = 1 + β_out × tanh(γ_out × z_out_d)          ← SYMMETRIC scalar β_out
        gate_final  = clamp(gate_in × gate_out, 0.05, 10.0)

    The OUTPUT arm in gelu211 is symmetric (same β for positive and negative deviations).
    gelu222 makes the OUTPUT arm ASYMMETRIC too:
        gate_out_d  = 1 + β_out_up*ReLU(tanh(γ_out*z_out_d)) − β_out_dn*ReLU(tanh(−γ_out*z_out_d))

    This doubles the expressiveness of the output gate:
        - β_out_up: how much to amplify channels with unusually HIGH output
        - β_out_dn: how much to suppress channels with unusually LOW output
        - Model can learn: "amplify both high-input AND high-output channels strongly"

CONJUNCTIVE INTERPRETATION:
    gate_final > 1 only when BOTH input AND output are ABOVE their means (double-novel)
    gate_final < 1 only when BOTH input AND output are BELOW their means (double-familiar)
    Mixed cases (input↑ but output↓, or input↓ but output↑): gate ≈ 1 (cancel out)

PARAMS: logit_decay, log_beta_up, log_beta_dn, log_gamma,
        log_beta_out_up, log_beta_out_dn, log_gamma_out  (7 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean (D,), _ema_out_sq (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU222(nn.Module):
    """gelu211 with fully asymmetric per-channel output arm (separate β_out_up and β_out_dn)."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        # Input-space asymmetric arms (same as gelu190/gelu211)
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Output-space asymmetric arms (new: separate up/down)
        self.log_beta_out_up = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_beta_out_dn = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val       = torch.sigmoid(self.logit_decay).detach().item()
        beta_up     = F.softplus(self.log_beta_up)
        beta_dn     = F.softplus(self.log_beta_dn)
        gamma       = F.softplus(self.log_gamma)
        beta_out_up = F.softplus(self.log_beta_out_up)
        beta_out_dn = F.softplus(self.log_beta_out_dn)
        gamma_out   = F.softplus(self.log_gamma_out)

        y = self._gelu(x)   # (B, T, D)

        # ── Init EMA ─────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                yf = y.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = yf.mean(0).clone()
                self._ema_out_sq   = yf.pow(2).mean(0).clone()
            self._ready = True
            return y

        # ── Input z-scores ───────────────────────────────────────────
        var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(self.eps_var)
        std_in  = var_in.sqrt()
        z_in    = (x - self._ema_mean) / (std_in + self.eps)   # (B, T, D)

        # ── Output z-scores ──────────────────────────────────────────
        var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(self.eps_var)
        std_out = var_out.sqrt()
        z_out   = (y - self._ema_out_mean) / (std_out + self.eps)   # (B, T, D)

        # ── Input asymmetric gate (per-channel) ──────────────────────
        up_in = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_in = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in = (1.0 + up_in - dn_in).clamp(0.05, 8.0)          # (B, T, D)

        # ── Output asymmetric gate (per-channel) ─────────────────────
        up_out = beta_out_up * F.relu(torch.tanh( gamma_out * z_out))
        dn_out = beta_out_dn * F.relu(torch.tanh(-gamma_out * z_out))
        gate_out = (1.0 + up_out - dn_out).clamp(0.05, 8.0)       # (B, T, D)

        # ── Product gate ─────────────────────────────────────────────
        gate = (gate_in * gate_out).clamp(0.05, 10.0)

        output = y * gate

        # ── Update EMAs ──────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            yf = y.detach().flatten(0, 1)
            self._ema_mean     = d_val*self._ema_mean     + (1-d_val)*xf.mean(0)
            self._ema_sq       = d_val*self._ema_sq       + (1-d_val)*xf.pow(2).mean(0)
            self._ema_out_mean = d_val*self._ema_out_mean + (1-d_val)*yf.mean(0)
            self._ema_out_sq   = d_val*self._ema_out_sq   + (1-d_val)*yf.pow(2).mean(0)

        return output
