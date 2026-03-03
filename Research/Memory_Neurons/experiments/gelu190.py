"""GELU190 – Asymmetric Bidirectional Per-Channel Vector Gate.

THE LIMITATION OF GELU181's SYMMETRIC GATE:
    gelu181: gate_d = clamp(1 + β × tanh(γ × z_d), 0.1, 5.0)

    This uses a SINGLE β parameter for both directions:
    - High z_d (above mean): gate_d = 1 + β × tanh(γ × z_d) > 1   (amplify)
    - Low  z_d (below mean): gate_d = 1 + β × tanh(γ × z_d) < 1   (suppress)

    The same β controls BOTH the up-modulation and down-modulation strength.
    But the OPTIMAL response to "unusually high" vs "unusually low" might differ.

THE NEW IDEA: Separate Excitatory and Inhibitory Arms
    Two independent parameters:
        β_up: controls amplification of high channels  (z > 0)
        β_dn: controls suppression of low channels     (z < 0)

    Per-channel gate:
        up_arm   = β_up × ReLU(tanh( γ × z_d))    — positive when z_d > 0
        dn_arm   = β_dn × ReLU(tanh(-γ × z_d))    — positive when z_d < 0 (note: -γz_d > 0)
        gate_d   = clamp(1 + up_arm - dn_arm, 0.05, 8.0)

    So:
        z_d > 0:  gate_d = 1 + β_up × tanh(γ × z_d)   — amplify with strength β_up
        z_d < 0:  gate_d = 1 - β_dn × tanh(γ× |z_d|)  — suppress with strength β_dn
        z_d = 0:  gate_d = 1                             — no-op (up_arm=dn_arm=0)

BIOLOGICAL ANALOGY:
    Excitatory neurons (β_up): fire MORE when active, less when inactive
    Inhibitory neurons (β_dn): specifically SILENCE low-activity channels
    The model learns the optimal ratio:
    - If β_up >> β_dn: primarily an amplifier of novel-high channels
    - If β_dn >> β_up: primarily a suppressor of familiar-low channels
    - If β_up ≈ β_dn: symmetric (≈ gelu181)

INITIALIZATION:
    β_up = β_dn = 0.5 initially (symmetric gelu181 behavior)
    The model is free to break symmetry during training.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma = 5 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU190(nn.Module):
    """Asymmetric bidirectional per-channel gate: separate β_up (excitatory) and β_dn (inhibitory)."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        # Both β_up and β_dn init to ≈0.5
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std  = var.sqrt().view(1, 1, D)
            mu_  = self._ema_mean.view(1, 1, D)
            z    = (x.detach() - mu_) / (std + self.eps)  # (B, T, D) signed

        # ── Asymmetric bidirectional gate ──────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z))   # (B, T, D) ≥ 0, active when z > 0
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z))   # (B, T, D) ≥ 0, active when z < 0
        gate_vec = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)   # (B, T, D)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)  # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
