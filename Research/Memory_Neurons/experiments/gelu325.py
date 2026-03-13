"""GELU325 – gelu211 with Gradient-Through x in Z-Scores.

gelu211's CRITICAL DETACHMENT: `z_in = (x.detach() - ema_mean) / std`

The `.detach()` on x means β_up, β_dn, γ receive gradient ONLY through how
gate_in = f(z_in) multiplies `out` — NOT through how the z_in values themselves
change when model weights shift x.

FIX: remove the detach() on x and out in the z-score computation:
    z_in  = (x   - ema_mean.detach()) / (std_in.detach()  + ε)   ← x keeps gradient
    z_out = (out - ema_out_mean.detach()) / (std_out.detach() + ε) ← out keeps gradient

GRADIENT PATH NOW:
    loss → output = out * gate_in(z_in(x)) * gate_out(z_out(out))
         → gradient flows through gate shape (as before)
         → AND through z_in = (x - μ) / σ  →  ∂z_in/∂x = 1/σ_in
         → β_up, β_dn, γ now update to shift WHERE in z-space the gate fires

This is the most targeted fix for gelu211's gradient deficit — same architecture,
same 7 params, just removing 2 detaches.

CAUSALITY: x is the current position's activation, not future tokens — safe.
z_in/z_out normalisation uses batch-level EMA stats (detached buffers) — safe.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma,
        log_beta_out, log_gamma_out  (7, identical to gelu211)
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq, _ema_out_dir
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU325(nn.Module):
    """gelu211 with x.detach() removed: full gradient through z-score normalization."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))  # init d ≈ 0.9
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
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

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

        # KEY CHANGE: use x (not x.detach()) so gradient flows through z-score
        var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var).detach()
        std_in  = var_in.sqrt().view(1, 1, D)
        mu_in   = self._ema_mean.detach().view(1, 1, D)
        z_in    = (x   - mu_in)  / (std_in  + self.eps)    # x NOT detached — gradient flows

        var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var).detach()
        std_out = var_out.sqrt().view(1, 1, D)
        mu_out  = self._ema_out_mean.detach().view(1, 1, D)
        z_out   = (out - mu_out) / (std_out + self.eps)    # out NOT detached — gradient flows

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
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
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
