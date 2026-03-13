"""GELU319 – Linear Mixed Z-Score Gate (α_in·z_in + α_out·z_out Combined Signal).

gelu211 computes gate_in and gate_out SEPARATELY then multiplies them:
    final = gate_in(z_in) × gate_out(z_out)

A FUNDAMENTALLY DIFFERENT ARCHITECTURE: combine z_in and z_out into a SINGLE
novelty signal first, then apply one gate:

    z_mix = z_in + α · z_out     — learnable contribution of output z-score
    gate  = asym(z_mix)          — asymmetric gate on combined signal
    output = out × gate × gate_cos

WHEN DOES THIS DIFFER FROM gelu211?
    gelu211: amplify when both z_in > 0 AND z_out > 0 (conjunctive)
    gelu319: amplify when z_in + α·z_out > 0 (additive combination)

    If α > 0: channel amplified when EITHER signal is high enough
    If α < 0: channel amplified when z_in dominates over z_out (input novelty despite output familiarity)
    α is learned — the model discovers the optimal projection

ADVANTAGES:
    - 1 gate instead of 2 (simpler gradient landscape)
    - α directly measures the "weight of output z-score" relative to input z-score
    - Still fully expressive (reduces to gelu190 when α→0)

PARAMS: logit_decay, log_tau, alpha (scalar, unconstrained), log_beta_up, log_beta_dn, log_gamma  (6)
STATE:  _ema_mean, _ema_sq, _ema_out_mean, _ema_out_sq, _ema_out_dir
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU319(nn.Module):
    """Single asymmetric gate on linearly mixed z_in + α*z_out combined signal."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.alpha       = nn.Parameter(torch.tensor(0.0))   # start: ignore z_out; model learns
        self.log_beta_up = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        alpha   = self.alpha                    # unconstrained — can amplify or invert z_out
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

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

        # Linear mixture: alpha gets gradient through gate computation
        z_mix = z_in + alpha * z_out            # (B, T, D) — alpha is differentiable

        gate = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_mix))
                    - beta_dn * F.relu(torch.tanh(-gamma * z_mix))).clamp(0.05, 8.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate * gate_cos

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
