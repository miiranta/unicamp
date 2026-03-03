"""GELU186 – Per-Channel Absolute Z-Score Vector Gate.

CONTRAST WITH GELU181 (signed vector gate):
    gelu181: gate_d = clamp(1 + β × tanh(γ × z_d),  0.1, 5.0)
        z_d > 0 → amplify;  z_d < 0 → suppress
        The direction of deviation matters.

    gelu186: gate_d = clamp(1 + β × tanh(γ × |z_d|), 1.0, 8.0)
        |z_d| is large → ALWAYS amplify, regardless of sign
        The gate is ALWAYS ≥ 1.0 — unusual channels are boosted, not suppressed.

THE HYPOTHESIS:
    Perhaps the model does not benefit from knowing WHICH direction the deviation is in,
    but just needs to know THAT a channel is unusual. Unusual channels carry information
    the EMA hasn't seen before. Amplifying them (not suppressing) lets the downstream
    attention layers decide what to do with the amplified signal.

    Suppression via gate_d < 1 (gelu181's behavior for z_d < 0) may remove information
    that is legitimately useful. A channel that is unusually LOW is also carrying a signal.

WHY CLAMP AT 1.0 (not 0.1):
    By flooring at 1.0, familiar channels pass through unchanged (gate=1, no modulation).
    Unusual channels get amplified. This is a one-sided intervention.

PARAMS: logit_decay, log_tau, log_beta, log_gamma = 4 scalars (same as gelu181)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,) unit vector
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU186(nn.Module):
    """Per-channel absolute z-score vector gate: always amplify unusual channels."""

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
            z        = (x.detach() - mu_) / (std + self.eps)                     # (B, T, D) signed
            abs_z    = z.abs()                                                    # (B, T, D)
            gate_vec = (1.0 + beta * torch.tanh(gamma * abs_z)).clamp(1.0, 8.0)  # always ≥ 1

            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)                    # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
