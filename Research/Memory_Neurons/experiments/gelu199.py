"""GELU199 – Surprise-Boosted Residual Deviation Gate.

ALL PREVIOUS EXPERIMENTS scale GELU(x) by a multiplicative gate:
    output = GELU(x) × gate(x)

GELU199 ADDS A RESIDUAL TERM instead:
    surprise_vec = GELU(x) - ema_out_vec          — direction of output deviation
    output = GELU(x) + α × surp_scalar × (surprise_vec / ema_rms)

GEOMETRIC INTERPRETATION:
    ema_out_vec: the "familiar" GELU output centroid
    (GELU(x) - ema_out_vec): how far this token's output departs from familiar
    α × surp × dev_normalized: an EXTRA PUSH in the direction of novelty

    Familiar tokens (low surp): output ≈ GELU(x) — small residual
    Novel tokens (high surp):   output = (1+α) × GELU(x) - α × ema_out_vec / ema_rms × ema_rms
                                ← amplifies output AND subtracts familiar direction

    This is explicit "de-familiarity": push the output AWAY from the mean centroid
    by an amount proportional to input z-score novelty.

WHY RESIDUAL BEATS SCALAR GATE:
    Scalar gate: output = GELU(x) × c    — uniform scaling in all directions
    Residual:    output = GELU(x) + α×surp×(GELU(x) − ema_out_vec) / ema_rms
                        — subtracts the familiar direction, not just scales uniformly

PARAMS: logit_decay, log_sigma, log_alpha = 3 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_vec (D,) unnormalized output centroid
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU199(nn.Module):
    """Surprise-boosted residual: GELU(x) + α × surp × (GELU(x) − ema_out_vec) / ema_rms."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_sigma   = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

        self._ema_mean:    torch.Tensor = None
        self._ema_sq:      torch.Tensor = None
        self._ema_out_vec: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean    = None
        self._ema_sq      = None
        self._ema_out_vec = None
        self._ready       = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        sigma = F.softplus(self.log_sigma)
        alpha = F.softplus(self.log_alpha)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean    = xf.mean(0).clone()
                self._ema_sq      = xf.pow(2).mean(0).clone()
                self._ema_out_vec = out.detach().flatten(0,1).mean(0).clone()
                self._ready       = True
            return out

        # ── Per-channel z-score → scalar surprise ─────────────────────────
        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_ = var.sqrt().view(1, 1, D)
            mu_  = self._ema_mean.view(1, 1, D)
            z    = (x.detach() - mu_) / (std_ + self.eps)
            mean_absz = z.abs().mean(-1)   # (B, T)

        surp = torch.tanh(sigma * mean_absz)   # (B, T)

        # ── Residual deviation from familiar output ────────────────────────
        ema_v   = self._ema_out_vec.view(1, 1, D)
        dev     = out - ema_v                                  # (B, T, D)
        ema_rms = (ema_v.pow(2).mean() + self.eps).sqrt()
        dev_n   = dev / (ema_rms + self.eps)

        output = out + alpha * surp.unsqueeze(-1) * dev_n

        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            self._ema_mean    = d_val * self._ema_mean    + (1-d_val) * xfl.mean(0)
            self._ema_sq      = d_val * self._ema_sq      + (1-d_val) * xfl.pow(2).mean(0)
            self._ema_out_vec = d_val * self._ema_out_vec + (1-d_val) * out.detach().flatten(0,1).mean(0)

        return output
