"""GELU196 – Pre-GELU Input Channel Modulation (Surprise in Activation Space).

ALL PREVIOUS EXPERIMENTS apply the gate to the GELU OUTPUT:
    output = GELU(x) × gate

GELU196 MOVES THE SURPRISE INTO THE GELU'S INPUT:
    x_mod  = x × (1 + α × tanh(γ × z))     — (B, T, D) per-channel modulation of x
    output = GELU(x_mod)                     — GELU sees a modified input

WHY THIS IS FUNDAMENTALLY DIFFERENT:
    GELU(x_d) ≈ 0 for x_d < -2 (dead zone). Post-GELU gates can't resurrect dead channels.
    But if x_mod_d is AMPLIFIED before GELU, a channel that was near the GELU threshold
    may cross it and produce meaningful output. Surprising channels that were "almost active"
    become active.

    Conversely, for familiar channels (z_d ≈ 0), x_mod ≈ x, so GELU(x_mod) ≈ GELU(x).
    The modification only matters for channels with large |z_d|.

THE NONLINEARITY OF GELU CREATES A THRESHOLD EFFECT:
    Linear regime (x_d > 0):  GELU(α×x) ≈ α × GELU(x)  — similar to post-GELU scaling
    Threshold regime (~-2 < x_d < 0):  small amplification can push x_mod past threshold
    Dead zone (x_d < -2):  amplification can REVIVE the channel

    Post-GELU gating completely misses the threshold and dead-zone effects.
    This experiment tests whether pre-GELU modulation improves on post-GELU gating.

PARAMS: logit_decay, log_tau, log_alpha, log_gamma = 4 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU196(nn.Module):
    """Pre-GELU per-channel input modulation: surprise shifts x before nonlinearity."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        # alpha: pre-GELU modulation strength; init small to be conservative
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(math.exp(0.2) - 1.0)))
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
        alpha = F.softplus(self.log_alpha)
        gamma = F.softplus(self.log_gamma)

        if not self._ready:
            out = self._gelu(x)
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score ────────────────────────────────────────────
        with torch.no_grad():
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)
            mu_ = self._ema_mean.view(1, 1, D)
            z   = (x.detach() - mu_) / (std + self.eps)               # (B, T, D) signed

        # ── Pre-GELU modulation ────────────────────────────────────────────
        mod   = (1.0 + alpha * torch.tanh(gamma * z)).clamp(0.1, 3.0)  # (B, T, D)
        x_mod = x * mod                                                  # grad flows through x
        out   = self._gelu(x_mod)

        # ── Cosine familiarity gate on output (scalar) ────────────────────
        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim)                        # (B, T)

        output = out * gate_cos.unsqueeze(-1)

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xfl.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xfl.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
