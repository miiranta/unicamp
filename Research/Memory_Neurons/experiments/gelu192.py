"""GELU192 – Output-Magnitude-Weighted Z-Score Aggregation.

THE LIMITATION OF UNIFORM CHANNEL AVERAGING (gelu80):
    gelu80: surp = tanh(σ × mean_d |z_d|)  — each channel equally weighted

    But GELU(x_d) is nearly zero for x_d < -2 (dead zone).
    Dead/near-zero channels contribute noise to the mean |z|:
    - They may have large z_d (deviation from mean) but GELU(x_d) ≈ 0
    - Their modulation has almost zero impact on the output vector
    - Yet they pollute the aggregated surprise signal

THE NEW IDEA: Weight Channels by GELU Output Magnitude
    Compute softmax of |out_d| at temperature T:
        w_d = softmax(|out_d| / temp)     — (B, T, D), data-dependent weights
    
    Weighted surprise:
        mean_w_z = sum_d (w_d × |z_d|)   — (B, T) surprise over ACTIVE channels
    
    This focuses surprise detection on the channels that are ACTUALLY doing something.
    If channel d has |out_d| ≈ 0, it contributes almost no weight → ignored.
    If channel d has large |out_d|, its z_d contributes proportionally more.

WHY SOFTMAX (NOT HARD SELECTION):
    Smooth weighting keeps gradients flowing. Temperature T is a learned parameter:
    - Low temp → nearly uniform (like gelu80)
    - High temp → near one-hot, focuses on the dominant channel

THIS IS DATA-DEPENDENT WEIGHTING:
    Unlike gelu187 (learned static alpha_d), this weighting changes every forward pass
    based on the CURRENT token's GELU activations. Different tokens get different
    channel weights → the surprise detector is context-sensitive.

PARAMS: logit_decay, log_tau, log_sigma, log_w_raw, log_temp = 5 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU192(nn.Module):
    """Output-magnitude-weighted z-score: surprise weighted toward active GELU channels."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma     = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Temperature for softmax weighting; init T≈1.0
        self.log_temp      = nn.Parameter(torch.tensor(0.0))

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
        sigma = F.softplus(self.log_sigma)
        w     = F.softplus(self.log_w_raw)
        temp  = self.log_temp.exp().clamp(min=0.01)   # temperature > 0

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score ────────────────────────────────────────────
        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_ = var.sqrt().view(1, 1, D)
            mu_  = self._ema_mean.view(1, 1, D)
            z    = (x.detach() - mu_) / (std_ + self.eps)   # (B, T, D)
            abs_z = z.abs()

        # ── Output-magnitude weights (data-dependent) ─────────────────────
        abs_out = out.detach().abs()                         # (B, T, D)
        wts     = F.softmax(abs_out / temp, dim=-1)          # (B, T, D) — sum to 1 over D
        mean_w_z    = (abs_z * wts).sum(-1)                  # (B, T) weighted surprise
        surp        = torch.tanh(sigma * mean_w_z)           # (B, T)

        # ── Cosine familiarity gate ────────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xfl.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xfl.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
