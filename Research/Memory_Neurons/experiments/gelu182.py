"""GELU182 – Per-Position EMA Z-Score Gate.

ORTHOGONAL AXIS NOT YET EXPLORED:
    All EMA experiments maintain statistics aggregated over ALL sequence positions.
    But position matters: position 0 almost always gets the start token; later positions
    accumulate context-specific content. A global EMA mixes position-0 stats with
    position-32 stats — "unusual at position 32" is different from "unusual globally."

PER-POSITION EMA:
    Maintain separate (μ_t, σ²_t) for each position t ∈ {0, ..., T-1}:
        _ema_mean (T, D),  _ema_sq (T, D)

    For each token at position t:
        z_{t,d} = (x_{b,t,d} - ema_mean[t,d]) / (std[t,d] + ε)
        surp_t  = tanh(σ × mean_d |z_{t,d}|)

    Gate: exp(-τ × cos(out_t, ema_out[t])) × (1 + w × surp_t)
    Where ema_out[t] is also per-position (T, D).

INTUITION:
    "Is this token unusual for where it appears in the sequence?"
    vs gelu80: "Is this token unusual globally?"
    The per-position reference is more specific and captures positional token distribution.

PARAMS: logit_decay, log_tau, log_sigma_raw, log_w_raw = 4 scalars
STATE:  _ema_mean (T, D), _ema_sq (T, D), _ema_out (T, D)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU182(nn.Module):
    """Per-position EMA z-score gate: separate statistics per sequence position."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None   # (T, D)
        self._ema_sq:   torch.Tensor = None   # (T, D)
        self._ema_out:  torch.Tensor = None   # (T, D) unit vectors per position
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
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xm  = x.detach().mean(0)        # (T, D)
                xsq = x.detach().pow(2).mean(0) # (T, D)
                om  = out.detach().mean(0)       # (T, D)
                self._ema_mean = xm.clone()
                self._ema_sq   = xsq.clone()
                self._ema_out  = F.normalize(om, dim=-1).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)  # (T, D)
            std  = var.sqrt().unsqueeze(0)           # (1, T, D)
            mu_  = self._ema_mean.unsqueeze(0)       # (1, T, D)
            z    = (x.detach() - mu_) / (std + self.eps)       # (B, T, D)
            surprise = torch.tanh(sigma * z.abs().mean(-1))    # (B, T)

            out_n    = F.normalize(out.detach(), dim=-1)        # (B, T, D)
            ema_n    = self._ema_out.unsqueeze(0)               # (1, T, D) already normalized
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)    # (B, T)
            gate_cos = torch.exp(-tau * cos_sim)

        output = out * (gate_cos * (1.0 + w * surprise)).unsqueeze(-1)

        with torch.no_grad():
            xm  = x.detach().mean(0)
            xsq = x.detach().pow(2).mean(0)
            om  = out.detach().mean(0)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xm
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xsq
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=-1)

        return output
