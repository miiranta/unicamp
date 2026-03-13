"""GELU338 – MAD-Robust Sigmoid Gate (4 params).

MOTIVATION: All prior gates normalize by VARIANCE (E[x²] - E[x]²). Variance is
SENSITIVE TO OUTLIERS — a single extreme activation squares to a large value,
inflating σ and shrinking every z-score. This makes the gate BLIND to moderate
but consistent deviations when outliers dominate the normalization.

SOLUTION: Normalize by MEAN ABSOLUTE DEVIATION (MAD) instead.
    MAD = E[|x - mean|]   — robust because |·| doesn't amplify outliers quadratically

    z_in_mad = (x - ema_mean) / (ema_mad + eps)
    gate_in   = 2σ(β_in * z_in_mad)

COMPARISON: For a Gaussian distribution, σ ≈ 1.25 * MAD, so the z-scores are
comparable in scale. For heavy-tailed distributions (which neural activations often
are after ReLU/GELU), MAD is more robust.

IMPLEMENTATION: 
    Track EMA of |x - ema_mean| as the scale estimate:
    ema_mad ← d * ema_mad + (1-d) * |x - ema_mean|.mean(0)

    This requires approximate alternating updates (compute residual using current ema_mean).

GATE:
    gate_in   = 2σ(β_in  * z_in_mad)
    gate_out  = 2σ(β_out * z_out_mad)
    gate_cos  = exp(−τ * cos)
    output    = GELU(x) × gate_in × gate_out × gate_cos

PARAMS: logit_decay, log_tau, log_beta_in, log_beta_out  (4 scalars — same as gelu333)
STATE:  _ema_mean (D), _ema_mad_in (D), _ema_out_mean (D), _ema_mad_out (D), _ema_out_dir (D)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU338(nn.Module):
    """MAD-robust z-score sigmoid gate: uses mean absolute deviation instead of variance."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_in = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

        self._ema_mean:    torch.Tensor = None
        self._ema_mad_in:  torch.Tensor = None
        self._ema_out_mean:torch.Tensor = None
        self._ema_mad_out: torch.Tensor = None
        self._ema_out_dir: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_mad_in = None
        self._ema_out_mean = self._ema_mad_out = self._ema_out_dir = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        beta_in  = F.softplus(self.log_beta_in)
        beta_out = F.softplus(self.log_beta_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
                bm_x = xf.mean(0); bm_o = of.mean(0)
                mad_x = (xf - bm_x.unsqueeze(0)).abs().mean(0).clamp(min=1e-4)
                mad_o = (of - bm_o.unsqueeze(0)).abs().mean(0).clamp(min=1e-4)
                self._ema_mean     = bm_x.clone()
                self._ema_mad_in   = mad_x.clone()
                self._ema_out_mean = bm_o.clone()
                self._ema_mad_out  = mad_o.clone()
                self._ema_out_dir  = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        # MAD-normalized z-scores
        with torch.no_grad():
            z_in  = (x.detach()   - self._ema_mean.view(1,1,D))     / (self._ema_mad_in.view(1,1,D)  + self.eps)
            z_out = (out.detach() - self._ema_out_mean.view(1,1,D)) / (self._ema_mad_out.view(1,1,D) + self.eps)

        gate_in  = 2.0 * torch.sigmoid(beta_in  * z_in)
        gate_out = 2.0 * torch.sigmoid(beta_out * z_out)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0); bm_o = of.mean(0)
            # Update MAD using current mean estimate
            mad_x = (xf - bm_x.unsqueeze(0)).abs().mean(0).clamp(min=1e-4)
            mad_o = (of - bm_o.unsqueeze(0)).abs().mean(0).clamp(min=1e-4)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * bm_x
            self._ema_mad_in   = d_val * self._ema_mad_in   + (1-d_val) * mad_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * bm_o
            self._ema_mad_out  = d_val * self._ema_mad_out  + (1-d_val) * mad_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(bm_o, dim=0)

        return output
