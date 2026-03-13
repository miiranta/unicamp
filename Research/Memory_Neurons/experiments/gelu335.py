"""GELU335 – Output Sigmoid Gate Only + Cosine (3 params).

ABLATION: Remove the INPUT gate from gelu333 entirely.

HYPOTHESIS: The output gate (gating GELU(x) based on how usual the output value is)
might be MORE important than the input gate. This directly tests the asymmetry.

The output gate embeds "was this output expected?" — a prediction error signal.
The input gate embeds "was this input unusual?" — a raw novelty signal.

If PPL(gelu335) > PPL(gelu334): output gate is more useful than input gate.
If PPL(gelu335) < PPL(gelu334): input gate is more useful.
Combined (gelu333) should be best, but the relative magnitudes are informative.

GATE:
    gate_out = 2σ(β * z_out)   — only output gate (single β, symmetric)
    gate_cos = exp(−τ * cos(out, ema_dir))
    output   = GELU(x) × gate_out × gate_cos
    [no gate_in]

STATE: _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)  — 3 buffers
PARAMS: logit_decay, log_tau, log_beta  (3 scalars)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU335(nn.Module):
    """Output sigmoid gate only + cosine: cross-ablation of gelu334, tests output vs input."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_out_mean = self._ema_out_sq = self._ema_out_dir = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _z(self, val, mean, sq):
        var = (sq - mean.pow(2)).clamp(min=self.eps_var)
        return (val - mean) / (var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                of = out.detach().flatten(0, 1)
                bm_o = of.mean(0)
                self._ema_out_mean = bm_o.clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        gate_out = 2.0 * torch.sigmoid(beta * z_out)  # ∈ (0, 2)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_out * gate_cos

        with torch.no_grad():
            of = out.detach().flatten(0, 1)
            bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(bm_o, dim=0)

        return output
