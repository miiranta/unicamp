"""GELU334 – Input Sigmoid Gate Only + Cosine (3 params).

ABLATION: Remove the output gate from gelu333 entirely.

HYPOTHESIS: Most of gelu211's benefit may come from the INPUT gate alone.
The output gate adds 1 param and some compute — is it actually needed?

If gelu334 ≈ gelu333 in PPL → output gate is redundant. Use gelu334 (3p) instead.
If gelu334 < gelu333 by ~0.3+ PPL → output gate earns its keep.

GATE:
    gate_in  = 2σ(β * z_in)   — only input gate (single β, symmetric)
    gate_cos = exp(−τ * cos(out, ema_dir))
    output   = GELU(x) × gate_in × gate_cos
    [no gate_out]

STATE: _ema_mean (D,), _ema_sq (D,), _ema_out_dir (D,)  — only 3 buffers (no output EMA)
PARAMS: logit_decay, log_tau, log_beta  (3 scalars)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU334(nn.Module):
    """Input sigmoid gate only + cosine: 3-param ablation removing output gate."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:    torch.Tensor = None
        self._ema_sq:      torch.Tensor = None
        self._ema_out_dir: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = self._ema_out_dir = None
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
                xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
                self._ema_mean    = xf.mean(0).clone()
                self._ema_sq      = xf.pow(2).mean(0).clone()
                self._ema_out_dir = F.normalize(of.mean(0), dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_in = self._z(x.detach(), self._ema_mean.view(1,1,D), self._ema_sq.view(1,1,D))

        gate_in  = 2.0 * torch.sigmoid(beta * z_in)  # ∈ (0, 2)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0); bm_o = of.mean(0)
            self._ema_mean    = d_val * self._ema_mean    + (1-d_val) * bm_x
            self._ema_sq      = d_val * self._ema_sq      + (1-d_val) * bsq_x
            self._ema_out_dir = d_val * self._ema_out_dir + (1-d_val) * F.normalize(bm_o, dim=0)

        return output
