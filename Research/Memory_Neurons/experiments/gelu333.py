"""GELU333 – Symmetric Sigmoid Gate (4 params).

MOTIVATION: gelu321 achieves 159.68 PPL with ASYMMETRIC sigmoid (4 β params + decay + τ = 6).
gelu314 shows that symmetric (tied β) input gate loses only 0.22 PPL vs 1 extra param.

HYPOTHESIS: Fully symmetric sigmoid for BOTH input AND output gates may match gelu321
while using only 4 params — the cleanest possible gelu211-class architecture.

GATE:
    gate_in  = 2σ(β_in  * z_in)    ∈ (0, 2) — symmetric: same β for push/pull
    gate_out = 2σ(β_out * z_out)   ∈ (0, 2) — symmetric output gate
    gate_cos = exp(−τ * cos(out, ema_dir))
    output   = GELU(x) × gate_in × gate_out × gate_cos

WHY SIGMOID OVER TANH:
    - 2σ(β·z) is smooth (0,2), never needs clamping
    - Gradient: 2·σ·(1−σ)·β — never zero for finite z
    - At z=0: gate=1 (identity) — stable init
    - Fewer hyperpriors: no separate γ sharpness needed (β encodes both magnitude and steepness)

PARAMS: logit_decay, log_tau, log_beta_in, log_beta_out  (4 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU333(nn.Module):
    """Symmetric sigmoid gate in+out+cos: simplest sigmoid variant of gelu211, 4 params."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))          # init d≈0.9
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_in = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # β≈1
        self.log_beta_out= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0))) # β≈0.5

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = None
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
        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        beta_in  = F.softplus(self.log_beta_in)
        beta_out = F.softplus(self.log_beta_out)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
                bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
                bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
                self._ema_mean = bm_x.clone(); self._ema_sq = bsq_x.clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = bsq_o.clone()
                self._ema_out_dir = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        gate_in  = 2.0 * torch.sigmoid(beta_in  * z_in)   # ∈ (0, 2)
        gate_out = 2.0 * torch.sigmoid(beta_out * z_out)  # ∈ (0, 2)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * bm_x
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * bsq_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(bm_o, dim=0)

        return output
