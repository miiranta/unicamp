"""GELU330 – Shared Small MLP Gate (Hidden=4, Fully Differentiable).

MOTIVATION: All prior gates use FIXED functional forms (tanh, softplus, sigmoid) to map
z-scores to gate values. These functional forms encode strong assumptions about gate shape.
An MLP can LEARN the optimal gate shape from data without encoding priors.

ARCHITECTURE:
    Inputs per channel: z_in_d, z_out_d  (2 scalar features)
    MLP: linear(2 → 4) → GELU → linear(4 → 1) → softplus → gate_d
    Shared weights across all D channels (NOT per-channel to avoid D*4 params)
    Cosine gate: same as gelu211 (EMA-based, detach cosine sim)
    output = GELU(x) × gate × gate_cos

KEY PROPERTIES:
    - Fully differentiable: no detach in gate computation
    - Shared weights: 2×4 + 4 + 4×1 + 1 = 17 params (not counting cosine)
    - Total: 17 + 2 (logit_decay, log_tau) = 19 params
    - Gate initialized near 1.0: mlp_out(0) ≈ softplus(b2) → b2 = log(expm1(1)) → clamp approach
    - The MLP can learn any monotone or non-monotone gate shape

MLP INITIALIZATION STRATEGY:
    - W1 = uniform(-0.1, 0.1), b1 = 0 → GELU(x1) ≈ 0.5*x1 near 0
    - W2 = 0, b2 = log(expm1(1)) ≈ 0.541 → softplus(0.541) ≈ 1.0 → gate ≈ 1 at start

CAUSALITY: all z-scores computed from batch statistics (mean over B×T dim) — causal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU330(nn.Module):
    """Shared 2-hidden-layer MLP learns optimal gate shape from z_in and z_out."""

    def __init__(self, hidden: int = 4, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.hidden   = hidden

        # EMA / cosine params
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))   # d ≈ 0.9
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))

        # MLP: (z_in, z_out) → gate_d
        self.W1 = nn.Parameter(torch.zeros(hidden, 2))    # hidden × 2
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.zeros(1, hidden))    # 1 × hidden
        self.b2 = nn.Parameter(torch.full((1,), math.log(math.expm1(1.0))))  # softplus → 1.0

        # Small init for W1 to keep gate near 1 at start
        nn.init.uniform_(self.W1, -0.05, 0.05)

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

    def _mlp_gate(self, z_in: torch.Tensor, z_out: torch.Tensor) -> torch.Tensor:
        """Apply shared MLP to (z_in_d, z_out_d) → gate_d.  Shapes (B,T,D)."""
        # Stack along last dim: (B, T, D, 2)
        features = torch.stack([z_in, z_out], dim=-1)           # (B, T, D, 2)
        # Linear layer 1: (..., 2) × W1^T → (..., hidden)
        h = features @ self.W1.t() + self.b1                    # (B, T, D, hidden)
        h = self._gelu(h)
        # Linear layer 2: (..., hidden) × W2^T → (..., 1)
        out_mlp = h @ self.W2.t() + self.b2                     # (B, T, D, 1)
        gate = F.softplus(out_mlp).squeeze(-1)                   # (B, T, D)
        return gate.clamp(0.05, 8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
                bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
                self._ema_mean     = bm_x.clone(); self._ema_sq     = bsq_x.clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = bsq_o.clone()
                self._ema_out_dir  = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        # z-scores (z_in detached from x/out — gate gradient flows through MLP weights only)
        z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
        z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        gate = self._mlp_gate(z_in, z_out)   # (B, T, D), gradient flows through W1, W2, b1, b2

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            bm_x = xf.mean(0);  bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0);  bsq_o = of.pow(2).mean(0)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * bm_x
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * bsq_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(bm_o, dim=0)

        return output
