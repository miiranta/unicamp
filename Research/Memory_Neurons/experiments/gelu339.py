"""GELU339 – Co-Novelty Gate (Joint z_in × z_out, 3 params).

CREATIVE IDEA: Rather than gating on input novelty AND output novelty INDEPENDENTLY
(which can contradict: novel input but expected output), gate on their JOINT novelty.

DEFINITION:
    z_joint = z_in × z_out  (element-wise product, shape B×T×D)

SEMANTICS of z_joint:
    z_joint > 0 (large):  BOTH novel input AND novel output       → "doubly surprising": AMPLIFY
    z_joint > 0 (small):  Both familiar input AND familiar output  → "routine": identity
    z_joint < 0 (large):  Novel input → SUPPRESSED output         → "unexpected suppression": suppress!
    z_joint < 0:          Familiar input → amplified output       → "unexplained amplification": suppress!

The co-novelty gate 2σ(β * z_joint) thus:
    - Amplifies channels where input and output AGREE in their novelty direction
    - Suppresses channels where novelty is INCONSISTENT (model is "confused")

This creates a CONSISTENCY FILTER: only pass through activations where the input
signal and the model's response are coherently anomalous together.

GATE:
    gate_joint = 2σ(β * z_in * z_out)    ∈ (0, 2)
    gate_cos   = exp(−τ * cos(out, ema_dir))
    output     = GELU(x) × gate_joint × gate_cos

PARAMS: logit_decay, log_tau, log_beta  (3 scalars — same as gelu334/335)
STATE:  5 EMA buffers (need both in and out z-scores)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU339(nn.Module):
    """Co-novelty gate: 2σ(β * z_in * z_out) — amplifies joint novelty, filters inconsistency."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # smaller init: z_joint can be large

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
        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
                bm_x = xf.mean(0); bm_o = of.mean(0)
                self._ema_mean = bm_x.clone(); self._ema_sq = xf.pow(2).mean(0).clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = of.pow(2).mean(0).clone()
                self._ema_out_dir = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))
            # Clamp individual z-scores before product to prevent extreme values
            z_in_c  = z_in.clamp(-3, 3)
            z_out_c = z_out.clamp(-3, 3)
            z_joint = z_in_c * z_out_c  # (B, T, D) — product z-score

        gate_joint = 2.0 * torch.sigmoid(beta * z_joint)  # ∈ (0, 2)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_joint * gate_cos

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
