"""GELU341 – Adaptive-τ Cosine Gate (Batch-Calibrated Novelty, 5 params).

MOTIVATION: In gelu211/321, τ is a fixed learnable scalar. But its effectiveness depends
on the VARIANCE of cosine similarities within each batch:

    - If all tokens in a batch are HIGHLY ALIGNED with ema_dir (cos_sim ≈ 0.9 for all),
      the cosine gate fires almost uniformly — it has low DISCRIMINATIVE POWER.
      A larger τ would help, but may overfit to batches where variance is accidentally high.

    - If tokens are DIVERSE (cos_sim spans -0.5 to +0.8), even a moderate τ produces
      a strong discriminative gate.

SOLUTION: Make τ SELF-CALIBRATE based on the batch's cosine distribution:
    cos_spread = std(cos_sim)  over all (B×T) positions in the batch
    τ_eff      = τ × (1 + scale × cos_spread)

    When cos_spread is HIGH (diverse batch): τ_eff > τ — applies stronger cosine gate
    When cos_spread is LOW (homogeneous batch): τ_eff ≈ τ — cosine gate is less relevant

GATE:
    cos_spread = std(cos_sim_BxT)         # scalar, fully differentiable
    τ_eff      = τ * (1 + scale * cos_spread)
    gate_cos   = exp(−τ_eff * cos_sim)   # shape (B,T,1)
    gate_in    = 2σ(β_in  * z_in)
    gate_out   = 2σ(β_out * z_out)
    output     = GELU(x) × gate_in × gate_out × gate_cos

GRADIENT NOTE: cos_spread is computed from gate-detached cos_sim values.
τ_eff and scale get gradient via the gate_cos computation path.

PARAMS: logit_decay, log_tau, log_beta_in, log_beta_out, log_scale  (5 scalars)
STATE:  5 EMA buffers (same as gelu333)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU341(nn.Module):
    """Adaptive-τ cosine gate: τ auto-scales with batch cosine spread; 5 params."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_in = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_scale   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))  # scale ≈ 1

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
        scale    = F.softplus(self.log_scale)

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

        gate_in  = 2.0 * torch.sigmoid(beta_in  * z_in)
        gate_out = 2.0 * torch.sigmoid(beta_out * z_out)

        # Adaptive τ: scale by batch cos_sim spread
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)   # (B, T)

        cos_spread = cos_sim.std().detach()   # scalar, detach → τ_eff gradient through scale only
        tau_eff    = tau * (1.0 + scale * cos_spread)        # still differentiable w.r.t. tau, scale
        gate_cos   = torch.exp(-tau_eff * cos_sim).unsqueeze(-1)

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
