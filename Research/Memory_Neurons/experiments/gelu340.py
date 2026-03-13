"""GELU340 – Log-Deviation Sigmoid Gate (Compressed z-score, 4 params).

MOTIVATION: Both tanh and sigmoid gate formulas saturate for large |z|:
    sigmoid: 2σ(β·z) → gradient 2·σ·(1-σ)·β → approaches 0 as |z| → ∞
    tanh:    β·tanh(γ·z) → gradient β·γ·sech²(γ·z) → 0 as |z| → ∞

For a heavy-tailed activation distribution (which post-GELU activations often are),
extreme z-scores (|z| > 3) saturate the gate AND lose gradient signal. This means
rare but potentially important activations are both clamped AND provide no gradient
for β/γ to learn from.

SOLUTION: Use a LOG-SPACE z-score:
    z_log = sign(z) × log(1 + |z|)

This compresses large z-scores logarithmically:
    z = 1   → z_log ≈ 0.69  — normal deviation
    z = 3   → z_log ≈ 1.39  — keeps gradient alive
    z = 10  → z_log ≈ 2.40  — still in sigmoid's active range (not saturated)
    z = 100 → z_log ≈ 4.61  — extremely deviant but STILL has gradient

GATE:
    z_log_in  = sign(z_in)  * log1p(|z_in|)   # monotone, same sign, compressed
    z_log_out = sign(z_out) * log1p(|z_out|)
    gate_in   = 2σ(β_in  * z_log_in)   ∈ (0, 2)
    gate_out  = 2σ(β_out * z_log_out)  ∈ (0, 2)
    gate_cos  = exp(−τ * cos)
    output    = GELU(x) × gate_in × gate_out × gate_cos

WHY THIS IS CREATIVE: It changes the EFFECTIVE SENSITIVITY CURVE — the gate responds
linearly to deviations near the mean, then transitions to log-sensitivity for outliers.
The gate shape itself is not symmetric (tanh-like), but the NORMALIZATION is softer.

PARAMS: logit_decay, log_tau, log_beta_in, log_beta_out  (4 scalars — same as gelu333)
STATE:  5 EMA buffers (same as gelu333)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU340(nn.Module):
    """Log-deviation sigmoid gate: sign(z)*log1p(|z|) keeps gradient alive for outliers."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_in = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

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

    @staticmethod
    def _log_z(z):
        """Compress z via sign(z) * log1p(|z|). Monotone, same sign, log-scale for outliers."""
        return z.sign() * torch.log1p(z.abs())

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
                self._ema_mean = bm_x.clone(); self._ema_sq = xf.pow(2).mean(0).clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = of.pow(2).mean(0).clone()
                self._ema_out_dir = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))
            # Apply log compression
            z_log_in  = self._log_z(z_in)
            z_log_out = self._log_z(z_out)

        gate_in  = 2.0 * torch.sigmoid(beta_in  * z_log_in)
        gate_out = 2.0 * torch.sigmoid(beta_out * z_log_out)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
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
