"""GELU332 – Asymmetric Output Clamp (Different Ranges for Positive/Negative z_out).

MOTIVATION: gelu211 uses a single clamp `(0.1, 5.0)` for gate_out regardless of whether
z_out is positive or negative. But the semantics of positive/negative output z-scores differ:

    z_out > 0:  output is ABOVE the average → novel / unexpected → may want wider amplification
    z_out < 0:  output is BELOW the average → familiar / expected → may want tighter suppression

HYPOTHESIS: Using DIFFERENT clamp bounds for the two regimes may better fit the
underlying distribution and allow more precise control:

    When z_out > 0:  allow gate_out ∈ [min_pos, max_uc]  (upper clamp controls amplification)
    When z_out < 0:  allow gate_out ∈ [min_lc, max_neg]  (lower clamp controls suppression)

IMPLEMENTATION: Smooth per-channel upper and lower bounds using learnable offsets:
    max_uc  = softplus(log_max_uc)   — upper bound for positive z_out channels
    min_lc  = sigmoid(logit_min_lc) * 0.9 + 0.05  — lower bound for negative z_out channels
    Combined via soft selection based on sign of z_out:
        gate_out = soft_max(gate_raw, gate_raw.clamp(...))

PRACTICAL IMPLEMENTATION: Param-efficient approach:
    gate_raw = 1 + β_out * tanh(γ_out * z_out)
    For positive z_out: allow up to max_up (learned ≥ 5.0)
    For negative z_out: allow down to min_dn (learned ≤ 0.1)
    Smooth split: weight = sigmoid(k * z_out) where k is large (sharp transition at z=0)
    gate_pos = clamp(gate_raw, 0.5, max_up)   — for positive side
    gate_neg = clamp(gate_raw, min_dn, 1.5)   — for negative side
    gate_out = weight * gate_pos + (1-weight) * gate_neg

PARAMS: 9 scalars — logit_decay, log_tau, log_beta_up (in), log_beta_dn (in), log_gamma (in),
        log_beta_out, log_gamma_out, log_max_up, logit_min_dn
STATE: same 5 EMA buffers as gelu211
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU332(nn.Module):
    """Asymmetric output clamp: separate upper/lower bounds for positive/negative output z-scores."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_max_up    = nn.Parameter(torch.tensor(math.log(math.exp(5.0) - 1.0)))   # init ≈ 5.0
        self.logit_min_dn  = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))              # init ≈ 0.1

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

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)
        max_up    = F.softplus(self.log_max_up)          # learned upper clamp for positive z_out
        min_dn    = torch.sigmoid(self.logit_min_dn)     # learned lower clamp for negative z_out ∈ (0,1)

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

        with torch.no_grad():
            z_in  = self._z(x.detach(),   self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)

        # Compute raw gate_out
        gate_raw = 1.0 + beta_out * torch.tanh(gamma_out * z_out)   # (B, T, D)

        # ASYMMETRIC CLAMP: soft-split by sign of z_out
        # weight = 1 when z_out >> 0 (positive z), weight = 0 when z_out << 0 (negative z)
        weight = torch.sigmoid(10.0 * z_out)   # sharp transition at z_out = 0

        # Positive-side gate: clamp to [0.5, max_up] — wider upper range for novel outputs
        gate_pos = gate_raw.clamp(0.5, max_up.item())

        # Negative-side gate: clamp to [min_dn, 1.5] — tighter suppression for familiar outputs
        gate_neg = gate_raw.clamp(min_dn.item(), 1.5)

        # Soft combination
        gate_out = weight * gate_pos + (1.0 - weight) * gate_neg

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_in * gate_out * gate_cos

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
