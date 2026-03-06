"""GELU212 – Multi-Scale Asymmetric Per-Channel Gate (gelu190 + gelu191 statistics).

MOTIVATION:
    gelu190 (PPL 160.54): asymmetric gate on SINGLE-SCALE input z-scores
    gelu191 (PPL 165.15): dual-speed EMA, scalar MAX surprise gate

    gelu191 uses MAX(surp_fast, surp_slow) — scalar gate.
    But the BEST mechanism uses per-channel VECTOR gates (gelu181 lineage).

    COMBINATION: use dual-speed EMA statistics to compute z-scores,
    then apply the ASYMMETRIC PER-CHANNEL gate on a WEIGHTED BLEND of
    fast and slow z-scores.

        z_fast_d = (x_d − μ_fast_d) / σ_fast_d
        z_slow_d = (x_d − μ_slow_d) / σ_slow_d
        z_blend_d = α × z_fast_d + (1−α) × z_slow_d    α ∈ (0,1) learnable

        gate_d = 1 + β_up×ReLU(tanh(γ×z_blend_d)) − β_dn×ReLU(tanh(−γ×z_blend_d))

    This captures regime changes (z_slow large) and local fluctuations (z_fast large)
    with the full per-channel asymmetric gate.

WHY α BLEND > MAX:
    MAX collapses to a scalar. Blend stays per-channel:
    - Channel 42 might be novel on the FAST scale but familiar on SLOW: z_blend_42 ≈ α×z_fast
    - Channel 7 might be drifting on the SLOW scale: z_blend_7 ≈ (1-α)×z_slow
    Each channel retains its own multi-scale signal.

INITIALIZATION:
    decay_fast ≈ 0.5  (logit ≈ 0)
    decay_slow ≈ 0.95
    α init = 0.5 (equal blend)
    β_up = β_dn = 0.5 (symmetric, then asymmetry learned)

PARAMS: logit_decay_fast, logit_decay_slow, logit_alpha, log_beta_up, log_beta_dn, log_gamma, log_tau
STATE:  _ema_mean_f (D,), _ema_sq_f (D,), _ema_mean_s (D,), _ema_sq_s (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU212(nn.Module):
    """Multi-scale asymmetric per-channel gate: blend fast+slow z-scores, then asymmetric gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # decay_fast ≈ 0.5, decay_slow ≈ 0.95
        self.logit_decay_fast = nn.Parameter(torch.tensor(0.0))
        self.logit_decay_slow = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))
        # α blend: init 0.5
        self.logit_alpha  = nn.Parameter(torch.tensor(0.0))
        # Asymmetric gate params
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))

        self._ema_mean_f: torch.Tensor = None
        self._ema_sq_f:   torch.Tensor = None
        self._ema_mean_s: torch.Tensor = None
        self._ema_sq_s:   torch.Tensor = None
        self._ema_out:    torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean_f = None
        self._ema_sq_f   = None
        self._ema_mean_s = None
        self._ema_sq_s   = None
        self._ema_out    = None
        self._ready      = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_f    = torch.sigmoid(self.logit_decay_fast).detach().item()
        d_s    = torch.sigmoid(self.logit_decay_slow).detach().item()
        alpha  = torch.sigmoid(self.logit_alpha)          # in-graph for blend
        tau    = self.log_tau.exp()
        beta_up= F.softplus(self.log_beta_up)
        beta_dn= F.softplus(self.log_beta_dn)
        gamma  = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                mu = xf.mean(0)
                sq = xf.pow(2).mean(0)
                self._ema_mean_f = mu.clone()
                self._ema_sq_f   = sq.clone()
                self._ema_mean_s = mu.clone()
                self._ema_sq_s   = sq.clone()
                self._ema_out    = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready      = True
            return out

        with torch.no_grad():
            var_f  = (self._ema_sq_f - self._ema_mean_f.pow(2)).clamp(min=self.eps_var)
            var_s  = (self._ema_sq_s - self._ema_mean_s.pow(2)).clamp(min=self.eps_var)
            z_fast = (x.detach() - self._ema_mean_f.view(1,1,D)) / (var_f.sqrt().view(1,1,D) + self.eps)
            z_slow = (x.detach() - self._ema_mean_s.view(1,1,D)) / (var_s.sqrt().view(1,1,D) + self.eps)

        # ── Blend z-scores per channel ─────────────────────────────────
        alpha_v = alpha.view(1, 1, 1)
        z_blend = alpha_v * z_fast + (1.0 - alpha_v) * z_slow   # (B, T, D)

        # ── Asymmetric per-channel gate ────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_blend))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_blend))
        gate_vec = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)     # (B, T, D)

        # ── Cosine EMA output gate (scalar) ────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            xm = xf.mean(0)
            xs = xf.pow(2).mean(0)
            self._ema_mean_f = d_f * self._ema_mean_f + (1-d_f) * xm
            self._ema_sq_f   = d_f * self._ema_sq_f   + (1-d_f) * xs
            self._ema_mean_s = d_s * self._ema_mean_s + (1-d_s) * xm
            self._ema_sq_s   = d_s * self._ema_sq_s   + (1-d_s) * xs
            om = out.detach().flatten(0,1).mean(0)
            self._ema_out    = d_s * self._ema_out    + (1-d_s) * F.normalize(om, dim=0)

        return output
