"""GELU194 – Anti-Trend Surprise: Deviation Opposing EMA Velocity.

A DIFFERENT KIND OF NOVELTY:
    gelu80 measures: "is x_d far from its historical mean?" (displacement)
    gelu194 measures: "does x_d OPPOSE the direction the mean is currently drifting?" (anti-trend)

THE INTUITION:
    Language has temporal structure. If the model is reading about "economics" for several
    sentences, the activation means drift gradually toward that semantic region.
    A token like "however" that OPPOSES this drift is a discourse marker — a structural signal.
    A token that continues the economic theme follows the trend (predictable, familiar).

    Surprise should be highest for tokens that CONTRADICT the current trajectory.

THE MATH:
    Track per-channel EMA mean: μ_d(t) with decay d
    Track per-channel EMA velocity: v_d(t) = μ_d(t) - μ_d(t-1) (one-step change in mean)
    Track EMA of velocity magnitude: ema_vel_d = EMA of |v_d| (typical speed of drift)

    Anti-trend signal for token x:
        trend_alignment_d = (x_d - μ_d) × v_d  / (|v_d| × σ_d + ε)
        When positive: x_d deviates IN THE SAME DIRECTION as the current drift (trend-following)
        When negative: x_d deviates AGAINST the drift (anti-trend = novel)

    Anti-trend surprise:
        anti_d = ReLU(-trend_alignment_d)   — only penalize trend-following, reward anti-trend
        surp   = tanh(σ × mean_d(anti_d))

STABILITY:
    At startup, v_d ≈ 0 (EMA hasn't moved yet), so trend_alignment ≈ 0 for all tokens.
    The surprise signal turns on gradually as the EMA builds up velocity.
    ReLU ensures that trend-following channels don't contribute a NEGATIVE surprise.

PARAMS: logit_decay (EMA mean), logit_decay_v (velocity EMA), log_tau, log_sigma, log_w_raw
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_prev_mean (D,), _ema_vel (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU194(nn.Module):
    """Anti-trend surprise: amplify tokens that deviate against the direction of EMA drift."""

    def __init__(self, ema_decay: float = 0.9, ema_decay_v: float = 0.8, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # Primary EMA decay (for mean/var)
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay   / (1.0 - ema_decay))))
        # Velocity EMA decay (for smoothing v_d)
        self.logit_decay_v = nn.Parameter(torch.tensor(math.log(ema_decay_v / (1.0 - ema_decay_v))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:      torch.Tensor = None   # (D,) current EMA mean
        self._ema_sq:        torch.Tensor = None   # (D,) current EMA mean-square
        self._ema_prev_mean: torch.Tensor = None   # (D,) EMA mean from previous step
        self._ema_vel:       torch.Tensor = None   # (D,) EMA of velocity |μ(t) - μ(t-1)|
        self._ema_out:       torch.Tensor = None   # (D,) unit vector
        self._ready = False

    def reset_state(self):
        self._ema_mean      = None
        self._ema_sq        = None
        self._ema_prev_mean = None
        self._ema_vel       = None
        self._ema_out       = None
        self._ready         = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        d_vel  = torch.sigmoid(self.logit_decay_v).detach().item()
        tau    = self.log_tau.exp()
        sigma  = F.softplus(self.log_sigma)
        w      = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                m  = xf.mean(0).clone()
                self._ema_mean      = m
                self._ema_sq        = xf.pow(2).mean(0).clone()
                self._ema_prev_mean = m.clone()
                self._ema_vel       = torch.zeros(D, device=x.device, dtype=x.dtype)
                self._ema_out       = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready         = True
            return out

        # ── Per-channel z-score and velocity ──────────────────────────────
        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std  = var.sqrt()                               # (D,)

            # Velocity: how much is the EMA mean drifting per step?
            v_d  = self._ema_mean - self._ema_prev_mean    # (D,) signed velocity

            xd   = x.detach()
            dev  = xd - self._ema_mean.view(1, 1, D)       # (B, T, D) signed deviation

        # Anti-trend alignment: negative = token opposes drift
        # trend_align_d = dev_d × v_d / (|v_d| × std_d + ε)
        v_mag  = self._ema_vel.abs().clamp(min=1e-6).view(1, 1, D)
        std_   = std.view(1, 1, D)
        v_sign = v_d.view(1, 1, D)
        denom  = v_mag * std_ + self.eps
        trend_alignment = (dev * v_sign) / denom           # (B, T, D)
        anti_d = F.relu(-trend_alignment)                  # (B, T, D) ≥ 0, anti-trend channels
        surp   = torch.tanh(sigma * anti_d.mean(-1))       # (B, T)

        # ── Cosine familiarity gate ────────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            new_mean = d_val * self._ema_mean + (1-d_val) * xfl.mean(0)
            new_vel  = new_mean - self._ema_mean   # velocity this step
            self._ema_prev_mean = self._ema_mean.clone()
            self._ema_mean      = new_mean
            self._ema_sq        = d_val * self._ema_sq + (1-d_val) * xfl.pow(2).mean(0)
            self._ema_vel       = d_vel * self._ema_vel + (1-d_vel) * new_vel.abs()
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out       = d_val * self._ema_out + (1-d_val) * F.normalize(om, dim=0)

        return output
