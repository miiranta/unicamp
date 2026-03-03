"""GELU191 – Dual-Speed EMA: Multi-Scale Temporal Surprise.

THE PROBLEM WITH SINGLE-SCALE EMA (gelu80):
    gelu80 maintains ONE EMA with a single learned decay d ≈ 0.9.
    This tracks a single "representative past" for each channel.

    But novelty has multiple time scales:
    - A token that's different from the LAST FEW tokens → locally novel
    - A token that's different from the LAST MANY tokens → globally novel
    - A token that's locally familiar but globally novel → REGIME CHANGE
    - A token that's locally novel but globally familiar → FLUCTUATION

    Single-scale EMA conflates all of these.

THE NEW IDEA: Two EMA Means at Different Speeds
    μ_fast_d: EMA with high decay (fast-forgetting, tracks recent ~5 tokens)
    μ_slow_d: EMA with low decay (slow-forgetting, tracks long-term ~50 tokens)

    Two sets of statistics → two z-scores:
        z_fast_d = (x_d - μ_fast_d) / (σ_fast_d + ε)   — vs recent history
        z_slow_d = (x_d - μ_slow_d) / (σ_slow_d + ε)   — vs long-term history

    Combined surprise (max across scales):
        surp = tanh(σ × max(mean_d|z_fast|, mean_d|z_slow|))

    This fires whenever a token is surprising at EITHER time scale.
    A token surprising at only fast scale: sudden spike (fluctuation)
    A token surprising at only slow scale: gradual drift detected by slow-scale
    A token surprising at BOTH: regime change or true novelty

REGIME CHANGE DETECTION:
    When the text topic shifts (e.g., end of one article, start of another in wiki),
    z_fast stays high for many tokens (each one differs from its recent neighbors),
    while z_slow fires on the first few tokens of the new regime.
    max(surp_fast, surp_slow) captures this robustly.

PARAMS: logit_decay_fast, logit_decay_slow, log_tau, log_sigma, log_w_raw = 5 scalars
    (init: decay_fast≈0.5, decay_slow≈0.95)
STATE:  _ema_mean_f (D,), _ema_sq_f (D,), _ema_mean_s (D,), _ema_sq_s (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU191(nn.Module):
    """Dual-speed EMA: max surprise across fast (recent) and slow (long-term) time scales."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # Fast decay≈0.5: logit=log(0.5/0.5)=0.0
        self.logit_decay_fast = nn.Parameter(torch.tensor(0.0))
        # Slow decay≈0.95: logit=log(0.95/0.05)=log(19)
        self.logit_decay_slow = nn.Parameter(torch.tensor(math.log(0.95 / 0.05)))
        self.log_tau          = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma        = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw        = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # Fast-scale stats
        self._ema_mean_f: torch.Tensor = None
        self._ema_sq_f:   torch.Tensor = None
        # Slow-scale stats
        self._ema_mean_s: torch.Tensor = None
        self._ema_sq_s:   torch.Tensor = None
        # Output EMA (shared)
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

        df = torch.sigmoid(self.logit_decay_fast).detach().item()
        ds = torch.sigmoid(self.logit_decay_slow).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                xm, xsq = xf.mean(0).clone(), xf.pow(2).mean(0).clone()
                self._ema_mean_f = xm
                self._ema_sq_f   = xsq
                self._ema_mean_s = xm.clone()
                self._ema_sq_s   = xsq.clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-scores at two scales ────────────────────────────
        with torch.no_grad():
            var_f = (self._ema_sq_f - self._ema_mean_f.pow(2)).clamp(min=self.eps_var)
            var_s = (self._ema_sq_s - self._ema_mean_s.pow(2)).clamp(min=self.eps_var)
            std_f = var_f.sqrt().view(1, 1, D)
            std_s = var_s.sqrt().view(1, 1, D)
            mu_f  = self._ema_mean_f.view(1, 1, D)
            mu_s  = self._ema_mean_s.view(1, 1, D)
            xd    = x.detach()
            z_f   = (xd - mu_f) / (std_f + self.eps)    # (B, T, D)
            z_s   = (xd - mu_s) / (std_s + self.eps)    # (B, T, D)

        mean_absz_f = z_f.abs().mean(-1)   # (B, T)
        mean_absz_s = z_s.abs().mean(-1)   # (B, T)
        # Max surprise across scales
        surp = torch.tanh(sigma * torch.max(mean_absz_f, mean_absz_s))

        # ── Cosine familiarity gate ────────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp)
        output = out * gate.unsqueeze(-1)

        # ── Update both EMA scales ────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            xfl_m  = xfl.mean(0)
            xfl_sq = xfl.pow(2).mean(0)
            self._ema_mean_f = df * self._ema_mean_f + (1-df) * xfl_m
            self._ema_sq_f   = df * self._ema_sq_f   + (1-df) * xfl_sq
            self._ema_mean_s = ds * self._ema_mean_s + (1-ds) * xfl_m
            self._ema_sq_s   = ds * self._ema_sq_s   + (1-ds) * xfl_sq
            om = out.detach().flatten(0, 1).mean(0)
            # Use slow decay for output EMA
            self._ema_out = ds * self._ema_out + (1-ds) * F.normalize(om, dim=0)

        return output
