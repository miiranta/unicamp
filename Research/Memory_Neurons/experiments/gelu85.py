"""GELU85 – Fusion Gate: Per-Channel Z-Score × Asymmetric Cosine.

MOTIVATION:
    gelu80 (best at 7.0%): captures INPUT novelty via per-channel z-score.
        surp = tanh(σ × mean_d(|z_d|)) where z_d = (x_d - μ_d)/σ_d
        gate = exp(-τ × cos_out) × (1 + w × surp)

    gelu78 (5.8%): captures DIRECTIONAL novelty via asymmetric output cosine.
        gate = exp(-τ_s × relu(cos)) × (1 + w_a × relu(-cos)) × (1 + w_surp × surp)
        KEY: anti-correlated tokens get AMPLIFIED above 1.0

    These two signals are complementary:
    - gelu80: "are the channel activations individually unusual?" (magnitude/variance novelty)
    - gelu78: "is the output direction opposite to what we usually see?" (directional novelty)

    Fusion:
        gate = [exp(-τ_s × relu(cos_out)) × (1 + w_a × relu(-cos_out))]   ← asymmetric cosine
               × (1 + w_surp × surp)                                         ← per-channel z-score
        output = GELU(x) × gate.unsqueeze(-1)

    The two signals interact: a token that is BOTH channel-surprising AND directionally
    anti-correlated will have gate >> 1, strongly amplified.
    A fully familiar token (low z + positive cosine) will be strongly suppressed.

    PROOF OF COMPLEMENTARITY:
    cos_out and mean_d(|z_d|) are largely independent:
    - cos_out depends on the OUTPUT direction relative to EMA
    - mean_d(|z_d|) depends on which INPUT channels deviated from their mean
    A token can have unusual channel magnitudes but typical output direction (e.g. shifted in
    a familiar axis → low cos gate but low asymmetric boost), or vice versa.
    The product captures BOTH axes of novelty.

Params: logit_decay, log_tau_s, log_w_a_raw, log_sigma_raw, log_w_surp = 5 scalars.
State: _ema_mean (D,), _ema_sq (D,), _ema_out (D,) unit vector.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU85(nn.Module):
    """Fusion: per-channel z-score surprise × asymmetric output cosine gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau_s     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_w_a_raw   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_surp    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        tau_s  = self.log_tau_s.exp()
        w_a    = F.softplus(self.log_w_a_raw)
        sigma  = F.softplus(self.log_sigma_raw)
        w_surp = F.softplus(self.log_w_surp)

        out = self._gelu(x)   # (B, T, D)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(of.mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score surprise ────────────────────────────────
        var      = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
        std      = var.sqrt()                                         # (D,)
        mu_      = self._ema_mean.view(1, 1, D)
        std_     = std.view(1, 1, D)
        z        = (x.detach() - mu_) / (std_ + self.eps)            # (B, T, D)
        mean_abs_z = z.abs().mean(dim=-1)                             # (B, T)
        surp     = torch.tanh(sigma * mean_abs_z)                     # (B, T)

        # ── Asymmetric output cosine gate ────────────────────────────────
        out_norm = F.normalize(out.detach(), dim=-1)                  # (B, T, D)
        ema_norm = F.normalize(self._ema_out.unsqueeze(0).unsqueeze(0), dim=-1)
        cos_out  = (out_norm * ema_norm).sum(dim=-1)                  # (B, T)

        supp_arm = torch.exp(-tau_s * cos_out.clamp(min=0.0))        # (B, T) ≤ 1
        ampl_arm = 1.0 + w_a * (-cos_out).clamp(min=0.0)             # (B, T) ≥ 1

        gate = supp_arm * ampl_arm * (1.0 + w_surp * surp)           # (B, T)
        result = out * gate.unsqueeze(-1)

        # ── EMA updates ─────────────────────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            of  = out.detach().flatten(0, 1)
            xm  = xf.mean(0)
            xsq = xf.pow(2).mean(0)
            om  = F.normalize(of.mean(0), dim=0)
            self._ema_mean = d_val * self._ema_mean + (1.0 - d_val) * xm
            self._ema_sq   = d_val * self._ema_sq   + (1.0 - d_val) * xsq
            self._ema_out  = F.normalize(d_val * self._ema_out + (1.0 - d_val) * of.mean(0), dim=0)

        return result
