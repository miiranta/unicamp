"""GELU195 – Log-Scale (Multiplicative) Surprise.

THE ADDITIVE ASSUMPTION OF Z-SCORES:
    gelu80: z_d = (x_d - μ_d) / σ_d   — measures ADDITIVE deviation from mean
    This is appropriate when channels have Gaussian-like distributions.

    But transformer FF activations after GELU can span several orders of magnitude.
    A channel with μ_d = 0.01 deviating to 0.02 is a 100% multiplicative change.
    A channel with μ_d = 10.0 deviating to 10.01 is a 0.1% multiplicative change.
    The z-score treats both as σ_d-scaled deviations — but the 100% change is arguably
    more semantically significant despite having the same z_d (if σ_d = 0.01, 10.0).

THE NEW IDEA: Log-Scale Surprise (Multiplicative Deviation)
    For each channel, measure the deviation in LOG space:
        log_ratio_d = log(|x_d| + ε) - log(|μ_d| + ε)
                    = log((|x_d| + ε) / (|μ_d| + ε))

    This measures how many octaves/decades the current value is from the mean.
    A 2× increase = log_ratio ≈ 0.693 regardless of base level.
    
    Sign: preserve direction of deviation using sign of (x_d - μ_d):
        signed_log_d = sign(x_d - μ_d) × |log_ratio_d|

    Surprise (mean absolute log-ratio):
        surp = tanh(σ × mean_d |log_ratio_d|)

    Cosine gate: same as gelu80.

WHY THIS MIGHT HELP:
    Small-magnitude channels with large relative changes are genuinely novel:
    E.g., a channel that normally holds ≈ 0.001 suddenly activating at 0.1 
    has the same z-score as a large channel's proportionally smaller change,
    but is 100× its normal amplitude — qualitatively more surprising.
    Log-scale captures this geometric deviation.

NUMERICAL CARE:
    |μ_d| may be near zero → log(0) problem.
    Solution: clamp denominator at ε_log = 0.01 (ensures ≥ 1% baseline).
    When μ_d ≈ 0, log( (|x_d| + ε) / ε_log ) measures log activation from zero.

PARAMS: logit_decay, log_tau, log_sigma, log_w_raw = 4 scalars (same as gelu80)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU195(nn.Module):
    """Log-scale multiplicative surprise: deviation measured in log space (octaves from mean)."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5, eps_log: float = 0.01):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.eps_log = eps_log    # floor for log-scale denominator
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

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

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Log-scale multiplicative deviation ────────────────────────────
        with torch.no_grad():
            xd      = x.detach()
            mu_abs  = self._ema_mean.abs().view(1, 1, D).clamp(min=self.eps_log)  # (D,) floor
            x_abs   = xd.abs() + self.eps                                          # (B, T, D)
            log_ratio = (x_abs / mu_abs).log()                                     # (B, T, D)
            # mean absolute log-ratio across channels
            mean_log_ratio = log_ratio.abs().mean(-1)                              # (B, T)

        surp = torch.tanh(sigma * mean_log_ratio)    # (B, T) ∈ (0, 1)

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
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xfl.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xfl.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
