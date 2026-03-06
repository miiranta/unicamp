"""GELU225 – Input-Output Divergence Gate.

MOTIVATION — DETECTING UNEXPECTED TRANSFORMATIONS:
    gelu190: gate based on z_in (input novelty)
    gelu221: gate based on z_out (output novelty)
    gelu211: gate based on z_in × z_out (conjunctive novelty)

    NEW HYPOTHESIS: The most INFORMATIVE signal is when input novelty and output
    novelty DIVERGE — i.e., the GELU transformation is doing something unexpected:

        Case A: z_in_d >> 0, z_out_d >> 0 → normal (high input → high output)
        Case B: z_in_d << 0, z_out_d << 0 → normal (low input → low output)
        Case C: z_in_d >> 0, z_out_d << 0 → UNEXPECTED: high input → low output
                (something is suppressing this channel despite high input)
        Case D: z_in_d << 0, z_out_d >> 0 → UNEXPECTED: low input → high output
                (channel is active despite being in "suppressed" input territory)

    The DIVERGENCE captures cases C and D:
        div_d = z_in_d - z_out_d

    When div_d >> 0 (Case C): input is above its mean but output is below → amplify
    When div_d << 0 (Case D): input is below mean but output is above → amplify
    When div_d ≈ 0: normal correlated transformation → gate ≈ 1

    GATE:
        div_d = z_in_d - z_out_d
        gate_d = clamp(1 + β_pos × ReLU(tanh(γ × |div_d|)), 1.0, 6.0)
        output = GELU(x) × gate

    KEY INSIGHT: This captures NONLINEAR INTERACTIONS that neither pure input
    nor pure output gates can detect. When GELU behaves "unexpectedly" for a channel,
    the model should pay more attention to that channel.

    Variant: Use signed divergence for asymmetric response:
        div_d = z_in_d - z_out_d
        gate_d = 1 + β_up × ReLU(tanh(γ × div_d)) − β_dn × ReLU(tanh(−γ × div_d))

PARAMS: logit_decay, log_beta_up, log_beta_dn, log_gamma = 4 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean (D,), _ema_out_sq (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU225(nn.Module):
    """Input-output divergence gate: amplifies channels where GELU transforms unexpectedly."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_beta_up = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        y = self._gelu(x)   # (B, T, D)

        # ── Init EMA ─────────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                yf = y.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = yf.mean(0).clone()
                self._ema_out_sq   = yf.pow(2).mean(0).clone()
            self._ready = True
            return y

        # ── Input z-scores ───────────────────────────────────────────
        var_in = (self._ema_sq - self._ema_mean.pow(2)).clamp(self.eps_var)
        std_in = var_in.sqrt()
        z_in   = (x - self._ema_mean) / (std_in + self.eps)     # (B, T, D)

        # ── Output z-scores ──────────────────────────────────────────
        var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(self.eps_var)
        std_out = var_out.sqrt()
        z_out   = (y - self._ema_out_mean) / (std_out + self.eps) # (B, T, D)

        # ── Divergence: how much does input novelty differ from output novelty? ─
        div = z_in - z_out    # (B, T, D)
        # div >> 0: high input but low output (unexpected suppression by GELU)
        # div << 0: low input but high output (unexpected activation by GELU)

        # ── Asymmetric divergence gate ───────────────────────────────
        up_arm = beta_up * F.relu(torch.tanh( gamma * div))
        dn_arm = beta_dn * F.relu(torch.tanh(-gamma * div))
        gate   = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)

        output = y * gate

        # ── Update EMA ────────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            yf = y.detach().flatten(0, 1)
            self._ema_mean     = d_val*self._ema_mean     + (1-d_val)*xf.mean(0)
            self._ema_sq       = d_val*self._ema_sq       + (1-d_val)*xf.pow(2).mean(0)
            self._ema_out_mean = d_val*self._ema_out_mean + (1-d_val)*yf.mean(0)
            self._ema_out_sq   = d_val*self._ema_out_sq   + (1-d_val)*yf.pow(2).mean(0)

        return output
