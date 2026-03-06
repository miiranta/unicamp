"""GELU224 – Squared-Activation Novelty Gate.

MOTIVATION — SIGN-SYMMETRIC NOVELTY DETECTION:
    All prior gates (gelu181, gelu190, gelu211, gelu221) measure signed deviation z_d:
        z_d = (x_d - mu_d) / sigma_d
        Positive z: channel is ABOVE mean (excited)
        Negative z: channel is BELOW mean (suppressed)

    The asymmetric gate then treats these differently (β_up vs β_dn).

    ALTERNATIVE: Measure novelty as departure in SQUARED magnitude:
        z_sq_d = (x_d² - EMA(x_d²)) / std(x_d²)

    This captures:
        - Channels with very HIGH magnitude (x_d >> 0 or x_d << 0)
        - REGARDLESS of sign
        - A channel firing at -3σ is just as "novel" as +3σ

    WHY THIS MIGHT HELP:
        - GELU(x) is asymmetric: GELU(x) ≈ 0 for x << 0, ≈ x for x >> 0
        - So negative input novelty (x_d << 0) produces near-zero output regardless
          → The signed gate modulates output of near-zero channels (less useful)
        - Squared gate recognizes: x_d = -3σ means the channel was STRONGLY INHIBITED
          → Amplifying or suppressing this at the output (≈0) has little effect...
        - BUT: x_d = -3σ means that channel is not contributing at ALL, which is itself
          an informative signal about WHAT IS NOT ACTIVATED for this token.

    ACTUAL MECHANISM:
        z_sq_d = (x_d² - mu_sq_d) / std_sq_d
        gate_d = clamp(1 + beta * ReLU(tanh(gamma * z_sq_d)), 1.0, max_gate)
        output = GELU(x) * gate

    Only AMPLIFIES (z_sq_d >= 0). No suppression arm — high squared novelty always amplifies.
    The EMA of x² is the second moment: tracks both variance and squared mean.

    When z_sq_d >> 0: something unusual happened (very high OR very low x_d)
    When z_sq_d ≈ 0: channel is in its normal operating range

PARAMS: logit_decay, log_beta, log_gamma = 3 scalars
STATE:  _ema_sq (D,), _ema_sqsq (D,)  [EMA of x² and x⁴ to get std(x²)]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU224(nn.Module):
    """Squared-activation novelty gate: amplify channels with unusual magnitude."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_sq   : torch.Tensor = None   # EMA(x²)
        self._ema_sq_sq: torch.Tensor = None   # EMA(x⁴) for var(x²)
        self._ready = False

    def reset_state(self):
        self._ema_sq    = None
        self._ema_sq_sq = None
        self._ready     = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        beta   = F.softplus(self.log_beta)
        gamma  = F.softplus(self.log_gamma)

        y = self._gelu(x)   # (B, T, D)

        # ── Init EMA on first call ────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                xf  = x.detach().flatten(0, 1)     # (B*T, D)
                x2  = xf.pow(2)
                self._ema_sq    = x2.mean(0).clone()
                self._ema_sq_sq = x2.pow(2).mean(0).clone()
            self._ready = True
            return y

        # ── Compute z-score in squared-activation space ───────────────
        x2     = x.pow(2)                           # (B, T, D)
        mu_sq  = self._ema_sq                       # (D,)
        sq_sq  = self._ema_sq_sq                    # (D,) = EMA(x⁴)
        # Var(x²) = E[x⁴] - (E[x²])²
        var_sq = (sq_sq - mu_sq.pow(2)).clamp(self.eps_var)
        std_sq = var_sq.sqrt()                      # (D,)

        z_sq = (x2 - mu_sq) / (std_sq + self.eps)  # (B, T, D)

        # ── Amplification-only gate ───────────────────────────────────
        # Only amplify channels with above-average squared magnitude
        gate = (1.0 + beta * F.relu(torch.tanh(gamma * z_sq))).clamp(1.0, 6.0)

        output = y * gate

        # ── Update EMA ────────────────────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            x2f = xf.pow(2)
            m_sq    = x2f.mean(0)
            m_sqsq  = x2f.pow(2).mean(0)
            self._ema_sq    = d_val*self._ema_sq    + (1-d_val)*m_sq
            self._ema_sq_sq = d_val*self._ema_sq_sq + (1-d_val)*m_sqsq

        return output
