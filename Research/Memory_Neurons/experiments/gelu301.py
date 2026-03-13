"""gelu301 – Per-Channel Input L2-Power Gate.

CONCEPT:
    gelu211 gates on the MEAN activation (first moment, z-score).
    The L2-POWER (second raw moment = mean squared amplitude) captures
    how much energy a channel carries.

    A channel with high power is "energetic" — it's processing a
    strong signal. A channel with low power is quiet. Novelty in power
    (above the expected energy level) indicates a new type of content.

MECHANISM:
    power_d = mean(x_d^2) over (B, T)   ← current batch energy per channel
    z_power = log(power_d) - log_baseline_d  ← log-ratio for scale invariance
    gate_d  = (1 + beta_up*relu(tanh(g*z)) - beta_dn*relu(tanh(-g*z))).clamp(...)
    return out * gate_d.view(1,1,D)

    log_baseline_d (D,): learned per-channel expected log-power (training distribution).

WHY LOG-RATIO:
    Power varies over many orders of magnitude across channels. log(power/baseline)
    is dimensionless and gives roughly Gaussian distribution → tanh saturation works well.

ASYMMETRIC GATE:
    Channels with power ABOVE baseline (z > 0): amplified via beta_up.
    Channels with power BELOW baseline (z < 0): suppressed via beta_dn.
    Both can be trained independently.

BENEFIT FROM BACKPROP:
    log_baseline (D,): gradient trains per-channel expected power from loss.
    log_beta_up, log_beta_dn, log_gamma: gate shape parameters.
    This is the power-domain analogue of gelu211's mean-domain gating.

EMA BASELINE ADAPTATION FOR Δ > 0:
    _power_ema (D,): slow EMA of observed power (d ≈ 0.995), persists across passes.
    effective_baseline = alpha * exp(log_baseline) + (1-alpha) * _power_ema
    As the eval EMA adapts to test power profile, the z-score approaches 0 for
    habituated channels → gate → 1 → less amplification → Δ > 0.

NO CAUSALITY LEAK:
    per-batch L2 power, causal across batches.

PARAMS:  log_baseline (D,), log_beta_up, log_beta_dn, log_gamma, logit_alpha, logit_d_pow.
STATE:   _power_ema (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU301(nn.Module):
    """Per-channel L2-power gate with asymmetric amplification and baseline EMA adaptation."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps  = eps
        self.D_FF = D_FF

        self.log_baseline = nn.Parameter(torch.zeros(D_FF))    # log-power baseline
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Blend between static and dynamic baseline
        self.logit_alpha  = nn.Parameter(torch.zeros(1))       # sigmoid(0) = 0.5
        # Slow EMA decay for power buffer
        self.logit_d_pow  = nn.Parameter(torch.tensor(math.log(0.995 / 0.005)))

        self._power_ema: torch.Tensor = None

    def reset_state(self):
        # Reset to trained static baseline power
        with torch.no_grad():
            self._power_ema = self.log_baseline.exp().detach().clone()

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        # Current per-channel L2 power (input space)
        power = x.pow(2).mean(dim=(0, 1)).clamp(min=self.eps)   # (D,)

        if self._power_ema is None:
            with torch.no_grad():
                self._power_ema = power.detach().clone()

        static_bl = self.log_baseline.exp().clamp(min=self.eps)            # (D,)
        alpha     = torch.sigmoid(self.logit_alpha)
        eff_bl    = alpha * static_bl + (1 - alpha) * self._power_ema.detach()

        # Log-ratio z-score
        z_power   = torch.log(power + self.eps) - torch.log(eff_bl + self.eps)  # (D,)

        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)
        up_arm  = beta_up * F.relu(torch.tanh( gamma * z_power))
        dn_arm  = beta_dn * F.relu(torch.tanh(-gamma * z_power))
        gate    = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)    # (D,)

        output  = out * gate.view(1, 1, D)

        # Update power EMA
        d_pow          = torch.sigmoid(self.logit_d_pow)
        new_power_ema  = d_pow * self._power_ema.detach() + (1 - d_pow) * power
        self._power_ema = new_power_ema.detach()

        return output
