"""GELU153 – Per-Channel Excess Kurtosis Gate (Heavy-Tail Detection).

THE KEY INSIGHT:
    Z-score (gelu80) uses the 2nd moment (variance) to measure channel spread.
    But variance penalizes all deviations equally — it can't distinguish between
    "uniformly spread" activations and "mostly near-mean with rare large spikes."

    Kurtosis = E[(x - μ)⁴] / σ⁴ = normalized 4th central moment.
    For a Gaussian: kurtosis = 3 (excess kurtosis = 0).
    Heavy-tailed: kurtosis > 3. Uniform / light-tailed: kurtosis < 3.

    HIGH EXCESS KURTOSIS per channel means:
        - The channel occasionally fires very strongly (heavy tail)
        - These rare large activations carry HIGH information (low prior probability)
        - A token that triggers such a spike is particularly novel

    So: channels with consistently high excess kurtosis are "surprise channels."
    Gate those channels up to amplify their rare strong signals.

IMPLEMENTATION:
    Track per-channel EMA of:
        _ema_mean  (D,)  — 1st moment
        _ema_sq    (D,)  — 2nd moment E[x²]
        _ema_m4    (D,)  — 4th moment E[x⁴]

    excess_kurt_d = (ema_m4_d - 4*ema_mean_d*ema_m3_d + ...) / var_d²
    Simplified using raw moments → central moment conversion:
        central_m4_d = ema_m4_d - 4*ema_mean_d*ema_m3_d + 6*ema_mean_d²*ema_sq_d
                       - 3*ema_mean_d⁴
    But tracking ema_m3 adds complexity. Use a simpler approximation:
        Track _ema_cm4 = EMA(|x - mean|⁴) directly.
        Then kurt_d = _ema_cm4_d / (var_d² + eps)
        excess = relu(kurt_d - 3.0)   # only super-Gaussian channels

    gate = 1 + alpha * tanh(sigma * mean_d(excess_kurt))
    Output = GELU(x) * gate

CAUSALITY: EMA updated AFTER forward with detached x. ✓
STABILITY: kurtosis can be large; tanh + eps clamps. Init with small alpha. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_mean (D,), _ema_sq (D,), _ema_cm4 (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
EPS_VAR   = 1e-4


class GELU153(nn.Module):
    """Per-channel excess kurtosis gate: rare large activations = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.d_ff = d_ff

        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # alpha = 1
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))  # sigma = 0.37, start small

        self._ema_mean: torch.Tensor = None   # (D,)
        self._ema_sq:   torch.Tensor = None   # (D,)
        self._ema_cm4:  torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_cm4  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)      # (N, D)
                mu = xf.mean(0)
                self._ema_mean = mu.clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_cm4  = (xf - mu).pow(4).mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # Compute per-channel variance and excess kurtosis from EMA stats
            var_d = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=EPS_VAR)  # (D,)
            kurt_d = self._ema_cm4 / (var_d.pow(2) + 1e-8)                     # (D,)
            excess_d = F.relu(kurt_d - 3.0)                                     # (D,) ≥ 0
            surp = excess_d.mean()                                              # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        # EMA update after forward
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)    # (N, D)
            mu_b = xf.mean(0)
            cm4_b = (xf - mu_b).pow(4).mean(0)

            d = EMA_DECAY
            self._ema_mean = d * self._ema_mean + (1 - d) * mu_b
            self._ema_sq   = d * self._ema_sq   + (1 - d) * xf.pow(2).mean(0)
            self._ema_cm4  = d * self._ema_cm4  + (1 - d) * cm4_b

        return output
