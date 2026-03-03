"""GELU167 – Per-Channel Platykurtosis Gate (Negative Excess Kurtosis = Bimodal).

THE KEY INSIGHT:
    gelu153 tracks POSITIVE excess kurtosis: heavy-tailed distributions where rare
    large spikes = novel events.

    But the OPPOSITE case is equally interesting: NEGATIVE excess kurtosis (platykurtosis)
    indicates a distribution that is flatter or more bimodal than Gaussian.

    A PLATYKURTIC channel (kurt < 3, excess < 0) means multiple distinct activation
    modes exist — the channel is being pulled in different directions by different tokens.
    This is characteristic of a channel that processes structurally DISTINCT input types
    and is therefore highly informative (low redundancy, high capacity utilization).

    When historical excess kurtosis is negative (platykurtic):
        The channel is already processing diverse content fairly
    When current batch causes kurtosis to become MORE negative than usual:
        Even more bimodal/diverse content → particularly novel batch

    Gate mechanism:
        EMA of per-channel excess kurtosis (signed).
        surp = mean_d(relu(-ema_excess_kurt_d))   — only negative kurtosis channels
        gate = 1 + alpha * tanh(sigma * surp)

    Contrast with gelu153 which uses relu(+excess_kurt_d): they target opposite tails.
    This gate fires when the DISTRIBUTION itself is unusual, not individual activations.

STATE: Track EMA of per-channel mean, variance, 4th central moment.
CAUSALITY: EMA updated after forward. ✓
GRADIENT: Kurtosis computation under no_grad. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_mean (D,), _ema_sq (D,), _ema_cm4 (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9
EPS_VAR   = 1e-4


class GELU167(nn.Module):
    """Per-channel platykurtosis gate: bimodal/uniform channel activation = novel regime."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))   # start small: sigma=0.37

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
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                mu = xf.mean(0)
                self._ema_mean = mu.clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_cm4  = (xf - mu).pow(4).mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            var_d = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=EPS_VAR)
            kurt_d = self._ema_cm4 / (var_d.pow(2) + 1e-8)    # (D,) raw kurtosis
            excess_d = kurt_d - 3.0                             # (D,) excess kurtosis
            # Platykurtic signal: only channels with negative excess kurtosis
            platy_d = F.relu(-excess_d)                         # (D,) ≥ 0
            surp    = platy_d.mean()                            # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        with torch.no_grad():
            xf   = x.detach().flatten(0, 1)
            mu_b = xf.mean(0)
            cm4_b = (xf - mu_b).pow(4).mean(0)
            d = EMA_DECAY
            self._ema_mean = d * self._ema_mean + (1 - d) * mu_b
            self._ema_sq   = d * self._ema_sq   + (1 - d) * xf.pow(2).mean(0)
            self._ema_cm4  = d * self._ema_cm4  + (1 - d) * cm4_b

        return output
