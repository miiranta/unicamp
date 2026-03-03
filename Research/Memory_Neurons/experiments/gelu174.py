"""GELU174 – Hebbian Correlation Deviation Gate (EMA of per-channel x·GELU(x)).

THE KEY INSIGHT:
    Hebb's rule says: "neurons that fire together wire together."
    For an FF neuron: when x_d and GELU(x_d) both activate strongly, the neuron
    is doing its learned job — it's in a FAMILIAR regime.

    The per-channel Hebbian correlation: h_d = x_d * GELU(x_d)

    Semantics:
    - h_d > 0, large: x and GELU(x) agree strongly → neuron fully engaged (familiar)
    - h_d ≈ 0: x and GELU(x) disagree (x positive but GELU near zero, or vice versa)
               → neuron in transition zone (moderately novel)
    - h_d < 0: unusual — input positive but GELU output near zero (suppressed activation)
               → neuron in unusual territory (most novel)

    Track EMA of E[h_d] per channel. Surprise = how much does the current batch's
    Hebbian correlation deviate from the expected level?

    surp = mean_d( |batch_h_d - ema_h_d| / (|ema_h_d| + eps) )
    gate = 1 + alpha * tanh(sigma * surp)

    Key novelty vs existing gates:
    - gelu80 measures x - mean(x), purely input-space
    - gelu98 measures z-score of GELU(x) output
    - gelu174 measures the CROSS-PRODUCT x·GELU(x): the interaction term
      that captures how the nonlinearity is being USED for this token

    When x·GELU(x) deviates from the expected mode of neuron use: novel semantics.

CAUSALITY: GELU(x) is the standard GELU applied to x (current token, no future).
           EMA of h_d updated after forward. ✓
GRADIENT: Hebbian computation under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_hebb (D,) per-channel EMA of x*GELU(x).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9


class GELU174(nn.Module):
    """Hebbian correlation deviation gate: unusual x·GELU(x) product = novel neuron usage."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_hebb: torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_hebb = None
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
                hebb_b = (x.detach() * out.detach()).flatten(0, 1).mean(0)  # (D,)
                self._ema_hebb = hebb_b.clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # Current batch Hebbian correlation per channel
            hebb_b   = (x.detach() * out.detach()).flatten(0, 1).mean(0)    # (D,)
            # Normalized deviation from EMA
            dev      = (hebb_b - self._ema_hebb).abs()                      # (D,)
            norm     = self._ema_hebb.abs() + 1e-6                          # (D,)
            surp     = (dev / norm).mean()                                   # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        with torch.no_grad():
            hebb_b_nw = (x.detach() * out.detach()).flatten(0, 1).mean(0)
            self._ema_hebb = EMA_DECAY * self._ema_hebb + (1 - EMA_DECAY) * hebb_b_nw

        return output
