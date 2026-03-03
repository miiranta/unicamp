"""GELU163 – EMA Channel-Sign Flip Rate Gate.

THE KEY INSIGHT:
    Each channel in the FF layer can be thought of as a "feature detector" that
    typically activates in one direction (positive or negative). When the network
    encounters familiar inputs, each channel fires in its habitual direction —
    the same direction it's been firing on average.

    When a NOVEL input arrives, some channels that normally fire positively suddenly
    fire negatively (or vice versa). This sign flip indicates the input is
    directionally inconsistent with the established pattern.

    Per-channel sign agreement:
        sign_current_d = sign(mean_{b,t}(x[b,t,d]))   — current batch direction
        sign_history_d = sign(ema_mean_d)               — historical direction
        agree_d = (sign_current == sign_history)        — did we flip?

    Fraction of channels that flipped:
        flip_rate = 1 - mean_d(agree_d.float())         ∈ [0, 1]

    Gate = 1 + alpha * tanh(sigma * flip_rate)

    When all channels consistent with history: flip_rate ≈ 0.
    When half channels flipped (novel regime): flip_rate ≈ 0.5.
    When all channels flipped (opposite regime): flip_rate = 1.

STABILITY: For zero-mean channels, sign is undefined. Use sign(mean)=0 → treat as
    no-flip (doesn't contribute to novelty signal). sign(0.0) = 0 in PyTorch,
    so (0 == sign_history) is True for positive history and False for negative.
    This is acceptable edge-case behavior.

CAUSALITY: EMA updated AFTER forward with detached x. ✓
GRADIENT: Flip rate computed with no_grad; gradients through alpha/sigma. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_mean (D,) per-channel historical mean.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_DECAY = 0.9


class GELU163(nn.Module):
    """Channel sign flip rate gate: channels pointing against history = novel regime."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_mean: torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
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
                self._ema_mean = x.detach().flatten(0, 1).mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # Current batch mean per channel
            cur_mean   = x.detach().flatten(0, 1).mean(0)      # (D,)
            # Sign agreement: did current batch direction match historical direction?
            sign_cur   = cur_mean.sign()                        # (D,)
            sign_hist  = self._ema_mean.sign()                  # (D,)
            # agree when both signs equal; treat sign=0 as having no mismatch
            agree      = ((sign_cur == sign_hist) | (sign_hist == 0) | (sign_cur == 0)).float()  # (D,)
            flip_rate  = 1.0 - agree.mean()                     # scalar ∈ [0, 1]

        gate   = 1.0 + alpha * torch.tanh(sigma * flip_rate)
        output = out * gate

        # EMA update after forward
        with torch.no_grad():
            cur_mean_nw = x.detach().flatten(0, 1).mean(0)
            self._ema_mean = EMA_DECAY * self._ema_mean + (1 - EMA_DECAY) * cur_mean_nw

        return output
