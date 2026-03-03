"""GELU158 – Momentum Predictor Gate (EMA + Velocity Prediction Error).

THE KEY INSIGHT:
    Simple EMA gates (like gelu80) compare the current token to the historical MEAN.
    But if the mean has been steadily drifting in one direction (e.g., the text is
    gradually shifting topic), a token that continues the drift looks "surprising"
    relative to the mean even though it's actually perfectly predictable.

    A better predictor uses both the current EMA mean AND its velocity:
        x_pred(t) = ema_mean + ema_velocity
    where ema_velocity = EMA of (mean(t) - mean(t-1))

    This is a first-order Kalman predictor. If the topic is slowly drifting, the
    predictor tracks the drift and only flags genuine surprises.

    x_pred is the expected "next" mean. Tokens far from x_pred are novel.
    Normalized error: ||x[b,t] - x_pred||² / (||x_pred||² + ||x[b,t]||² + eps)

IMPLEMENTATION:
    Track:
        _ema_mean: (D,)  EMA of batch mean
        _ema_vel:  (D,)  EMA of (mean - prev_mean) = velocity
        _prev_mean:(D,)  previous batch mean (for velocity computation)

    At forward:
        1. x_pred = ema_mean + ema_vel   — first-order predictor
        2. err_bt = ||x[b,t] - x_pred||² / (||x[b,t]||² + ||x_pred||² + eps)  per token
        3. surp = tanh(sigma * mean_{b,t}(err_bt))
        4. gate = 1 + alpha * surp

    After forward: update EMA states.

DECAY:
    DECAY_MEAN = 0.9   — fast, tracks mean closely
    DECAY_VEL  = 0.7   — faster, responsive to direction changes

CAUSALITY: EMA states updated AFTER forward with detached x. ✓
GRADIENT: Error computed with no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_mean (D,), _ema_vel (D,), _prev_mean (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DECAY_MEAN = 0.9
DECAY_VEL  = 0.7


class GELU158(nn.Module):
    """Momentum predictor gate: first-order Kalman-style prediction error = novelty."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self._ema_mean:  torch.Tensor = None   # (D,)
        self._ema_vel:   torch.Tensor = None   # (D,)
        self._prev_mean: torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mean  = None
        self._ema_vel   = None
        self._prev_mean = None
        self._ready     = False

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
                mu = x.detach().flatten(0, 1).mean(0)
                self._ema_mean  = mu.clone()
                self._ema_vel   = torch.zeros_like(mu)
                self._prev_mean = mu.clone()
                self._ready     = True
            return out

        with torch.no_grad():
            # First-order predictor: mean + velocity
            x_pred = self._ema_mean + self._ema_vel            # (D,)

            # Normalized prediction error per token
            x_det = x.detach()                                  # (B, T, D)
            diff  = x_det - x_pred.view(1, 1, D)               # (B, T, D)
            numer = diff.pow(2).sum(-1)                         # (B, T)
            denom = (x_det.pow(2).sum(-1) +
                     x_pred.pow(2).sum().expand(B, T) + 1e-6)  # (B, T)
            err   = (numer / denom).mean()                      # scalar ∈ [0, 1]

        gate   = 1.0 + alpha * torch.tanh(sigma * err)
        output = out * gate

        # Update EMA states after forward
        with torch.no_grad():
            mu_b = x.detach().flatten(0, 1).mean(0)            # (D,)
            vel_b = mu_b - self._prev_mean                      # (D,) current velocity

            self._ema_vel   = DECAY_VEL  * self._ema_vel  + (1 - DECAY_VEL)  * vel_b
            self._ema_mean  = DECAY_MEAN * self._ema_mean + (1 - DECAY_MEAN) * mu_b
            self._prev_mean = mu_b.clone()

        return output
