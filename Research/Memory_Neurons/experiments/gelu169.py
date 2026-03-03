"""GELU169 – Angular Velocity Gate (Token Trajectory Speed in Activation Space, Stateless).

THE KEY INSIGHT:
    Every token at position t occupies a point in D-dimensional activation space.
    Consecutive tokens form a TRAJECTORY through this space.

    A FAMILIAR sequence (repetitive, predictable) has low angular velocity:
        each step moves in nearly the same direction as the previous step.
        cos(x_t, x_{t-1}) ≈ 1 for small position changes.

    A NOVEL sequence (topic change, surprising word) has high angular velocity:
        the trajectory suddenly changes direction in activation space.
        cos(x_t, x_{t-1}) ≈ 0 or negative for sharp turns.

    This is a kinematic analog: the activation vector traces a path through space,
    and its angular speed measures how rapidly it's changing direction.

    Key distinction from gelu152 (lag-1 autocorrelation):
    - gelu152: per-CHANNEL correlation of activation values (scalar x time series)
    - gelu169: whole-VECTOR cosine similarity (angular displacement of full D-dim vector)
    gelu169 captures rotational change in the full representation space,
    not per-channel value changes.

IMPLEMENTATION (stateless):
    x0 = normalize(x[:, :-1, :], dim=-1)   (B, T-1, D) unit vectors at t-1
    x1 = normalize(x[:, 1:, :], dim=-1)    (B, T-1, D) unit vectors at t

    cos_sim = (x0 * x1).sum(-1)            (B, T-1) ∈ [-1, 1]
    angular = 1.0 - cos_sim                (B, T-1) ∈ [0, 2]; 0=same dir, 2=opposite
    surp    = angular.mean()               scalar ∈ [0, 2]; baseline ≈ 1.0 (random)

    gate = 1 + alpha * tanh(sigma * surp)

CAUSALITY: Fully within-sequence computation. ✓
NOTE: Like gelu152 and gelu154, the gate at position t uses statistics from all positions.
STATELESS: No EMA. ✓

Params: log_alpha, log_sigma (2 scalars).
State: None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU169(nn.Module):
    """Angular velocity gate: the faster the activation trajectory turns, the more novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def reset_state(self):
        pass   # fully stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if T < 2:
            return out

        with torch.no_grad():
            x_d = x.detach()
            # Normalize to unit vectors for pure angular measurement
            x0  = F.normalize(x_d[:, :-1, :], dim=-1)    # (B, T-1, D)
            x1  = F.normalize(x_d[:, 1:, :],  dim=-1)    # (B, T-1, D)

            cos_sim  = (x0 * x1).sum(-1)                  # (B, T-1) ∈ [-1, 1]
            angular  = 1.0 - cos_sim                       # (B, T-1) ∈ [0, 2]
            surp     = angular.mean()                      # scalar ∈ [0, 2]

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
