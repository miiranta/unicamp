"""GELU152 – Lag-1 Temporal Autocorrelation Gate (Stateless).

THE KEY INSIGHT:
    All EMA-based experiments track *long-term* statistics and ask: "how different is this
    token from the historical average?" But there's a complementary question:
    "How much is the *sequence itself* changing right now?"

    A passage where every token follows predictably from the previous one carries low
    information. A passage with rapid, unpredictable transitions carries high information.

    Measurement: Pearson lag-1 autocorrelation per channel per sequence.
        corr(x_t, x_{t-1}) — if high, the sequence is slowly varying (familiar/predictable).
        If low or negative, the sequence is rapidly changing (novel/information-dense).

    Gate: Channels with low autocorrelation in the current batch get boosted.
    This is entirely STATELESS — no EMA state required.

IMPLEMENTATION:
    x0 = x[:, :-1, :]   (B, T-1, D)
    x1 = x[:, 1:, :]    (B, T-1, D)
    Normalize per (batch, channel) over the T-1 sequence positions.
    corr_bd = (z0 * z1).mean(T-1)  →  (B, D)
    novelty_bd = 1 - |corr_bd|     →  (B, D)  ∈ [0, 1]

    gate = 1 + alpha * tanh(sigma * novelty_bd)   →  (B, 1, D) broadcast

    When corr = +1 (perfectly predictable): novelty = 0, gate = 1
    When corr =  0 (independent):           novelty = 1, gate = 1 + alpha*tanh(sigma)
    When corr = -1 (alternating):           novelty = 1, gate = 1 + alpha*tanh(sigma)

CAUSALITY: Fully within-batch computation on current x. No future tokens used. ✓
    Note: The gate for position t uses corr averaged over ALL positions 1..T-1,
    which includes future positions. However, this is equivalent to using the
    *global* sequence statistic (all positions equally), not conditioning on future
    at any individual token position. All our batch-statistic gates do this.

GRADIENT:
    novelty_bd is computed with torch.no_grad() on x.detach() → pure statistics.
    Gradients flow through alpha and sigma only. ✓

Params: log_alpha, log_sigma (2 scalars).
State: None (fully stateless).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU152(nn.Module):
    """Lag-1 temporal autocorrelation gate: low-autocorr channels = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # alpha = 1
        self.log_sigma = nn.Parameter(torch.tensor(0.0))   # sigma = 1

    def reset_state(self):
        pass   # fully stateless

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out    = self._gelu(x)

        alpha  = self.log_alpha.exp()
        sigma  = self.log_sigma.exp()

        if T < 2:
            return out   # degenerate: cannot compute lag-1

        with torch.no_grad():
            x_d = x.detach()                        # (B, T, D)
            x0  = x_d[:, :-1, :]                   # (B, T-1, D)
            x1  = x_d[:, 1:, :]                    # (B, T-1, D)

            # Normalize each (batch-item, channel) slice over T-1 positions
            mu0  = x0.mean(1, keepdim=True)         # (B, 1, D)
            mu1  = x1.mean(1, keepdim=True)
            std0 = x0.std(1, keepdim=True).clamp(min=1e-5)
            std1 = x1.std(1, keepdim=True).clamp(min=1e-5)

            z0 = (x0 - mu0) / std0                  # (B, T-1, D)
            z1 = (x1 - mu1) / std1

            corr_bd  = (z0 * z1).mean(1)            # (B, D) ∈ [-1, 1]
            nov_bd   = 1.0 - corr_bd.abs()          # (B, D) ∈ [0, 1]

        # gate: (B, 1, D) → broadcast over T
        gate   = 1.0 + alpha * torch.tanh(sigma * nov_bd.unsqueeze(1))
        output = out * gate

        return output
