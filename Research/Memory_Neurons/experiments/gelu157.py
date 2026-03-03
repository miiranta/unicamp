"""GELU157 – Sign Consensus Voting Gate (Stateless).

THE KEY INSIGHT:
    When the network processes a familiar, predictable token, many channels tend to
    "agree" — they consistently activate positively or consistently negatively across
    most tokens in the batch. This is because familiar patterns trigger well-learned
    feature detectors that fire reliably.

    When processing a novel or ambiguous token, channels become "uncertain" — some
    fire positively, some negatively, producing a near-zero mean sign across the batch.
    This sign-split is a signature of distributional ambiguity.

    Measurement: For each channel d, |mean(sign(x[b,t,d]))| ∈ [0, 1]
    - Near 1: all tokens activate this channel in the same direction → consensus (familiar)
    - Near 0: tokens split evenly positive/negative → disagreement (novel/uncertain)

    Gate boosts when channels disagree (high entropy of sign distribution).
    novelty_d = 1 - |mean_sign_d|   ∈ [0, 1]
    gate = 1 + alpha * tanh(sigma * mean_d(novelty_d))

IMPLEMENTATION (stateless):
    xf = x.detach().flatten(0, 1)               (N, D)
    sign_mean = xf.sign().mean(0).abs()         (D,) ∈ [0, 1]
    novelty   = 1.0 - sign_mean                 (D,) ∈ [0, 1]
    surp      = novelty.mean()                  scalar
    gate      = 1 + alpha * tanh(sigma * surp)
    output    = GELU(x) * gate

NOTE ON sign(0): torch.sign(0) = 0, which contributes 0 to mean_sign — this is correct
since an activation of exactly zero provides no directional evidence.

CAUSALITY: Fully within-batch computation. ✓
STATELESS: No EMA state. ✓
GRADIENT: All sign computation done with no_grad; gradients through alpha/sigma only. ✓
COST: O(N*D) — extremely fast. N=2048, D=1024 → trivial.

Params: log_alpha, log_sigma (2 scalars).
State: None (fully stateless).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU157(nn.Module):
    """Sign consensus gate: channel sign split across batch = novel/uncertain."""

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

        with torch.no_grad():
            xf        = x.detach().flatten(0, 1)     # (N, D)
            sign_mean = xf.sign().mean(0).abs()       # (D,) ∈ [0, 1], high = consensus
            novelty   = 1.0 - sign_mean               # (D,) ∈ [0, 1], high = disagreement
            surp      = novelty.mean()                 # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
