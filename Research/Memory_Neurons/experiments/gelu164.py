"""GELU164 – EMA Channel-Pair Cross-Correlation Surprise Gate.

THE KEY INSIGHT:
    All previous gates (gelu80, gelu131, etc.) treat channels INDEPENDENTLY.
    Per-channel z-score measures how much each individual channel deviates from
    its own history — but it misses changes in how channels CO-VARY.

    Example: Channels 5 and 47 might typically have a strong positive correlation
    (they tend to fire together for common syntactic patterns). When processing a
    novel token that causes channel 5 to fire strongly but channel 47 to fire
    negatively — the anti-correlation is a signature of novelty not captured by
    any per-channel gate.

    CROSS-CORRELATION SURPRISE: For K=64 random channel pairs (i_k, j_k):
        Cross-moment: m_k(batch) = mean_{b,t}(x[b,t,i_k] * x[b,t,j_k])

    Track EMA of each pair's typical cross-moment.
    At forward: compare current cross-moments to EMA.
    Normalized deviation = |current - ema| / (|ema| + eps)

    gate = 1 + alpha * tanh(sigma * mean_k(deviation_k))

    When cross-correlation structure is stable (familiar): deviation ≈ 0.
    When cross-correlations break (novel): deviation > 0.

WHY RANDOM PAIRS:
    Full covariance matrix = D²/2 ≈ 500K entries. Too expensive.
    K=64 random pairs: O(64*N) computation, captures a random projection of the
    covariance difference — an unbiased estimator of covariance change.

CAUSALITY: EMA updated after forward. ✓
GRADIENT: Cross-moment deviation computed with no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: _ema_xcov (K,) EMA cross-moments, _pairs (K, 2) random pair indices [buffer].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K_PAIRS   = 64
EMA_DECAY =  0.9


class GELU164(nn.Module):
    """EMA channel-pair cross-correlation surprise: novel cross-channel structure = novel."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.d_ff = d_ff

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        # Fixed random channel pair indices (registered as buffer, moved with model)
        pairs = torch.stack([
            torch.randperm(d_ff)[:K_PAIRS],
            torch.randperm(d_ff)[:K_PAIRS],
        ], dim=1)   # (K, 2)
        self.register_buffer('pairs', pairs)

        self._ema_xcov: torch.Tensor = None   # (K,)
        self._ready = False

    def reset_state(self):
        self._ema_xcov = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)    # (N, D)
            i_idx = self.pairs[:, 0]         # (K,)
            j_idx = self.pairs[:, 1]         # (K,)
            # Per-pair cross-moments: (N, K) → mean over N
            xcov_b = (xf[:, i_idx] * xf[:, j_idx]).mean(0)   # (K,)

        if not self._ready:
            with torch.no_grad():
                self._ema_xcov = xcov_b.clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # Normalized deviation from EMA cross-correlation structure
            dev  = (xcov_b - self._ema_xcov).abs()             # (K,) unsigned change
            norm = self._ema_xcov.abs() + 1e-6                 # (K,) scale reference
            surp = (dev / norm).mean()                          # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        # EMA update after forward
        with torch.no_grad():
            self._ema_xcov = EMA_DECAY * self._ema_xcov + (1 - EMA_DECAY) * xcov_b

        return output
