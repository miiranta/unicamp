"""GELU165 – Intra-Batch IQR Extremeness Gate (Non-Parametric, Stateless).

THE KEY INSIGHT:
    Both z-score (gelu80) and rank extremeness (gelu161) treat all deviations
    symmetrically. But the IQR (interquartile range) method is the classical
    statistician's approach to outlier detection: it's distribution-free,
    resistant to the very outliers it's trying to detect, and has an
    interpretable threshold.

    For each channel d in the current batch:
        Q1_d = 25th percentile of x[:, d] across batch
        Q3_d = 75th percentile
        IQR_d = Q3_d - Q1_d   (robust spread measure)
        Tukey fence: lower = Q1_d - fence*IQR_d, upper = Q3_d + fence*IQR_d

    A token at x[b,t,d] is an outlier if it falls outside the fences.
    Extremeness per token-channel:
        ext_d = relu(|x[b,t,d] - median_d| - fence*IQR_d/2) / (IQR_d + eps)

    This is zero for inliers and grows proportionally to how far outside the
    fence a token falls.

    gate = 1 + alpha * tanh(sigma * mean_{b,t,d}(ext))

    Using fence=1.5 (Tukey's standard, 99.3% inlier rate for Gaussian data).

IMPLEMENTATION (stateless):
    q1 = torch.quantile(xf, 0.25, dim=0)  (D,)
    q3 = torch.quantile(xf, 0.75, dim=0)  (D,)
    med = (q1 + q3) / 2                   (D,) approx median
    iqr = (q3 - q1).clamp(min=1e-5)      (D,)
    fence_half = 0.75 * iqr               (D,)  fence=1.5 * IQR/2 = 0.75*IQR
    ext_bt = relu(|x - med| - fence_half) / (iqr + eps)  (B, T, D)
    surp = ext_bt.mean()                  scalar

NOTE: torch.quantile on (2048×1024) tensor is O(N*D*log(N)) — acceptable on GPU.

CAUSALITY: Fully within-batch. ✓
STATELESS: No EMA. ✓
GRADIENT: All IQR computation under no_grad; alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FENCE = 1.5   # Tukey fence factor


class GELU165(nn.Module):
    """IQR extremeness gate: tokens outside Tukey fences per channel = outliers = novel."""

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
            xf  = x.detach().flatten(0, 1)          # (N, D)

            # Per-channel quartiles
            q1  = torch.quantile(xf, 0.25, dim=0)  # (D,)
            q3  = torch.quantile(xf, 0.75, dim=0)  # (D,)
            med = (q1 + q3) * 0.5                   # (D,) approximate median

            iqr        = (q3 - q1).clamp(min=1e-5)         # (D,)
            fence_half = (FENCE * iqr * 0.5).view(1, 1, D) # (1,1,D) half-fence width
            med_       = med.view(1, 1, D)                  # (1,1,D)
            iqr_       = iqr.view(1, 1, D)                  # (1,1,D)

            # Extremeness: normalized distance beyond fences
            ext  = F.relu((x.detach() - med_).abs() - fence_half) / (iqr_ + 1e-6)  # (B,T,D)
            surp = ext.mean()                                # scalar

        gate   = 1.0 + alpha * torch.tanh(sigma * surp)
        output = out * gate

        return output
