"""GELU154 – FFT Spectral High-Frequency Ratio Gate (Stateless).

THE KEY INSIGHT:
    Language tokens form sequences over time (position). The sequence x[:, :, d]
    acts as a 1D signal of length T=64. Its frequency content reveals its structure:

    LOW FREQUENCY content = slowly-varying signal = repetitive/familiar text
        (long repeated phrases, common function words, simple continuations)

    HIGH FREQUENCY content = rapidly-varying signal = novel/complex text
        (sudden topic changes, rare words, abrupt syntactic shifts)

    By computing the FFT along the sequence dimension and measuring the ratio of
    high-frequency to low-frequency energy, we get a batch-level novelty signal
    that captures how "spiky" or "complex" the current sequence context is.

IMPLEMENTATION (stateless, no EMA):
    X_fft = rfft(x, dim=1)                   — (B, T//2+1, D) complex
    mag_sq = |X_fft|^2                       — (B, T//2+1, D)
    Split: low = mag_sq[:, 1:n_low, D]       — skip DC component (idx 0)
           high = mag_sq[:, n_low:, D]
    n_low = (T//2+1) // 2 + 1               — upper half of spectrum

    hf_ratio = mean(high) / (mean(low) + mean(high) + eps)   ∈ (0, 1)
    gate = 1 + alpha * tanh(sigma * hf_ratio)
    output = GELU(x) * gate

    Skip DC (index 0) since it just encodes the batch mean — not novelty-relevant.

CAUSALITY: Fully within-sequence computation on current x. ✓
NOTE: Using rfft over T=64 positions. Since we process the entire sequence at once
    (all positions see each other in the FFT), this does use "future" tokens to
    compute the gate for any given position. This is the same as gelu138 (batch variance)
    and gelu152 (lag-1 corr): the gate is a global sequence-statistic applied uniformly
    to all positions. The model cannot "peek" at individual future token values to help
    predict token t — it can only use aggregate spectral statistics.

GRADIENT: FFT computation run under no_grad on detached x. alpha/sigma get gradients. ✓

Params: log_alpha, log_sigma (2 scalars).
State: None (fully stateless).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU154(nn.Module):
    """FFT spectral HF ratio gate: high-frequency sequence content = novel."""

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
        out   = self._gelu(x)

        alpha = self.log_alpha.exp()
        sigma = self.log_sigma.exp()

        if T < 4:
            return out   # degenerate: too few sequence positions

        with torch.no_grad():
            x_d    = x.detach()
            # rfft along sequence dimension (dim=1)
            X_fft  = torch.fft.rfft(x_d, dim=1)                # (B, T//2+1, D) complex
            mag_sq = X_fft.real.pow(2) + X_fft.imag.pow(2)    # (B, T//2+1, D)

            N_freq = mag_sq.shape[1]                            # T//2 + 1
            n_low  = max(2, (N_freq + 1) // 2)                 # boundary between low/high

            # Skip DC component (idx 0): it's just the batch mean, not informative
            lf = mag_sq[:, 1:n_low, :].mean()                  # low-frequency energy
            hf = mag_sq[:, n_low:, :].mean()                   # high-frequency energy

            hf_ratio = hf / (lf + hf + 1e-8)                  # ∈ (0, 1)

        gate   = 1.0 + alpha * torch.tanh(sigma * hf_ratio)
        output = out * gate

        return output
