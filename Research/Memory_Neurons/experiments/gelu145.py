"""
GELU145 — Signal-to-noise ratio of per-channel surprise.

gelu80: gate based on mean_d(|z_d|) — how much do channels deviate on average?
gelu144: gate based on top-K(|z_d|)  — what is the peak surprise?

GELU145 measures SIGNAL-TO-NOISE of the surprise distribution across channels:

    SNR = mean_d(|z_d|) / (std_d(|z_d|) + eps)

Interpretation:
    - High SNR: many channels have a SIMILAR level of surprise.
                → coherent global deviation (clear unusual input pattern)
    - Low SNR:  a FEW channels have huge surprise, rest have near-zero.
                → sparse/noisy; individual channel anomaly, not token-level novelty

    BUT: switching perspective — low SNR with high mean means CONCENTRATED SURPRISE.
    So we combine both: gate depends on both mean and inverse-CV (coefficient of variation):

    gate = 1 + alpha * tanh(sigma_m * mean|z|) * (1 + beta * tanh(sigma_s * SNR))

Params: log_alpha, log_sigma_m, log_beta, log_sigma_s = 4 scalars
State:  _ema_mean (D,), _ema_sq (D,)
"""

import torch
import torch.nn as nn


class GELU145(nn.Module):
    def __init__(self, d_ff: int = 1024):
        super().__init__()
        self.d_ff = d_ff
        self.log_alpha   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_m = nn.Parameter(torch.tensor(0.0))  # sensitivity to mean surprise
        self.log_beta    = nn.Parameter(torch.tensor(0.0))  # sensitivity to SNR
        self.log_sigma_s = nn.Parameter(torch.tensor(0.0))  # sensitivity to SNR scale
        self._gelu = nn.GELU()

        self.register_buffer("_ema_mean", torch.zeros(d_ff))
        self.register_buffer("_ema_sq",   torch.ones(d_ff))
        self._decay  = 0.99
        self._warmup = True

    def reset_state(self):
        self._ema_mean.zero_()
        self._ema_sq.fill_(1.0)
        self._warmup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._gelu(x)

        if self._warmup:
            self._ema_mean.copy_(x.detach().mean(dim=(0, 1)))
            self._ema_sq.copy_((x.detach() ** 2).mean(dim=(0, 1)))
            self._warmup = False
            return base

        ema_var = (self._ema_sq - self._ema_mean ** 2).clamp(min=1e-6)
        ema_std = ema_var.sqrt()
        z_abs   = (x - self._ema_mean).abs() / ema_std    # (B, T, D)

        # mean and std of |z| across channels (per token)
        z_mean = z_abs.mean(dim=-1, keepdim=True)          # (B, T, 1)
        z_std  = z_abs.std(dim=-1, keepdim=True) + 1e-6    # (B, T, 1)
        snr    = z_mean / z_std                             # (B, T, 1) — signal-to-noise

        alpha   = torch.exp(self.log_alpha)
        sigma_m = torch.exp(self.log_sigma_m)
        beta    = torch.exp(self.log_beta)
        sigma_s = torch.exp(self.log_sigma_s)

        # gate: mean-based novelty, modulated by structural coherence (SNR)
        mean_term = torch.tanh(sigma_m * z_mean)            # (B, T, 1)
        snr_mod   = 1.0 + beta * torch.tanh(sigma_s * snr) # (B, T, 1)
        gate = 1.0 + alpha * mean_term * snr_mod
        out  = base * gate

        with torch.no_grad():
            d = self._decay
            self._ema_mean.mul_(d).add_(x.detach().mean(dim=(0,1)) * (1-d))
            self._ema_sq.mul_(d).add_((x.detach()**2).mean(dim=(0,1)) * (1-d))

        return out
