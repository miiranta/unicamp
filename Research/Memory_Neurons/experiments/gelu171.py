"""GELU171 – Per-Channel Adaptive EMA Decay Gate.

THE KEY INSIGHT:
    gelu80 (best, 7%) uses a SINGLE global EMA decay for all D=1024 channels.
    But different channels have very different dynamics:

    - SYNTACTIC channels: stable, slow-changing — benefit from LONG memory (high decay)
    - SEMANTIC/CONTENT channels: rapidly varying with topic — benefit from SHORT memory (low decay)
    - POSITIONAL channels: predictably oscillating — might need very specific timescale

    A single decay=0.9 is a compromise. The optimal decay differs per channel.

    Solution: Give each channel its own learnable EMA decay parameter.
        decay_d = sigmoid(logit_d)   where logit_d is a D-dimensional learnable parameter

    The model can then learn:
        decay_d ≈ 0.99 for stable channels (forget slowly)
        decay_d ≈ 0.5  for volatile channels (forget quickly)

    This doubles gelu80's expressive power with only D additional parameters.

    Gate (same structure as gelu80 but with per-channel means and adaptive decay):
        z[b,t,d] = (x[b,t,d] - ema_mean_d) / (std_d + eps)
        gate = 1 + alpha * tanh(sigma * mean_d(|z[b,t,d]|))
        output = GELU(x) * gate

    EMA update: ema_mean_d(t+1) = decay_d * ema_mean_d(t) + (1-decay_d) * batch_mean_d

STABILITY:
    logit_d initialized to logit(0.9) = log(9) ≈ 2.197 → all decays start at 0.9.
    Model can move them up or down during training.

Params: log_alpha, log_sigma (2 scalars), logit_decay (D,) per-channel decay logits.
State: _ema_mean (D,), _ema_sq (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_VAR = 1e-4


class GELU171(nn.Module):
    """Per-channel adaptive EMA decay: each channel learns its own memory timescale."""

    def __init__(self, d_ff: int):
        super().__init__()
        self.d_ff = d_ff

        # Scalar gate parameters
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

        # Per-channel decay logit (D,), init at logit(0.9) ≈ 2.197
        init_logit = math.log(0.9 / 0.1)   # = log(9) ≈ 2.197
        self.logit_decay = nn.Parameter(torch.full((d_ff,), init_logit))

        self._ema_mean: torch.Tensor = None   # (D,)
        self._ema_sq:   torch.Tensor = None   # (D,)
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out   = self._gelu(x)
        alpha = self.log_alpha.exp()
        sigma = F.softplus(self.log_sigma)

        # Per-channel decay values (detach for EMA, but logit_decay still gets gradient via gate)
        decay = torch.sigmoid(self.logit_decay)          # (D,) ∈ (0, 1), in-graph

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)             # (N, D)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            return out

        # Per-channel z-score using per-channel adaptive EMA statistics
        with torch.no_grad():
            var_d = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=EPS_VAR)
            std_d = var_d.sqrt()                          # (D,)

        mu_  = self._ema_mean.view(1, 1, D)
        std_ = std_d.view(1, 1, D)
        z    = (x.detach() - mu_) / (std_ + 1e-5)       # (B, T, D) no grad from stats

        # Gradient path for logit_decay:
        # Weight each channel's |z| by (1 - decay_d): channels with fast decay (low decay)
        # contribute MORE to surprise — they're reacting to short-term patterns.
        # This gives logit_decay direct gradient via the gate value.
        w_d       = (1.0 - decay).view(1, 1, D)         # (1, 1, D) weight per channel
        mean_abs_z = (z.abs() * w_d).sum(-1) / (w_d.sum(-1) + 1e-8)  # (B, T) weighted mean
        surprise   = torch.tanh(sigma * mean_abs_z)      # (B, T) ∈ (0, 1)
        gate       = 1.0 + alpha * surprise              # (B, T)
        output     = out * gate.unsqueeze(-1)

        # Update per-channel EMA with per-channel adaptive decay
        with torch.no_grad():
            d_val = decay.detach()                        # (D,) detached for EMA update
            xf = x.detach().flatten(0, 1)                # (N, D)
            bm = xf.mean(0)                              # (D,) current batch mean
            bsq = xf.pow(2).mean(0)                      # (D,) current batch sq mean
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * bm
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * bsq

        return output
