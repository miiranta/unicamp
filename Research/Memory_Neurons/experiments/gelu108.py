"""
gelu108 – Cross-layer novelty (input residual deviation gate)
─────────────────────────────────────────────────────────────────────────────
All prior experiments measure novelty of x relative to an EMA of x.
This variant measures how much x has CHANGED from a "prior representation"
by comparing x to an EMA of x _before_ the residual connection added to it.
In a transformer FFN: x_in = residual stream → the gate tracks whether the
attention output has caused a large DEVIATION from the running stream.

Concretely:
    delta_d  = x_d – ema_prev_d           (deviation from running stream mean)
    ema_rms  = EMA( rms(delta) )           (running rms of deviation)
    novelty  = rms(delta) / (ema_rms + ε)  (how unusual is today's deviation?)
    surp     = tanh(σ × (novelty – 1))    (0 when typical, >0 when large deviance)
    gate     = 1 + w × relu(surp)
    result   = GELU(x) × gate

Parameters: logit_decay, log_sigma_raw, log_w_raw  →  3 scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU108(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.D = d_model
        self.logit_decay   = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.log_w_raw     = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('_ema_stream', torch.zeros(d_model))  # running mean of x
        self.register_buffer('_ema_rms',    torch.tensor(1.0))     # running rms of delta
        self._initialised = False

    def _gelu(self, x):
        return F.gelu(x)

    def forward(self, x):
        B, T, D = x.shape
        sigma = F.softplus(self.log_sigma_raw) + 0.01
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        # ── deviation from running stream ──────────────────────────────────
        ema_stream = self._ema_stream.detach()          # (D,)
        ema_rms    = self._ema_rms.detach().clamp(min=1e-6)   # scalar

        delta    = x - ema_stream                       # (B,T,D) with grad
        cur_rms  = delta.pow(2).mean(-1, keepdim=True).sqrt()   # (B,T,1)
        novelty  = cur_rms / ema_rms                    # relative magnitude
        surp     = torch.tanh(sigma * (novelty - 1.0))
        gate     = 1.0 + w * F.relu(surp)              # only amplify, never suppress

        result = out * gate

        # ── EMA update (no_grad) ───────────────────────────────────────────
        x_flat    = x.detach().reshape(-1, D)
        xb        = x_flat.mean(0)
        delta_det = (x_flat - self._ema_stream.unsqueeze(0))
        rms_batch = delta_det.pow(2).mean().sqrt()

        decay = torch.sigmoid(self.logit_decay).detach().item()
        with torch.no_grad():
            if not self._initialised:
                self._ema_stream.copy_(xb)
                self._ema_rms.fill_(rms_batch.item())
                self._initialised = True
            else:
                self._ema_stream.mul_(decay).add_((1 - decay) * xb)
                self._ema_rms.mul_(decay).add_((1 - decay) * rms_batch)
        return result
