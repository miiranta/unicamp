"""GELU210 – Surprise-Shifted SiLU.

MEMORY EMBEDDED IN THE NONLINEARITY ITSELF (not a gate on top).

In gelu189/206-208 we compute out = base_activation(x) × gate.
The gate is a multiplicative correction applied after the activation.

Here we take a different approach: shift the INPUT to SiLU per-channel,
so that the activation threshold itself adapts to the channel's history.

    z     = (x − ema_mean) / ema_std          per-channel z-score
    shift = β × tanh(γ × z)                   dense, bounded shift
    out   = SiLU(x + shift)
          = (x + shift) × σ(x + shift)

INTERPRETATION:
    - If z > 0 (x is above its mean → novel), shift > 0 → push activation HIGHER
      (amplify the novel signal by pushing it further into the linear regime of SiLU)
    - If z < 0 (x is below its mean → familiar-negative), shift < 0 → push LOWER
      (suppress the familiar-negative signal)
    - tanh bounds the shift: |shift| ≤ β (prevents runaway)

This is a DENSE operation — ALL D=1024 channels are modified.
There is no sparse selection, no cosine gate, no explicit output scaling.
The memory modulates where on the SiLU curve each channel evaluation happens.

PARAMS: logit_decay, log_beta, log_gamma (3 scalars)
STATE:  _ema_mean (D,), _ema_sq (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU210(nn.Module):
    """Surprise-shifted SiLU: out = (x + β·tanh(γ·z)) × σ(x + β·tanh(γ·z))."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_beta    = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ready    = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ready    = True
            # Fallback: vanilla SiLU on first batch
            return x * torch.sigmoid(x)

        with torch.no_grad():
            var = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std = var.sqrt().view(1, 1, D)
            mu_ = self._ema_mean.view(1, 1, D)
            z   = (x.detach() - mu_) / (std + self.eps)         # (B, T, D)

        # Compute surprise-driven shift (dense, all channels)
        shift   = beta * torch.tanh(gamma * z)                  # (B, T, D)
        shifted = x + shift                                      # (B, T, D)
        output  = shifted * torch.sigmoid(shifted)              # SiLU(shifted)

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)

        return output
