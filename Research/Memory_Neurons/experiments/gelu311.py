"""GELU311 – Input Absolute-Value Power Gate (Creative: Magnitude-Only Novelty).

CREATIVE APPROACH: Rather than normalizing by EMA mean/std to get z-scores,
gate directly on the L2 POWER of the input relative to a running power baseline.

MOTIVATION:
    - Z-score gating measures WHICH DIRECTION is novel (above/below mean)
    - Power gating measures HOW ENERGETIC the activation is, direction-agnostic
    - Some channels may be "normally quiet" (low power) and "fire loudly" only for
      novel inputs — the power ratio captures this directly

MECHANISM:
    power_in_d = x_d^2                              [instantaneous per-channel power]
    ema_power_d = EMA of power_in_d                 [running baseline]
    ratio_d = power_in_d / (ema_power_d + ε)        [relative power]
    z_power_d = log(ratio_d + ε)                    [log-ratio = log-power novelty]

    gate_power = clamp(1 + β_p * tanh(γ_p * z_power), 0.05, 8.0)

    Combined with output cosine gate:
    output = GELU(x) * gate_power * gate_cos

WHY DIFFERENT FROM gelu211:
    - No subtraction of mean (centering) — pure energy comparison
    - Log ratio is symmetric in log-space: log(power/ema)
    - Naturally handles channels with very different amplitude scales
    - Much simpler: only 1 EMA buffer (per-channel power baseline)

CAUSALITY: EMA power is batch-level statistic → safe.

PARAMS: logit_decay, log_tau, log_beta_p, log_gamma_p  (4)
STATE:  _ema_power (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU311(nn.Module):
    """Relative log-power gate: gates on per-channel input energy vs. EMA baseline."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.logit_decay = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau     = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_p  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma_p = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_power:   torch.Tensor = None
        self._ema_out_dir: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_power   = None
        self._ema_out_dir = None
        self._ready       = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_p  = F.softplus(self.log_beta_p)
        gamma_p = F.softplus(self.log_gamma_p)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_power   = xf.pow(2).mean(0).clone()      # (D,)
                self._ema_out_dir = F.normalize(of.mean(0), dim=0).clone()
                self._ready       = True
            return out

        with torch.no_grad():
            power_in = x.detach().pow(2).flatten(0, 1).mean(0)     # (D,) batch-avg power
            ratio    = power_in / (self._ema_power + self.eps)
            z_power  = ratio.log().view(1, 1, D)                    # (1, 1, D) — detached

        gate_power = (1.0 + beta_p * torch.tanh(gamma_p * z_power)).clamp(0.05, 8.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_power * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_power   = d_val * self._ema_power   + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_dir = d_val * self._ema_out_dir + (1 - d_val) * F.normalize(of.mean(0), dim=0)

        return output
