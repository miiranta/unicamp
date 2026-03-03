"""GELU198 – Output-Space Z-Score Per-Channel Vector Gate.

ALL PREVIOUS VECTOR GATE EXPERIMENTS use INPUT z-scores:
    z_d^in = (x_d - ema_mean_d) / (std_in_d + ε)    — deviation of input from its EMA

GELU198 GATES ON OUTPUT Z-SCORES:
    z_d^out = (GELU(x)_d - ema_out_mean_d) / (std_out_d + ε)   — deviation of output from its EMA

    gate_d = clamp(1 + β × tanh(γ × z_d^out), 0.1, 5.0)
    output = GELU(x) × gate_vec × gate_cos

WHY OUTPUT Z-SCORES MIGHT BE BETTER:
    The input x is the raw pre-activation. GELU introduces nonlinearity:
    - Channels where x_d < -2: GELU ≈ 0 regardless of input z-score
    - Channels where x_d ≈ 0: GELU ≈ 0.5 × x_d (linear-ish)
    - Channels where x_d > 2: GELU ≈ x_d (identity-ish)

    Output z-scores NATURALLY account for the GELU nonlinearity:
    - Dead channels have ema_out_mean ≈ 0, std_out ≈ small → small z_out
    - Active channels carry the real signal; their deviations fire the gate
    - Output z-score directly measures what PROPAGATES to the next layer

COMPLEMENTARY TO gelu193:
    gelu193 used z_in × z_out (joint product gate)
    gelu198 uses only z_out (pure output-space gate)

PARAMS: logit_decay, log_tau, log_beta, log_gamma = 4 scalars
STATE:  _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU198(nn.Module):
    """Per-channel output-space z-score vector gate: tracks GELU output statistics."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                of = out.detach().flatten(0, 1)
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        # ── Per-channel output z-score ─────────────────────────────────────
        with torch.no_grad():
            var_out  = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            std_out  = var_out.sqrt().view(1, 1, D)
            mu_out   = self._ema_out_mean.view(1, 1, D)
            z_out    = (out.detach() - mu_out) / (std_out + self.eps)       # (B, T, D)

        gate_vec = (1.0 + beta * torch.tanh(gamma * z_out)).clamp(0.1, 5.0) # (B, T, D)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)              # (B, T, 1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            ofl = out.detach().flatten(0, 1)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * ofl.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * ofl.pow(2).mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(ofl.mean(0), dim=0)

        return output
