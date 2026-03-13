"""GELU310 – Sigmoid-Softmax Mixture Gate (Creative: Multi-Mode Gating).

CREATIVE APPROACH: Instead of a single gate formula, learn K=4 "gate modes" and
combine them with a softmax mixture using a LEARNABLE context vector.

MOTIVATION: Different parts of the activation distribution may benefit from
different gating strategies. For example:
    - Mode 1: Linear amplification (gate ≈ β * z_in)
    - Mode 2: Habituating suppression of repeats (sigmoid of cosine)
    - Mode 3: Pass-through (gate ≈ 1)
    - Mode 4: Output-based normalization

The softmax mixture weights are GLOBAL scalars learned per mode, so the model
discovers which mode combination works best across the whole distribution.

GATE:
    mode_0 = gate_in  (asymmetric input z-score gate from gelu211)
    mode_1 = gate_out (symmetric output z-score gate from gelu211)
    mode_2 = gate_cos (scalar cosine gate from gelu211)
    mode_3 = 1.0      (identity / no gating)

    # Softmax mixture weights (4 scalars)
    w = softmax(log_w)   — shape (4,)

    # Weighted sum in LOG space for numerical stability
    log_modes = stack([log(gate_in), log(gate_out), log(gate_cos), zeros], dim=-1)  (B,T,D,4) or (B,T,1,4)
    log_gate  = (w * log_modes).sum(-1)   ← weighted average of log-gates

    output = out * exp(log_gate.clamp(-3, 2.3))

WHY THIS IS CREATIVE: The model learns to BLEND three gating strategies, and
the gradient can shift weight toward whichever combination minimizes perplexity.

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma,
        log_beta_out, log_gamma_out, log_w (4 scalars)  → 11 total
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU310(nn.Module):
    """Softmax-mixture of gelu211's three gate modes in log-space."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Mixture weights: start near equal mix of all 4 modes
        self.log_w         = nn.Parameter(torch.zeros(4))  # softmax → 0.25 each initially

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)
        w         = torch.softmax(self.log_w, dim=0)   # (4,) sums to 1

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            var_in  = (self._ema_sq  - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach()   - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

        gate_in  = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_in))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_in))).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)   # (B, T, 1)

        # Stack log-gate modes: (B, T, D, 4)
        log_gin  = gate_in.log()                        # (B, T, D)
        log_gout = gate_out.log()                       # (B, T, D)
        log_gcos = gate_cos.log().expand(B, T, D)      # (B, T, D)
        log_id   = torch.zeros_like(log_gin)             # mode 3: identity

        # Weighted mixture in log-space: w[0]*log_gin + w[1]*log_gout + w[2]*log_gcos + w[3]*0
        log_gate = w[0] * log_gin + w[1] * log_gout + w[2] * log_gcos + w[3] * log_id

        output = out * torch.exp(log_gate.clamp(-3.0, 2.3))

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
