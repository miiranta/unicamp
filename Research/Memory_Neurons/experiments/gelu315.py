"""GELU315 – Two-Timescale Velocity Gate (Fast − Slow EMA Trend Signal).

MOTIVATION: gelu211 gates on STATIC novelty: |x − ema_mean| / std.
This measures whether the current activation is unusual vs. the historical average.

A DIFFERENT SIGNAL: Instead of measuring deviation from a fixed baseline, measure
CHANGE IN THE BASELINE ITSELF. If the fast EMA is pulling away from the slow EMA,
the activation distribution is TRENDING — a different kind of novelty.

VELOCITY SIGNAL:
    fast_mean  = EMA(x, d_fast ≈ 0.5)   — responds quickly to recent batches
    slow_mean  = EMA(x, d_slow ≈ 0.99)  — long-term baseline
    velocity   = fast_mean − slow_mean     — direction of drift

    z_vel = velocity / (slow_std + ε)     — normalised by slow variance

    gate_vel = asym(z_vel)                — amplify channels drifting upward, suppress drifting down

This is like a first-order derivative of the activation distribution.

COMBINED WITH cosine gate:
    output = out × gate_vel × gate_cos

CAUSALITY: both EMAs are batch-level statistics → safe.
NO torch.no_grad on gate computation (gate_vel gets gradient via β, γ params).

PARAMS: log_tau, log_d_fast, log_d_slow (two separate decays!), log_beta_up, log_beta_dn, log_gamma  (6)
STATE:  _fast_mean (D,), _fast_sq (D,), _slow_mean (D,), _slow_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU315(nn.Module):
    """Two-timescale velocity gate: gates on fast-EMA − slow-EMA drift signal."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        # Two independent EMA decays
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.logit_d_fast = nn.Parameter(torch.tensor(math.log(0.5 / 0.5)))      # d_fast ≈ 0.5
        self.logit_d_slow = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))    # d_slow ≈ 0.99
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._fast_mean:  torch.Tensor = None
        self._fast_sq:    torch.Tensor = None
        self._slow_mean:  torch.Tensor = None
        self._slow_sq:    torch.Tensor = None
        self._ema_out_dir: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._fast_mean  = None
        self._fast_sq    = None
        self._slow_mean  = None
        self._slow_sq    = None
        self._ema_out_dir = None
        self._ready      = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_fast  = torch.sigmoid(self.logit_d_fast).detach().item()
        d_slow  = torch.sigmoid(self.logit_d_slow).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                batch_mean = xf.mean(0)
                self._fast_mean  = batch_mean.clone()
                self._fast_sq    = xf.pow(2).mean(0).clone()
                self._slow_mean  = batch_mean.clone()
                self._slow_sq    = xf.pow(2).mean(0).clone()
                self._ema_out_dir = F.normalize(of.mean(0), dim=0).clone()
                self._ready = True
            return out

        with torch.no_grad():
            # Velocity = fast_mean − slow_mean (normalised by slow variance)
            slow_var = (self._slow_sq - self._slow_mean.pow(2)).clamp(min=self.eps_var)
            velocity = self._fast_mean - self._slow_mean
            z_vel    = (velocity / (slow_var.sqrt() + self.eps)).view(1, 1, D)

        # Asymmetric gate on velocity (same structure as gelu211 input gate)
        gate_vel = (1.0 + beta_up * F.relu(torch.tanh( gamma * z_vel))
                       - beta_dn * F.relu(torch.tanh(-gamma * z_vel))).clamp(0.05, 8.0)

        with torch.no_grad():
            out_n    = F.normalize(out.detach(), dim=-1)
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vel * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            bm = xf.mean(0)
            self._fast_mean  = d_fast * self._fast_mean + (1 - d_fast) * bm
            self._fast_sq    = d_fast * self._fast_sq   + (1 - d_fast) * xf.pow(2).mean(0)
            self._slow_mean  = d_slow * self._slow_mean + (1 - d_slow) * bm
            self._slow_sq    = d_slow * self._slow_sq   + (1 - d_slow) * xf.pow(2).mean(0)
            self._ema_out_dir = d_slow * self._ema_out_dir + (1 - d_slow) * F.normalize(of.mean(0), dim=0)

        return output
