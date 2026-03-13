"""gelu298 – gelu211 + Per-Channel Learnable Bypass (Skip Gate).

CONCEPT:
    The fundamental question about any gating experiment: "does this channel
    BENEFIT from gating at all?"  For some channels, the gate is always near 1
    (no habituation needed); applying gating to them just adds noise.

    This experiment adds a PER-CHANNEL BYPASS parameter alpha_d ∈ (0,1):
        output_d = alpha_d * (out_d * gate_211_d) + (1 - alpha_d) * out_d
                 = out_d * (1 - alpha_d + alpha_d * gate_211_d)
                 = out_d * effective_gate_d

    When alpha_d → 0: channel bypasses the gate entirely.
    When alpha_d → 1: channel gets the full gelu211 gate.

    Gradient flows directly from loss to alpha_d, teaching the model to
    include gating exactly for the channels where it helps.

BENEFIT FROM BACKPROP:
    alpha (D,) = sigmoid(logit_alpha) — D independent bypass parameters.
    This is the most direct use of backprop: the loss directly trains
    which channels should be gated, bypassing EMA heuristics.

CAUSALITY:
    alpha is a fixed learned parameter, not computed from current activations.
    No causality issue.

SEQUENTIAL ADAPTATION:
    alpha is static across passes (trained parameter) → Δ ≈ 0 from alpha itself.
    But: bypassing noisy channels → cleaner pass-1 PPL → cleaner reference for
    pass-2 state-based adaptation (if used with a stateful mechanism later).
    Here standalone: Δ ≈ 0, benefit is base PPL.

PARAMS:  gelu211 params (7) + logit_alpha (D,).
STATE:   gelu211 state (5 EMA buffers).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU298(nn.Module):
    """gelu211 + per-channel learned bypass: output = alpha*gated + (1-alpha)*ungated."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.D_FF    = D_FF

        # ── gelu211 params ──────────────────────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Bypass parameters ───────────────────────────────────────────
        # Init logit_alpha = 0 → alpha = 0.5 (half bypass, half gated)
        # Model will learn to push alpha_d toward 0 (bypass) or 1 (gate)
        self.logit_alpha = nn.Parameter(torch.zeros(D_FF))

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.flatten(0,1); of = out.flatten(0,1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready = True
            return out

        d_val    = torch.sigmoid(self.logit_decay).detach().item()
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        gamma    = F.softplus(self.log_gamma)
        beta_out = F.softplus(self.log_beta_out)
        gamma_out= F.softplus(self.log_gamma_out)

        with torch.no_grad():
            xf = x.detach().flatten(0,1); of = out.detach().flatten(0,1)
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1,1)
            gate_cos= torch.exp(-tau.detach() * cos_sim).unsqueeze(-1)

        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        gate_211 = gate_in * gate_out * gate_cos    # full gelu211 gate

        # ── Bypass blend ─────────────────────────────────────────────────
        alpha          = torch.sigmoid(self.logit_alpha).view(1, 1, D)  # (1,1,D) ∈ (0,1)
        effective_gate = (1.0 - alpha) + alpha * gate_211              # (B,T,D)
        output         = out * effective_gate

        with torch.no_grad():
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
