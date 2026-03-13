"""gelu291 – gelu282 with Fully Differentiable Per-Channel Depletion.

THE CHANGE FROM gelu282:
    gelu282 wraps the depletion accumulation in torch.no_grad():
        excess = relu(gate_raw - 1.0)   (detached)
        depl ← d * depl + (1-d) * excess.mean(0)   (no_grad)
        depl_gate = exp(-U * depl)

    Here, the depletion participates in the computation graph:
        excess      = relu(gate_raw - 1.0)           ← gradient from gate_raw
        new_depl    = d * _depl.detach() + (1-d) * excess.mean(dim=(0,1))
                                                      ← gradient through excess, d, (1-d)
        depl_gate   = exp(-U * new_depl)              ← gradient through U and new_depl
        gate_final  = gate_raw * depl_gate            ← gradient through both arms
        _depl       = new_depl.detach()               ← stored for next batch (no BPTT)

WHY THIS MATTERS:
    The gradient path: loss → gate_final → depl_gate → new_depl → excess → gate_raw
    creates a "circular" incentive:
        - Amplifying a channel (gate_raw > 1) increases excess → increases depl →
          depl_gate decreases → gate_final decreases for future occurrences.
        - The model learns to time amplification for channels where the resulting
          depletion in pass-2 produces a net LOWER LOSS (better sequential Δ).

    Without this gradient path (gelu282), gate_raw is trained ignoring the
    downstream consequence of depletion.  With it, gate_raw and depletion params
    are jointly optimised for coordinated adaptation.

CAUSALITY:
    Depletion accumulates causally across batches — identical to gelu282.
    No within-sequence future information used.

PARAMS:  all gelu211 params + logit_d_depl + log_U (same as gelu282).
STATE:   five gelu211 EMA buffers + _depl (D,) — reset to zeros on reset_state().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU291(nn.Module):
    """gelu282 variant: depletion update participates in the computation graph."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        # ── gelu211 params ──────────────────────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Depletion params ─────────────────────────────────────────────
        self.logit_d_depl = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))
        self.log_U        = nn.Parameter(torch.tensor(math.log(1.0)))

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        # ── Depletion buffer ─────────────────────────────────────────────
        self._depl: torch.Tensor = None

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False
        self._depl         = None

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        out = self._gelu(x)

        # ── First batch: initialise buffers, return ungated ──────────────
        if not self._ready:
            with torch.no_grad():
                xf = x.flatten(0, 1); of = out.flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._depl         = torch.zeros(D, device=x.device, dtype=x.dtype)
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
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1, 1, D)) / (var_in.sqrt().view(1, 1, D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau.detach() * cos_sim).unsqueeze(-1)

        # ── gelu211 gate_raw (with gradient through gate params) ─────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)
        gate_raw = gate_in * gate_out * gate_cos   # (B, T, D)

        # ── Differentiable depletion accumulation ────────────────────────
        d_depl  = torch.sigmoid(self.logit_d_depl)
        U       = self.log_U.exp()

        excess      = F.relu(gate_raw - 1.0)                                     # (B, T, D) with grad
        new_depl    = d_depl * self._depl.detach() + (1 - d_depl) * excess.mean(dim=(0, 1))
        depl_gate   = torch.exp(-U * new_depl)                                   # (D,)

        output = out * (gate_raw * depl_gate.view(1, 1, D)).clamp(0.05, 10.0)

        # ── Store state (detached to prevent BPTT explosion) ─────────────
        self._depl = new_depl.detach()
        with torch.no_grad():
            xf = x.detach().flatten(0, 1); of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1 - d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1 - d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1 - d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1 - d_val) * F.normalize(om, dim=0)

        return output
