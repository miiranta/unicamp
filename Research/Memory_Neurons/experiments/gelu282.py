"""gelu282 – Per-Channel Depletion via Novelty Accumulation.

THE MECHANISM BRIDGE BETWEEN gelu253 AND gelu211:
    gelu253 (PPL=182.60, Δ=+7.34): SCALAR depletion. Achieves huge Δ by
        accumulating a single depletion variable that gates ALL channels equally.
        Hurts base PPL because it mutes channels indiscriminately.

    gelu211 (PPL=159.35, Δ≈0):    no depletion. Per-channel asymmetric gate
        amplifies novel channels (z_d > 0) and suppresses familiar ones (z_d < 0).
        No adaptation because the gate re-fires each pass with similar strengths.

THE INSIGHT:
    In gelu211, channels that were AMPLIFIED on pass 1 (gate_d > 1) are the
    "test-novel" channels — they were above the training EMA. On pass 2 they
    are NO LONGER novel: same content, same context. They should be suppressed.
    But gelu211 doesn't know they were amplified before; the EMA has drifted.

THE FIX — ACCUMULATE AMPLIFICATION → DEPLETE:
    Track EMA of how much each channel was AMPLIFIED above its baseline:

        excess_d = ReLU(gate_raw_d − 1.0)          ← how much was amplified
        depl_d ← d_r × depl_d + (1−d_r) × excess_d ← slow EMA accumulates novelty

    Then apply a DEPLETION FACTOR to future gates:

        depl_gate_d = exp(−U × depl_d)             ∈ (0, 1]
        gate_final  = gate_raw_d × depl_gate_d      (clamped)

BEHAVIOUR:
    Pass 1: depl_d accumulates as gelu211 amplifies test-novel channels.
            For the FIRST occurrence (training → test transition), depl_d is low
            → gate_final ≈ gate_raw → base PPL preserved.
            Near end of pass 1: depl_d has grown for consistently-amplified channels.
    Pass 2: depl_d is ALREADY HIGH for test-typical novel channels.
            depl_gate_d < 1 → combined gate SMALLER than gate_raw.
            These channels (previously amplified) are now SUPPRESSED.
            The model sees less of the "previously novel" content → less residual
            noise from over-amplification → LOWER LOSS → PPL IMPROVES.

KEY DIFFERENCES FROM gelu253:
    1. PER-CHANNEL depletion (D-dimensional) vs gelu253's scalar gate.
       Only channels that were AMPLIFIED deplete; suppressed channels are unaffected.
    2. Depletion triggered by AMPLIFICATION (excess above 1.0), not mere firing.
       Training content that's near-baseline (gate≈1) never builds depletion.
    3. No ring buffer / pass detection needed — purely continuous EMA mechanism.

CALIBRATION:
    d_r ≈ 0.99 (very slow decay): depletion accumulates over many batches and
    persists across passes within the same eval run.
    U (learnable): controls how strongly depletion penalises amplified channels.
    Both are trained parameters; gradient descent learns the right operating point.

PARAMS: all gelu211 params + logit_d_depl (depletion EMA decay) + log_U (strength).
STATE:  gelu211 state + _depl (D,), initialised to zero (no depletion initially).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU282(nn.Module):
    """gelu211 + per-channel depletion EMA: amplified channels deplete across passes."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
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
        # d_depl: EMA decay for accumulated novelty; very slow (≈0.99) so
        # depletion persists across batches and across eval passes.
        self.logit_d_depl = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))
        # U: depletion strength — how hard accumulated novelty suppresses
        self.log_U        = nn.Parameter(torch.tensor(math.log(1.0)))

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        # ── Depletion state: zero = no depletion initially ───────────────
        self._depl: torch.Tensor = None

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False
        self._depl         = None   # reset accumulated novelty on eval restart

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)
        d_depl    = torch.sigmoid(self.logit_d_depl).detach().item()
        U         = self.log_U.exp()

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
                self._depl         = torch.zeros(D, device=x.device, dtype=x.dtype)
                self._ready        = True
            return out

        if self._depl is None:
            self._depl = torch.zeros(D, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)

        # ── gelu211 gate ─────────────────────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        gate_raw = (gate_in * gate_out * gate_cos).clamp(0.01, 20.0)   # (B, T, D)

        # ── Depletion gate ────────────────────────────────────────────────
        # Channels with accumulated excess amplification (depl_d > 0) → suppressed
        depl_gate = torch.exp(-U * self._depl).view(1, 1, D)        # (1, 1, D) ∈ (0, 1]
        gate_final = (gate_raw * depl_gate).clamp(0.01, 20.0)

        output = out * gate_final

        # ── State updates ─────────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)
            # Accumulate excess amplification: channels amplified above 1.0 in this batch
            excess = F.relu(gate_raw.detach().flatten(0, 1) - 1.0).mean(0)   # (D,)
            self._depl = d_depl * self._depl + (1 - d_depl) * excess

        return output
