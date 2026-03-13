"""gelu279 – Per-Channel Gate Momentum (Self-Reinforcing Contrast).

THE MISSING TEMPORAL AXIS IN ALL PRIOR Z-SCORE EXPERIMENTS:
    Every experiment computes the gate from the CURRENT TOKEN vs. the running EMA.
    The gate formula is stateless w.r.t. the gate itself — it has no memory of
    what it has been doing.

    Pass 1 → Pass 2: the EMA shifts slightly (test contamination), but the FORMULA
    is the same. Result: gate ≈ same each pass → Δ ≈ 0.

THE FIX — GATE MEMORY:
    Track an EMA of the per-channel gate vector itself. The combined gate is:

        gate_raw_d     = gelu211_gate_d(z_in, z_out)          ∈ [0.05, 8.0]
        gate_ema_d     = EMA_d(gate_raw_d)                      ∈ [0.05, 8.0]
        gate_final_d   = gate_raw_d × gate_ema_d ^ κ           (clamped)

    In log space: log(gate_final_d) = log(gate_raw_d) + κ × log(gate_ema_d)

    Interpretation: gate_ema_d encodes how THIS CHANNEL has been gated historically.
        gate_ema_d < 1 (familiar channel, consistently suppressed):
            → gate_final < gate_raw → DEEPER SUPPRESSION
        gate_ema_d > 1 (novel channel, consistently amplified):
            → gate_final > gate_raw → STRONGER AMPLIFICATION

    The self-reinforcing contrast grows with each pass.

WHY THIS GIVES SEQUENTIAL ADAPTATION:
    Pass 1: gate_ema starts at 1.0 (neutral); gate_final ≈ gate_raw.
            gate_ema converges toward the typical test-content gate pattern.
    Pass 2: gate_ema now reflects pass-1.
            Channels suppressed in pass 1 (familiar-to-test, gate_ema < 1):
                → gate_final < gate_raw → EXTRA SUPPRESSION ← PPL benefit
            Channels amplified in pass 1 (globally novel, gate_ema > 1):
                → gate_final > gate_raw → EXTRA AMPLIFICATION ← sharper signal
    Pass 3: gate_ema has absorbed passes 1 and 2 → pattern sharpens further.

    The effect is PROPORTIONAL TO κ and to how consistently the gate has fired.
    No ring buffer. No discrete detection. Fully continuous.

RELATION TO PRIOR WORK:
    gelu276 (Δ=+0.77): freezes mean/variance EMA at pass-1 end to reduce z-scores.
        gelu279: does not freeze; uses gate EMA as implicit channel-level memory.
    gelu238 (Δ=+0.75): adds episodic ring buffer + hard scalar gate on pass 2.
        gelu279: per-channel gate momentum — finer resolution, no detection needed.
    gelu253 (Δ=+7.34, PPL=182.60): accumulates scalar depletion.
        gelu279: same idea but per-channel and without disrupting pass-1 PPL.

PARAMS: all gelu211 params + logit_d_gate (gate EMA decay) + log_kappa (momentum).
STATE:  gelu211 state + _gate_ema (D,), initialised to 1.0 (neutral).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU279(nn.Module):
    """Per-channel gate momentum: gate_final = gate_raw × gate_ema^κ."""

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

        # ── Gate momentum params ────────────────────────────────────────
        # d_gate: how quickly gate_ema adapts; slow (≈0.99) → long historical memory
        self.logit_d_gate = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))
        # κ: momentum strength — how much gate_ema influences gate_final (init 0.5)
        self.log_kappa    = nn.Parameter(torch.tensor(math.log(0.5)))

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        # ── Gate EMA state: initialised to 1.0 (no momentum initially) ─
        self._gate_ema: torch.Tensor = None

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False
        self._gate_ema     = None

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
        d_gate    = torch.sigmoid(self.logit_d_gate).detach().item()
        kappa     = self.log_kappa.exp()

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
                self._gate_ema     = torch.ones(D, device=x.device, dtype=x.dtype)
                self._ready        = True
            return out

        if self._gate_ema is None:
            self._gate_ema = torch.ones(D, device=x.device, dtype=x.dtype)

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
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)   # (B, T, 1)

        gate_raw = (gate_in * gate_out * gate_cos).clamp(0.01, 20.0)   # (B, T, D)

        # ── Gate momentum ────────────────────────────────────────────────
        # gate_ema^κ: amplifies suppression where historically suppressed, amplification where amplified
        log_gate_ema = self._gate_ema.clamp(0.01, 20.0).log().view(1, 1, D)
        momentum     = torch.exp(kappa * log_gate_ema)             # (1, 1, D)
        gate_final   = (gate_raw * momentum).clamp(0.01, 20.0)

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
            # Gate EMA: updated from mean gate across (B*T) — per channel
            gate_mean       = gate_raw.detach().flatten(0, 1).mean(0)   # (D,)
            self._gate_ema  = d_gate * self._gate_ema + (1 - d_gate) * gate_mean

        return output
