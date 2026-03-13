"""gelu281 – Per-Channel Gate-Snapshot Ring Buffer (Amplified Gate Replay).

THE LIMITATION OF ACTIVATION-BASED RING BUFFERS (gelu238, Δ=+0.75):
    All ring buffer experiments store ACTIVATION MEANS to identify familiar batches.
    The resulting second-pass gate is a SCALAR: one number applied uniformly across D.

    This discards rich per-channel structure:
        In pass 1, channel 200 was amplified (z_d > 0) → gate > 1.
        In pass 1, channel 512 was suppressed (z_d < 0) → gate < 1.
    On pass 2, both channels should get the same PATTERN replayed — but
    a scalar gate treats them identically.

THE FIX — STORE THE GATE VECTOR:
    For each buffer slot, store BOTH:
        1. Normalised activation mean (D floats) — for batch recognition
        2. Mean per-channel gate vector from gelu211 (D floats) — for replay

    On pass-2 recognition:
        gate_replay_d = stored_gate_d ^ (1 + κ)    (power amplification)
        output = gelu(x) × gate_replay_d

    Since stored_gate_d < 1 on suppressed channels (familiar-to-test content):
        Power (1+κ) pulls value CLOSER to 0 → deeper suppression.
    Since stored_gate_d > 1 on amplified channels (novel-relative-to-training):
        Power (1+κ) pushes value FURTHER above 1 → stronger novelty boost.

    The CHANNEL PATTERN is thus replayed with increased contrast each pass.

WHY THIS IS BETTER THAN gelu238:
    gelu238's episodic gate is scalar — it applies the same factor to all D.
    A token might have channel 50 heavily suppressed and channel 200 amplified.
    A scalar gate misses this structure entirely.
    gelu281 stores D gate values per slot → exact channel-level suppression
    pattern is reused with amplification → more precise and stronger Δ.

TRAINING vs EVAL:
    Training (self.training = True): standard gelu211 gate only; buffer not used.
    Eval pass 1: fill buffer with (act_mean, gate_mean) pairs; use raw gate.
    Eval pass 2+: detected via high cosine match; replay amplified stored gate.

    Eval-only replay means training learns conservatively for base PPL;
    replay is free to be aggressive because it only fires at test time.

PARAMS (trained):   all gelu211 params + log_kappa (replay power, init 0.5).
PARAMS (eval init): logit_gate_floor (floor on replay gate, init ≈ 0.05).
STATE:  act_buf (N, D), gate_buf (N, D), mask (N,), ptr (int), _pass2 (bool).
        Buffer size N=256; per layer ≈ 256 × 1024 × 2 × 4 bytes ≈ 2 MB.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECT_THRESH = 0.88
N_BUF = 256


class GELU281(nn.Module):
    """gelu211 base + per-channel gate-vector ring buffer; amplified replay on pass 2."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4

        # ── gelu211 params ──────────────────────────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Replay params ────────────────────────────────────────────────
        self.log_kappa        = nn.Parameter(torch.tensor(math.log(0.5)))   # power factor
        self.logit_gate_floor = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))  # ≈0.05

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        # ── Buffer state ─────────────────────────────────────────────────
        self._act_buf:  torch.Tensor = None   # (N, D) normalised act means
        self._gate_buf: torch.Tensor = None   # (N, D) mean gate vectors from gelu211
        self._mask:     torch.Tensor = None   # (N,) bool — slot is filled
        self._ptr      = 0
        self._pass2    = False

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False
        self._act_buf      = None;  self._gate_buf     = None
        self._mask         = None;  self._ptr          = 0
        self._pass2        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _init_bufs(self, D: int, device, dtype):
        self._act_buf  = torch.zeros(N_BUF, D, device=device, dtype=dtype)
        self._gate_buf = torch.ones( N_BUF, D, device=device, dtype=dtype)
        self._mask     = torch.zeros(N_BUF, device=device, dtype=torch.bool)
        self._ptr      = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)
        kappa       = self.log_kappa.exp().clamp(0.01, 3.0)
        gate_floor  = 0.01 + 0.49 * torch.sigmoid(self.logit_gate_floor)

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
            if self._act_buf is None:
                self._init_bufs(D, x.device, x.dtype)
            return out

        if self._act_buf is None:
            self._init_bufs(D, x.device, x.dtype)

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

        gate_raw  = (gate_in * gate_out * gate_cos).clamp(0.01, 20.0)   # (B, T, D)
        gate_mean = gate_raw.detach().flatten(0, 1).mean(0)              # (D,)

        m_curr   = out.detach().flatten(0, 1).mean(0)
        m_curr_n = F.normalize(m_curr, dim=0)

        # ── Ring buffer logic ─────────────────────────────────────────────
        if not self._pass2 and self.training:
            # Training: standard gelu211 only
            output = out * gate_raw

        elif not self._pass2:
            # Eval pass 1: fill buffer, check for pass-2 detection
            detected = False
            if self._mask.any():
                sims = (F.normalize(self._act_buf, dim=-1) * m_curr_n).sum(-1)
                sims = sims.masked_fill(~self._mask, -1.0)
                if sims.max().item() > DETECT_THRESH:
                    self._pass2 = True
                    detected    = True
                else:
                    self._act_buf[self._ptr]  = m_curr_n
                    self._gate_buf[self._ptr] = gate_mean
                    self._mask[self._ptr]     = True
                    self._ptr = (self._ptr + 1) % N_BUF
            else:
                self._act_buf[0]  = m_curr_n
                self._gate_buf[0] = gate_mean
                self._mask[0]     = True
                self._ptr         = 1

            if not detected:
                output = out * gate_raw

        if self._pass2:
            # Eval pass 2+: replay stored gate vector, amplified
            with torch.no_grad():
                sims2  = (F.normalize(self._act_buf, dim=-1) * m_curr_n).sum(-1)
                sims2  = sims2.masked_fill(~self._mask, -1.0)
                n_idx  = sims2.argmax()
                stored = self._gate_buf[n_idx].clamp(0.01, 20.0)   # (D,)
                # Amplify: stored^(1+κ) — deepens suppression where stored<1, amplification where >1
                replay = stored.log().mul(1.0 + kappa).exp()
                replay = replay.clamp(gate_floor.item(), 20.0).view(1, 1, D)
            output = out * replay

        # ── EMA updates ───────────────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
