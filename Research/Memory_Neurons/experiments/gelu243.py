"""GELU243 – gelu211 PPL Gate + Depletion Adaptation (Best-of-Both).

DESIGN GOAL: Achieve BOTH best PPL (like gelu211=159.35) AND large Δ1→3.

    Current situation:
        gelu211: best PPL 159.35, Δ1→3 ≈ +0.05 (small, EMA-based)
        gelu239: expected PPL ≈ control (pass-1 gate=1.0), large Δ1→3 via depletion

    This experiment is a two-layer stack:
        Layer 1 (always active): gelu211 product gate → improves PPL
        Layer 2 (eval-only):     depletion buffer → improves Δ1→3 without affecting pass-1

LAYER 1: gelu211 gate (input × output EMA product gate)
    Active during BOTH training AND eval.
    Reduced to its core: input z-score gate × output z-score gate.
    Same learned parameters as gelu211.
    
LAYER 2: depletion ring buffer
    INACTIVE during training (self.training=True → returns 1.0 multiplier).
    Eval pass 1: builds buffer, gate=1.0 for all fresh slots.
    Eval pass 2+: depletion fires for matched slots → gate < 1.0.
    
    The depletion gate is applied ON TOP of the gelu211 output:
        output = y * gate_211 * gate_depl

    During pass 1: gate_depl = 1.0 → PPL ≈ gelu211 (no change from layer 2).
    During pass 2: gate_depl < 1.0 → additional suppression → Δppl > 0.

LAYER 2 PARAMS (receive ~zero gradient during training since layer 2 is inactive):
    log_k_depl: depletion gate strength, init 2.0

LAYER 1 PARAMS (fully learned):
    Same as gelu211: logit_decay, log_beta_up, log_beta_dn, log_gamma,
                     log_beta_out, log_gamma_out, log_tau
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU243(nn.Module):
    """gelu211 product gate (PPL) + eval-only depletion gate (adaptation)."""

    def __init__(self, ema_decay: float = 0.9, buffer_size: int = 512,
                 depletion_rate: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4

        # ── Layer 1: gelu211 params (trained) ─────────────────────────
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_beta_out  = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_gamma_out = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        # ── Layer 2: depletion params (init only, ~zero training gradient) ──
        self.log_k_depl = nn.Parameter(torch.tensor(math.log(2.0)))

        # ── Layer 1 state ─────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._l1_ready = False

        # ── Layer 2 state ─────────────────────────────────────────────
        self._N    = buffer_size
        self._DR   = depletion_rate
        self._buf:  torch.Tensor = None
        self._depl: torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._l2_ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._l1_ready     = False
        self._buf          = None
        self._depl         = None
        self._mask         = None
        self._ptr          = 0
        self._l2_ready     = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # ── Layer 1 hyperparams ────────────────────────────────────────
        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_up   = F.softplus(self.log_beta_up)
        beta_dn   = F.softplus(self.log_beta_dn)
        gamma     = F.softplus(self.log_gamma)
        beta_out  = F.softplus(self.log_beta_out)
        gamma_out = F.softplus(self.log_gamma_out)

        out = self._gelu(x)

        # ── Layer 1: gelu211 product gate ─────────────────────────────
        if not self._l1_ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._l1_ready     = True
            gate_211 = out                           # pass-through first call
        else:
            with torch.no_grad():
                var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
                z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
                var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
                z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)

            up_arm  = beta_up * F.relu(torch.tanh(gamma * z_in))
            dn_arm  = beta_dn * F.relu(torch.tanh(-gamma * z_in))
            g_in    = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
            g_out   = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

            with torch.no_grad():
                out_n  = F.normalize(out.detach(), dim=-1)
                ema_n  = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
                cos_s  = (out_n * ema_n).sum(-1).clamp(-1, 1)
                g_cos  = torch.exp(-tau * cos_s).unsqueeze(-1)

            gate_211 = out * g_in * g_out * g_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            of = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        # ── Layer 2: depletion gate (INACTIVE during training) ─────────
        if self.training:
            return gate_211   # layer 2 OFF during training

        k_depl = self.log_k_depl.exp().clamp(0.1, 8.0)
        m_curr = out.detach().flatten(0, 1).mean(0)

        if not self._l2_ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=out.dtype)
                self._depl = torch.ones(self._N,    device=x.device, dtype=out.dtype)
                self._mask = torch.zeros(self._N,   device=x.device, dtype=torch.bool)
                self._buf[0]  = F.normalize(m_curr, dim=0)
                self._depl[0] = 1.0
                self._mask[0] = True
            self._ptr    = 1
            self._l2_ready = True
            return gate_211   # first eval call: layer 2 gate = 1.0

        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            depl_level  = self._depl[nearest_idx].item()

        gate_depl = math.exp(-k_depl.item() * (1.0 - depl_level))

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._depl[nearest_idx] *= self._DR
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._depl[self._ptr] = 1.0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return gate_211 * gate_depl
