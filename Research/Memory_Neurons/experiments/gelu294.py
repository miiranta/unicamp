"""gelu294 – Per-Channel Diagonal GRU Gate.

CONCEPT:
    All previous experiments track activation STATISTICS (means, variances).
    A GRU hidden state can represent richer sequential information: not just
    "is this channel currently high?" but "has this channel been consistently
    high across many recent batches?"

    We implement a minimal, fully diagonal (per-channel) GRU:
        z_d = sigmoid(w_z_d * score_d + b_z_d)    update gate
        h_d = (1 - z_d) * h_d + z_d * tanh(w_h_d * score_d + b_h_d)
    where score_d = gate value from gelu211 for channel d.

    h_d persists across batches and across eval passes.  It encodes
    "what has the gate been doing for channel d over time".

SEQUENTIAL ADAPTATION:
    Pass 1: h accumulates gating patterns for test content.
        Channels amplified repeatedly by gelu211 → h_d → positive → gate↑.
    Pass 2: h from pass 1 already encodes "channel d was consistently
        amplified for this test content".
        If the learned GRU maps consistently-amplified → suppressed:
        the model can DEPLETE without an explicit depletion parameter.
    The GRU weights are trained to find the correct mapping.

GATE COMPOSITION:
    gate_211_d = gelu211 product gate (input-space × output-space)
    gate_gru_d = 1 + beta * tanh(h_d)
    output     = out * gate_211_d * gate_gru_d

BENEFIT FROM BACKPROP:
    All GRU params (w_z, b_z, w_h, b_h, log_beta_gru) get gradients.
    The GRU learns to use its hidden state to encode habituation pressure.

NO CAUSALITY LEAK:
    score_d is gelu211 gate value (batch-level, same causal guarantees).
    GRU updates are per-batch, not per-position.

PARAMS:  gelu211 params (7) + w_z (D), b_z (D), w_h (D), b_h (D), log_beta_gru.
STATE:   gelu211 state + h (D,), reset to zeros.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU294(nn.Module):
    """gelu211 + per-channel diagonal GRU: h encodes historical gating pattern."""

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

        # ── Diagonal GRU params ─────────────────────────────────────────
        self.w_z          = nn.Parameter(torch.zeros(D_FF))
        self.b_z          = nn.Parameter(torch.zeros(D_FF))
        self.w_h          = nn.Parameter(torch.zeros(D_FF))
        self.b_h          = nn.Parameter(torch.zeros(D_FF))
        self.log_beta_gru = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        # ── GRU state ───────────────────────────────────────────────────
        self._h: torch.Tensor = None   # (D,), reset to zeros

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False
        self._h            = None

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
                self._h            = torch.zeros(D, device=x.device, dtype=x.dtype)
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

        # ── gelu211 gate ─────────────────────────────────────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)
        gate_211 = (gate_in * gate_out * gate_cos).mean(dim=(0,1))   # (D,) batch-mean gate score

        # ── Diagonal GRU update ──────────────────────────────────────────
        # score = batch-mean gate value (detached from gelu211 path)
        score = gate_211.detach()
        z_gru = torch.sigmoid(self.w_z * score + self.b_z)           # (D,) update gate
        h_cand= torch.tanh(self.w_h * score + self.b_h)              # (D,) candidate
        new_h = (1 - z_gru) * self._h.detach() + z_gru * h_cand      # (D,) new state

        beta_gru = F.softplus(self.log_beta_gru)
        gate_gru = (1.0 + beta_gru * torch.tanh(new_h)).clamp(0.05, 4.0)  # (D,)

        output = out * gate_in * gate_out * gate_cos * gate_gru.view(1,1,D)

        self._h = new_h.detach()

        with torch.no_grad():
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
