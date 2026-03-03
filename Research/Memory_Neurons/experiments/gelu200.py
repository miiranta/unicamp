"""GELU200 – Surprise Momentum Gate (Habituation-Based).

THE SINGLE-STEP SURPRISE PROBLEM:
    All previous experiments compute surprise from x[b,t] vs. the EMA mean.
    A 5-token run of highly surprising tokens receives the same per-token gate
    as a single surprising token in isolation.

    But in a language model, a long run of surprising tokens is a context:
    - After 10 surprising tokens, the model should HABITUATE (reduce amplification)
      because this new level of surprise is becoming the "new normal"
    - A SINGLE unexpected token after a run of familiar tokens is maximally novel

THE NEW IDEA: Surprise Momentum (Two-Level Surprise)
    Track an EMA of recent batch-mean surprise values:
        surp_ema(t) = d_s × surp_ema(t-1) + (1-d_s) × mean(surp_raw(t))

    Modulated surprise (habituation):
        surp_mod(t) = ReLU(surp_raw(t) - surp_ema(t))

    When surp_raw >> surp_ema:  token is MORE surprising than recent ones → fire hard
    When surp_raw ≈  surp_ema:  consistent surprise → habituated → reduced gate
    When surp_raw <  surp_ema:  LESS surprising → near-zero modulation

RESULT: Gate is highest for tokens that BREAK A PATTERN.

INITIALIZATION:
    surp_ema = 0 → first surprising token always fires at full strength.
    If every token has surp ≈ 0.3, surp_ema → 0.3, and surp_mod → 0.

PARAMS: logit_decay, log_tau, logit_decay_s, log_sigma, log_w_raw = 5 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,), _surp_ema (scalar)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU200(nn.Module):
    """Surprise momentum gate: habituates to sustained surprise, fires on surprise above recent baseline."""

    def __init__(self, ema_decay: float = 0.9, surp_decay: float = 0.7, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay  / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.logit_decay_s = nn.Parameter(torch.tensor(math.log(surp_decay / (1.0 - surp_decay))))
        self.log_sigma     = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._surp_ema: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._surp_ema = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        d_surp = torch.sigmoid(self.logit_decay_s).detach().item()
        tau    = self.log_tau.exp()
        sigma  = F.softplus(self.log_sigma)
        w      = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._surp_ema = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                self._ready    = True
            return out

        # ── Per-channel z-score → raw surprise ────────────────────────────
        with torch.no_grad():
            var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_ = var.sqrt().view(1, 1, D)
            mu_  = self._ema_mean.view(1, 1, D)
            z    = (x.detach() - mu_) / (std_ + self.eps)
            mean_absz = z.abs().mean(-1)

        surp_raw = torch.tanh(sigma * mean_absz)                 # (B, T)

        # ── Habituation: subtract recent surprise baseline ─────────────────
        surp_mod = F.relu(surp_raw - self._surp_ema)             # (B, T) excess above recent avg

        # ── Cosine familiarity gate ────────────────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp_mod)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xfl.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xfl.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)
            batch_surp = surp_raw.detach().mean()
            self._surp_ema = d_surp * self._surp_ema + (1-d_surp) * batch_surp

        return output
