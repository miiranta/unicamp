"""gelu283 – gelu211 with Fully Differentiable EMA Updates.

THE CHANGE FROM gelu211:
    gelu211 wraps all EMA statistics updates in torch.no_grad(), which blocks
    gradient flow to learnable parameters (logit_decay, log_tau, etc.) through
    the EMA reference values.

    Here we remove torch.no_grad() and instead use:
        new_ema = d * old_ema.detach() + (1 - d) * current_batch_stats

    The .detach() on old_ema prevents unbounded BPTT across batches, but
    gradient still flows through:
        • d = sigmoid(logit_decay)  via  (1 - d) * current_stats
        • current_batch_stats       via  z_score = (x - new_ema) / std

    This gives logit_decay a direct gradient signal: "how much should the
    current batch shift the reference used to compute novelty?"

    All five EMA buffers (ema_mean, ema_sq, ema_out_mean, ema_out_sq,
    ema_out_dir) are updated differentiably; the cosine gate is also
    computed without no_grad.

CAUSALITY:
    Per-batch statistics (mean/var over all B×T positions) are causal across
    batches and across eval passes — identical to gelu211. No new causality
    issues are introduced.

SEQUENTIAL ADAPTATION:
    state persists across eval passes (not reset between them);
    same mechanism as gelu211, but now logit_decay is optimised more
    precisely → the decay rate may be tuned for better pass-1 PPL.

PARAMS:  same seven as gelu211 (logit_decay, log_tau, log_beta_up/dn,
         log_gamma, log_beta_out, log_gamma_out).
STATE:   five EMA buffers, identical to gelu211.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU283(nn.Module):
    """gelu211 with differentiable EMA: gradient flows through decay rate d."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
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
                self._ready = True
            return out

        d        = torch.sigmoid(self.logit_decay)          # scalar, has gradient
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        gamma    = F.softplus(self.log_gamma)
        beta_out = F.softplus(self.log_beta_out)
        gamma_out= F.softplus(self.log_gamma_out)

        xf = x.flatten(0, 1)                               # (B*T, D)
        of = out.flatten(0, 1)                             # (B*T, D)

        # ── Differentiable EMA update ─────────────────────────────────
        # old state detached (no BPTT); current stats contribute gradient
        new_ema_mean     = d * self._ema_mean.detach()     + (1 - d) * xf.mean(0)
        new_ema_sq       = d * self._ema_sq.detach()       + (1 - d) * xf.pow(2).mean(0)
        new_ema_out_mean = d * self._ema_out_mean.detach() + (1 - d) * of.mean(0)
        new_ema_out_sq   = d * self._ema_out_sq.detach()   + (1 - d) * of.pow(2).mean(0)
        new_ema_out_dir  = d * self._ema_out_dir.detach()  + (1 - d) * F.normalize(of.mean(0), dim=0)

        # ── Z-scores using updated EMAs (gradient flows through d) ────
        var_in  = (new_ema_sq  - new_ema_mean.pow(2)).clamp(min=self.eps_var)
        z_in    = (x   - new_ema_mean)  / (var_in.sqrt()  + self.eps)   # (B, T, D)

        var_out = (new_ema_out_sq - new_ema_out_mean.pow(2)).clamp(min=self.eps_var)
        z_out   = (out - new_ema_out_mean) / (var_out.sqrt() + self.eps)  # (B, T, D)

        # ── Input-space asymmetric gate ───────────────────────────────
        up_arm  = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm  = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)

        # ── Output-space symmetric gate ───────────────────────────────
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        # ── Cosine output-direction gate ──────────────────────────────
        out_n    = F.normalize(out, dim=-1)
        dir_n    = F.normalize(new_ema_out_dir, dim=0).view(1, 1, D)
        cos_sim  = (out_n * dir_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)   # (B, T, 1)

        output = out * gate_in * gate_out * gate_cos

        # ── Store EMAs for next batch (detached to prevent BPTT) ─────
        self._ema_mean     = new_ema_mean.detach()
        self._ema_sq       = new_ema_sq.detach()
        self._ema_out_mean = new_ema_out_mean.detach()
        self._ema_out_sq   = new_ema_out_sq.detach()
        self._ema_out_dir  = new_ema_out_dir.detach()

        return output
