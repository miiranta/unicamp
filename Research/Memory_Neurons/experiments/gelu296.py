"""gelu296 – Block-Wise EMA Decay with Differentiable Update.

CONCEPT:
    gelu283 (differentiable EMA) uses a SINGLE decay rate d shared across
    all D channels.  But different channels may have very different optimal
    timescales: channels encoding low-frequency patterns need slow decay
    (d→1), while channels encoding transient patterns need fast decay (d→0).

    This experiment divides the D_FF channels into B=8 equal blocks.
    Each block gets its own learnable logit_decay_b (8 scalars total).
    Everything else is identical to gelu283: differentiable EMA update with
    the old state detached and current batch contributing gradient.

BENEFIT:
    Block-structured decay: minimal parameter overhead (+7 params vs gelu283)
    but 8× richer temporal modelling.  Gradient can push some blocks to fast
    decay (adapt quickly) and others to slow decay (maintain stable baseline).
    This is especially useful for sequential adaptation: some blocks may benefit
    from tracking test content fast (quick adaptation), others may benefit from
    remembering training distribution (stable reference).

NO CAUSALITY LEAK:
    per-batch statistics, identical to gelu211.

BENEFIT FROM BACKPROP:
    8 logit_decay_b values are trained to optimise PPL; each block finds
    its own optimal timescale.  All gate params also receive gradient.

PARAMS:  logit_decay_b (B=8), log_tau, log_beta_up, log_beta_dn, log_gamma,
         log_beta_out, log_gamma_out.
STATE:   five EMA buffers (D,) — identical to gelu211.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_N_BLOCKS = 8


class GELU296(nn.Module):
    """gelu283 with per-block (8 blocks) independent EMA decay rates."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.D_FF    = D_FF
        self.n_blocks = _N_BLOCKS

        # One decay per block; initialise all near ema_decay
        init_logit = math.log(ema_decay / (1.0 - ema_decay))
        self.logit_decay_b = nn.Parameter(torch.full((_N_BLOCKS,), init_logit))

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

    def _block_decay(self, device, dtype):
        """Return per-channel decay tensor by repeating block values."""
        d_b   = torch.sigmoid(self.logit_decay_b)             # (B,)
        block_size = self.D_FF // self.n_blocks
        # Each block d_b[i] applies to block_size channels
        d_ch  = d_b.repeat_interleave(block_size)[:self.D_FF] # (D,)
        return d_ch

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

        d     = self._block_decay(x.device, x.dtype)          # (D,) per-channel decay
        tau      = self.log_tau.exp()
        beta_up  = F.softplus(self.log_beta_up)
        beta_dn  = F.softplus(self.log_beta_dn)
        gamma    = F.softplus(self.log_gamma)
        beta_out = F.softplus(self.log_beta_out)
        gamma_out= F.softplus(self.log_gamma_out)

        xf = x.flatten(0,1); of = out.flatten(0,1)

        # Differentiable block-wise EMA update
        xf_mean  = xf.mean(0); xf_sq  = xf.pow(2).mean(0)
        of_mean  = of.mean(0); of_sq  = of.pow(2).mean(0)
        of_dir_n = F.normalize(of_mean, dim=0)

        new_ema_mean     = d * self._ema_mean.detach()     + (1 - d) * xf_mean
        new_ema_sq       = d * self._ema_sq.detach()       + (1 - d) * xf_sq
        new_ema_out_mean = d * self._ema_out_mean.detach() + (1 - d) * of_mean
        new_ema_out_sq   = d * self._ema_out_sq.detach()   + (1 - d) * of_sq
        new_ema_out_dir  = d * self._ema_out_dir.detach()  + (1 - d) * of_dir_n

        var_in  = (new_ema_sq  - new_ema_mean.pow(2)).clamp(min=self.eps_var)
        z_in    = (x   - new_ema_mean)  / (var_in.sqrt()  + self.eps)
        var_out = (new_ema_out_sq - new_ema_out_mean.pow(2)).clamp(min=self.eps_var)
        z_out   = (out - new_ema_out_mean) / (var_out.sqrt() + self.eps)

        up_arm  = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm  = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out= (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)

        out_n   = F.normalize(out, dim=-1)
        dir_n   = F.normalize(new_ema_out_dir, dim=0).view(1,1,D)
        cos_sim = (out_n * dir_n).sum(-1).clamp(-1,1)
        gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output  = out * gate_in * gate_out * gate_cos

        self._ema_mean     = new_ema_mean.detach()
        self._ema_sq       = new_ema_sq.detach()
        self._ema_out_mean = new_ema_out_mean.detach()
        self._ema_out_sq   = new_ema_out_sq.detach()
        self._ema_out_dir  = new_ema_out_dir.detach()

        return output
