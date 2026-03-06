"""GELU215 – Dual-Space Asymmetric Intersection Gate.

MOTIVATION — UPGRADING gelu205's SYMMETRIC GATE TO ASYMMETRIC:
    gelu205 (PPL 161.10): dual-space intersection gate
        - Top-K channels by |z_in| AND |z_out| simultaneously
        - Symmetric gate: 1 + β×tanh(γ×z_in_d) on intersection
        - This is the best SPARSE method (K_each=64, ~4 intersection channels)

    gelu190 (PPL 160.54): asymmetric DENSE gate
        - β_up for amplification, β_dn for suppression
        - All D channels modulated

    HYPOTHESIS: The intersection constraint of gelu205 reduces noise (fewer channels)
    while the asymmetric arms of gelu190 give finer control. Combining them:

        DUAL-SPACE ASYMMETRIC INTERSECTION GATE:
        1. Find top-K channels by |z_in| (novel in input space)
        2. Find top-K channels by |z_out| (novel in output space)
        3. Intersection = channels in BOTH top-K sets
        4. Apply gelu190's asymmetric arms ON THE INTERSECTION only:
            gate_d = 1 + β_up×ReLU(tanh(γ×z_in_d)) − β_dn×ReLU(tanh(−γ×z_in_d))
        5. Channels OUTSIDE intersection: gate_d = 1 (pass-through)

    More selective signal (intersection) + asymmetric control = ideally best of both.

SIGNED Z-SCORE CHOICE:
    We gate on z_in (signed: whether channel is above or below its mean).
    The intersection SELECTS which channels to touch (large |z_in| AND |z_out|).
    The asymmetric arms decide HOW to modulate (amplify high-z, suppress low-z).

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


K_EACH = 64   # same as gelu205


class GELU215(nn.Module):
    """Dual-space intersection (top-K by both |z_in| and |z_out|) + asymmetric per-channel gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.k_each  = K_EACH
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean     = None
        self._ema_sq       = None
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k = min(self.k_each, D)

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

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
            return out

        with torch.no_grad():
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            z_in    = (x.detach() - self._ema_mean.view(1,1,D)) / (var_in.sqrt().view(1,1,D) + self.eps)
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1,1,D)) / (var_out.sqrt().view(1,1,D) + self.eps)

            # Per-channel mean |z| scores
            abs_z_in_mean  = z_in.abs().mean(dim=(0, 1))    # (D,)
            abs_z_out_mean = z_out.abs().mean(dim=(0, 1))   # (D,)

            # Top-K masks
            mask_in  = torch.zeros(D, device=x.device, dtype=torch.bool)
            mask_out = torch.zeros(D, device=x.device, dtype=torch.bool)
            mask_in[abs_z_in_mean.topk(k).indices]  = True
            mask_out[abs_z_out_mean.topk(k).indices] = True
            # Intersection
            mask_inter = (mask_in & mask_out).view(1, 1, D).expand(B, T, D)

        # ── Asymmetric gate on intersection channels ───────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_raw = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)    # (B, T, D)

        gate_vec = torch.where(mask_inter, gate_raw, torch.ones_like(gate_raw))

        # ── Cosine output EMA gate (scalar) ────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vec * gate_cos

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
