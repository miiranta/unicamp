"""GELU213 – Sparse Top-K Asymmetric Gate on OUTPUT Z-Scores.

MOTIVATION — COMPLEMENT TO gelu202:
    gelu202 (PPL 162.16): sparse top-K=16 asymmetric gate on INPUT z-scores
        — find 16 most novel INPUT channels, apply β_up/β_dn on them

    gelu198 (PPL 161.89): dense symmetric gate on OUTPUT z-scores

    QUESTION: What if we apply the ASYMMETRIC sparse gate in OUTPUT space instead?

    RATIONALE:
        Input space: measures "is the pre-activation unusual?"
        Output space: measures "is the post-GELU activation unusual?"

        The GELU nonlinearity compresses/expands differently at different regimes:
        - Saturated channels (|x| >> 0): |GELU(x)| ≈ |x|
        - Threshold channels (x ≈ 0): GELU(x) ≈ 0, so output deviations are small
        - Negative channels (x << 0): GELU(x) ≈ 0, surprises invisible in output space

        Selecting TOP-K by OUTPUT z-score targets channels WHERE the output is actually
        DOING SOMETHING significant — not just input surprises that get squashed to zero
        by the nonlinearity.

MECHANISM:
    out_base = GELU(x)
    z_out_d  = (out_base_d − μ_out_d) / σ_out_d       per-channel output z-score
    top_k_mask = top-K channels by |z_out_d| mean over tokens    (D,) → (B,T,D)
    gate_d = 1 + β_up×ReLU(tanh(γ×z_out_d)) − β_dn×ReLU(tanh(−γ×z_out_d))  [on top-K only]
    gate_d = 1 elsewhere

PARAMS: logit_decay, log_tau, log_beta_up, log_beta_dn, log_gamma
STATE:  _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


K_EACH = 16


class GELU213(nn.Module):
    """Sparse top-K asymmetric gate on OUTPUT z-scores."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.k       = K_EACH
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_up  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_beta_dn  = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_out_mean = None
        self._ema_out_sq   = None
        self._ema_out_dir  = None
        self._ready        = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k = min(self.k, D)

        d_val   = torch.sigmoid(self.logit_decay).detach().item()
        tau     = self.log_tau.exp()
        beta_up = F.softplus(self.log_beta_up)
        beta_dn = F.softplus(self.log_beta_dn)
        gamma   = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                of = out.detach().flatten(0, 1)
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            z_out   = (out.detach() - self._ema_out_mean.view(1, 1, D)) / (var_out.sqrt().view(1, 1, D) + self.eps)

            # Select top-K channels by mean |z_out| across batch×time
            abs_z_mean = z_out.abs().mean(dim=(0, 1))          # (D,)
            topk_idx   = abs_z_mean.topk(k).indices            # (K,)
            mask       = torch.zeros(D, device=x.device, dtype=torch.bool)
            mask[topk_idx] = True
            mask_bt    = mask.view(1, 1, D).expand(B, T, D)   # (B, T, D)

        # ── Asymmetric gate on selected OUTPUT channels ────────────────
        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_out))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_out))
        gate_raw = 1.0 + up_arm - dn_arm                       # (B, T, D) — full

        gate_vec = torch.where(mask_bt, gate_raw.clamp(0.05, 8.0),
                               torch.ones_like(gate_raw))       # pass-through outside top-K

        # ── Cosine output EMA gate (scalar) ────────────────────────────
        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos= torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * gate_vec * gate_cos

        with torch.no_grad():
            of = out.detach().flatten(0, 1)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
