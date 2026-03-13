"""GELU342 – Pre + Post GELU Gate (Dual Gate Position, 4 params).

CREATIVE IDEA: All prior experiments gate AFTER GELU(x). What if we additionally
gate the INPUT to GELU — changing WHICH REGION of the nonlinearity is traversed?

GELU is highly non-uniform: activations near x=0 are in the most nonlinear region
(where gradient ≈ 0.5 and the soft-zero behavior is dominant), while large positive
x gets near-linear treatment, and large negative x is nearly zeroed.

By applying a PRE-GATE to x, we shift the effective input distribution:
    x_gated = x × 2σ(β_pre × z_in)

If a channel is UNUSUALLY ACTIVE (z_in > 0, pre_gate > 1):
    x_gated > x → channel is pushed further into the GELU linear regime
    GELU(x_gated) ≈ x_gated for large x → cleaner, less-saturated signal

If a channel is BELOW AVERAGE (z_in < 0, pre_gate < 1):
    x_gated < x → channel is pushed toward the GELU zero-suppression zone
    GELU(x_gated) ≈ 0 for very negative x → true gating effect

This creates a SOFT ROUTING mechanism: unusual channels are pushed into the
linear regime (passed through cleanly), while familiar channels are damped into
the nonlinear suppression zone.

Then a POST-GATE (based on the ORIGINAL GELU(x) z-score for clean EMA) provides
output-level control:
    out_gated = GELU(x × post_gate_pre) × 2σ(β_post × z_out_original) × gate_cos

The z-scores for both gates use the ORIGINAL x and ORIGINAL GELU(x), so EMA
statistics remain clean and comparable across training.

GATE:
    pre_gate = 2σ(β_pre * z_in)          — applied to x before GELU
    out      = GELU(x × pre_gate)        — modified nonlinear response
    post_gate= 2σ(β_post * z_out)        — z_out from EMA of ORIGINAL GELU(x)
    gate_cos = exp(−τ * cos(GELU(x), ema_dir))   — cosine based on ORIGINAL GELU(x)
    output   = out × post_gate × gate_cos

PARAMS: logit_decay, log_tau, log_beta_pre, log_beta_post  (4 scalars)
STATE:  _ema_mean (D), _ema_sq (D), _ema_out_mean (D), _ema_out_sq (D), _ema_out_dir (D)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU342(nn.Module):
    """Pre+post GELU gate: gates input before AND output after GELU for soft routing."""

    def __init__(self, D: int = None, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.eps_var  = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(9.0)))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta_pre = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # smaller: subtle pre-gate
        self.log_beta_post= nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = self._ema_sq = None
        self._ema_out_mean = self._ema_out_sq = self._ema_out_dir = None
        self._ready = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _z(self, val, mean, sq):
        var = (sq - mean.pow(2)).clamp(min=self.eps_var)
        return (val - mean) / (var.sqrt() + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        tau       = self.log_tau.exp()
        beta_pre  = F.softplus(self.log_beta_pre)
        beta_post = F.softplus(self.log_beta_post)

        # ORIGINAL outputs for EMA tracking — not used in gated path
        out_orig = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1); of = out_orig.detach().flatten(0, 1)
                bm_x = xf.mean(0); bm_o = of.mean(0)
                self._ema_mean = bm_x.clone(); self._ema_sq = xf.pow(2).mean(0).clone()
                self._ema_out_mean = bm_o.clone(); self._ema_out_sq = of.pow(2).mean(0).clone()
                self._ema_out_dir = F.normalize(bm_o, dim=0).clone()
                self._ready = True
            return out_orig

        with torch.no_grad():
            # z-scores from ORIGINAL x and ORIGINAL GELU(x) for clean EMA comparison
            z_in  = self._z(x.detach(),        self._ema_mean.view(1,1,D),     self._ema_sq.view(1,1,D))
            z_out = self._z(out_orig.detach(), self._ema_out_mean.view(1,1,D), self._ema_out_sq.view(1,1,D))

        # Pre-gate: apply to x BEFORE GELU — shifts activation into/out of nonlinear zone
        pre_gate = 2.0 * torch.sigmoid(beta_pre * z_in)   # ∈ (0, 2)
        out      = self._gelu(x * pre_gate)                # ← gated GELU input

        # Post-gate: applied to the gated GELU output
        post_gate = 2.0 * torch.sigmoid(beta_post * z_out) # z_out from ORIGINAL GELU(x)

        with torch.no_grad():
            out_n    = F.normalize(out_orig.detach(), dim=-1)  # cosine still from original
            ema_n    = F.normalize(self._ema_out_dir, dim=0).view(1,1,D)
            cos_sim  = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)

        output = out * post_gate * gate_cos

        with torch.no_grad():
            xf = x.detach().flatten(0, 1); of = out_orig.detach().flatten(0, 1)
            bm_x = xf.mean(0); bsq_x = xf.pow(2).mean(0)
            bm_o = of.mean(0); bsq_o = of.pow(2).mean(0)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * bm_x
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * bsq_x
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * bm_o
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * bsq_o
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(bm_o, dim=0)

        return output
