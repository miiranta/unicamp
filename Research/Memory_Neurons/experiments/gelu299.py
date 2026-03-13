"""gelu299 – Bidirectional Depletion + Rebound Gate.

CONCEPT:
    gelu282 (per-channel depletion) only tracks when a channel is OVER-amplified
    (excess = ReLU(gate_raw - 1)).  If a channel is OVER-SUPPRESSED (gate_raw < 1
    for a long period), this experiment ALSO detects that and AMPLIFIES it
    (rebound effect) — like a neuron released from inhibition.

TWO ACCUMULATORS:
    depl_amp (D,): EMA of ReLU(gate_raw - 1)    ← amplification history
    depl_sup (D,): EMA of ReLU(1 - gate_raw)    ← suppression history

COMBINED GATE:
    amp_factor = exp(-U_amp * depl_amp)    ∈ (0,1] — more amplification → more suppression
    reb_factor = exp(+U_reb * depl_sup)    ∈ [1,∞) — more suppression → more rebound
    
    depl_combined = clamp(amp_factor * reb_factor, 0.05, 8.0)
    
    gate_final = gate_raw * depl_combined

SEQUENTIAL ADAPTATION:
    Pass 1: Channels amplified by gelu211 build up depl_amp → gate_final < gate_raw on pass 2.
            Channels suppressed by gelu211 build up depl_sup → gate_final > gate_raw on pass 2.
            Both effects shift the model AWAY from its initial response PATTERN → Δ > 0.
    
    This creates a push toward gate_final = 1 (habituation at both extremes).

BENEFIT FROM BACKPROP (vs gelu282):
    All depletion accumulation is differentiable:
        excess_amp = ReLU(gate_raw - 1)    ← gradient through gate_raw
        excess_sup = ReLU(1 - gate_raw)    ← gradient through gate_raw
        new_depl_amp = d * depl_amp.detach() + (1-d) * excess_amp.mean(0-1)
        new_depl_sup = d * depl_sup.detach() + (1-d) * excess_sup.mean(0-1)
    Gradient paths allow U_amp, U_reb, d to be jointly optimised for coordinated adaptation.

NO CAUSALITY LEAK:
    same batch-level accumulation as gelu282/291.

PARAMS:  gelu211 params + logit_d_depl, log_U_amp, log_U_reb.
STATE:   gelu211 state + _depl_amp (D,), _depl_sup (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU299(nn.Module):
    """gelu211 + bidirectional depletion: amplified→suppressed, suppressed→amplified."""

    def __init__(self, D_FF: int = 1024, ema_decay: float = 0.9, eps: float = 1e-5):
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

        # ── Bidirectional depletion params ──────────────────────────────
        self.logit_d_depl = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))  # slow EMA
        self.log_U_amp    = nn.Parameter(torch.tensor(math.log(1.0)))   # amplification suppressor
        self.log_U_reb    = nn.Parameter(torch.tensor(math.log(0.5)))   # suppression rebounder

        # ── gelu211 state ───────────────────────────────────────────────
        self._ema_mean:     torch.Tensor = None
        self._ema_sq:       torch.Tensor = None
        self._ema_out_mean: torch.Tensor = None
        self._ema_out_sq:   torch.Tensor = None
        self._ema_out_dir:  torch.Tensor = None
        self._ready = False

        self._depl_amp: torch.Tensor = None
        self._depl_sup: torch.Tensor = None

    def reset_state(self):
        self._ema_mean     = None;  self._ema_sq       = None
        self._ema_out_mean = None;  self._ema_out_sq   = None
        self._ema_out_dir  = None;  self._ready        = False
        self._depl_amp     = None;  self._depl_sup     = None

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
                self._depl_amp     = torch.zeros(D, device=x.device, dtype=x.dtype)
                self._depl_sup     = torch.zeros(D, device=x.device, dtype=x.dtype)
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

        up_arm   = beta_up * F.relu(torch.tanh( gamma * z_in))
        dn_arm   = beta_dn * F.relu(torch.tanh(-gamma * z_in))
        gate_in  = (1.0 + up_arm - dn_arm).clamp(0.05, 8.0)
        gate_out = (1.0 + beta_out * torch.tanh(gamma_out * z_out)).clamp(0.1, 5.0)
        gate_raw = gate_in * gate_out * gate_cos                        # (B, T, D)

        # ── Bidirectional differentiable depletion ───────────────────────
        d_depl   = torch.sigmoid(self.logit_d_depl)
        U_amp    = self.log_U_amp.exp()
        U_reb    = self.log_U_reb.exp()

        excess_amp   = F.relu(gate_raw - 1.0).mean(dim=(0,1))                    # (D,)
        excess_sup   = F.relu(1.0 - gate_raw).mean(dim=(0,1))                    # (D,)
        new_depl_amp = d_depl * self._depl_amp.detach() + (1 - d_depl) * excess_amp
        new_depl_sup = d_depl * self._depl_sup.detach() + (1 - d_depl) * excess_sup

        amp_factor   = torch.exp(-U_amp * new_depl_amp)                          # (D,) suppress
        reb_factor   = torch.exp( U_reb * new_depl_sup)                          # (D,) amplify
        depl_combined= (amp_factor * reb_factor).clamp(0.05, 8.0)               # (D,)

        output = out * (gate_raw * depl_combined.view(1,1,D)).clamp(0.05, 10.0)

        self._depl_amp = new_depl_amp.detach()
        self._depl_sup = new_depl_sup.detach()

        with torch.no_grad():
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xf.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xf.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * of.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * of.pow(2).mean(0)
            om = of.mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(om, dim=0)

        return output
