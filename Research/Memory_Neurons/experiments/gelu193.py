"""GELU193 – Joint Input-Output Deviation Product Gate (Per-Channel).

THE MISSING SIGNAL IN GELU80:
    gelu80 computes surprise from the INPUT z-score: z_d = (x_d - μ_d) / σ_d
    gelu80 computes familiarity from the OUTPUT cosine: cos(out, ema_out)

    These are applied as SEPARATE multiplicative terms. There is no signal that asks:
    "Is this channel SIMULTANEOUSLY unusual in input AND unusual in output?"

THE NEW IDEA: Joint Deviation Signal
    Track per-channel EMA of GELU output values:
        ema_out_mean_d = EMA of out_d   — (D,) per-channel output mean

    Per-channel output deviation:
        delta_out_d = out_d - ema_out_mean_d   — (B, T, D)

    Joint signal (product of input and output deviations):
        joint_d = z_d × norm(delta_out_d)
        (where norm = delta_out_d / (ema_out_std_d + ε))

    joint_d > 0 when:
        - x_d is above mean AND out_d is above mean  → double novelty signal
        - x_d is below mean AND out_d is below mean  → double suppression signal
    joint_d < 0 when:
        - x_d is above mean but out_d is below mean  → conflicted (GELU is suppressing it)

    Per-channel vector gate:
        gate_d = clamp(1 + β × tanh(γ × joint_d), 0.1, 5.0)

WHY THIS MATTERS:
    GELU's nonlinearity means that a channel may have high z_d (unusual input)
    but if x_d < -1, GELU ≈ 0 so the output deviation is small (GELU already silenced it).
    The joint product naturally down-weights those channels — they're unusual in input
    but GELU already dealt with them.
    We only amplify channels where BOTH the input deviation AND the output deviation agree.

PARAMS: logit_decay, log_tau, log_beta, log_gamma = 4 scalars
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out_mean (D,), _ema_out_sq (D,), _ema_out_dir (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU193(nn.Module):
    """Joint input-output per-channel deviation product gate."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_beta     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_gamma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:     torch.Tensor = None   # (D,) input mean
        self._ema_sq:       torch.Tensor = None   # (D,) input mean-square
        self._ema_out_mean: torch.Tensor = None   # (D,) output mean
        self._ema_out_sq:   torch.Tensor = None   # (D,) output mean-square
        self._ema_out_dir:  torch.Tensor = None   # (D,) output unit vector (for cosine gate)
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

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        beta  = F.softplus(self.log_beta)
        gamma = F.softplus(self.log_gamma)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf  = x.detach().flatten(0, 1)
                of  = out.detach().flatten(0, 1)
                self._ema_mean     = xf.mean(0).clone()
                self._ema_sq       = xf.pow(2).mean(0).clone()
                self._ema_out_mean = of.mean(0).clone()
                self._ema_out_sq   = of.pow(2).mean(0).clone()
                self._ema_out_dir  = F.normalize(of.mean(0), dim=0).clone()
                self._ready        = True
            return out

        with torch.no_grad():
            # Input z-score
            var_in  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std_in  = var_in.sqrt().view(1, 1, D)
            mu_in   = self._ema_mean.view(1, 1, D)
            z_in    = (x.detach() - mu_in) / (std_in + self.eps)        # (B, T, D)

            # Output z-score
            var_out = (self._ema_out_sq - self._ema_out_mean.pow(2)).clamp(min=self.eps_var)
            std_out = var_out.sqrt().view(1, 1, D)
            mu_out  = self._ema_out_mean.view(1, 1, D)
            z_out   = (out.detach() - mu_out) / (std_out + self.eps)    # (B, T, D)

            # Joint signal: product of signed z-scores
            joint   = z_in * z_out                                       # (B, T, D)

        # Per-channel vector gate based on joint signal
        gate_vec = (1.0 + beta * torch.tanh(gamma * joint)).clamp(0.1, 5.0)  # (B, T, D)

        with torch.no_grad():
            out_n   = F.normalize(out.detach(), dim=-1)
            ema_n   = F.normalize(self._ema_out_dir, dim=0).view(1, 1, D)
            cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
            gate_cos = torch.exp(-tau * cos_sim).unsqueeze(-1)          # (B, T, 1)

        output = out * gate_vec * gate_cos

        # ── Update EMA statistics ─────────────────────────────────────────
        with torch.no_grad():
            xfl = x.detach().flatten(0, 1)
            ofl = out.detach().flatten(0, 1)
            self._ema_mean     = d_val * self._ema_mean     + (1-d_val) * xfl.mean(0)
            self._ema_sq       = d_val * self._ema_sq       + (1-d_val) * xfl.pow(2).mean(0)
            self._ema_out_mean = d_val * self._ema_out_mean + (1-d_val) * ofl.mean(0)
            self._ema_out_sq   = d_val * self._ema_out_sq   + (1-d_val) * ofl.pow(2).mean(0)
            self._ema_out_dir  = d_val * self._ema_out_dir  + (1-d_val) * F.normalize(ofl.mean(0), dim=0)

        return output
