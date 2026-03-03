"""GELU188 – Relative Per-Channel Surprise (|z_d| / EMA(|z_d|)).

THE PROBLEM WITH ABSOLUTE Z-SCORES:
    gelu80 measures: z_d = (x_d - μ_d) / σ_d   — how many standard deviations from mean?
    
    But σ_d is the OVERALL per-channel standard deviation. If a channel is always noisy
    (high σ_d), then even a "surprising" deviation might look small in z-score terms.
    
    More precisely: suppose channel d is always erratic (large fluctuations every step).
    Then σ_d is large → z_d stays small even when x_d deviates a lot.
    
    What we REALLY want: is THIS INPUT more surprising than the model's typical noise level?

THE NEW IDEA: Relative Surprise
    Track a second EMA: EMA of |z_d| itself → ema_absz_d

    Relative surprise at time t:
        rel_d = |z_d(t)| / EMA(|z_d|)_d     — "is this more surprising than usual?"
    
    If rel_d > 1: this token's deviation in channel d is ABOVE the historical norm
    If rel_d < 1: this token's deviation is BELOW the historical norm (familiar)
    
    Only the excess above the historical norm contributes to surprise:
        excess_d = ReLU(rel_d - 1.0)         — (B, T, D) ≥ 0
        surp = tanh(σ × mean_d(excess_d))     — (B, T)

    This is a second-order surprise: "more surprising than I'm used to being surprised by".

WHY THIS COULD WORK:
    In training a language model, some sequence positions are intrinsically hard (e.g.,
    first token after a very unpredictable context). Those positions will have systematically
    high z-scores. EMA(|z|) will be high for those positions, so they won't accumulate
    false surprise. Relative surprise only fires when a token is MORE unexpected than
    the model's current calibration.

PARAMS: logit_decay (primairy EMA), logit_decay2 (absz EMA), log_tau, log_sigma, log_w_raw
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,), _ema_absz (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU188(nn.Module):
    """Relative per-channel surprise: excess above historical |z| baseline."""

    def __init__(self, ema_decay: float = 0.9, ema_decay2: float = 0.95, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay  = nn.Parameter(torch.tensor(math.log(ema_decay  / (1.0 - ema_decay))))
        self.logit_decay2 = nn.Parameter(torch.tensor(math.log(ema_decay2 / (1.0 - ema_decay2))))
        self.log_tau      = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw    = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))

        self._ema_mean:  torch.Tensor = None
        self._ema_sq:    torch.Tensor = None
        self._ema_out:   torch.Tensor = None
        self._ema_absz:  torch.Tensor = None   # (D,) EMA of mean |z_d| over batches
        self._ready = False

    def reset_state(self):
        self._ema_mean  = None
        self._ema_sq    = None
        self._ema_out   = None
        self._ema_absz  = None
        self._ready     = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val  = torch.sigmoid(self.logit_decay).detach().item()
        d_val2 = torch.sigmoid(self.logit_decay2).detach().item()
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
                self._ema_absz = torch.ones(D, device=x.device, dtype=x.dtype)  # init to 1 (neutral)
                self._ready    = True
            return out

        # ── Per-channel z-score ────────────────────────────────────────────
        var   = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
        std   = var.sqrt().view(1, 1, D)
        mu_   = self._ema_mean.view(1, 1, D)
        z     = (x.detach() - mu_) / (std + self.eps)     # (B, T, D)
        abs_z = z.abs()                                    # (B, T, D)

        # ── Relative surprise ─────────────────────────────────────────────
        ema_absz_ = self._ema_absz.view(1, 1, D).clamp(min=0.1)
        rel_z     = abs_z / ema_absz_                      # (B, T, D) ratio
        excess    = F.relu(rel_z - 1.0)                    # above-baseline only
        surp      = torch.tanh(sigma * excess.mean(-1))    # (B, T)

        # ── Cosine familiarity gate ────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)

        gate   = gate_cos * (1.0 + w * surp)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA statistics (no grad) ───────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)
            # Update EMA of per-channel mean |z|
            cur_absz = abs_z.detach().flatten(0, 1).mean(0)   # (D,) mean over batch×time
            self._ema_absz = d_val2 * self._ema_absz + (1-d_val2) * cur_absz

        return output
