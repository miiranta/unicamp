"""GELU187 – Learned Per-Channel Importance Weights (α_d).

THE LIMITATION OF GELU80'S AGGREGATION:
    gelu80: surp = tanh(σ × mean_d |z_d|)   — uniform average over all D channels
    All 1024 channels contribute equally to the surprise signal.

    But in a transformer FF layer, channels are NOT equal:
    - Some channels may be specialized for syntactic patterns (highly predictive → low z on familiar input)
    - Some channels may be "catch-all" channels with high variance (contribute noise to mean|z|)
    - Some channels may carry the most semantically relevant deviation signals

    If we DOWN-WEIGHT noisy channels and UP-WEIGHT informative channels,
    the weighted mean|z| becomes a better surprise detector.

THE NEW IDEA: Gradient-Trained Channel Weights
    alpha_raw: nn.Parameter of shape (D,) — learnable, initialized to log(1) = 0

    weights_d = softmax-normalized softplus(alpha_raw):
        weights_d = softplus(alpha_raw) / sum_d(softplus(alpha_raw))   — (D,)
        ≡ a learned probability distribution over channels

    Weighted surprise:
        mean_w_z = sum_d (|z_d| × weights_d)     — scalar per (b,t)
        surp     = tanh(σ × mean_w_z)             — (B, T)

    At initialization: alpha_raw = 0 → softplus(0) = log(2) ≈ 0.693 for all d
        weights_d = 1/D for all d  →  mean_w_z = uniform mean = exactly gelu80's surp
    So gelu187 begins IDENTICAL to gelu80 and then learns channel importance.

GRADIENT PATH:
    output = GELU(x) × gate × (1 + w × surp)
    d(loss)/d(alpha_raw_d) propagates through surp → mean_w_z → weights_d → alpha_raw_d
    Channels that, when up-weighted, improve loss get positive gradient → higher alpha

PARAMS: logit_decay, log_tau, log_sigma_raw, log_w_raw (4 scalars) + alpha_raw (D,)
STATE:  _ema_mean (D,), _ema_sq (D,), _ema_out (D,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU187(nn.Module):
    """Learned per-channel importance weights: gradient-trained channel weighting for z-score aggregation."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau       = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.3) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # alpha_raw lazily initialized on first forward when D is known
        self.alpha_raw: nn.Parameter = None

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_out:  torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_out  = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Lazy init of alpha_raw
        if self.alpha_raw is None:
            self.alpha_raw = nn.Parameter(torch.zeros(D, device=x.device, dtype=x.dtype))

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        tau   = self.log_tau.exp()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_out  = F.normalize(out.detach().flatten(0,1).mean(0), dim=0).clone()
                self._ready    = True
            return out

        # ── Per-channel z-score ────────────────────────────────────────────
        var  = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
        std  = var.sqrt()
        mu_  = self._ema_mean.view(1, 1, D)
        std_ = std.view(1, 1, D)
        z    = (x.detach() - mu_) / (std_ + self.eps)   # (B, T, D) — detach from EMA
        abs_z = z.abs()                                  # (B, T, D)

        # ── Learned channel weights ────────────────────────────────────────
        sp    = F.softplus(self.alpha_raw)               # (D,) positive
        wts   = sp / (sp.sum() + self.eps)               # (D,) normalized weights
        mean_w_z   = (abs_z * wts.view(1, 1, D)).sum(-1) # (B, T) weighted mean |z|
        surprise   = torch.tanh(sigma * mean_w_z)        # (B, T) ∈ (0, 1)

        # ── Cosine familiarity gate ────────────────────────────────────────
        out_n   = F.normalize(out.detach(), dim=-1)
        ema_n   = F.normalize(self._ema_out, dim=0).view(1, 1, D)
        cos_sim = (out_n * ema_n).sum(-1).clamp(-1, 1)
        gate_cos = torch.exp(-tau * cos_sim)              # (B, T)

        gate   = gate_cos * (1.0 + w * surprise)
        output = out * gate.unsqueeze(-1)

        # ── Update EMA (no grad) ───────────────────────────────────────────
        with torch.no_grad():
            xf = x.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1-d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1-d_val) * xf.pow(2).mean(0)
            om = out.detach().flatten(0, 1).mean(0)
            self._ema_out  = d_val * self._ema_out  + (1-d_val) * F.normalize(om, dim=0)

        return output
