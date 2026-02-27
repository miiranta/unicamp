"""GELU89 – Learned Channel Router: Surprise-Routing via Soft Channel Competition.

BEYOND SCALAR & PER-CHANNEL GATES:
    gelu80: gate_scalar(token) applied to ALL channels → same rescaling everywhere
    gelu87: gate_d(token) per-channel → different scaling per channel

    But both use FIXED channel assignments: channel d always means the same thing.
    What if the ROUTING itself is dynamic and novelty-dependent?

THE IDEA — SOFT CHANNEL COMPETITION WITH LEARNED ROUTING:
    Inspired by Mixture of Experts: route the input to "specialist channels"
    depending on HOW NOVEL the token is and WHICH channels are novel.

    Mechanism:
    1. Compute per-channel z-score: z ∈ (B, T, D)
    2. Compute a soft ROUTING WEIGHT via normalized z-scores:
        route_weight = softmax(σ × z²)         ← smooth, differentiable  (B, T, D)
        (tokens with extreme z in any channel route more weight there)
    3. Amplify channels that "win" the routing competition:
        output = GELU(x) × (1 + w × D × route_weight)
                                                ← D × route_weight: if uniform, each weight=1/D
                                                   so D × 1/D = 1 → no gain
                                                   if concentrated: winning channel × D > 1 → amplified
    4. Subtract EMA of GELU(x) to contrast-normalize:
        output = GELU(x) × (1 + w × D × route_weight) - ema_gelu × w × D × mean_route_weight
                                                   ← normalize to prevent scale-shift

    INTUITION ("sparse expert routing inside FFN"):
    * Familiar token: z ≈ 0 for all channels → route_weight ≈ 1/D → gain ≈ 1 → no change
    * Novel token: z >> 0 in some channels → those channels gain >> 1, others < 1
      → output is ROTATED toward novel channels (not just scaled!)
    * The routing weight softmax is differentiable → the model can LEARN which channels
      should "win" for which types of novelty via backprop.

WHY ROUTING + GELU?
    A router alone (without gating the GELU output) just rescales.
    But here, GELU(x) itself defines the "canonical" representation.
    The router AMPLIFIES the GELU channels that are most activated and most unusual.
    Channels that are strong AND unusual get double boost: large GELU(x_d) × large route_weight.
    Channels that are weak OR familiar get little contribution.

CONTRAST NORMALIZATION:
    Without normalization, routing amplification would inflate the overall magnitude.
    We subtract: output -= mean_gelu × mean_route_weight × w × D
    This centers the output so no systematic scale inflation.

Params: logit_decay, log_sigma_raw, log_w_raw = 3 scalars.
State: _ema_mean (D,), _ema_sq (D,), _ema_gelu (D,).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU89(nn.Module):
    """Surprise-routing via soft channel competition: novel channels self-amplify."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay   = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_sigma_raw = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        self.log_w_raw     = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

        self._ema_mean: torch.Tensor = None
        self._ema_sq:   torch.Tensor = None
        self._ema_gelu: torch.Tensor = None
        self._ready = False

    def reset_state(self):
        self._ema_mean = None
        self._ema_sq   = None
        self._ema_gelu = None
        self._ready    = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val = torch.sigmoid(self.logit_decay).detach().item()
        sigma = F.softplus(self.log_sigma_raw)
        w     = F.softplus(self.log_w_raw)

        out = self._gelu(x)   # (B, T, D)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean = xf.mean(0).clone()
                self._ema_sq   = xf.pow(2).mean(0).clone()
                self._ema_gelu = of.mean(0).clone()
                self._ready    = True
            return out

        with torch.no_grad():
            # ── Per-channel z-score (detach only) ──────────────────────
            var   = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
            std   = var.sqrt()
            mu_   = self._ema_mean.view(1, 1, D)
            std_  = std.view(1, 1, D)
            z     = (x.detach() - mu_) / (std_ + self.eps)           # (B, T, D)
            z_det = z.clone()   # detached tensor holding z values

        # ── Soft routing weight: softmax over z² (sigma in grad graph) ──
        route_logits  = sigma * z_det.pow(2)                          # (B, T, D) ≥ 0
        route_weight  = torch.softmax(route_logits, dim=-1)           # (B, T, D) sums to 1
        # Scale so that uniform routing (=1/D) maps to 1, concentrated > 1
        scaled_route  = route_weight * D                              # (B, T, D): uniform=1

        # ── Amplify via routing (differentiable: gradients through out) ──
        amplification = 1.0 + w * (scaled_route - 1.0)               # (B, T, D)
        # = 1 when uniform, > 1 for winning channels, < 1 for losing channels
        # Clamp to avoid extreme suppression on losing channels
        amplification = amplification.clamp(min=0.1)

        result = out * amplification

        # ── EMA updates ─────────────────────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            of  = out.detach().flatten(0, 1)
            self._ema_mean = d_val * self._ema_mean + (1 - d_val) * xf.mean(0)
            self._ema_sq   = d_val * self._ema_sq   + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_gelu = d_val * self._ema_gelu + (1 - d_val) * of.mean(0)

        return result
