"""GELU88 – Relative Surprise Gate (Surprise Normalized by Running Mean Surprise).

THE PROBLEM WITH ABSOLUTE SURPRISE:
    In gelu80, the gate is:
        surp = tanh(σ × mean_d(|z_d|))
        gate = exp(-τ × cos_out) × (1 + w × surp)

    This uses ABSOLUTE z-scores: how far is this token from the global mean?
    But in language, the MODEL ALWAYS sees surprising inputs early in training
    and less surprising inputs later (as it learns). The EMA adapts, but the
    absolute z-score level changes throughout training.

    More fundamentally: different semantic domains have different "typical surprise"
    levels. A list of rare vocabulary words seems "surprising" by absolute z-score
    even if those words consistently repeat in the same context (low local novelty).

THE FIX — RELATIVE SURPRISE:
    Normalize the surprise by the running mean surprise:
        z_score(token)     = mean_d(|z_d|)                      ← absolute
        EMA_zscore         = running mean of z_score values      ← typical level
        relative_surp      = z_score / (EMA_zscore + eps)        ← RELATIVE: is this MORE surprising than usual?

    If relative_surp > 1: this token is MORE surprising than usual → amplify
    If relative_surp < 1: this token is LESS surprising than usual → suppress
    If relative_surp = 1: neutral → no change

    Gate:
        surp_centered = tanh(σ × (relative_surp - 1))           ← centered at 0, bounded
        gate = exp(-τ × cos_out) × (1 + w × surp_centered)

    WHY THIS MATTERS:
    Suppose the model is processing a technical science paper.
    ALL tokens have high z-score (unfamiliar scientific vocabulary).
    Absolute surprise: ALL tokens get amplified equally → no discrimination.
    Relative surprise: WITHIN the paper, the MOST unusual tokens get amplified,
    common ones (even if globally unusual) get less boost.
    
    This is like a LOCAL contrast enhancement: amplify relative deviations,
    not just absolute deviations.

BIOLOGICAL ANALOGY:
    Visual cortex contrast gain control: neurons adapt to the mean luminance
    of the local surround. A bright spot in a dark scene drives a strong response,
    but the same absolute luminance in a bright scene drives a weaker response.
    "Bright" is RELATIVE to context.

STABILITY:
    relative_surp = z_score / EMA_zscore is initialized = 1 (EMA starts at first z_score).
    tanh(σ × (1-1)) = 0 initially → no gate effect at start.
    EMA_zscore is tracked with a slower decay (d_rel = 0.99) for stability.

Params: logit_decay, log_tau, log_sigma_raw, log_w_raw, logit_decay_rel = 5 scalars.
State: _ema_mean (D,), _ema_sq (D,), _ema_out (D,), _ema_zscore (scalar).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU88(nn.Module):
    """Relative surprise gate: z-score normalized by running mean z-score."""

    def __init__(self, ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.eps     = eps
        self.eps_var = 1e-4
        self.logit_decay     = nn.Parameter(torch.tensor(math.log(ema_decay / (1.0 - ema_decay))))
        self.log_tau         = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_sigma_raw   = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.log_w_raw       = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        # Slower decay for the running z-score baseline (more stable normalization)
        self.logit_decay_rel = nn.Parameter(torch.tensor(math.log(0.99 / 0.01)))  # init ≈ 0.99

        self._ema_mean:   torch.Tensor = None
        self._ema_sq:     torch.Tensor = None
        self._ema_out:    torch.Tensor = None
        self._ema_zscore: torch.Tensor = None   # scalar: running mean of mean_d(|z_d|)
        self._ready = False

    def reset_state(self):
        self._ema_mean   = None
        self._ema_sq     = None
        self._ema_out    = None
        self._ema_zscore = None
        self._ready      = False

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        d_val     = torch.sigmoid(self.logit_decay).detach().item()
        d_rel     = torch.sigmoid(self.logit_decay_rel).detach().item()
        tau       = self.log_tau.exp()
        sigma     = F.softplus(self.log_sigma_raw)
        w         = F.softplus(self.log_w_raw)

        out = self._gelu(x)

        if not self._ready:
            with torch.no_grad():
                xf = x.detach().flatten(0, 1)
                of = out.detach().flatten(0, 1)
                self._ema_mean   = xf.mean(0).clone()
                self._ema_sq     = xf.pow(2).mean(0).clone()
                self._ema_out    = F.normalize(of.mean(0), dim=0).clone()
                self._ema_zscore = torch.tensor(1.0, device=x.device)  # neutral init
                self._ready      = True
            return out

        # ── Per-channel z-score ───────────────────────────────────────────
        var       = (self._ema_sq - self._ema_mean.pow(2)).clamp(min=self.eps_var)
        std       = var.sqrt()
        mu_       = self._ema_mean.view(1, 1, D)
        std_      = std.view(1, 1, D)
        z         = (x.detach() - mu_) / (std_ + self.eps)             # (B, T, D)
        abs_z_mean = z.abs().mean(dim=-1)                               # (B, T)

        # ── Relative surprise ─────────────────────────────────────────────
        # Convert to relative: how much above/below typical?
        rel_surp   = abs_z_mean / (self._ema_zscore + self.eps)         # (B, T) ~1 on average
        surp       = torch.tanh(sigma * (rel_surp - 1.0))               # (B, T) centered at 0

        # ── Output cosine gate ────────────────────────────────────────────
        out_norm  = F.normalize(out.detach(), dim=-1)
        ema_norm  = self._ema_out.view(1, 1, D)
        cos_out   = (out_norm * ema_norm).sum(dim=-1)                   # (B, T)

        gate   = torch.exp(-tau * cos_out) * (1.0 + w * surp)          # (B, T)
        result = out * gate.unsqueeze(-1)

        # ── EMA updates ───────────────────────────────────────────────────
        with torch.no_grad():
            xf  = x.detach().flatten(0, 1)
            of  = out.detach().flatten(0, 1)
            batch_z_mean = abs_z_mean.detach().mean()                   # scalar batch mean
            self._ema_mean   = d_val * self._ema_mean   + (1 - d_val) * xf.mean(0)
            self._ema_sq     = d_val * self._ema_sq     + (1 - d_val) * xf.pow(2).mean(0)
            self._ema_out    = F.normalize(d_val * self._ema_out + (1 - d_val) * of.mean(0), dim=0)
            self._ema_zscore = d_rel * self._ema_zscore + (1 - d_rel) * batch_z_mean

        return result
