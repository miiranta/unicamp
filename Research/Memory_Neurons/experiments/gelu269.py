"""GELU269 – Modern Hopfield Retrieval (Soft Associative Memory Playback).

═══════════════════════════════════════════════════════════════════════════
INSPIRATION: Modern Hopfield Networks (Ramsauer et al., 2020 "Hopfield
Networks Is All You Need") — energy-based content-addressable memory that
uses an attention-like update to retrieve stored patterns.
═══════════════════════════════════════════════════════════════════════════

CORE IDEA  [f(x) = m(s(x,c), g(x))]:
    g(x) = GELU(x)            — standard activation
    s(x,c) = softmax(β·K·q)   — SOFT attention over all stored keys K
               (not just nearest-neighbor; a continuous, differentiable
               form of "how familiar is the current batch?")
    m      = additive injection:
               output = g(x) + t * (retrieved_pattern - global_mean)

    In pass 1: t=0 (no injection). Buffer fills with batch means.
    In pass 2+: t = k * log(hit_count + 1)  compounding log-scale injection.

MECHANISM:
    Keys    K = (N, D): normalized batch means from pass 1 (ring buffer).
    Values  V = (N, D): raw (unnormalized) batch means from pass 1.
    Query   q = (D,): normalized current batch mean.

    attn_logits = β * (K @ q)                    [shape: (N,)]
    attn_weights = softmax(attn_logits)           [shape: (N,)]
    retrieved   = attn_weights ⊤ @ V             [shape: (D,)]

    In pass 2, the nearest stored slot gets almost all the weight (β controls
    sharpness). The retrieved vector blends nearby patterns softly.

WHY SOFT RETRIEVAL IS BETTER THAN HARD NEAREST-NEIGHBOR:
    Hard NN: only 1 slot contributes → brittle, ignores partial matches.
    Hopfield attention: ALL slots contribute weighted by similarity →
    smoother, more robust retrieval; graceful degradation if exact match
    is not in buffer.
    
    Also: the energy function E = -softmax(β·K·q)⊤·V is minimized by
    the retrieved vector — a principled mathematical guarantee of convergence
    to the nearest stored attractor.

WHY THIS SHOULD IMPROVE PPL ACROSS PASSES:
    Pass 1: t=0 → output = GELU(x). Buffer builds.
    Pass 2: t > 0 → output += t * (retrieved - global_mean).
            retrieved ≈ pass-1 activation for this batch.
            Injecting the stored activations reduces model uncertainty.
    Pass 3: t × RATE → stronger injection, lower PPL still.

    The retrieved-global_mean subtraction injects only the SPECIFIC
    content of this episode (the residual from average), not the baseline.

MONOTONICITY:
    t = k_inject * log(hit_count + 1), hit_count incremented PRE-FIRE.
    hit_count: 0 (pass1) → 1 (pass2) → 2 (pass3)
    t: 0, k·log(2)≈0.69k, k·log(3)≈1.10k → Δ1→3 > Δ1→2 ✓

PARAMS: log_beta (Hopfield sharpness, init β=10),
        log_k_inject (injection scale, init k=0.1)
STATE:  _buf_keys (N,D) normalized,  _buf_vals (N,D) raw,
        _hit_count (N,) int,  _mask (N,) bool,  _ptr int,
        _pass1_complete bool,  _global_sum (D,),  _n_global int
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
N_BUF       = 512


class GELU269(nn.Module):
    """Modern Hopfield retrieval + additive injection (pre-fire log-scale)."""

    def __init__(self, buffer_size: int = N_BUF):
        super().__init__()
        self._N = buffer_size

        self.log_beta     = nn.Parameter(torch.tensor(math.log(10.0)))   # Hopfield sharpness
        self.log_k_inject = nn.Parameter(torch.tensor(math.log(0.1)))    # injection scale

        self._buf_keys:  torch.Tensor = None  # (N, D) normalized
        self._buf_vals:  torch.Tensor = None  # (N, D) raw
        self._hit_count: torch.Tensor = None  # (N,) int counts
        self._mask:      torch.Tensor = None  # (N,) bool
        self._global_sum: torch.Tensor = None  # (D,) sum of all pass-1 means
        self._n_global   = 0
        self._ptr        = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf_keys   = None
        self._buf_vals   = None
        self._hit_count  = None
        self._mask       = None
        self._global_sum = None
        self._n_global   = 0
        self._ptr        = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        y       = self._gelu(x)
        m_curr  = y.detach().flatten(0, 1).mean(0)   # (D,)
        m_raw   = m_curr.clone()

        # ── Init ────────────────────────────────────────────────────────
        if self._buf_keys is None:
            with torch.no_grad():
                self._buf_keys  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_vals  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hit_count = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask      = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
                self._global_sum = torch.zeros(D,         device=x.device, dtype=y.dtype)
            self._ptr = 0

        # ── Pass-1 building ─────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True   # freeze buffer
                    else:
                        self._buf_keys[self._ptr]  = F.normalize(m_curr, dim=0)
                        self._buf_vals[self._ptr]  = m_raw
                        self._mask[self._ptr]      = True
                        self._global_sum          += m_raw
                        self._n_global            += 1
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf_keys[0]   = F.normalize(m_curr, dim=0)
                    self._buf_vals[0]   = m_raw
                    self._mask[0]       = True
                    self._global_sum   += m_raw
                    self._n_global      = 1
                    self._ptr           = 1
                    return y

        # ── Pass-2+ Hopfield retrieval ──────────────────────────────────
        beta     = self.log_beta.exp().clamp(1.0, 50.0)
        k_inject = self.log_k_inject.exp().clamp(0.001, 2.0)

        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)           # (1, D)
            logits      = beta * (self._buf_keys * q).sum(-1)                 # (N,)
            logits      = logits.masked_fill(~self._mask, -1e9)
            attn        = torch.softmax(logits, dim=0)                        # (N,)
            retrieved   = (attn.unsqueeze(-1) * self._buf_vals).sum(0)       # (D,)

            # Find nearest for PRE-FIRE hit-count update
            nearest_idx = logits.argmax()
            sims_plain  = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
            if sims_plain[nearest_idx].item() > FIRE_THRESH:
                self._hit_count[nearest_idx] += 1

            hit = self._hit_count[nearest_idx].item()
            t   = k_inject * math.log(float(hit) + 1.0)   # 0 at hit=0, k*log2 at hit=1

            global_mean = self._global_sum / max(self._n_global, 1)
            delta       = retrieved - global_mean   # (D,) specific episode content

        # Inject episode delta broadcast over all tokens
        output = y + t * delta.view(1, 1, D)
        return output
