"""GELU248 – Per-Token Hit Count Gate (Token-Level Granularity).

FINER GRANULARITY THAN BATCH-LEVEL HIT COUNTING:

    All previous experiments (gelu239-247) track familiarity at BATCH LEVEL:
    one depletion/hit counter per buffer slot = one counter per ~(batch_size × seq_len)
    = one counter per ~2048 tokens.

    gelu248 tracks familiarity at INDIVIDUAL TOKEN LEVEL within the batch.
    Instead of asking "has this BATCH been seen before?", it asks:
    "has THIS SPECIFIC TOKEN been seen in a FAMILIAR context before?"

    Per-token approach:
        For each position (b, t):
            tok_sim(b,t) = cos(y[b,t], nearest_slot_mean)   ∈ [-1, 1]
        
        High tok_sim → this token's activation is very similar to what
                        the model saw at this batch position last time.
        Low  tok_sim → this token's activation is dissimilar even if the
                        batch-level context matches.

    Combined gate:
        depl_gate   = exp(-k_d * (1 - depl_nearest))       BATCH-LEVEL depletion
        tok_gate(b,t) = exp(-k_t * tok_sim(b,t).clamp(0,1)) TOKEN-LEVEL cosine
        
        Final: gate(b,t) = depl_gate × tok_gate(b,t)
        
    PASS-1:
        depl=1.0 → depl_gate = 1.0
        tok_sim(b,t) for cross-batch comparison ≈ 0.2-0.5
        tok_gate ≈ exp(-k_t * 0.35) ≈ moderate suppression for tok_sim

        Wait - this would HURT pass-1 PPL! The tok_gate is NOT zero for pass-1.

    REVISION: Use depletion gate ALONE as the scalar, and use tok_sim only to
    MODULATE the depletion effect within a batch:

        gate(b,t) = exp(-k * (1-depl) * tok_sim(b,t).clamp(0,1))

        Pass-1 (depl=1.0): gate(b,t) = exp(0) = 1.0   ← zero change ✓
        Pass-2 (depl=0.5):
            familiar token (tok_sim=0.9): gate = exp(-k*0.5*0.9) ← suppressed
            novel    token (tok_sim=0.1): gate = exp(-k*0.5*0.1) ≈ 1.0  ← not suppressed

    This is exactly the same as gelu240 but here we also COUNT hits to make
    the effect COMPOUND across passes (like gelu241's integer counter):

        gate(b,t) = exp(-k * h_nearest * tok_sim(b,t).clamp(0,1))

        h=0 (pass 1): gate = 1.0                            ← zero change ✓
        h=1 (pass 2): gate = exp(-k * tok_sim)              ← selective per-token
        h=2 (pass 3): gate = exp(-2k * tok_sim)             ← stronger

    ADVANTAGE: The hit-count compounding is STRONGER than depletion-rate:
        Depletion at h=2: depl=0.25 → 1-depl=0.75
        Hit count at h=2: h=2 → even larger penalty

PARAMS: log_k (1 scalar, controls per-token suppression strength)
STATE:  _buf, _hits, _mask, _ptr
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85


class GELU248(nn.Module):
    """Per-token hit-count gate: exp(-k * h * tok_sim); zero change on pass 1."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k = nn.Parameter(torch.tensor(math.log(1.0)))  # k=1.0

        self._buf:  torch.Tensor = None
        self._hits: torch.Tensor = None
        self._mask: torch.Tensor = None
        self._ptr   = 0
        self._ready = False

    def reset_state(self):
        self._buf   = None
        self._hits  = None
        self._mask  = None
        self._ptr   = 0
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_gate = self.log_k.exp().clamp(0.1, 5.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hits = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
                self._buf[0]  = F.normalize(m_curr, dim=0)
                self._hits[0] = 0
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            h           = self._hits[nearest_idx].item()
            nearest_vec = self._buf[nearest_idx].clone()

        # Per-token similarity
        y_flat  = y.flatten(0, 1)
        y_n     = F.normalize(y_flat, dim=-1)
        nv_n    = F.normalize(nearest_vec.view(1, D), dim=-1)
        tok_sim = (y_n * nv_n).sum(-1).clamp(0.0, 1.0).view(B, T)   # (B, T)

        # gate(b,t) = exp(-k * h * tok_sim(b,t))
        # h=0 (pass 1): gate = 1.0 for ALL tokens (zero PPL change)
        gate = torch.exp(-k_gate * h * tok_sim)   # (B, T)
        output = y * gate.unsqueeze(-1)

        with torch.no_grad():
            if max_sim > FIRE_THRESH:
                self._hits[nearest_idx] += 1
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._hits[self._ptr] = 0
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
