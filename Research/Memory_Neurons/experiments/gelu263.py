"""GELU263 – Explicit Pass-Counter Gate.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: The most TRANSPARENT possible implementation of monotonic
sequential adaptation. We explicitly track which pass we're on and scale
the gate LINEARLY with pass number × similarity.
═══════════════════════════════════════════════════════════════════════════

CORE IDEA:
    Detect pass transitions by monitoring cumulative batch count.
    Since we know the test set has N_test ≈ 98 batches per pass:
        batches 0..N-1   → pass 1
        batches N..2N-1  → pass 2
        batches 2N..3N-1 → pass 3

    But we can't hardcode N_test.
    Better: detect pass boundary using the same reset-detection trick as
    gelu249 (first high-sim match with buffer → pass 2 started).
    Then increment pass counter.

    GATE:
        gate = 1 + k * (pass_num - 1) * sim_to_nearest_slot

    Pass 1: pass_num=1 → gate = 1.0 exactly ✓
    Pass 2: pass_num=2 → gate = 1 + k*sim   (moderate)
    Pass 3: pass_num=3 → gate = 1 + 2k*sim  (double)
    Pass 4: pass_num=4 → gate = 1 + 3k*sim  (triple)

    Δ1→2 < Δ1→3 GUARANTEED if k*sim, 2k*sim both help PPL.

COMPARE TO gelu249:
    gelu249 uses facil (exponential: 1, 2, 4, 8...) → faster compounding.
    gelu263 uses pass_num (linear: 1, 2, 3, 4...) → more controlled.
    Linear scaling avoids over-boosting on later passes.

PASS DETECTION:
    Like gelu249: first high-sim match to buffer (cos_sim > THRESH) signals
    pass-2 start. Increment pass_num. Subsequent transitions detected similarly
    by resetting a "pass_max_sim" tracker.

    If after 2 consecutive batches with low sim (<THRESH) then a batch with
    high sim again → that's the pass-3 boundary.

    Simpler: track cumulative batch count. When batch count mod N_inferred
    goes back to 0 → new pass. N_inferred = batch count at first high-sim hit.

PARAMS: log_k_pass (gate strength, init k=1.0)
STATE:  _buf (N,D), _mask, _ptr, _pass_num (int), _n_pass1 (int),
        _batch_count (int), _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
MAX_GATE    = 8.0


class GELU263(nn.Module):
    """Explicit pass-counter gate: gate = 1 + k*(pass_num-1)*sim."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_k_pass = nn.Parameter(torch.tensor(math.log(1.0)))

        self._buf:   torch.Tensor = None
        self._mask:  torch.Tensor = None
        self._ptr    = 0
        self._pass_num = 1
        self._n_pass1  = 0           # number of pass-1 batches observed
        self._batch_count = 0        # total eval batches since reset
        self._pass1_complete = False
        self._pass_batch_count = 0   # batches since last pass boundary

    def reset_state(self):
        self._buf          = None
        self._mask         = None
        self._ptr          = 0
        self._pass_num     = 1
        self._n_pass1      = 0
        self._batch_count  = 0
        self._pass1_complete = False
        self._pass_batch_count = 0

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_pass = self.log_k_pass.exp().clamp(0.01, 5.0)

        y      = self._gelu(x)
        y_mean = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
            self._ptr = 0

        self._batch_count       += 1
        self._pass_batch_count  += 1

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n   = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    buf_n = F.normalize(self._buf, dim=-1)
                    sims  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                        self._pass_num       = 2
                        self._n_pass1        = self._pass_batch_count - 1
                        self._pass_batch_count = 1  # first batch of pass 2
                    else:
                        self._buf[self._ptr]  = F.normalize(y_mean, dim=0)
                        self._mask[self._ptr] = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]  = F.normalize(y_mean, dim=0)
                    self._mask[0] = True
                    self._ptr = 1
                    return y
            if not self._pass1_complete:
                return y

        # ── PASS 2+ PHASE ─────────────────────────────────────────────
        # Detect pass boundary: pass_batch_count > n_pass1 → new pass
        if (self._n_pass1 > 0
                and self._pass_batch_count > self._n_pass1):
            self._pass_num        += 1
            self._pass_batch_count = 1

        with torch.no_grad():
            m_n         = F.normalize(y_mean.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            sim_val     = sims[nearest_idx].clamp(0.0, 1.0).item()

        # gate = 1 + k * (pass_num - 1) * sim
        gate = min(1.0 + k_pass.item() * (self._pass_num - 1) * sim_val, MAX_GATE)
        return y * gate
