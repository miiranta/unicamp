"""GELU275 – Per-Position Token Memory Gate.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Every prior buffer experiment stores PER-BATCH MEANS
(aggregated over all B*T tokens). This loses positional structure.
This experiment stores PER-POSITION means (T-dimensional slots):
    _pos_buf[t] = mean of GELU(x)[*, t, *] across all pass-1 batches.

At pass 2, facilitation at position t depends on how similar the current
token at position t is to the stored position-t mean from pass 1.
═══════════════════════════════════════════════════════════════════════════

HYPOTHESIS:
    In a causal language model, position within the sequence matters.
    Position 0 is always the start token.
    Position T/2 is typically mid-sentence.
    Position T-1 is always the last token before prediction.

    The model's activations at each position have CONSISTENT statistical
    fingerprints across all batches (position 0 of WikiText batches all
    start with a similar distribution — newlines, article starts, etc.).

    By storing the mean activation AT EACH POSITION separately, we have
    T=64 specialized memory slots, each tailored to that position.

    In pass 2:
        sim_t = cosine(gelu(x)[b,t], pos_buf[t])
        facil grows as pos_buf[t] gets more hits

    gate[b,t,d] = 1 + k * (pos_facil[t] - 1) * sim_t

    This is a FINE-GRAINED position-aware facilitation that adapts differently
    at each sequence position.

MECHANISM:
    Pass 1: accumulate position-wise means into _pos_buf (T, D).
            Also track _pos_count (T,) for online average.
    
    Pass-2 detection: same mean-over-all-positions batch detection.
    
    Pass 2+: PRE-FIRE pos_facil[t] *= FACIL_RATE for each position t.
             gate[b,t] = 1 + k * (pos_facil[t]-1) * sim_t

COMPARE TO ALL PRIOR EXPERIMENTS:
    All prior: buffer stores batch-level means (N, D), lookup by batch.
    gelu275: stores position-level means (T, D), lookup by position.
    This is orthogonal to batch-level memory. Combined: gelu264 analogy.

PASS-1 DETECTION:
    Use a separate pass-detection ring buffer (N=512) with batch means,
    purely for detecting when pass 2 starts. Doesn't affect the gate itself.

PARAMS: log_k_pos (gate scale, init k=0.5)
STATE:  _pos_buf (T, D) position-wise means,
        _pos_count (T,) int count of pass-1 batches contributing,
        _pos_facil (T,) facilitation per position,
        _det_buf (N, D) detection ring buffer,
        _det_mask (N,) bool, _det_ptr int,
        _pass1_complete bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0
N_DET       = 512


class GELU275(nn.Module):
    """Per-position token memory with position-wise facilitation gate."""

    def __init__(self):
        super().__init__()
        self.log_k_pos = nn.Parameter(torch.tensor(math.log(0.5)))

        self._pos_buf:   torch.Tensor = None   # (T, D)
        self._pos_count: torch.Tensor = None   # (T,) int counts
        self._pos_facil: torch.Tensor = None   # (T,) facilitation

        # Detection ring buffer (batch-level)
        self._det_buf:  torch.Tensor = None
        self._det_mask: torch.Tensor = None
        self._det_ptr   = 0
        self._pass1_complete = False

    def reset_state(self):
        self._pos_buf   = None
        self._pos_count = None
        self._pos_facil = None
        self._det_buf   = None
        self._det_mask  = None
        self._det_ptr   = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_pos   = self.log_k_pos.exp().clamp(0.01, 5.0)

        y       = self._gelu(x)
        y_det   = y.detach()

        # ── Init ──────────────────────────────────────────────────────
        if self._pos_buf is None:
            with torch.no_grad():
                self._pos_buf   = torch.zeros(T, D, device=x.device, dtype=y.dtype)
                self._pos_count = torch.zeros(T,    device=x.device, dtype=torch.long)
                self._pos_facil = torch.ones( T,    device=x.device, dtype=y.dtype)
                self._det_buf   = torch.zeros(N_DET, D, device=x.device, dtype=y.dtype)
                self._det_mask  = torch.zeros(N_DET,    device=x.device, dtype=torch.bool)
            self._det_ptr = 0

        # ── Pass-1 building ────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                m_curr = y_det.mean(dim=(0, 1))   # (D,) batch mean (for detection)

                # Detection check
                if self._det_mask.any():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (F.normalize(self._det_buf, dim=-1) * q).sum(-1).masked_fill(~self._det_mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                        # Don't return early — fall through to pass-2 gate
                    else:
                        # Accumulate position-wise online mean
                        # pos_mean_new = (count * pos_mean + batch_mean_t) / (count+1)
                        batch_pos_mean = y_det.mean(dim=0)    # (T, D): mean over batch dim
                        for t in range(T):
                            c = self._pos_count[t].item()
                            self._pos_buf[t] = (c * self._pos_buf[t] + batch_pos_mean[t]) / (c + 1)
                            self._pos_count[t] += 1
                        # Update detection buffer
                        m_norm = F.normalize(m_curr, dim=0)
                        self._det_buf[self._det_ptr]  = m_norm
                        self._det_mask[self._det_ptr] = True
                        self._det_ptr = (self._det_ptr + 1) % N_DET
                        return y
                else:
                    # Very first batch
                    batch_pos_mean = y_det.mean(dim=0)
                    self._pos_buf    = batch_pos_mean.clone()
                    self._pos_count += 1
                    m_norm = F.normalize(y_det.mean(dim=(0,1)), dim=0)
                    self._det_buf[0]   = m_norm
                    self._det_mask[0]  = True
                    self._det_ptr      = 1
                    return y

        # ── Pass-2+ per-position facilitation ─────────────────────────
        with torch.no_grad():
            # Normalize stored position means for similarity
            pos_buf_n = F.normalize(self._pos_buf, dim=-1)    # (T, D)
            y_n       = F.normalize(y_det, dim=-1)            # (B, T, D)

            # sim[b, t] = cosine(y[b,t], pos_buf[t])
            sim = (y_n * pos_buf_n.unsqueeze(0)).sum(-1)      # (B, T)

            # PRE-FIRE: update pos_facil for ALL positions with sim > FIRE_THRESH
            fire_mask = sim.mean(0) > FIRE_THRESH              # (T,) bool
            self._pos_facil[fire_mask] = (self._pos_facil[fire_mask] * FACIL_RATE).clamp(max=16.0)

            facil_t = self._pos_facil.clone()                  # (T,)

        # gate[b, t] = 1 + k * (facil_t[t] - 1) * sim[b,t]
        # shape: (B, T)
        gate = (1.0 + k_pos * (facil_t.unsqueeze(0) - 1.0) * sim).clamp(max=MAX_GATE)
        return y * gate.unsqueeze(-1)
