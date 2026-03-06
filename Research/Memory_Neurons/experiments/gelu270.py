"""GELU270 – Contrastive Slot Gate.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: All previous buffer experiments gate on ABSOLUTE similarity
to the nearest slot. This experiment gates on the RELATIVE similarity:
    contrast = sim_nearest  -  mean(sim_all_other_slots)

A high contrast means: "this IS the stored batch, and nothing else matches."
A low contrast means: "similar to everything generally" → don't boost.
═══════════════════════════════════════════════════════════════════════════

MOTIVATION:
    Consider two scenarios at pass 2:
    (A) sim_nearest=0.95, mean_others=0.50 → contrast=0.45 (genuine match)
    (B) sim_nearest=0.85, mean_others=0.82 → contrast=0.03 (weak selectivity)

    Current experiments (gelu249 etc.) give the SAME gate for (A) and (B)
    if max_sim > FIRE_THRESH, because they only check the threshold.
    Contrastive gating gives MUCH HIGHER gate for (A) vs (B), properly
    rewarding SPECIFIC episodic recall over generic familiarity.

    This maps cleanly to the thesis s(x,c):
        s(x,c) = sim(x, c_nearest) - mean_j[sim(x, c_j)]
    — a contrastive similarity that measures SPECIFIC recognition.

GATE FORMULA:
    contrast     = sim_nearest - (sum_all_sims - sim_nearest) / (n_valid - 1)
    gate         = 1 + k * (facil - 1) * sigmoid(sharpness * contrast)

    sigmoid(sharpness * contrast):
        contrast=0: sigmoid(0)=0.5 → gate=1+k*(facil-1)*0.5 (moderate)
        contrast>0.3: sigmoid≈1.0  → gate≈1+k*(facil-1)    (full boost)
        contrast<0: sigmoid≈0      → gate≈1.0              (no boost)

    Pre-fire facilitation (×FACIL_RATE per pass) ensures monotonicity.

PARAMS: log_k_gate (gate scale, init k=0.5),
        log_sharpness (contrast sharpness, init s=5.0)
STATE:  ring buffer + facilitation (same as gelu249) + _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0
N_BUF       = 512


class GELU270(nn.Module):
    """Contrastive slot gate: nearest-sim minus mean-other-sim drives facilitation."""

    def __init__(self, buffer_size: int = N_BUF):
        super().__init__()
        self._N = buffer_size

        self.log_k_gate    = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(5.0)))

        self._buf:   torch.Tensor = None
        self._facil: torch.Tensor = None
        self._mask:  torch.Tensor = None
        self._ptr    = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf    = None
        self._facil  = None
        self._mask   = None
        self._ptr    = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k_gate    = self.log_k_gate.exp().clamp(0.01, 5.0)
        sharpness = self.log_sharpness.exp().clamp(0.5, 20.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf   = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask  = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── Pass-1 building ────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (self._buf * q).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]   = F.normalize(m_curr, dim=0)
                        self._facil[self._ptr] = 1.0
                        self._mask[self._ptr]  = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]   = F.normalize(m_curr, dim=0)
                    self._facil[0] = 1.0
                    self._mask[0]  = True
                    self._ptr      = 1
                    return y

        # ── Pass-2+ contrastive gate ───────────────────────────────────
        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)
            sims        = (self._buf * q).sum(-1)            # (N,) raw; normalize below
            sims_masked = sims.masked_fill(~self._mask, -1.0)
            nearest_idx = sims_masked.argmax()
            sim_nearest = sims_masked[nearest_idx].item()

            # Contrastive: sim_nearest - mean(others)
            n_valid = self._mask.sum().item()
            if n_valid > 1:
                sum_others  = sims.masked_fill(~self._mask, 0.0).sum() - sims[nearest_idx]
                mean_others = (sum_others / (n_valid - 1)).item()
                contrast    = sim_nearest - mean_others
            else:
                contrast = 0.0

            # PRE-FIRE facilitation update
            if sim_nearest > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE

            facil_level = self._facil[nearest_idx].item()

        # gate = 1 + k * (facil-1) * sigmoid(sharpness * contrast)
        selectivity = torch.sigmoid(torch.tensor(sharpness.item() * contrast))
        gate = min(1.0 + k_gate.item() * (facil_level - 1.0) * selectivity.item(), MAX_GATE)
        return y * gate
