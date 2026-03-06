"""GELU259 – Pre-GELU Input Nudging (Input-Space Correction).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: Every prior experiment modifies POST-GELU activations.
This experiment corrects the INPUT to GELU, before non-linearity is applied.
═══════════════════════════════════════════════════════════════════════════

CORE IDEA:
    In pass 1, store the mean INPUT vector per batch: x̄_slot ∈ ℝᴰ.
    Also track the global EMA input mean across all of pass 1: x̄_global.

    In pass 2+, when batch k matches slot s with high similarity:
        1. Compute nudge direction: Δ = x̄_slot[s] - x̄_global
           (how this specific batch's input deviates from average)
        2. Apply pre-GELU correction: x_adj = x + inject * Δ
        3. Compute: output = GELU(x_adj)

    On each pass-n (pre-fire): inject *= NUDGE_RATE (e.g. ×2)
    → inject = 0 at pass 1, small at pass 2, larger at pass 3.

WHY THIS SHOULD WORK:
    The GELU non-linearity is the crucial computation. Its output depends
    STRONGLY on where exactly x sits relative to the activation threshold.
    A small shift in the INPUT can produce large, structured output changes.

    By nudging x toward where it WAS during pass 1 (relative to global mean),
    we reinforce the activation pattern that produced the original output.
    The GELU then "sees" input that is closer to the remembered pattern,
    producing a more coherent activation with lower loss.

    Critical difference from post-GELU injection (gelu251):
        Post-GELU: adds a fixed bias to the output (may be inconsistent with
                   the projection that follows GELU).
        Pre-GELU:  shifts input → GELU recomputes → output is consistent
                   with the natural function, just from a shifted start point.

MONOTONICITY GUARANTEE:
    inject=0 at pass 1 → gate=1.0, no change.
    inject > 0 at pass 2 (pre-fire: inject 0→NUDGE_BASE).
    inject larger at pass 3 (pre-fire: inject NUDGE_BASE→NUDGE_BASE*NUDGE_RATE).
    → Δ1→3 > Δ1→2 > 0 if nudging helps. ✓

PARAMS: log_inject_base (learnable nudge strength, init = 0.1)
STATE:  _xbuf (N, D) raw input means, _buf (N, D) normalized GELU means (keys),
        _inj (N,) per-slot inject level, _xglobal (D,) global input EMA,
        _mask (N,), _ptr, _pass1_complete
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH  = 0.85
NUDGE_RATE   = 2.0    # inject doubles on each pass
NUDGE_BASE   = 0.1    # inject level on first retrieval in pass 2


class GELU259(nn.Module):
    """Pre-GELU input nudging: corrects the input before non-linearity."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_inject = nn.Parameter(torch.tensor(math.log(NUDGE_BASE)))

        self._buf:    torch.Tensor = None   # (N, D) normalized GELU means (keys)
        self._xbuf:   torch.Tensor = None   # (N, D) raw input means  (values)
        self._xglobal: torch.Tensor = None  # (D,)   global EMA of inputs
        self._inj:    torch.Tensor = None   # (N,)   per-slot inject level
        self._mask:   torch.Tensor = None   # (N,)   bool
        self._ptr     = 0
        self._pass1_complete = False
        self._n_pass1 = 0    # count of pass-1 batches (for EMA weight)

    def reset_state(self):
        self._buf    = None
        self._xbuf   = None
        self._xglobal = None
        self._inj    = None
        self._mask   = None
        self._ptr    = 0
        self._pass1_complete = False
        self._n_pass1 = 0

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        inject_base = self.log_inject.exp().clamp(1e-3, 2.0)

        x_mean  = x.detach().flatten(0, 1).mean(0)   # (D,) raw input mean
        y_plain = self._gelu(x)
        y_mean  = y_plain.detach().flatten(0, 1).mean(0)

        # Episodic layer is EVAL-ONLY: without this guard, epoch-2 training
        # batches trigger the buffer and apply input nudges during training,
        # corrupting all downstream gradients -> loss explosion.
        if self.training:
            return y_plain

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf     = torch.zeros(self._N, D, device=x.device, dtype=y_plain.dtype)
                self._xbuf    = torch.zeros(self._N, D, device=x.device, dtype=y_plain.dtype)
                self._inj     = torch.zeros(self._N,    device=x.device, dtype=y_plain.dtype)
                self._mask    = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
                self._xglobal = x_mean.clone()
            self._ptr, self._n_pass1 = 0, 0

        # ── PASS-1 BUILDING PHASE ─────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n   = F.normalize(y_mean.unsqueeze(0), dim=-1)
                    buf_n = F.normalize(self._buf, dim=-1)
                    sims  = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]  = F.normalize(y_mean, dim=0)
                        self._xbuf[self._ptr] = x_mean.clone()
                        self._inj[self._ptr]  = 0.0            # not yet activated
                        self._mask[self._ptr] = True
                        self._ptr = (self._ptr + 1) % self._N
                        # update global EMA
                        n = self._n_pass1
                        self._xglobal = (self._xglobal * n + x_mean) / (n + 1)
                        self._n_pass1 += 1
                        return y_plain
                else:
                    self._buf[0]  = F.normalize(y_mean, dim=0)
                    self._xbuf[0] = x_mean.clone()
                    self._inj[0]  = 0.0
                    self._mask[0] = True
                    self._xglobal = x_mean.clone()
                    self._n_pass1 = 1
                    self._ptr     = 1
                    return y_plain

        # ── PASS-2+ PHASE: frozen buffer, pre-fire nudge ─────────────
        with torch.no_grad():
            m_n         = F.normalize(y_mean.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()

            if max_sim > FIRE_THRESH:
                # PRE-FIRE: update inject level BEFORE compute
                if self._inj[nearest_idx].item() < 1e-6:
                    self._inj[nearest_idx] = inject_base
                else:
                    self._inj[nearest_idx] *= NUDGE_RATE

            inj_level = self._inj[nearest_idx].item()
            stored_x  = self._xbuf[nearest_idx].clone()   # (D,)

        if inj_level < 1e-6:
            # No nudge yet; shouldn't happen in pass 2 but safety guard
            return y_plain

        # Nudge direction: stored_x - global_mean → batch-specific deviation
        # Normalize to unit direction so inject_level controls the EXACT scale
        # (raw input norms can be O(10-50) for D=1024, blowing up GELU input)
        nudge_dir  = stored_x - self._xglobal                         # (D,)
        nudge_norm = nudge_dir.norm(p=2).clamp(min=1e-6)
        nudge_unit = nudge_dir / nudge_norm                            # (D,) unit vec

        # Per-token similarity weighting (tokens similar to stored context get bigger nudge)
        x_n       = F.normalize(x.detach().flatten(0, 1), dim=-1)     # (B*T, D)
        sn_n      = F.normalize(stored_x.unsqueeze(0), dim=-1)        # (1, D)
        tok_sim   = (x_n * sn_n).sum(-1).clamp(0.0, 1.0).view(B, T)  # (B, T)

        # x_adj = x + inj * tok_sim * nudge_unit  (controlled-magnitude nudge)
        x_adj  = x + inj_level * tok_sim.unsqueeze(-1) * nudge_unit.view(1, 1, D)
        output = self._gelu(x_adj)

        return output
