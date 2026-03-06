"""GELU278 – Predictive Coding Delta Injection.

═══════════════════════════════════════════════════════════════════════════
INSPIRATION: PREDICTIVE CODING (Rao & Ballard, 1999).
In predictive coding, the brain maintains predictions of sensory input and
only propagates RESIDUALS (differences from prediction) upward.
Familiarity = small residual. Novelty = large residual.

This experiment INVERTS the mechanism for test-time adaptation:
    "I know what's coming. Inject the STORED PREDICTION to help the model."
═══════════════════════════════════════════════════════════════════════════

MECHANISM  [f(x) = m(s(x,c), g(x))]:

    During pass 1, store per batch:
        stored_val[s] = mean GELU activation           (the prediction)
        global_mean   = mean of ALL stored_val across all pass-1 slots

    "Prediction residual" for slot s:
        delta_s = stored_val[s] - global_mean
        = WHAT MAKES THIS BATCH'S ACTIVATION PROFILE UNIQUE

    During pass 2+, when batch matches slot s:
        output = GELU(x) + inject * delta_s

    inject = k * (pass_num - 1)  [linear scaling: 0 at pass1, k at pass2, 2k at pass3]
    OR
    inject = k * log(hit_count + 1)  [log scaling]

    The injection is ADDITIVE: it injects the SPECIFIC CONTENT of this episode
    as a bias onto the GELU output. The model "remembers" what makes this
    batch different from the average, and the memory is injected as a bias.

WHY THIS IS DISTINCT FROM gelu251 (additive injection) AND gelu269 (Hopfield):
    gelu251: injects stored_val (the full absolute mean)
             — contains BOTH global baseline AND specific content
    gelu269: Hopfield retrieval = weighted sum of all stored vals
             then injects retrieved - global_mean (similar delta approach)
             — but uses soft attention over ALL slots and compounding facil
    gelu278: injects (stored_val - global_mean) using PASS-COUNTER (linear)
             — simpler, linear scaling, no exponential growth
             — linear Δ1→2 < Δ1→3 explicitly by design (inject = k, 2k)

    The key difference: LINEAR scaling (not exponential) + DELTA (not absolute value).

DELTA vs ABSOLUTE:
    Absolute injection (gelu251):
        output += k * stored_val
        Adds EVERYTHING in stored_val, including the global average.
        Dimensions where all slots are similar get boosted for no reason.
    
    Delta injection (gelu278):
        output += k * (stored_val - global_mean)
        Adds ONLY the deviation from average. Dims that are the same for
        all batches contribute ~0. Only DISTINCTIVE dims get injected.
        More signal, less noise.

BROADCAST STRATEGY:
    delta_s ∈ ℝᴰ is broadcast across all B*T tokens:
        output[b,t,:] = GELU(x[b,t,:]) + inject * delta_s

    Optional: scale by per-token similarity for selectivity:
        output[b,t,:] = GELU(x[b,t,:]) + inject * delta_s * tok_sim[b,t]
    → tokens most similar to stored slot get stronger injection.
    (Using this for extra selectivity.)

PARAMS: log_k_inject (injection scale, init k=0.3)
STATE:  _buf_keys (N,D) normalized, _buf_vals (N,D) raw,
        _global_sum (D,), _n_global int,
        _hit_count (N,) int,
        _mask (N,) bool, _ptr int, _pass1_complete bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
N_BUF       = 512
MAX_INJECT  = 5.0


class GELU278(nn.Module):
    """Predictive coding delta injection: additive pass-1 deviation injection."""

    def __init__(self, buffer_size: int = N_BUF):
        super().__init__()
        self._N = buffer_size

        self.log_k_inject = nn.Parameter(torch.tensor(math.log(0.3)))

        self._buf_keys:   torch.Tensor = None
        self._buf_vals:   torch.Tensor = None
        self._hit_count:  torch.Tensor = None
        self._mask:       torch.Tensor = None
        self._global_sum: torch.Tensor = None
        self._n_global    = 0
        self._ptr         = 0
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
        k_inj   = self.log_k_inject.exp().clamp(0.001, MAX_INJECT)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf_keys is None:
            with torch.no_grad():
                self._buf_keys  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_vals  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hit_count = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask      = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
                self._global_sum = torch.zeros(D,         device=x.device, dtype=y.dtype)
            self._n_global = 0
            self._ptr      = 0

        # ── Pass-1 building ────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True   # freeze, fall through
                    else:
                        self._buf_keys[self._ptr]  = F.normalize(m_curr, dim=0)
                        self._buf_vals[self._ptr]  = m_curr.clone()
                        self._hit_count[self._ptr] = 0
                        self._mask[self._ptr]      = True
                        self._global_sum += m_curr
                        self._n_global   += 1
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf_keys[0]  = F.normalize(m_curr, dim=0)
                    self._buf_vals[0]  = m_curr.clone()
                    self._hit_count[0] = 0
                    self._mask[0]      = True
                    self._global_sum   = m_curr.clone()
                    self._n_global     = 1
                    self._ptr          = 1
                    return y

        # ── Pass-2+ predictive delta injection ────────────────────────
        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)
            sims        = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()

            if max_sim > FIRE_THRESH:
                self._hit_count[nearest_idx] += 1

            count        = self._hit_count[nearest_idx].item()
            stored_val   = self._buf_vals[nearest_idx]       # (D,)
            global_mean  = self._global_sum / max(self._n_global, 1)
            delta        = stored_val - global_mean           # (D,) deviation

            # inject = k * hit_count (linear: 0 at pass1, k at pass2, 2k at pass3)
            inject_scale = k_inj.detach() * float(count)

            # Per-token similarity for selective injection
            y_flat  = y.detach().flatten(0, 1)              # (B*T, D)
            y_n     = F.normalize(y_flat, dim=-1)
            sv_n    = F.normalize(stored_val.unsqueeze(0), dim=-1)
            tok_sim = (y_n * sv_n).sum(-1).clamp(0, 1)      # (B*T,) ≥ 0

        if inject_scale < 1e-6:
            return y

        # output = y + inject_scale * delta * tok_sim (broadcast)
        # Using k_inj (requires grad) for backprop
        inject  = k_inj * float(count)
        delta_b = delta.view(1, 1, D)
        tok_w   = tok_sim.view(B, T, 1)
        output  = y + inject * delta_b * tok_w
        return output
