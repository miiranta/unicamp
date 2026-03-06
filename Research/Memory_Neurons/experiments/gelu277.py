"""GELU277 – Sparse Top-K Neuron Amplification.

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: SELECTIVE NEURON AMPLIFICATION.
In all previous D-dimensional gate experiments (gelu261, gelu265, gelu267),
the gate is a CONTINUOUS function of stored statistics.
This experiment identifies the TOP-K MOST IMPORTANT DIMENSIONS from pass 1
and applies a BINARY gate: amplify those K dims, leave others unchanged.
═══════════════════════════════════════════════════════════════════════════

MOTIVATION (Lottery Ticket Hypothesis + Sparse Coding):
    Neural networks exhibit SPARSE ACTIVATION: in a typical MLP forward pass,
    ~80% of GELU units are near-zero. Only ~20% carry the meaningful signal.
    
    For a given text passage, a SPECIFIC set of neurons is consistently active.
    These are the "important neurons for this content."
    
    In pass 2, the SAME neurons should be active (same text → same features).
    If we selectively amplify only those neurons, we boost the signal-to-noise
    ratio WITHOUT affecting the inactive dimensions (which add no information).

SPARSE GATE:
    During pass 1, per slot s, identify:
        top_k_mask[s] = top-K dimensions of |mean_activation| over the B*T tokens

    During pass 2+, when slot s fires:
        gate_d = 1.0 + (k_amp - 1.0) * top_k_mask[s][d]
               = k_amp  for d in top-K
               = 1.0    for d not in top-K

    Where k_amp = 1 + strength * (facil - 1):
        facil grows ×2 each pass (pre-fire).
        facil=1.0 → k_amp=1.0 (no effect, pass 1)
        facil=2.0 → k_amp=1+strength (moderate)
        facil=4.0 → k_amp=1+3*strength (strong)

    Only K/D fraction of dimensions are affected. With K=64, D=1024: ~6%.
    The OTHER 94% of dimensions are exactly GELU(x).

WHY SPARSE IS BETTER THAN DENSE:
    Dense gates (gelu249, gelu255) apply to ALL D=1024 dimensions equally.
    Most of these dimensions carry DIFFERENT content each pass. Boosting
    them uniformly adds noise from irrelevant dimensions.
    
    Sparse gate: only the K dimensions that SPECIFICALLY represent this
    content are boosted. Orthogonal noise dims are untouched.
    
    This is essentially "select the neurons that matter, amplify them."

TOP-K SELECTION:
    top_k_vals, top_k_idx = torch.topk(stored_mean.abs(), K, dim=-1)
    top_k_mask[s] = zeros except at top_k_idx[s] = 1.0

PARAMS: log_k_strength (amplification per-unit k), log_top_k (soft K selection)
        Actually: K is fixed at init (e.g., K=64, typical sparse coding recommendation)
        log_strength (gate strength: k_amp = 1+strength*(facil-1), init strength=1.0)
STATE:  _buf_keys (N,D), _buf_masks (N,D) 0/1 bool, _facil (N,),
        _mask (N,) bool, _ptr int, _pass1_complete bool
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0
MAX_GATE    = 8.0
N_BUF       = 512
TOP_K_FRAC  = 0.10    # amplify top 10% of dimensions


class GELU277(nn.Module):
    """Sparse top-K neuron amplification: only the most active dims get boosted."""

    def __init__(self, buffer_size: int = N_BUF):
        super().__init__()
        self._N         = buffer_size
        self._top_k_abs = None   # set from D at first forward call

        self.log_strength = nn.Parameter(torch.tensor(math.log(1.0)))   # strength=1.0

        self._buf_keys:  torch.Tensor = None
        self._buf_means: torch.Tensor = None   # (N, D) raw means
        self._buf_masks: torch.Tensor = None   # (N, D) 0/1 float top-K mask
        self._facil:     torch.Tensor = None
        self._mask:      torch.Tensor = None
        self._ptr        = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf_keys  = None
        self._buf_means = None
        self._buf_masks = None
        self._facil     = None
        self._mask      = None
        self._ptr       = 0
        self._pass1_complete = False
        # Keep _top_k_abs — it's a function of D, which doesn't change

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def _compute_mask(self, mean_vec: torch.Tensor, k: int) -> torch.Tensor:
        """Return a (D,) float tensor: 1.0 at top-k |mean| dims, 0.0 elsewhere."""
        _, top_idx = mean_vec.abs().topk(k, dim=-1)
        mask = torch.zeros_like(mean_vec)
        mask[top_idx] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        strength = self.log_strength.exp().clamp(0.01, 5.0)

        # Set top-K count from D
        if self._top_k_abs is None:
            self._top_k_abs = max(1, int(D * TOP_K_FRAC))

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf_keys is None:
            with torch.no_grad():
                self._buf_keys  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_means = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_masks = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil     = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask      = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # ── Pass-1 building ────────────────────────────────────────────
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        topk_mask = self._compute_mask(m_curr, self._top_k_abs)
                        self._buf_keys[self._ptr]  = F.normalize(m_curr, dim=0)
                        self._buf_means[self._ptr] = m_curr
                        self._buf_masks[self._ptr] = topk_mask
                        self._facil[self._ptr]     = 1.0
                        self._mask[self._ptr]      = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    topk_mask = self._compute_mask(m_curr, self._top_k_abs)
                    self._buf_keys[0]  = F.normalize(m_curr, dim=0)
                    self._buf_means[0] = m_curr
                    self._buf_masks[0] = topk_mask
                    self._facil[0]     = 1.0
                    self._mask[0]      = True
                    self._ptr          = 1
                    return y

        # ── Pass-2+ sparse top-K gate ──────────────────────────────────
        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)
            sims        = (self._buf_keys * q).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()

            if sims[nearest_idx].item() > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE

            facil       = self._facil[nearest_idx].item()
            topk_mask_s = self._buf_masks[nearest_idx]    # (D,)

        # k_amp = 1 + strength * (facil - 1) for top-K dims, 1.0 otherwise
        # gate_d = 1 + (k_amp - 1) * topk_mask_d
        k_amp = min(1.0 + strength.item() * (facil - 1.0), MAX_GATE)
        gate  = 1.0 + (k_amp - 1.0) * topk_mask_s   # (D,)

        return y * gate.view(1, 1, D)
