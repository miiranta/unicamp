"""GELU226 – Ring Buffer with Per-Channel Familiarity Gate.

UPGRADE FROM gelu54 (best adaptation: Δ1→3 = +0.030):
    gelu54 uses a SCALAR cosine similarity for the gate:
        sim_scalar = cosine(mean_BT(GELU(x)), nearest_buffer_episode)   ← single scalar
        gate_token = cosine(GELU(x_token), nearest_buffer_episode)       ← scalar per token
        gate_final = (1-α) + α × exp(-τ × gate_token)                   ← scalar per token

    The scalar gate applies the SAME modulation to ALL D=1024 channels, even though
    different channels may have very different degrees of familiarity.

    gelu226 UPGRADES to PER-CHANNEL familiarity:
        nearest_buf_d = buffer[nearest_idx]_d               ← (D,) per-channel episode mean
        dev_d = |GELU(x_token_d) - nearest_buf_d| / (pooled_std_d + eps)   ← per-channel deviation
        fam_d = exp(-dev_d²)                                ← per-channel familiarity ∈ (0,1]
        gate_d = 1 - alpha_d × fam_d                        ← per-channel gate, learnable alpha

    SEMANTICS:
        familiar channel (GELU(x_d) ≈ nearest_buf_d):  fam_d ≈ 1 → gate_d ≈ 1 - alpha_d [suppress]
        novel channel    (GELU(x_d) >> nearest_buf_d):  fam_d ≈ 0 → gate_d ≈ 1            [pass-through]

    ADAPTATION ADVANTAGE:
        After test pass 1: ring buffer contains test episode means per channel.
        Pass 2: Each channel individually recognized as familiar → stronger per-channel suppression
        vs pass 1 (where buffer had training episodes → most channels appear novel → gate ≈ 1)
        → Stronger positive Δ than gelu54's scalar approach.

PARAMS: log_alpha (scalar gate strength, shared), log_tau (sharpness) = 2 scalars
        pooled_std is estimated from the ring buffer entries themselves.
STATE:  ring buffer (N, D), mask (N,), pointer (int)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU226(nn.Module):
    """Per-channel ring buffer familiarity gate: stronger adaptation than gelu54."""

    def __init__(self, buffer_size: int = 32):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None   # (N, D) — raw (non-normalized) episode means
        self._mask: torch.Tensor = None   # (N,) bool
        self._ptr   = 0
        self._ready = False

        # Learnable parameters
        self.log_alpha = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))  # init α ≈ 0.5
        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))                   # sharpness τ

    def reset_state(self):
        self._buf   = None
        self._mask  = None
        self._ptr   = 0
        self._ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        alpha = F.softplus(self.log_alpha)   # ∈ (0, ∞) — max suppression depth
        tau   = self.log_tau.exp()

        y = self._gelu(x)   # (B, T, D)

        # Batch mean of GELU output for buffer update
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Initialise on first call ──────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0] = m_curr.clone()
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        # ── Find nearest buffer episode (scalar cosine like gelu54) ──
        filled = self._mask                                        # (N,)
        m_norm = F.normalize(m_curr.unsqueeze(0), dim=-1)         # (1, D)
        b_norm = F.normalize(self._buf, dim=-1)                   # (N, D)
        sims   = (b_norm * m_norm).sum(-1).masked_fill(~filled, -1.0)
        nearest_idx = sims.argmax()                                # scalar int

        nearest_ep = self._buf[nearest_idx].detach()   # (D,) — detach: state tensor, not parameter

        # ── Estimate per-channel std from buffer entries ───────────────
        if filled.sum() > 1:
            buf_filled = self._buf[filled]                         # (k, D)
            # Pooled std across filled episodes, each of which is an episode mean
            # This underestimates token-level std, but is fast and differentiable
            buf_std = buf_filled.std(0).clamp(1e-4)               # (D,)
        else:
            buf_std = torch.ones(D, device=x.device, dtype=y.dtype) * 0.1

        # ── Per-channel familiarity gate ──────────────────────────────
        # Deviation of each token from nearest buffer episode, per channel
        dev  = (y - nearest_ep) / (buf_std + 1e-5)               # (B, T, D)
        fam  = torch.exp(-tau * dev.pow(2))                       # (B, T, D) ∈ (0,1]
        gate = (1.0 - alpha * fam).clamp(0.05, 1.05)             # (B, T, D)
        # familiar: fam≈1 → gate ≈ 1-α (suppressed)
        # novel:    fam≈0 → gate ≈ 1.0 (pass-through)

        output = y * gate

        # ── Update ring buffer ────────────────────────────────────────
        with torch.no_grad():
            self._buf[self._ptr]  = m_curr.clone()
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
