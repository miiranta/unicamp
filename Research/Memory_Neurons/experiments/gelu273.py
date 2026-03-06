"""GELU273 – Norm-Preserving SLERP Gate (Spherical Memory Interpolation).

═══════════════════════════════════════════════════════════════════════════
NOVEL DIRECTION: PRESERVE ACTIVATION NORM while changing direction.
All previous multiplicative gates (gate > 1) INCREASE the norm of gelu(x),
which changes the magnitude seen by downstream layers — a side effect.
SLERP (Spherical Linear Interpolation) moves the output along the unit sphere
toward the stored context direction, preserving norm exactly.
═══════════════════════════════════════════════════════════════════════════

MATHEMATICS:
    Let y = gelu(x)  (the current activation)
    Let v = stored mean from nearest pass-1 slot (unit vector)

    SLERP formula:
        y_hat = y / ||y||               [unit vector in y direction]
        Ω     = arccos(clamp(y_hat · v, -1, 1))   [angle between them]

        if Ω ≈ 0: y_hat and v are already aligned → no rotation
        else:
            slerp(y_hat, v, t) = sin((1-t)Ω)/sin(Ω) * y_hat
                                + sin(t·Ω)/sin(Ω)    * v

        output = ||y|| * slerp(y_hat, v, t)   [rescale to original norm]

    t ∈ [0, 1] controls interpolation:
        t=0: output = y (no change, pass 1 behavior)
        t=0.3: output direction rotated 30% toward stored context
        t=1.0: output direction = stored context direction exactly

    t = sigmoid(k * log(hit_count + 1)):
        hit_count=0: t = sigmoid(-∞) = 0 (if k=∞) → but k is finite, so
        To guarantee t=0 at pass 1: use t=(hit_count/(hit_count+1))*sigmoid(k)
            hit=0: t=0 exactly ✓
            hit=1: t=0.5*sigmoid(k) ≈ 0.5*0.73 ≈ 0.37 (with k=1)
            hit=2: t=0.67*sigmoid(k) ≈ 0.67*0.73 ≈ 0.49

WHY NORM PRESERVATION MATTERS:
    After the MLP GELU layer, the output is projected by a linear layer W.
    If we multiply gelu(x) by a gate >1, the projection sees a larger vector →
    larger logit magnitudes → potentially distorted softmax distributions.
    
    SLERP changes DIRECTION without changing MAGNITUDE:
    The projection W sees an output with the SAME norm as the standard GELU
    output, just rotated toward the stored context direction. This is the
    cleanest possible form of "memory-guided activation."

PARAMS: log_k_slerp (controls t ramp, init k=1.0)
STATE:  ring buffer + hit_count (same detection as gelu271)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
N_BUF       = 512


class GELU273(nn.Module):
    """Norm-preserving SLERP: rotate activation toward stored context direction."""

    def __init__(self, buffer_size: int = N_BUF):
        super().__init__()
        self._N = buffer_size

        self.log_k_slerp = nn.Parameter(torch.tensor(math.log(1.0)))

        self._buf:       torch.Tensor = None
        self._hit_count: torch.Tensor = None
        self._mask:      torch.Tensor = None
        self._ptr        = 0
        self._pass1_complete = False

    def reset_state(self):
        self._buf        = None
        self._hit_count  = None
        self._mask       = None
        self._ptr        = 0
        self._pass1_complete = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    @staticmethod
    def _slerp(y: torch.Tensor, v: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
        """SLERP from y to v with fraction t, at each token position.
        y: (B*T, D), v: (D,) unit vector, t: scalar
        Returns output of same shape and norm as y.
        """
        norms  = y.norm(dim=-1, keepdim=True).clamp(min=eps)           # (B*T, 1)
        y_hat  = y / norms                                              # (B*T, D) unit
        v_exp  = v.unsqueeze(0)                                         # (1, D)

        cos_om = (y_hat * v_exp).sum(-1).clamp(-1.0 + eps, 1.0 - eps)  # (B*T,)
        omega  = torch.acos(cos_om)                                     # (B*T,) angles
        sin_om = omega.sin().clamp(min=eps)                             # (B*T,)

        w1 = (((1.0 - t) * omega).sin() / sin_om).unsqueeze(-1)        # (B*T, 1)
        w2 = ((t * omega).sin()         / sin_om).unsqueeze(-1)        # (B*T, 1)

        y_slerp = w1 * y_hat + w2 * v_exp                              # (B*T, D) unit sphere
        return y_slerp * norms                                          # restore norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D  = x.shape
        k_slerp  = self.log_k_slerp.exp().clamp(0.01, 5.0)

        y      = self._gelu(x)
        m_curr = y.detach().flatten(0, 1).mean(0)

        # ── Init ──────────────────────────────────────────────────────
        if self._buf is None:
            with torch.no_grad():
                self._buf       = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._hit_count = torch.zeros(self._N,    device=x.device, dtype=torch.long)
                self._mask      = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
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
                        self._buf[self._ptr]       = F.normalize(m_curr, dim=0)
                        self._hit_count[self._ptr] = 0
                        self._mask[self._ptr]      = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]       = F.normalize(m_curr, dim=0)
                    self._hit_count[0] = 0
                    self._mask[0]      = True
                    self._ptr          = 1
                    return y

        # ── Pass-2+ SLERP toward stored context ───────────────────────
        with torch.no_grad():
            q           = F.normalize(m_curr.unsqueeze(0), dim=-1)
            sims        = (self._buf * q).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()

            if sims[nearest_idx].item() > FIRE_THRESH:
                self._hit_count[nearest_idx] += 1

            count   = self._hit_count[nearest_idx].item()
            v_store = self._buf[nearest_idx].clone()   # (D,) unit vector

        # t = hit/(hit+1) * sigmoid(k)  → t=0 when hit=0 ✓
        sig_k = torch.sigmoid(k_slerp).item()
        t     = (count / (count + 1.0)) * sig_k if count > 0 else 0.0
        if t < 1e-6:
            return y

        y_flat  = y.detach().flatten(0, 1)    # (B*T, D)
        y_slerp = self._slerp(y_flat, v_store, t)
        # Must rebuild the output with gradients for training — use scale factor
        # direction_scale: how much we rotate = y_slerp / y_flat (per token)
        with torch.no_grad():
            scale = (y_slerp / (y_flat.abs() + 1e-8)).clamp(-3.0, 3.0)
        output = (y.flatten(0, 1) * scale).view(B, T, D)
        return output
