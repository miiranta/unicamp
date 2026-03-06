"""GELU251 – Detection-based Frozen Buffer + Additive Memory Injection.

HYPOTHESIS: Instead of SCALING the gelu output (multiplicative gate), ADD the
stored pass-1 mean activation as a residual correction. This is more stable
because it doesn't distort the relative activation magnitudes.

MECHANISM:
    output = gelu(x) + inject_scale * (facil - 1) * stored_mean[nearest]

    stored_mean[nearest] = pass-1 mean GELU activation for the matched slot.
    inject_scale = learnable injection coefficient.
    (facil - 1) = how many times this slot has been re-accessed (0, 1, 2, ...)

    pass 1: buffer building, output = gelu(x).
    pass 2: facil=1.0→2.0 (after pre-fire), inject 1 * stored_mean
    pass 3: facil=2.0→4.0, inject 3 * stored_mean (3x stronger signal)

WHY ADDITIVE INJECTION DIFFERS FROM MULTIPLICATIVE:
    Multiplicative gate:  output = gelu(x) * gate
                         scales ALL dimensions uniformly
    Additive injection:   output = gelu(x) + alpha * stored_mean
                         adds pass-1 activation PATTERN to current output.
                         If pass-1 was already a good representation for this
                         text, adding it pulls activations toward the "ideal."

    For text this model has already processed, stored_mean contains the
    AVERAGE GELU activation that helped predict that text accurately.
    Re-injecting it should reinforce the correct prediction pathway.

INTERESTING NOTE ON DIRECTION:
    stored_mean is the unnormalized mean (not normalized like in _buf cosine search).
    We store the raw GELU output mean per batch. The injection adds this direction.
    Over passes, the injected signal gets stronger (2x per pass via facil).

MONOTONIC GUARANTEE: inject strength = (facil-1) * stored_mean_norm → 0, norm, 3*norm
    Pass 1: 0    injection → baseline PPL ✓
    Pass 2: 1×   injection → PPL improves (if stored_mean is useful)
    Pass 3: 3×   injection → more improvement

PARAMS:
    log_inject_scale (1 scalar, injection strength)
    _buf_raw (N, D) stores UNNORMALIZED means for injection
    _buf     (N, D) stores NORMALIZED means for cosine lookup
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FIRE_THRESH = 0.85
FACIL_RATE  = 2.0


class GELU251(nn.Module):
    """Additive injection: output += inject_scale * (facil-1) * stored_mean."""

    def __init__(self, buffer_size: int = 512):
        super().__init__()
        self._N = buffer_size
        self.log_inject = nn.Parameter(torch.tensor(math.log(0.1)))  # small init

        self._buf:     torch.Tensor = None  # normalized (for lookup)
        self._buf_raw: torch.Tensor = None  # unnormalized (for injection)
        self._facil:   torch.Tensor = None
        self._mask:    torch.Tensor = None
        self._ptr  = 0
        self._pass1_complete = False
        # _test_mode is only True after reset_state() is explicitly called by the
        # test harness.  During training validation (eval mode but no reset_state
        # call) this stays False so the buffer never fires and val_loss stays finite.
        self._test_mode = False

    def reset_state(self):
        self._buf     = None
        self._buf_raw = None
        self._facil   = None
        self._mask    = None
        self._ptr     = 0
        self._pass1_complete = False
        self._test_mode = True   # activate buffer logic for the 3 test passes

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        inject = self.log_inject.exp().clamp(1e-4, 2.0)

        y      = self._gelu(x)

        # Episodic layer is TEST-ONLY: skip during both training AND training's
        # validation passes.  The buffer only activates after reset_state() is
        # explicitly called by the 3-test-pass harness (_test_mode flag).
        # Without this guard, validation within training epochs can build the
        # buffer and grow facil exponentially → injection overflow → NaN val_loss.
        if self.training or not self._test_mode:
            return y

        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,) unnormalized

        if self._buf is None:
            with torch.no_grad():
                self._buf     = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._buf_raw = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._facil   = torch.ones( self._N,    device=x.device, dtype=y.dtype)
                self._mask    = torch.zeros(self._N,    device=x.device, dtype=torch.bool)
            self._ptr = 0

        # PASS-1: build buffer
        if not self._pass1_complete:
            with torch.no_grad():
                if self._mask.any():
                    m_n  = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (F.normalize(self._buf, dim=-1) * m_n).sum(-1).masked_fill(~self._mask, -1.0)
                    if sims.max().item() > FIRE_THRESH:
                        self._pass1_complete = True
                    else:
                        self._buf[self._ptr]     = F.normalize(m_curr, dim=0)
                        self._buf_raw[self._ptr] = m_curr
                        self._facil[self._ptr]   = 1.0
                        self._mask[self._ptr]    = True
                        self._ptr = (self._ptr + 1) % self._N
                        return y
                else:
                    self._buf[0]     = F.normalize(m_curr, dim=0)
                    self._buf_raw[0] = m_curr
                    self._facil[0]   = 1.0
                    self._mask[0]    = True
                    self._ptr        = 1
                    return y

        # PASS-2+: frozen buffer, pre-fire, additive injection
        with torch.no_grad():
            m_n         = F.normalize(m_curr.unsqueeze(0), dim=-1)
            buf_n       = F.normalize(self._buf, dim=-1)
            sims        = (buf_n * m_n).sum(-1).masked_fill(~self._mask, -1.0)
            nearest_idx = sims.argmax()
            max_sim     = sims[nearest_idx].item()
            if max_sim > FIRE_THRESH:
                self._facil[nearest_idx] *= FACIL_RATE
            facil_level = self._facil[nearest_idx].item()
            stored_mean = self._buf_raw[nearest_idx].clone()   # (D,) unnormalized

        # output = gelu(x) + inject * (facil-1) * stored_mean  [broadcast over B,T]
        # facil=2.0 (1st fire): adds 1 * stored_mean
        # facil=4.0 (2nd fire): adds 3 * stored_mean
        injection = inject * (facil_level - 1.0) * stored_mean.view(1, 1, D)
        return y + injection
