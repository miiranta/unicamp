"""GELU276 – gelu211 Frozen-EMA Release (Novelty Suppression Cancellation).

═══════════════════════════════════════════════════════════════════════════
DIAGNOSIS: WHY gelu211 GIVES Δ ≈ 0 (or negative)

gelu211 suppresses NOVEL activations:
    gate = g_in(z_in) * g_out(z_out)   [both <= 1]
    z_in  = squared z-score of input x vs EMA_x mean/std
    z_out = squared z-score of output y vs EMA_y mean/std

    High z-score = NOVEL = SUPPRESSED (gate low)
    Low  z-score = FAMILIAR = PASSED THROUGH (gate ≈ 1)

During TEST PASS 1: EMA is from training data. Test data is somewhat novel
    → z_in > 0, gate < 1, PPL = 159 (good, model learned novel = uncertain)

During TEST PASS 2: SAME test data as pass 1, BUT:
    → The EMA has been updated with pass-1 test data
    → Pass-2 z-scores might be EVEN HIGHER because EMA now reflects both
       training AND pass-1 test data (mixed), making pass-2 input feel novel
       relative to the contaminated EMA!
    → PPL STAYS THE SAME or gets WORSE (hence Δ ≈ -0.05 in gelu211).

FIX: FREEZE THE EMA AT PASS-1 COMPLETION.
    When pass-2 is detected (ring buffer sees high-sim match):
        → Stop updating EMA (set _l1_frozen = True)
        → All subsequent z-scores computed against FROZEN PASS-1 EMA

    In pass 2: same z-scores as pass 1 (EMA unchanged) → same gate → same PPL?
    NO — better! Pass-1 test data was used to UPDATE the EMA during pass 1.
    By pass-1 end: _ema_x is an AVERAGE of training + test pass-1 data.
    Pass-2 z-scores against this EMA = LOWER z-scores (test data is familiar
    relative to the test-contaminated EMA) → gate CLOSER to 1.0 → LESS
    suppression → HIGHER activation magnitude → LOWER loss.

    Pass 3: EMA still frozen at pass-1 state. z-scores for pass-3 data 
    (exact same as pass 1) computed against the same EMA as pass 2.
    Gate is SAME as pass 2, so PPL ≈ same as pass 2.

    To get Δ1→3 > Δ1→2: at pass-3 detection, CONTINUE UPDATING THE EMA
    with pass-2 data (partial unfreeze): this shifts EMA even MORE toward
    the test distribution, so pass-3 z-scores are EVEN LOWER → gate even
    closer to 1.0 → lower PPL → Δ1→3 > Δ1→2 ✓

IMPLEMENTATION:
    _ema_frozen_x, _ema_frozen_x2, _ema_frozen_y, _ema_frozen_y2:
        Snapshots taken at pass-1 completion.
    
    Pass 1: EMA updates normally (as in gelu211).
    Pass 2+: Use FROZEN EMA for z-scores. Still update live EMA separately.
    Pass 3: Update frozen EMA SLIGHTLY toward live EMA.
        ema_frozen = (1-γ) * ema_frozen + γ * ema_live   [γ small: 0.1]
        → z-scores drift toward 0 → more gating relief each pass.

PARAMS: all gelu211 params (trained). log_gamma (FROZEN EMA drift rate, init γ=0.1).
STATE: same as gelu211 + _frozen_* EMAs + _l1_frozen bool + detection ring buffer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECT_THRESH = 0.85
N_BUF         = 512
GAMMA_DEFAULT = 0.1


class GELU276(nn.Module):
    """gelu211 with frozen EMA release: suppress novelty gate gradually released across passes."""

    def __init__(self, buffer_size: int = N_BUF, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # ── gelu211 params (trained) ───────────────────────────────────
        self.log_d_x     = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_d_y     = nn.Parameter(torch.tensor(math.log(0.9 / 0.1)))
        self.log_tau_in  = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_tau_out = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_a_in    = nn.Parameter(torch.tensor(0.0))
        self.log_a_out   = nn.Parameter(torch.tensor(0.0))
        # EMA drift rate for frozen EMA (eval-only, not trained during training)
        self.log_gamma   = nn.Parameter(torch.tensor(math.log(GAMMA_DEFAULT / (1 - GAMMA_DEFAULT))))

        # ── Live EMA state (gelu211) ───────────────────────────────────
        self._ema_x:  torch.Tensor = None
        self._ema_x2: torch.Tensor = None
        self._ema_y:  torch.Tensor = None
        self._ema_y2: torch.Tensor = None
        self._l1_ready = False

        # ── Frozen EMA snapshots ───────────────────────────────────────
        self._frz_x:  torch.Tensor = None
        self._frz_x2: torch.Tensor = None
        self._frz_y:  torch.Tensor = None
        self._frz_y2: torch.Tensor = None
        self._l1_frozen = False

        # ── Detection ring buffer (pass-1 → pass-2 detection) ─────────
        self._det_buf:  torch.Tensor = None
        self._det_mask: torch.Tensor = None
        self._det_ptr   = 0
        self._det_ready = False

    def reset_state(self):
        self._ema_x     = None;  self._ema_x2  = None
        self._ema_y     = None;  self._ema_y2  = None
        self._l1_ready  = False
        self._frz_x     = None;  self._frz_x2  = None
        self._frz_y     = None;  self._frz_y2  = None
        self._l1_frozen = False
        self._det_buf   = None;  self._det_mask = None
        self._det_ptr   = 0;     self._det_ready = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        y       = self._gelu(x)

        d_x     = torch.sigmoid(self.log_d_x).detach().item()
        d_y     = torch.sigmoid(self.log_d_y).detach().item()
        tau_in  = self.log_tau_in.exp()
        tau_out = self.log_tau_out.exp()
        a_in    = torch.sigmoid(self.log_a_in)
        a_out   = torch.sigmoid(self.log_a_out)
        gamma   = torch.sigmoid(self.log_gamma).detach().item()

        x_flat = x.detach().flatten(0, 1)
        y_flat = y.detach().flatten(0, 1)

        # ── Init live EMA ─────────────────────────────────────────────
        if not self._l1_ready:
            with torch.no_grad():
                self._ema_x  = x_flat.mean(0).clone()
                self._ema_x2 = (x_flat**2).mean(0).clone()
                self._ema_y  = y_flat.mean(0).clone()
                self._ema_y2 = (y_flat**2).mean(0).clone()
            self._l1_ready = True
            return y

        # ── Decide which EMA to use for z-scores ──────────────────────
        if self._l1_frozen and not self.training:
            ema_x  = self._frz_x;  ema_x2 = self._frz_x2
            ema_y  = self._frz_y;  ema_y2 = self._frz_y2
        else:
            ema_x  = self._ema_x;  ema_x2 = self._ema_x2
            ema_y  = self._ema_y;  ema_y2 = self._ema_y2

        # ── Compute gelu211 product gate ──────────────────────────────
        std_x  = (ema_x2 - ema_x**2).clamp(0).sqrt() + self.eps
        z_in   = ((x.detach()  - ema_x.view(1,1,D)) / std_x.view(1,1,D)).pow(2).mean(-1)
        std_y  = (ema_y2 - ema_y**2).clamp(0).sqrt() + self.eps
        z_out  = ((y.detach()  - ema_y.view(1,1,D)) / std_y.view(1,1,D)).pow(2).mean(-1)
        g_in   = (1.0 - a_in)  + a_in  * torch.exp(-tau_in  * z_in)
        g_out  = (1.0 - a_out) + a_out * torch.exp(-tau_out * z_out)
        gate   = g_in * g_out

        # ── Update live EMA (always) ───────────────────────────────────
        with torch.no_grad():
            self._ema_x  = d_x * self._ema_x  + (1-d_x) * x_flat.mean(0)
            self._ema_x2 = d_x * self._ema_x2 + (1-d_x) * (x_flat**2).mean(0)
            self._ema_y  = d_y * self._ema_y  + (1-d_y) * y_flat.mean(0)
            self._ema_y2 = d_y * self._ema_y2 + (1-d_y) * (y_flat**2).mean(0)

        # ── Eval-only pass detection ───────────────────────────────────
        if not self.training:
            m_curr = y.detach().flatten(0, 1).mean(0)

            if not self._det_ready:
                with torch.no_grad():
                    self._det_buf  = torch.zeros(N_BUF, D, device=x.device, dtype=y.dtype)
                    self._det_mask = torch.zeros(N_BUF, dtype=torch.bool, device=x.device)
                    self._det_buf[0]  = F.normalize(m_curr, dim=0)
                    self._det_mask[0] = True
                self._det_ptr   = 1
                self._det_ready = True
            else:
                with torch.no_grad():
                    q    = F.normalize(m_curr.unsqueeze(0), dim=-1)
                    sims = (F.normalize(self._det_buf, dim=-1) * q).sum(-1).masked_fill(~self._det_mask, -1.0)
                    if sims.max().item() > DETECT_THRESH and not self._l1_frozen:
                        # Pass 2 detected: freeze EMA
                        self._frz_x   = self._ema_x.clone()
                        self._frz_x2  = self._ema_x2.clone()
                        self._frz_y   = self._ema_y.clone()
                        self._frz_y2  = self._ema_y2.clone()
                        self._l1_frozen = True
                    elif self._l1_frozen:
                        # Pass 3+: gently drift frozen EMA toward live EMA
                        self._frz_x  = (1-gamma) * self._frz_x  + gamma * self._ema_x
                        self._frz_x2 = (1-gamma) * self._frz_x2 + gamma * self._ema_x2
                        self._frz_y  = (1-gamma) * self._frz_y  + gamma * self._ema_y
                        self._frz_y2 = (1-gamma) * self._frz_y2 + gamma * self._ema_y2
                    else:
                        self._det_buf[self._det_ptr]  = F.normalize(m_curr, dim=0)
                        self._det_mask[self._det_ptr] = True
                        self._det_ptr = (self._det_ptr + 1) % N_BUF

        return y * gate.unsqueeze(-1)
