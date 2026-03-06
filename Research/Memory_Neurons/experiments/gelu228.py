"""GELU228 – Dual-Buffer: Training-Frozen + Test-Exclusive Gate.

CORE INSIGHT — WHY ALL EMA-BASED ADAPTATION ATTEMPTS FAILED (gelu216-220):
    gelu218, 219, 220 tried to make the gate adapt to test distribution.
    All degraded. Why?
    Because during TRAINING, the same adaptation mechanisms are active → the model
    learns to expect the ADAPTED gate → test pass 1 behaves strangely → worse calibration.

    ROOT CAUSE: No clean separation between "training memory" and "test memory".

gelu228 SOLVES THIS with strict train/test memory separation:
    - TRAIN BUF (N1=16):  Updated ONLY when self.training == True (ring FIFO)
    - TEST BUF (N2=8):    Updated ONLY when self.training == False (ring FIFO)
                          Cleared when first test batch arrives after training

    GATE DURING TRAINING (self.training):
        gate = gelu54-style based on train_buf → model calibrates to training ring buffer
        test_buf is NEVER used → gate behaves as if test_buf doesn't exist

    GATE DURING TEST PASS 1 (first test):
        - test_buf is empty → gate = gelu54-style based on train_buf ONLY
        - Same behavior as training → well-calibrated predictions
        - test_buf accumulates test episode means as batches run

    GATE DURING TEST PASS 2+:
        - test_buf now has test episodes → gate adds EXTRA suppression for test-familiar content
        - gate_test = (1-α_te) + α_te × exp(-τ × max_cos_to_test_buf)
        - gate_train = (1-α_tr) + α_tr × exp(-τ × max_cos_to_train_buf)
        - COMBINED gate = gate_train × gate_test (product → lower gate = more suppression)
        - More suppression for test-familiar content on pass 2 → positive adaptation Δ

    CRITICAL: The model was trained WITHOUT the test gate factor.
    During training: gate = gate_train (test gate = 1 always)
    During test pass 1: gate ≈ gate_train (test gate ≈ 1 since test_buf is building up)
    During test pass 2: gate = gate_train × gate_test < gate_train

    Since gelu54's mechanism shows that MORE suppression on pass 2 gives positive Δ,
    gelu228 should give EVEN STRONGER positive Δ than gelu54.

PARAMS: log_tau, log_blend_train, log_blend_test = 3 scalars
STATE:  train_buf (N1, D), test_buf (N2, D), respective masks and pointers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU228(nn.Module):
    """Dual-buffer gate: training-frozen ring buffer + test-accumulating ring buffer."""

    def __init__(self, train_buffer_size: int = 16, test_buffer_size: int = 8):
        super().__init__()
        self._N_train = train_buffer_size
        self._N_test  = test_buffer_size

        # Train buffer
        self._train_buf:  torch.Tensor = None
        self._train_mask: torch.Tensor = None
        self._train_ptr   = 0

        # Test buffer (empty at inference start)
        self._test_buf:  torch.Tensor = None
        self._test_mask: torch.Tensor = None
        self._test_ptr   = 0
        self._was_training = True   # track transition from train to eval

        self._ready = False

        self.log_tau         = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend_train = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α_tr ≈ 0.3
        self.log_blend_test  = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α_te ≈ 0.3

    def reset_state(self):
        self._train_buf   = None
        self._train_mask  = None
        self._train_ptr   = 0
        self._test_buf    = None
        self._test_mask   = None
        self._test_ptr    = 0
        self._was_training = True
        self._ready        = False

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0/math.pi) * (x + 0.044715 * x.pow(3))))

    def _max_cos_gate(self, y_token: torch.Tensor, buf: torch.Tensor,
                      mask: torch.Tensor, alpha: torch.Tensor,
                      tau: torch.Tensor) -> torch.Tensor:
        """Compute gelu54-style scalar cosine gate for one buffer.
        y_token: (B*T, D),  buf: (N, D),  mask: (N,) bool
        Returns: gate (B*T,)
        """
        BT, D = y_token.shape
        y_n   = F.normalize(y_token, dim=-1)          # (B*T, D)
        buf_n = F.normalize(buf, dim=-1)               # (N, D)

        sims  = (y_n @ buf_n.T)                        # (B*T, N)
        # Mask unfilled slots
        sims.masked_fill_(~mask.unsqueeze(0), -1.0)
        max_sim, _ = sims.max(dim=-1)                  # (B*T,)

        gate = (1.0 - alpha) + alpha * torch.exp(-tau * max_sim)
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        tau          = self.log_tau.exp()
        alpha_train  = torch.sigmoid(self.log_blend_train)
        alpha_test   = torch.sigmoid(self.log_blend_test)

        y = self._gelu(x)   # (B, T, D)
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Detect training → eval transition ─────────────────────────
        is_training = self.training
        if self._was_training and not is_training:
            # Just transitioned to eval: clear test buffer
            with torch.no_grad():
                if self._test_buf is not None:
                    self._test_buf.zero_()
                    self._test_mask.zero_()
                    self._test_ptr = 0
        self._was_training = is_training

        # ── Initialise buffers on first call ──────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._train_buf  = torch.zeros(self._N_train, D, device=x.device, dtype=y.dtype)
                self._train_mask = torch.zeros(self._N_train, dtype=torch.bool, device=x.device)
                self._test_buf   = torch.zeros(self._N_test,  D, device=x.device, dtype=y.dtype)
                self._test_mask  = torch.zeros(self._N_test,  dtype=torch.bool, device=x.device)
                m_n = F.normalize(m_curr, dim=0)
                self._train_buf[0]  = m_n;  self._train_mask[0] = True
                self._train_ptr = 1;  self._test_ptr = 0
            self._ready = True
            return y

        # ── Compute gate ──────────────────────────────────────────────
        y_flat = y.flatten(0, 1).detach()   # (B*T, D)

        # Gate from training buffer (always active)
        gate_train = self._max_cos_gate(y_flat, self._train_buf,
                                         self._train_mask, alpha_train, tau)  # (B*T,)

        # Gate from test buffer (only active when test_buf has entries)
        test_has_data = self._test_mask.any().item()
        if test_has_data and not is_training:
            gate_test = self._max_cos_gate(y_flat, self._test_buf,
                                            self._test_mask, alpha_test, tau)  # (B*T,)
        else:
            gate_test = torch.ones(B * T, device=x.device, dtype=y.dtype)

        # Combined gate (product → more suppression on pass 2)
        gate_combined = (gate_train * gate_test).view(B, T, 1)

        output = y * gate_combined

        # ── Update buffers ───────────────────────────────────────────
        with torch.no_grad():
            m_n = F.normalize(m_curr, dim=0)
            if is_training:
                self._train_buf[self._train_ptr]  = m_n
                self._train_mask[self._train_ptr] = True
                self._train_ptr = (self._train_ptr + 1) % self._N_train
            else:
                self._test_buf[self._test_ptr]  = m_n
                self._test_mask[self._test_ptr] = True
                self._test_ptr = (self._test_ptr + 1) % self._N_test

        return output
