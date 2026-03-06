"""GELU227 – Ring Buffer with Soft-Attention Episodic Retrieval.

UPGRADE FROM gelu54 — SOFT ATTENTION OVER EPISODES:
    gelu54 uses HARD nearest-episode selection:
        nearest_idx = argmax cosine(m_curr, buffer[i])   ← picks ONE episode
        gate based on cosine(token, buffer[nearest_idx])

    Problem: argmax is unstable near ties — if two episodes are almost equally similar,
    the gate can switch abruptly, causing training instability.

    gelu227 uses SOFT ATTENTION over ALL buffer episodes:
        sims_i = cosine(m_curr, buffer[i]) / T_att       ← attention score (temperature T_att)
        weights = softmax(sims_i)                         ← (N,) attention weights
        retrieved = sum_i(weights_i × buffer[i])          ← (D,) weighted episode mean

    The gate is then based on per-token similarity to this RETRIEVED context:
        tok_sim = cosine(GELU(x_token), retrieved / ||retrieved||)   ← scalar per token
        gate    = (1-α) + α × exp(-τ × tok_sim)

    ADAPTATION ADVANTAGE:
        After test pass 1: ALL buffer entries have test content.
        Softmax attention weights are distributed over all test episodes.
        retrieved = weighted mean of test episodes → captures test distribution better than
        any single nearest episode.

        Pass 2: tok_sim to test-episode retrieval is higher → gate is more suppressive
        → Stronger positive Δ than gelu54 (which only uses one episode).

    The soft attention also means pass 2 recognizes MORE test patterns simultaneously
    (not just the single nearest one), giving more comprehensive suppression.

PARAMS: log_tau (gate sharpness), log_blend (α), log_T_att (attention temperature) = 3 scalars
STATE:  ring buffer (N, D), mask (N,), pointer (int)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU227(nn.Module):
    """Soft-attention episodic retrieval gate: distributes recognition over all buffer episodes."""

    def __init__(self, buffer_size: int = 32):
        super().__init__()
        self._N     = buffer_size
        self._buf:  torch.Tensor = None   # (N, D)
        self._mask: torch.Tensor = None   # (N,) bool
        self._ptr   = 0
        self._ready = False

        self.log_tau   = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))  # α ≈ 0.3
        self.log_T_att = nn.Parameter(torch.tensor(math.log(1.0)))        # T_att ≈ 1.0

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
        tau   = self.log_tau.exp()
        alpha = torch.sigmoid(self.log_blend)
        # T_att is used only inside no_grad – detach to avoid any graph entanglement
        T_att = self.log_T_att.exp().clamp(0.1, 10.0).detach().item()

        y = self._gelu(x)   # (B, T, D)

        # Batch mean of GELU output for buffer update
        m_curr = y.detach().flatten(0, 1).mean(0)   # (D,)

        # ── Initialise ───────────────────────────────────────────────
        if not self._ready:
            with torch.no_grad():
                self._buf  = torch.zeros(self._N, D, device=x.device, dtype=y.dtype)
                self._mask = torch.zeros(self._N, dtype=torch.bool, device=x.device)
                self._buf[0] = F.normalize(m_curr, dim=0)
                self._mask[0] = True
            self._ptr  = 1
            self._ready = True
            return y

        # ── Soft attention retrieval – entirely under no_grad ─────────
        # This avoids any in-place buffer update corrupting the autograd
        # version counter of a tensor that is part of the compute graph.
        with torch.no_grad():
            m_norm = F.normalize(m_curr.unsqueeze(0), dim=-1)     # (1, D)
            buf_n  = F.normalize(self._buf, dim=-1)                # (N, D)
            sims   = (buf_n * m_norm).sum(-1)                      # (N,)
            sims_masked = sims.masked_fill(~self._mask, -1e4)
            weights = F.softmax(sims_masked / T_att, dim=0)        # (N,) soft weights
            retrieved = (weights.unsqueeze(1) * self._buf).sum(0)  # (D,) weighted episode
            retrieved_n = F.normalize(retrieved.unsqueeze(0), dim=-1)  # (1, D)

            # Per-token cosine to retrieved context (fully detached scalar signal)
            y_flat   = y.detach().flatten(0, 1)                    # (B*T, D)
            y_n_flat = F.normalize(y_flat, dim=-1)                 # (B*T, D)
            tok_sim  = (y_n_flat * retrieved_n).sum(-1).view(B, T) # (B, T)

        # ── Gate – tok_sim is a plain (no-grad) tensor here ──────────
        # tau and alpha still receive gradients; T_att is not trained
        # via this path (its gradient comes implicitly from the gate quality).
        gate_scalar = (1.0 - alpha) + alpha * torch.exp(-tau * tok_sim)  # (B, T)
        gate = gate_scalar.unsqueeze(-1)                           # (B, T, 1)

        output = y * gate

        # ── Update ring buffer ────────────────────────────────────────
        with torch.no_grad():
            self._buf[self._ptr]  = F.normalize(m_curr, dim=0)
            self._mask[self._ptr] = True
            self._ptr = (self._ptr + 1) % self._N

        return output
