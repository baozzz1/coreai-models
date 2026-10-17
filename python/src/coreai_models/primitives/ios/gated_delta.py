# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Chunked form of the gated delta recurrence.

The recurrence carries a ``[d_v, d_k]`` state per head::

    S_t = g_t S_{t-1} + k_t u_t^T,   u_t = b_t (v_t - (g_t S_{t-1})^T k_t),   y_t = S_t^T q_t

Written step by step it is a loop, and a loop region is scheduled on the CPU. Written
per chunk it is straight-line matrix arithmetic that the Neural Engine takes whole.

Within a chunk, substituting the unrolled state into its own update turns the sequence of
``u_t`` into one triangular system::

    (I + L) U = M,    L = diag(beta) (R ⊙ K K^T ⊙ strict_lower),
                      M = diag(beta) (V - K (diag(A) S))

where ``A_t`` is the decay accumulated from the chunk's first token and ``R[t, j] = A_t / A_j``.
``L`` is strictly lower triangular, so ``L^C = 0`` and the Neumann series
``U = Σ (-L)^n M`` terminates: ``chunk_size - 1`` Horner steps solve the system exactly, and
fewer is a truncation whose error this module's callers are expected to have measured.

Every decay factor used here is a ratio over a window that ends no later than it starts,
so all of them are at most 1. Folding the decay into ``K`` instead — ``k_j / A_j``, the usual
chunked formulation — would divide by an accumulated decay and leave fp16 immediately.
"""

import torch
from torch import nn


class ChunkedGatedDelta(nn.Module):
    """Gated delta recurrence over fixed-size chunks.

    Args:
        chunk_size: tokens per chunk. The sequence length must be a multiple of it.
        order: Horner steps per chunk. ``None`` means ``chunk_size - 1``, which solves the
            triangular system exactly; a smaller value truncates the Neumann series.
    """

    def __init__(self, chunk_size: int, order: int | None = None) -> None:
        super().__init__()
        if chunk_size < 1:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        self.chunk_size = chunk_size
        self.order = chunk_size - 1 if order is None else order
        if not 0 <= self.order <= chunk_size - 1:
            msg = f"order must be in [0, {chunk_size - 1}], got {self.order}"
            raise ValueError(msg)

        # Built on the CPU whatever device the model is being assembled on: these are
        # constants no checkpoint carries, so a meta device would leave them unfilled.
        with torch.device("cpu"):
            c = chunk_size
            lower = torch.tril(torch.ones(c, c))
            self.register_buffer("_prefix_sum", lower, persistent=False)
            self.register_buffer("_lower_incl", lower, persistent=False)
            self.register_buffer(
                "_lower_strict", torch.tril(torch.ones(c, c), -1), persistent=False
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence.

        Args:
            query, key: ``[b, h, t, d_k]``.
            value: ``[b, h, t, d_v]``.
            log_g: ``[b, h, t]``, the log of the per-step decay.
            beta: ``[b, h, t]``.
            state: ``[b, h, d_k, d_v]``. Held key-major so the two reads of it are
                plain matmuls; a transposed state would put a rank-4 transpose of the
                largest tensor in the layer in front of them.

        Returns:
            ``[b, h, t, d_v]`` outputs and the state after the last token.
        """
        chunk = self.chunk_size
        seq_len = query.shape[-2]
        if seq_len % chunk:
            msg = f"sequence length {seq_len} is not a multiple of chunk size {chunk}"
            raise ValueError(msg)

        dtype = query.dtype
        prefix_sum = self._prefix_sum.to(dtype)
        lower_incl = self._lower_incl.to(dtype)
        lower_strict = self._lower_strict.to(dtype)
        zero = torch.zeros((), dtype=dtype, device=query.device)

        outputs = []
        for start in range(0, seq_len, chunk):
            stop = start + chunk
            q = query[..., start:stop, :]
            k = key[..., start:stop, :]
            v = value[..., start:stop, :]
            lg = log_g[..., start:stop].unsqueeze(-1)
            b = beta[..., start:stop].unsqueeze(-1)

            # Decay accumulated from the chunk's first token, as a matmul against a
            # constant lower-triangular matrix rather than a scan.
            cum = prefix_sum @ lg
            # ratio[t, j] = A_t / A_j, clamped at 1 so the masked-out upper half cannot
            # overflow on its way to being multiplied by zero.
            ratio = torch.exp(torch.minimum(cum - cum.transpose(-1, -2), zero))
            decay = torch.exp(cum)
            tail = torch.exp(cum[..., -1:, :] - cum)

            # The decay rides on q and k rather than on the products, so the two reads of
            # the state stay plain matmuls.
            decayed_k = decay * k
            decayed_q = decay * q
            lhs = b * (ratio * (k @ k.transpose(-1, -2)) * lower_strict)
            rhs = b * (v - decayed_k @ state)

            weights = rhs
            for _ in range(self.order):
                weights = rhs - lhs @ weights

            attn = ratio * (q @ k.transpose(-1, -2)) * lower_incl
            outputs.append(decayed_q @ state + attn @ weights)
            state = decay[..., -1:, :] * state + (tail * k).transpose(-1, -2) @ weights

        return torch.cat(outputs, dim=-2), state
