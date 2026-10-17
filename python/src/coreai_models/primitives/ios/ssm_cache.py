# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Per-layer state slabs for the linear-attention block.

Unlike the KV cache, which grows along the sequence and is written at an offset, both
states a gated delta layer carries are rewritten whole on every call: the conv window is
the last ``kernel_size - 1`` columns of its own input, and the recurrent matrix is a fixed
``[d_k, d_v]`` per head. So there is one index to compute, the layer's, and the update is a
slice assignment at it.
"""

import torch
from torch import nn
from typing_extensions import Self

from coreai_models.primitives._ops import mutable_slice_update


class SSMStateHandler:
    """Holds one state tensor of shape ``(n_layers, batch, *state_dims)``."""

    def __init__(self: Self, n_layers: int) -> None:
        self._states: torch.Tensor | None = None
        self._n_layers = n_layers
        with torch.device("cpu"):
            self._layer_begin = nn.Buffer(
                torch.arange(n_layers, dtype=torch.int32).unsqueeze(1), persistent=False
            )
            self._layer_end = nn.Buffer(
                torch.arange(1, n_layers + 1, dtype=torch.int32).unsqueeze(1), persistent=False
            )

    def register(self: Self, states: torch.Tensor) -> None:
        if states.size(0) != self._n_layers:
            msg = f"state tensor holds {states.size(0)} layers, expected {self._n_layers}"
            raise ValueError(msg)
        self._states = states

    @property
    def states(self) -> torch.Tensor:
        assert self._states is not None, "register the state tensor before using it"
        return self._states

    def fetch(self: Self, layer_idx: int) -> torch.Tensor:
        torch._check_is_size(layer_idx)
        return self.states.narrow(0, layer_idx, 1).squeeze(0)

    def update(self: Self, layer_idx: int, new_state: torch.Tensor) -> None:
        cache = self.states
        torch._check_is_size(layer_idx)
        torch._check(layer_idx < cache.size(0))
        device = new_state.device
        zeros = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(cache.dim() - 1)]
        sizes = [
            torch.tensor((cache.size(i),), dtype=torch.int32, device=device)
            for i in range(1, cache.dim())
        ]
        mutable_slice_update(
            x=cache,
            update=new_state.unsqueeze(0),
            begin=torch.cat([self._layer_begin[layer_idx].to(device), *zeros]),
            end=torch.cat([self._layer_end[layer_idx].to(device), *sizes]),
        )
