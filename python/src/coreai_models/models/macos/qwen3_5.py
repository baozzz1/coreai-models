# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.5 text decoder for CoreAI model export.

Hybrid-attention decoder: `full_attention_interval` layers of gated DeltaNet
linear attention for every one layer of gated full attention. Architecture
features:
- Layer types from `config.layer_types`: 18 `linear_attention` + 6 `full_attention`
- Gated DeltaNet: depthwise causal conv1d + gated delta rule + gated RMSNorm
- Gated full attention: `q_proj` emits query and gate interleaved per head
- Partial RoPE on full-attention layers (`partial_rotary_factor` of head_dim)
- `(1 + weight)` RMSNorm scaling
- Interleaved MRoPE collapses to plain RoPE when the three position grids carry
  the same ids, which is what a flat `position_ids` row gives them. Text
  positions are that by construction; image positions are that only because the
  VLM runtime numbers the merged sequence flat, in place of the checkpoint's
  2-D image grid

Runtime state is four tensors: the KV cache for the full-attention layers, plus
a conv state and a recurrent (delta-rule) state for the linear-attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import GatedDeltaUpdate
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFQwen3_5ForConditionalGeneration,
)
from typing_extensions import Self, override

from coreai_models._constants import (
    KEY_CACHE_NAME,
    MAIN_GRAPH_NAME,
    VALUE_CACHE_NAME,
)
from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM, TraceSpec
from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNormGated, RMSNormPlusOne
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA

LINEAR_ATTENTION = "linear_attention"

# Runner-visible names for the two linear-attention states. Neither says "cache":
# the runtime takes a static-shape state with that word in its name for a
# truncatable cache, and these two cannot be rewound to an earlier token.
CONV_STATE_NAME = "convState"
RECURRENT_STATE_NAME = "recurrentState"


def _rotary_dims(config: Qwen3_5TextConfig) -> int:
    """Number of head-dim elements RoPE rotates; the rest pass through."""
    factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
    return int(config.head_dim * factor)


def _linear_layer_indices(config: Qwen3_5TextConfig) -> list[int]:
    """Global layer indices of the linear-attention layers, in order."""
    return [i for i, t in enumerate(config.layer_types) if t == LINEAR_ATTENTION]


def _full_layer_indices(config: Qwen3_5TextConfig) -> list[int]:
    """Global layer indices of the full-attention layers, in order."""
    return [i for i, t in enumerate(config.layer_types) if t != LINEAR_ATTENTION]


class Attention(nn.Module):
    """Gated full attention: the query projection also emits a per-head output gate."""

    def __init__(self, config: Qwen3_5TextConfig, cache_idx: int) -> None:
        super().__init__()
        self.cache_idx = cache_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        bias = config.attention_bias
        # Query and gate share one projection: (..., n_heads, 2 * head_dim).
        self.q_proj = nn.Linear(dim, n_heads * head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        self.q_norm = RMSNormPlusOne(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormPlusOne(head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(is_causal=True)
        self.rope = initialize_rope(
            dims=_rotary_dims(config),
            base=float(resolve_rope_theta(config)),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads, head_dim = self.n_heads, self.n_kv_heads, self.head_dim

        query, gate = torch.chunk(
            self.q_proj(x).reshape(batch_size, query_len, n_heads, head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch_size, query_len, n_heads * head_dim)

        query = self.q_norm(query).permute(0, 2, 1, 3)
        key = self.k_norm(
            self.k_proj(x).reshape(batch_size, query_len, n_kv_heads, head_dim)
        ).permute(0, 2, 1, 3)
        value = (
            self.v_proj(x).reshape(batch_size, query_len, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        )

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query = self.rope(query, position_ids=rope_positions)
        key = self.rope(key, position_ids=rope_positions)

        key, value = cache.update_and_fetch(
            self.cache_idx, offset, key, value, seq_len=seq_len, query_len=query_len
        )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * head_dim)
        )
        return self.o_proj(output * torch.sigmoid(gate))


class GatedDeltaNet(nn.Module):
    """Linear attention: depthwise causal conv1d feeding the gated delta rule.

    Carries two pieces of state. ``conv_state`` holds the ``kernel_size - 1``
    channel values of left context the causal conv needs; ``recurrent_state``
    holds the delta rule's key-value matrix.

    Query and key are projected at the key head count and the recurrence runs at
    the value head count, so a group of value heads shares one key head.
    """

    def __init__(self, config: Qwen3_5TextConfig, cache_idx: int) -> None:
        super().__init__()
        self.cache_idx = cache_idx

        dim = config.hidden_size
        self.n_k_heads = config.linear_num_key_heads
        self.n_v_heads = config.linear_num_value_heads
        self.head_repeat = self.n_v_heads // self.n_k_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.n_k_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        # The conv reads its left context from the state, so it pads nothing itself.
        self.conv_state_len = config.linear_conv_kernel_dim - 1

        self.in_proj_qkv = nn.Linear(dim, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(dim, self.n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(dim, self.n_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=config.linear_conv_kernel_dim,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.n_v_heads))

        self.gated_delta_update = GatedDeltaUpdate()
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        conv_cache: SSMState,
        recurrent_cache: SSMState,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        torch._check_is_size(query_len)

        conv_state = conv_cache.states.narrow(0, self.cache_idx, 1).squeeze(0)
        recurrent_state = recurrent_cache.states.narrow(0, self.cache_idx, 1).squeeze(0)

        # Prepend the cached left context so the causal conv sees it, then keep the
        # tail of that same buffer as the next state.
        conv_input = torch.cat([conv_state, self.in_proj_qkv(x).transpose(1, 2)], dim=-1)
        conv_cache.update_states(
            self.cache_idx, conv_input.narrow(-1, query_len, self.conv_state_len)
        )
        mixed = F.silu(self.conv1d(conv_input)).transpose(1, 2)

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, query_len, self.n_k_heads, self.head_k_dim).permute(
            0, 2, 1, 3
        )
        key = key.reshape(batch_size, query_len, self.n_k_heads, self.head_k_dim).permute(
            0, 2, 1, 3
        )
        value = value.reshape(batch_size, query_len, self.n_v_heads, self.head_v_dim).permute(
            0, 2, 1, 3
        )
        if self.head_repeat > 1:
            # Value head v reads key head v // head_repeat, so each key head is repeated
            # consecutively: k0, k0, k1, k1.
            query = query.repeat_interleave(self.head_repeat, dim=1)
            key = key.repeat_interleave(self.head_repeat, dim=1)

        beta = self.in_proj_b(x).sigmoid().transpose(1, 2)
        # A might overflow to -inf in fp16, so the decay is computed in fp32.
        a = self.in_proj_a(x).float()
        g = (-self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())).transpose(1, 2)

        core_attn_out, new_recurrent_state = self.gated_delta_update(
            query, key, value, g.to(query.dtype), beta, recurrent_state
        )
        recurrent_cache.update_states(self.cache_idx, new_recurrent_state)

        z = self.in_proj_z(x).reshape(batch_size, query_len, self.n_v_heads, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, query_len, self.value_dim)
        return self.out_proj(core_attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, cache_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.is_linear = config.layer_types[layer_idx] == LINEAR_ATTENTION
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config, cache_idx=cache_idx)
        else:
            self.self_attn = Attention(config, cache_idx=cache_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNormPlusOne(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormPlusOne(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache,
        conv_cache: SSMState,
        recurrent_cache: SSMState,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        if self.is_linear:
            r = self.linear_attn(normed, conv_cache, recurrent_cache)
        else:
            r = self.self_attn(normed, position_ids, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)

        # Each layer indexes its own state bank, so linear and full layers are
        # numbered separately from the global layer index.
        linear_count = 0
        full_count = 0
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == LINEAR_ATTENTION:
                cache_idx, linear_count = linear_count, linear_count + 1
            else:
                cache_idx, full_count = full_count, full_count + 1
            layers.append(TransformerBlock(config, layer_idx, cache_idx))
        self.layers = nn.ModuleList(layers)

        self.norm = RMSNormPlusOne(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache,
        conv_cache: SSMState,
        recurrent_cache: SSMState,
    ) -> torch.Tensor:
        return self.forward_embeds(
            self.embed_tokens(input_ids), position_ids, cache, conv_cache, recurrent_cache
        )

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache,
        conv_cache: SSMState,
        recurrent_cache: SSMState,
    ) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache, conv_cache, recurrent_cache)
        return self.norm(h)


class Qwen3_5ForCausalLM(BaseForCausalLM):
    """Engine-compatible Qwen3.5 text decoder.

    The VL checkpoint keeps the text weights under ``model.language_model.``;
    the vision tower and the MTP head are dropped.
    """

    _HF_MODEL_CLASS = HFQwen3_5ForConditionalGeneration

    exports_prefill_graph = True

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if max_context_length is not None:
            config.max_position_embeddings = max_context_length
        if num_layers is not None:
            config.num_hidden_layers = num_layers
            config.layer_types = config.layer_types[:num_layers]
        return config

    @override
    def _init_model(self, config: Qwen3_5TextConfig) -> None:
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor | tuple:
        return self._head(
            self.model(
                input_ids,
                position_ids,
                KVCache(k_cache, v_cache),
                SSMState(conv_state),
                SSMState(recurrent_state),
            )
        )

    def _head(self, out: torch.Tensor) -> torch.Tensor | tuple:
        if self.prefill_mode:
            # A bare `return` causes torch export to trace a leaf node with value
            # `None` rather than having no leaf nodes whatsoever. Remedied with
            # empty tuple.
            return ()
        return self.lm_head(out)

    # ------------------------------------------------------------------
    # Export contract
    # ------------------------------------------------------------------

    @classmethod
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        return {
            MAIN_GRAPH_NAME: (
                KEY_CACHE_NAME,
                VALUE_CACHE_NAME,
                CONV_STATE_NAME,
                RECURRENT_STATE_NAME,
            )
        }

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict]:
        n_full = len(_full_layer_indices(config))
        n_linear = len(_linear_layer_indices(config))
        linear_attn = next(layer.linear_attn for layer in self.model.layers if layer.is_linear)

        input_ids = torch.randint(1, config.vocab_size, (1, spec.query_len), dtype=torch.int32)
        position_ids = (
            torch.arange(spec.query_len + spec.offset, dtype=torch.int32)
            .unsqueeze(0)
            .expand(1, spec.query_len + spec.offset)
        )
        cache_shape = (n_full, 1, config.num_key_value_heads, spec.cache_seq_len, config.head_dim)
        return {
            MAIN_GRAPH_NAME: {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "k_cache": torch.zeros(cache_shape, dtype=target_dtype),
                "v_cache": torch.zeros(cache_shape, dtype=target_dtype),
                "conv_state": torch.zeros(
                    n_linear,
                    1,
                    linear_attn.conv_dim,
                    linear_attn.conv_state_len,
                    dtype=target_dtype,
                ),
                "recurrent_state": torch.zeros(
                    n_linear,
                    1,
                    linear_attn.n_v_heads,
                    linear_attn.head_k_dim,
                    linear_attn.head_v_dim,
                    dtype=target_dtype,
                ),
            }
        }

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict:
        shapes = super().build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
        # Both linear-attention states are fixed-size: the conv state holds the
        # kernel's left context and the recurrent state is a k×v matrix.
        shapes["conv_state"] = None
        shapes["recurrent_state"] = None
        return {MAIN_GRAPH_NAME: shapes}

    @override
    def validate_export_contract(self, reference_inputs: dict, dynamic_shapes: dict) -> None:
        super().validate_export_contract(reference_inputs, dynamic_shapes)
        # Only full-attention layers write the key/value cache. A trace leaves a state
        # nothing writes out of its mutated inputs, and the converter then rejects the
        # declared state names.
        if not _full_layer_indices(self.config):
            raise ValueError(
                f"{type(self).__name__}: layer_types {self.config.layer_types} has no "
                "full-attention layer to write the key/value cache. Export at least "
                "through the first full-attention layer."
            )

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Raw checkpoint keys keep the text weights under "model.language_model.";
        # from_hf_memory_efficient hands them over already under "model.", having
        # stripped the registry prefix.
        for key in list(state_dict):
            if key.startswith("model.language_model."):
                state_dict["model." + key[len("model.language_model.") :]] = state_dict.pop(key)
            elif key.startswith(("model.visual.", "mtp.")):
                del state_dict[key]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


class Qwen3_5ForCausalLMEmbeddings(Qwen3_5ForCausalLM):
    """Qwen3.5 decoder entered at the merged embedding sequence.

    The VLM runtime embeds the prompt itself and splices the vision tower's
    features into the image-token positions, so this entrypoint starts one step
    later than the token-id decoder. The four states, the prefill entrypoint and
    the logits are the same graph.
    """

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor | tuple:
        return self._head(
            self.model.forward_embeds(
                inputs_embeds,
                position_ids,
                KVCache(k_cache, v_cache),
                SSMState(conv_state),
                SSMState(recurrent_state),
            )
        )

    @classmethod
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        return {MAIN_GRAPH_NAME: ("inputs_embeds", "position_ids")}

    @override
    def build_reference_inputs(
        self,
        config,
        target_dtype: torch.dtype,
        spec: TraceSpec,
    ) -> dict[str, dict]:
        inputs = super().build_reference_inputs(config, target_dtype, spec)[MAIN_GRAPH_NAME]
        inputs.pop("input_ids")
        embeds = torch.randn(1, spec.query_len, config.hidden_size, dtype=target_dtype)
        # The dict binds positionally to forward, so the embeddings lead.
        return {MAIN_GRAPH_NAME: {"inputs_embeds": embeds, **inputs}}

    @override
    def build_dynamic_shapes(self, config, spec: TraceSpec) -> dict:
        shapes = super().build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
        # The sequence axis is the same one the token ids moved along; the hidden
        # axis is the model's own width and stays static.
        return {MAIN_GRAPH_NAME: {"inputs_embeds": shapes.pop("input_ids"), **shapes}}
