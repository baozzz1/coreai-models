# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.5 for the iOS static-shape export path.

Six of the twenty-four blocks are gated full attention; the other eighteen are linear
attention, and those carry two states the KV cache cannot express: the depthwise conv's
left context and the delta rule's key-value matrix. Both are rewritten whole each call, so
they ride in their own tensors alongside the KV cache.

The recurrence is the chunked form from ``primitives.ios.gated_delta``, one chunk per call.
Query length and chunk size are therefore the same number, and the graph is traced at it —
the Horner chain's length is ``chunk - 1``, a Python constant, so it cannot be left to a
symbolic dimension the way the cache length can.

What the caller owes ``extend``: the ``rope_cos`` / ``rope_sin`` rows for the call's
positions in place of ``position_ids``, the ``conv_cache`` / ``recurrent_cache`` states
zeroed at the start of a session and carried from call to call like the KV cache, and
``IOS_QUERY_LEN`` real tokens per call. The linear-attention blocks have no mask: every
position in the query advances both of their states, a padded one included.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from coreai._compiler.types import AllocationType, HardwareConstraints
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFQwen3_5,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextConfig,
)
from typing_extensions import override

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    KEY_CACHE_INPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    POSITION_IDS_INPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
)
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.gated_delta import ChunkedGatedDelta
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA
from coreai_models.primitives.ios.ssm_cache import SSMStateHandler

LINEAR_ATTENTION = "linear_attention"

ROPE_COS_INPUT_NAME = "rope_cos"
ROPE_SIN_INPUT_NAME = "rope_sin"
CONV_CACHE_INPUT_NAME = "conv_cache"
RECURRENT_CACHE_INPUT_NAME = "recurrent_cache"
CONV_CACHE_OUTPUT_NAME = "new_conv_cache"
RECURRENT_CACHE_OUTPUT_NAME = "new_recurrent_cache"


def rotary_dims(config: Qwen3_5TextConfig) -> int:
    """Head-dim elements RoPE rotates; the rest pass through."""
    return int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Unit-length rows over the head dimension, as the delta rule's kernel does."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


def linear_layer_indices(config: Qwen3_5TextConfig) -> list[int]:
    return [i for i, t in enumerate(config.layer_types) if t == LINEAR_ATTENTION]


def full_layer_indices(config: Qwen3_5TextConfig) -> list[int]:
    return [i for i, t in enumerate(config.layer_types) if t != LINEAR_ATTENTION]


class InterleavedMRoPE(nn.Module):
    """Qwen3.5's three-row rotary embedding.

    The three position rows — time, height, width — are not split into contiguous bands of
    the frequency axis but interleaved along it: frequency ``f`` takes its position from
    row ``f mod 3``. A text prompt passes the same row three times and the result is the
    one-dimensional embedding, at the same cost.

    The rows themselves are an input. Where they come from — a token index, or a grid
    position inside an image — is the caller's business, and computing them needs the
    image layout, which the graph does not have.

    This runs on the host and the graph takes the cos and sin it returns. In the graph the
    table lookup is a composite the runtime resolves with a pass of its own, and that pass
    survives one lookup fed straight from the position input. A second lookup, or one fed a
    slice of that input, crashes delegate specialization on the GPU and on the Neural
    Engine alike.
    """

    def __init__(self, rotary_dims: int, max_positions: int, theta: float) -> None:
        super().__init__()
        self.cache = RoPECache(rotary_dims, max_positions, theta)
        with torch.device("cpu"):
            frequency = torch.arange(rotary_dims) % (rotary_dims // 2)
            sections = torch.stack([(frequency % 3 == s).float() for s in range(3)])
            self.register_buffer("_sections", sections, persistent=False)

    def forward(self, *rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = None, None
        for section, row in enumerate(rows):
            row_cos, row_sin = self.cache.gather_cos_sin(row)
            mask = self._sections[section]
            cos = row_cos * mask if cos is None else cos + row_cos * mask
            sin = row_sin * mask if sin is None else sin + row_sin * mask
        return cos, sin


class Attention(nn.Module):
    """Gated full attention: the query projection also emits a per-head output gate."""

    def __init__(self, config: Qwen3_5TextConfig, cache_idx: int) -> None:
        super().__init__()
        self.cache_idx = cache_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dims = rotary_dims(config)

        bias = config.attention_bias
        self.q_proj = nn.Conv2d(dim, self.n_heads * self.head_dim * 2, 1, bias=bias)
        self.k_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, 1, bias=bias)
        self.v_proj = nn.Conv2d(dim, self.n_kv_heads * self.head_dim, 1, bias=bias)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, dim, 1, bias=bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sdpa = SDPA(head_dim=self.head_dim)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rotated = apply_rope(x.narrow(-1, 0, self.rotary_dims), cos, sin)
        pass_through = x.narrow(-1, self.rotary_dims, self.head_dim - self.rotary_dims)
        return torch.cat([rotated, pass_through], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler,
    ) -> torch.Tensor:
        batch, query_len, _, _ = x.shape
        head_dim, n_heads, n_kv = self.head_dim, self.n_heads, self.n_kv_heads

        x = x.transpose(-3, -1)
        # Query and gate share one projection, interleaved per head.
        query, gate = torch.chunk(
            self.q_proj(x).transpose(-3, -1).reshape(batch, query_len, n_heads, head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch, query_len, 1, n_heads * head_dim).transpose(-3, -1)

        query = self._rope(self.q_norm(query).transpose(-2, -3), rope_cos, rope_sin)
        key = self._rope(
            self.k_norm(
                self.k_proj(x).transpose(-3, -1).reshape(batch, query_len, n_kv, head_dim)
            ).transpose(-2, -3),
            rope_cos,
            rope_sin,
        )

        query = (
            query.transpose(-2, -3)
            .reshape(batch, query_len, 1, n_heads * head_dim)
            .transpose(-3, -1)
        )
        key = key.transpose(-2, -3).reshape(batch, query_len, 1, n_kv * head_dim).transpose(-3, -1)
        value = self.v_proj(x)

        key, value = cache.update_and_fetch(self.cache_idx, in_step, key, value, query_len)
        output = self.sdpa(query, key, value, causal_mask) * torch.sigmoid(gate)
        return self.o_proj(output).transpose(-3, -1)


class GatedDeltaNet(nn.Module):
    """Linear attention: depthwise causal conv feeding the chunked gated delta rule.

    Query and key are projected at the key head count and the recurrence runs at the value
    head count, so a group of value heads shares one key head.
    """

    def __init__(self, config: Qwen3_5TextConfig, cache_idx: int, chunk_size: int) -> None:
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
        self.conv_state_len = config.linear_conv_kernel_dim - 1

        self.in_proj_qkv = nn.Conv2d(dim, self.conv_dim, 1, bias=False)
        self.in_proj_z = nn.Conv2d(dim, self.value_dim, 1, bias=False)
        self.in_proj_b = nn.Conv2d(dim, self.n_v_heads, 1, bias=False)
        self.in_proj_a = nn.Conv2d(dim, self.n_v_heads, 1, bias=False)
        # The conv reads its left context from the state, so it pads nothing itself.
        self.conv1d = nn.Conv2d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=(1, config.linear_conv_kernel_dim),
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.n_v_heads))

        self.gated_delta = ChunkedGatedDelta(chunk_size)
        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Conv2d(self.value_dim, dim, 1, bias=False)

    def _heads(self, x: torch.Tensor, n_heads: int, head_dim: int, seq: int) -> torch.Tensor:
        """``(b, heads * dim, 1, s)`` to ``(b, heads, s, dim)``."""
        return x.reshape(x.shape[0], n_heads, head_dim, seq).transpose(-1, -2)

    def forward(
        self,
        x: torch.Tensor,
        conv_cache: SSMStateHandler,
        recurrent_cache: SSMStateHandler,
    ) -> torch.Tensor:
        batch, query_len, _, _ = x.shape
        x = x.transpose(-3, -1)

        conv_input = torch.cat([conv_cache.fetch(self.cache_idx), self.in_proj_qkv(x)], dim=-1)
        conv_cache.update(self.cache_idx, conv_input.narrow(-1, query_len, self.conv_state_len))
        mixed = F.silu(self.conv1d(conv_input))

        query, key, value = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=1)
        query = self._heads(query, self.n_k_heads, self.head_k_dim, query_len)
        key = self._heads(key, self.n_k_heads, self.head_k_dim, query_len)
        value = self._heads(value, self.n_v_heads, self.head_v_dim, query_len)

        # The rule normalises the rows of q and k to unit length before the recurrence;
        # q additionally carries the attention scale.
        query = l2_normalize(query) * self.head_k_dim**-0.5
        key = l2_normalize(key)
        if self.head_repeat > 1:
            # Value head v reads key head v // head_repeat, so each key head is repeated
            # consecutively: k0, k0, k1, k1. The norm and the scale act along the head
            # dimension, which makes a copy of a normalized head the normalized copy.
            query = query.repeat_interleave(self.head_repeat, dim=-3)
            key = key.repeat_interleave(self.head_repeat, dim=-3)

        beta = torch.sigmoid(self.in_proj_b(x)).reshape(batch, self.n_v_heads, query_len)
        a = self.in_proj_a(x).reshape(batch, self.n_v_heads, query_len)
        log_g = -torch.exp(self.A_log).unsqueeze(-1) * F.softplus(a + self.dt_bias.unsqueeze(-1))

        out, new_state = self.gated_delta(
            query, key, value, log_g, beta, recurrent_cache.fetch(self.cache_idx)
        )
        recurrent_cache.update(self.cache_idx, new_state)

        gate = self._heads(self.in_proj_z(x), self.n_v_heads, self.head_v_dim, query_len)
        out = self.norm(out) * F.silu(gate)
        out = out.transpose(-1, -2).reshape(batch, self.value_dim, 1, query_len)
        return self.out_proj(out).transpose(-3, -1)


class TransformerBlock(nn.Module):
    def __init__(
        self, config: Qwen3_5TextConfig, layer_idx: int, cache_idx: int, chunk_size: int
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.is_linear = config.layer_types[layer_idx] == LINEAR_ATTENTION
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config, cache_idx, chunk_size)
        else:
            self.self_attn = Attention(config, cache_idx)
        self.mlp = MLP(dim=hidden, hidden_dim=config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler,
        conv_cache: SSMStateHandler,
        recurrent_cache: SSMStateHandler,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        if self.is_linear:
            r = self.linear_attn(normed, conv_cache, recurrent_cache)
        else:
            r = self.self_attn(normed, rope_cos, rope_sin, in_step, causal_mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, chunk_size: int) -> None:
        super().__init__()
        cache_idx = dict.fromkeys(range(config.num_hidden_layers), 0)
        for order, indices in (
            (0, full_layer_indices(config)),
            (1, linear_layer_indices(config)),
        ):
            del order
            for slot, layer in enumerate(indices):
                cache_idx[layer] = slot
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer, cache_idx[layer], chunk_size)
                for layer in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler,
        conv_cache: SSMStateHandler,
        recurrent_cache: SSMStateHandler,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, rope_cos, rope_sin, in_step, causal_mask, cache, conv_cache, recurrent_cache
            )
        return self.norm(x)


class Qwen3_5Extend(nn.Module):
    """The transformer contract: one call over a fixed query length, four states."""

    def __init__(self, config: Qwen3_5TextConfig, chunk_size: int) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config, chunk_size)
        self.prefill_mode = False

        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

        n_full = len(full_layer_indices(config))
        self.kv_cache = KVCacheHandler(n_full, config.num_key_value_heads * config.head_dim)
        n_linear = len(linear_layer_indices(config))
        self.conv_cache = SSMStateHandler(n_linear)
        self.recurrent_cache = SSMStateHandler(n_linear)

    def forward(  # noqa: PLR0913
        self,
        transformer_input: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_cache: torch.Tensor,
        recurrent_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        self.conv_cache.register(conv_cache)
        self.recurrent_cache.register(recurrent_cache)

        batch, query_len, _, hidden = transformer_input.shape
        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            self.kv_cache,
            self.conv_cache,
            self.recurrent_cache,
        )
        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            return self.lm_head(out.transpose(-2, -3))

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
            )
        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )
        out = out.transpose(-3, -1).reshape(batch, 1, hidden, query_len)
        return (embedding_table @ out).transpose(-2, -1)


class Qwen3_5ForCausalLMForiOS(BaseForCausalLMForiOS):
    """Qwen3.5 for iOS.

    The query length is the chunk size the linear-attention blocks run at, so unlike the
    dense models this one is specialized over cache length alone.
    """

    _HF_MODEL_CLASS = HFQwen3_5

    #: Tokens per call, and the chunk the gated delta recurrence runs as one block.
    IOS_QUERY_LEN = 16
    IOS_STATIC_QUERY_LENS = (16,)

    #: Norms Qwen3.5 stores as an offset from one.
    _PLUS_ONE_NORMS = (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    )

    @classmethod
    def _get_reauthored_config(
        cls, hf_config, max_context_length: int | None = None, num_layers: int | None = None
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
        self.extend = Qwen3_5Extend(config, self.IOS_QUERY_LEN)

    def forward(  # noqa: PLR0913
        self,
        input_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_cache: torch.Tensor,
        recurrent_cache: torch.Tensor,
    ) -> torch.Tensor:
        table = self.load_embeddings.embedding_table
        return self.extend(
            self.gather_embeddings(input_ids, table),
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            conv_cache,
            recurrent_cache,
            table,
        )

    # ------------------------------------------------------------------
    # Export contract
    # ------------------------------------------------------------------

    @classmethod
    @override
    def export_input_names(cls) -> dict[str, tuple[str, ...]]:
        names = super().export_input_names()
        inputs = list(names[EXTEND_FUNCTION_NAME])
        at = inputs.index(POSITION_IDS_INPUT_NAME)
        names[EXTEND_FUNCTION_NAME] = tuple(
            inputs[:at] + [ROPE_COS_INPUT_NAME, ROPE_SIN_INPUT_NAME] + inputs[at + 1 :]
        )
        return names

    @classmethod
    @override
    def export_state_names(cls) -> dict[str, tuple[str, ...]]:
        names = super().export_state_names()
        names[EXTEND_FUNCTION_NAME] = (
            *names[EXTEND_FUNCTION_NAME],
            CONV_CACHE_INPUT_NAME,
            RECURRENT_CACHE_INPUT_NAME,
        )
        return names

    @classmethod
    @override
    def export_state_output_names(cls) -> dict[str, tuple[str, ...]]:
        names = super().export_state_output_names()
        names[EXTEND_FUNCTION_NAME] = (
            *names[EXTEND_FUNCTION_NAME],
            CONV_CACHE_OUTPUT_NAME,
            RECURRENT_CACHE_OUTPUT_NAME,
        )
        return names

    @classmethod
    def _ssm_shapes(cls, config) -> tuple[tuple[int, ...], tuple[int, ...]]:
        n_linear = len(linear_layer_indices(config))
        conv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim + (
            config.linear_num_value_heads * config.linear_value_head_dim
        )
        conv = (n_linear, 1, conv_dim, 1, config.linear_conv_kernel_dim - 1)
        recurrent = (
            n_linear,
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )
        return conv, recurrent

    @override
    def build_reference_inputs(self, config, target_dtype, spec):
        query_len = self.IOS_QUERY_LEN
        max_ctx = spec.max_context_length
        head_dim = self._head_dim(config)
        table = self.load_embeddings.embedding_table
        # `target_dtype` is read off the first parameter, which is the int8 embedding
        # table. The states carry activations, so they take the transformer's dtype.
        cache_dtype = self.extend.model.norm.weight.dtype
        rotary = rotary_dims(config)
        input_ids = torch.randint(1, config.vocab_size, (1, query_len), dtype=torch.int32)
        key_cache = torch.zeros(
            len(full_layer_indices(config)),
            1,
            config.num_key_value_heads * head_dim,
            1,
            max_ctx,
            dtype=cache_dtype,
        )
        conv_shape, recurrent_shape = self._ssm_shapes(config)
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {
                "input_ids": input_ids,
                "embedding_table": table,
            },
            EXTEND_FUNCTION_NAME: {
                "transformer_input": self.gather_embeddings(input_ids, table),
                # Time, height and width end to end. Equal rows are the text case.
                "rope_cos": torch.zeros(1, query_len, rotary, dtype=cache_dtype),
                "rope_sin": torch.zeros(1, query_len, rotary, dtype=cache_dtype),
                "in_step": torch.zeros((1,), dtype=torch.int32),
                "causal_mask": torch.zeros(1, max_ctx, 1, query_len, dtype=cache_dtype),
                "key_cache": key_cache,
                "value_cache": key_cache.clone(),
                "conv_cache": torch.zeros(conv_shape, dtype=cache_dtype),
                "recurrent_cache": torch.zeros(recurrent_shape, dtype=cache_dtype),
                "embedding_table": table,
            },
        }

    @override
    def build_dynamic_shapes(self, config, spec):
        cache_len = torch.export.Dim("cache_len", max=spec.max_context_length)
        # The gather keeps a symbolic sequence length: the fused embedding gather is a
        # template op whose signature is written against one, and it holds no chunked
        # arithmetic that would need the length as a constant.
        seq_len = torch.export.Dim("seq_len", max=spec.max_context_length)
        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {"input_ids": {1: seq_len}, "embedding_table": None},
            EXTEND_FUNCTION_NAME: {
                "transformer_input": None,
                "rope_cos": None,
                "rope_sin": None,
                "in_step": None,
                "causal_mask": {1: cache_len},
                "key_cache": {4: cache_len},
                "value_cache": {4: cache_len},
                "conv_cache": None,
                "recurrent_cache": None,
                "embedding_table": None,
            },
        }

    @override
    def validate_export_contract(self, reference_inputs: dict, dynamic_shapes: dict) -> None:
        super().validate_export_contract(reference_inputs, dynamic_shapes)
        # Only full-attention layers write the key/value cache. A trace leaves a state
        # nothing writes out of its mutated inputs, and the converter then rejects the
        # declared state names; the prefill entry cannot even read the empty cache.
        if not full_layer_indices(self.config):
            raise ValueError(
                f"{type(self).__name__}: layer_types {self.config.layer_types} has no "
                "full-attention layer to write the key/value cache. Export at least "
                "through the first full-attention layer."
            )

    @classmethod
    @override
    def export_static_shape_configs(cls, config, max_context_length: int):
        query_len = cls.IOS_QUERY_LEN
        kv_embed = config.num_key_value_heads * cls._head_dim(config)
        n_full = len(full_layer_indices(config))
        conv_shape, recurrent_shape = cls._ssm_shapes(config)

        transformer = {}
        cache_len = cls.IOS_STATIC_MIN_CACHE_LEN
        while cache_len <= max_context_length:
            transformer[f'"{cache_len}_{query_len}"'] = {
                TRANSFORMER_INPUT_NAME: (1, query_len, 1, config.hidden_size),
                ROPE_COS_INPUT_NAME: (1, query_len, rotary_dims(config)),
                ROPE_SIN_INPUT_NAME: (1, query_len, rotary_dims(config)),
                CAUSAL_MASK_INPUT_NAME: (1, cache_len, 1, query_len),
                KEY_CACHE_INPUT_NAME: (n_full, 1, kv_embed, 1, cache_len),
                VALUE_CACHE_INPUT_NAME: (n_full, 1, kv_embed, 1, cache_len),
                CONV_CACHE_INPUT_NAME: conv_shape,
                RECURRENT_CACHE_INPUT_NAME: recurrent_shape,
            }
            cache_len *= 2

        return {
            LOAD_EMBEDDINGS_FUNCTION_NAME: {},
            GATHER_EMBEDDINGS_FUNCTION_NAME: {
                f'"{query_len}"': {TOKEN_IDS_INPUT_NAME: (1, query_len)}
            },
            EXTEND_FUNCTION_NAME: transformer,
        }

    @classmethod
    @override
    def export_hardware_constraints(cls, max_context_length: int):
        constraints = super().export_hardware_constraints(max_context_length)
        slab = HardwareConstraints(AllocationType.IOSurface, interleave=[1] * 5, alignments=[1] * 6)
        for name in (
            CONV_CACHE_INPUT_NAME,
            RECURRENT_CACHE_INPUT_NAME,
            CONV_CACHE_OUTPUT_NAME,
            RECURRENT_CACHE_OUTPUT_NAME,
        ):
            constraints[EXTEND_FUNCTION_NAME][name] = slab
        return constraints

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        # The VL checkpoint keeps the text weights under "model.language_model.".
        for key in list(state_dict):
            if key.startswith("model.language_model."):
                state_dict["model." + key[len("model.language_model.") :]] = state_dict.pop(key)
            elif key.startswith(("model.visual.", "mtp.")):
                del state_dict[key]

        # Qwen3.5 stores these norms as an offset from one; folding the one in here keeps
        # an add per norm per token out of the graph.
        for key in list(state_dict):
            if key == "model.norm.weight" or any(
                key.endswith(f"{name}.weight") for name in self._PLUS_ONE_NORMS
            ):
                state_dict[key] = state_dict[key] + 1.0

        for key in list(state_dict):
            if key.endswith("conv1d.weight"):
                # Depthwise over the sequence: [channels, 1, taps] to a 1xtaps Conv2d.
                state_dict[key] = state_dict[key].unsqueeze(-2)
            elif (
                key.endswith(".weight")
                and state_dict[key].dim() == 2
                and not key.endswith("lm_head.weight")
                and "embed_tokens" not in key
            ):
                # Every remaining rank-2 weight belongs to a 1x1 Conv2d.
                state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)

        table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
        if self.disable_embedding_quantization:
            scale = torch.tensor(1.0, dtype=table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)
        else:
            table, scale, zero_point = quantize_per_tensor(table, nbits=8, symmetric=True)
        state_dict["load_embeddings.embedding_table"] = table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point
        state_dict.pop("model.embed_tokens.weight")

        moved = {}
        for key in list(state_dict):
            if key.startswith("model.") and "gather_embeddings" not in key:
                moved[f"extend.{key}"] = state_dict.pop(key)
        state_dict.update(moved)

        if not self.config.tie_word_embeddings:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]
        state_dict.pop("lm_head.weight", None)
