# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.5 iOS static-shape variant: numerics, carried states and the export contract.

A tiny random checkpoint in the Hugging Face layout -- text weights under
``model.language_model.``, beside a vision tower -- stands in for the real one. The
numerical tests run it at fp32 through ``transformers`` and through the iOS class and
compare; none of these need the Core AI runtime, they stop before the converter.
"""

import inspect
from typing import NamedTuple

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from coreai_models._constants import (
    EXTEND_FUNCTION_NAME,
    KEY_CACHE_INPUT_NAME,
    PROMPT_OPT_FUNCTION_NAME,
    VALUE_CACHE_INPUT_NAME,
)
from coreai_models._hf import resolve_rope_theta
from coreai_models.export import pipeline as export_pipeline
from coreai_models.models.base import TraceSpec
from coreai_models.models.ios.qwen3_5 import (
    CONV_CACHE_INPUT_NAME,
    RECURRENT_CACHE_INPUT_NAME,
    InterleavedMRoPE,
    Qwen3_5ForCausalLMForiOS,
    rotary_dims,
)
from coreai_models.models.registry import get_model_entry
from coreai_models.primitives.ios.ssm_cache import SSMStateHandler

MAX_CTX = 256
# Three linear-attention layers, then the first full-attention one.
NUM_LAYERS = 4
TEXT_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "vocab_size": 128,
    "max_position_embeddings": MAX_CTX,
    "tie_word_embeddings": True,
}

# Key and value head counts for the linear-attention layers: one head each, then a group
# of value heads per key head. Qwen3.5-0.8B and 2B are the first; 4B is the second.
HEAD_COUNTS = [(4, 4), (2, 4)]
HEAD_COUNT_IDS = ["one-to-one", "grouped"]

#: ``max |got - reference| / max |reference|``. Both sides run the same weights in fp32,
#: so the only gap is summation order; it measures around 2e-7 on this checkpoint.
REL_TOLERANCE = 5e-6

QUERY_LEN = Qwen3_5ForCausalLMForiOS.IOS_QUERY_LEN


def text_config(num_layers: int | None = None, **overrides) -> Qwen3_5TextConfig:
    return Qwen3_5ForCausalLMForiOS._get_reauthored_config(
        Qwen3_5TextConfig(**{**TEXT_CONFIG, **overrides}), MAX_CTX, num_layers=num_layers
    )


def write_checkpoint(path, config: dict) -> str:
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=config,
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": config["hidden_size"],
            },
            tie_word_embeddings=True,
        )
    )
    # The default init decays the recurrent state by orders of magnitude per token, which
    # leaves nothing of it for a later call to read. A slow decay makes the carried state
    # carry.
    for layer in model.model.language_model.layers:
        if hasattr(layer, "linear_attn"):
            with torch.no_grad():
                layer.linear_attn.A_log.uniform_(-4, -1)
    model.save_pretrained(path)
    return str(path)


class Pair(NamedTuple):
    """One checkpoint through the reference implementation and the iOS class."""

    reference: Qwen3_5ForConditionalGeneration
    ios: Qwen3_5ForCausalLMForiOS


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Loads and caches a ``Pair`` per linear-attention key/value head count."""
    loaded: dict[tuple[int, int], Pair] = {}

    def load(n_k_heads: int, n_v_heads: int) -> Pair:
        if (n_k_heads, n_v_heads) not in loaded:
            path = write_checkpoint(
                tmp_path_factory.mktemp(f"qwen3_5_{n_k_heads}_{n_v_heads}"),
                {
                    **TEXT_CONFIG,
                    "linear_num_key_heads": n_k_heads,
                    "linear_num_value_heads": n_v_heads,
                },
            )
            loaded[n_k_heads, n_v_heads] = Pair(
                Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval(),
                Qwen3_5ForCausalLMForiOS.from_hf(
                    path,
                    max_context_length=MAX_CTX,
                    target_dtype=torch.float32,
                    disable_embedding_quantization=True,
                ).eval(),
            )
        return loaded[n_k_heads, n_v_heads]

    return load


def assert_matches(got: torch.Tensor, reference: torch.Tensor, what: str) -> None:
    """Relative-error check that fails rather than pass when there is nothing to compare."""
    assert tuple(got.shape) == tuple(reference.shape), (
        f"{what}: shape {tuple(got.shape)} against {tuple(reference.shape)}"
    )
    assert torch.isfinite(got).all(), f"{what}: result is not finite"
    assert torch.isfinite(reference).all(), f"{what}: reference is not finite"
    scale = reference.abs().max()
    assert scale > 0, f"{what}: reference is all zero"
    error = (got - reference).abs().max() / scale
    assert error <= REL_TOLERANCE, f"{what}: relative error {error:.2e} over {REL_TOLERANCE:.0e}"


def hidden_states(seq_len: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, seq_len, TEXT_CONFIG["hidden_size"])


def token_ids(seq_len: int) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(1, TEXT_CONFIG["vocab_size"], (1, seq_len), dtype=torch.int32)


def reference_linear_attn(model, hidden: torch.Tensor) -> torch.Tensor:
    """The first linear-attention layer of the reference model, uncached."""
    with torch.no_grad():
        return model.model.language_model.layers[0].linear_attn(hidden)


def ios_linear_attn(model, hidden: torch.Tensor, chunk: int) -> torch.Tensor:
    """The first linear-attention layer of the iOS variant, ``chunk`` tokens per call.

    Both states start zeroed and are carried from one call to the next, which is the
    contract the exported graph is written against.
    """
    layer = model.extend.model.layers[0].linear_attn
    conv = SSMStateHandler(1)
    conv.register(torch.zeros(1, 1, layer.conv_dim, 1, layer.conv_state_len))
    recurrent = SSMStateHandler(1)
    recurrent.register(torch.zeros(1, 1, layer.n_v_heads, layer.head_k_dim, layer.head_v_dim))
    outputs = []
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk):
            step = hidden[:, start : start + chunk].unsqueeze(-2)
            outputs.append(layer(step, conv, recurrent).squeeze(-2))
    return torch.cat(outputs, dim=1)


def ios_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """``IOS_QUERY_LEN`` tokens per call, every state carried across the calls."""
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX, query_len=QUERY_LEN)
    inputs = model.build_reference_inputs(model.config, torch.float32, spec)[EXTEND_FUNCTION_NAME]
    table = model.load_embeddings.embedding_table
    inputs["embedding_table"] = table
    rope = InterleavedMRoPE(
        rotary_dims(model.config),
        model.config.max_position_embeddings,
        float(resolve_rope_theta(model.config)),
    ).eval()
    cache_positions = torch.arange(MAX_CTX).reshape(1, MAX_CTX, 1, 1)

    outputs = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[1], QUERY_LEN):
            positions = torch.arange(start, start + QUERY_LEN, dtype=torch.int32).unsqueeze(0)
            # A text prompt sends the same row as time, height and width.
            cos, sin = rope(positions, positions, positions)
            inputs["transformer_input"] = model.gather_embeddings(
                input_ids[:, start : start + QUERY_LEN], table
            )
            inputs["rope_cos"], inputs["rope_sin"] = cos.float(), sin.float()
            inputs["in_step"] = torch.tensor([start], dtype=torch.int32)
            inputs["causal_mask"] = torch.where(
                cache_positions <= positions.reshape(1, 1, 1, QUERY_LEN), 0.0, float("-inf")
            )
            outputs.append(model.extend(**inputs).reshape(1, QUERY_LEN, -1))
    return torch.cat(outputs, dim=1)


def build_contract(model, config):
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX)
    return (
        model.build_reference_inputs(config, torch.float16, spec),
        model.build_dynamic_shapes(config, spec),
    )


def test_registry_maps_dense_qwen3_5_to_the_ios_class() -> None:
    assert get_model_entry("qwen3_5").ios_class is Qwen3_5ForCausalLMForiOS
    # The iOS class has no expert routing.
    assert get_model_entry("qwen3_5_moe").ios_class is None


@pytest.mark.parametrize(("n_k_heads", "n_v_heads"), HEAD_COUNTS, ids=HEAD_COUNT_IDS)
class TestGatedDeltaNet:
    """The linear-attention layer against the reference one, same weights, fp32."""

    def test_matches_the_reference(self, pair, n_k_heads, n_v_heads) -> None:
        models = pair(n_k_heads, n_v_heads)
        hidden = hidden_states(QUERY_LEN)

        assert_matches(
            ios_linear_attn(models.ios, hidden, QUERY_LEN),
            reference_linear_attn(models.reference, hidden),
            "iOS gated delta net",
        )

    def test_carries_its_states_between_calls(self, pair, n_k_heads, n_v_heads) -> None:
        """Two calls of the fixed query length cover what the reference does in one pass.

        The conv state has to hand the second call its left context and the recurrent
        state the delta rule's matrix, or the second half drifts.
        """
        models = pair(n_k_heads, n_v_heads)
        hidden = hidden_states(2 * QUERY_LEN)

        assert_matches(
            ios_linear_attn(models.ios, hidden, QUERY_LEN),
            reference_linear_attn(models.reference, hidden),
            "iOS gated delta net over two calls",
        )


@pytest.mark.parametrize(("n_k_heads", "n_v_heads"), HEAD_COUNTS, ids=HEAD_COUNT_IDS)
def test_logits_match_the_reference(pair, n_k_heads, n_v_heads) -> None:
    """A fixed number of tokens per call, so a 32-token prompt is two calls carrying the
    key/value cache and both linear-attention states, through a layer stack that reaches
    the first full-attention layer."""
    models = pair(n_k_heads, n_v_heads)
    ids = token_ids(2 * QUERY_LEN)

    with torch.no_grad():
        reference = models.reference(input_ids=ids.long()).logits

    assert_matches(ios_logits(models.ios, ids), reference, "iOS logits")


class TestExportNeedsAFullAttentionLayer:
    """Only full-attention layers write the key/value cache, which the contract declares."""

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_a_linear_attention_only_stack_is_rejected(self, num_layers) -> None:
        config = text_config(num_layers)
        model = Qwen3_5ForCausalLMForiOS(config, model_device="cpu").to(torch.float16).eval()
        # Eager use of the short stack still works; only export is refused.
        built = build_contract(model, config)
        with pytest.raises(ValueError, match="no full-attention layer"):
            model.validate_export_contract(*built)

    def test_a_stack_through_the_first_full_attention_layer_validates(self) -> None:
        config = text_config(NUM_LAYERS)
        model = Qwen3_5ForCausalLMForiOS(config, model_device="cpu").to(torch.float16).eval()
        model.validate_export_contract(*build_contract(model, config))

    def test_traces_write_every_declared_state(self, disable_hf_impl_for_coreai) -> None:
        from coreai_models.export.ios import _export_programs

        config = text_config(NUM_LAYERS)
        model = Qwen3_5ForCausalLMForiOS(config, model_device="cpu").to(torch.float16).eval()
        programs = _export_programs(model, *build_contract(model, config))

        declared = model.export_state_names()[EXTEND_FUNCTION_NAME]
        for entrypoint in (EXTEND_FUNCTION_NAME, PROMPT_OPT_FUNCTION_NAME):
            mutated = programs[entrypoint].graph_signature.user_inputs_to_mutate.values()
            assert len(set(mutated)) == len(declared), entrypoint


# A deeper stack than the checkpoint above: linear x3, full, linear x3, full.
DEEP_LAYERS = 8
# Built depth -> (full-attention layers, linear-attention layers) among those built.
BUILT_LAYER_COUNTS = {4: (1, 3), 5: (1, 4), 8: (2, 6)}


def deep_config(num_layers: int) -> Qwen3_5TextConfig:
    """The deep stack lowered to ``num_layers`` the way the export pipeline lowers it:
    ``num_hidden_layers`` alone, with ``layer_types`` still listing every layer."""
    config = Qwen3_5TextConfig(
        **{**TEXT_CONFIG, "num_hidden_layers": DEEP_LAYERS, "max_position_embeddings": MAX_CTX}
    )
    config.num_hidden_layers = num_layers
    return config


STATE_NAMES = (
    KEY_CACHE_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
    CONV_CACHE_INPUT_NAME,
    RECURRENT_CACHE_INPUT_NAME,
)


def state_rows(states: dict) -> tuple[int, ...]:
    """Leading (layer) dim of each state, in contract order; tensors or shape tuples."""
    return tuple(
        states[name][0] if isinstance(states[name], tuple) else states[name].shape[0]
        for name in STATE_NAMES
    )


class TestStateRowsCountTheBuiltLayers:
    """``--num-layers`` lowers ``num_hidden_layers`` and leaves ``layer_types`` whole, so
    every per-layer-type count reads the first ``num_hidden_layers`` entries only."""

    @pytest.mark.parametrize("num_layers", sorted(BUILT_LAYER_COUNTS))
    def test_contract_and_model_agree(self, num_layers) -> None:
        n_full, n_linear = BUILT_LAYER_COUNTS[num_layers]
        config = deep_config(num_layers)
        assert len(config.layer_types) == DEEP_LAYERS
        expected = (n_full, n_full, n_linear, n_linear)

        static = Qwen3_5ForCausalLMForiOS.export_static_shape_configs(config, MAX_CTX)
        for rung, shapes in static[EXTEND_FUNCTION_NAME].items():
            assert state_rows(shapes) == expected, rung

        model = Qwen3_5ForCausalLMForiOS(config, model_device="cpu").to(torch.float32).eval()
        assert len(model.extend.model.layers) == num_layers
        reference, shapes = build_contract(model, config)
        model.validate_export_contract(reference, shapes)
        states = reference[EXTEND_FUNCTION_NAME]
        assert state_rows(states) == expected
        # The handlers the layers write through hold the same rows, or binding fails.
        with torch.no_grad():
            model.extend(**states)


def ios_export_config(path: str, **options) -> export_pipeline.ExportConfig:
    return export_pipeline.ExportConfig(
        hf_model_id=path,
        variant="iOS",
        compute_precision="float32",
        num_layers=NUM_LAYERS,
        **options,
    )


class TestExportPipeline:
    """The iOS class under ``resolve_model`` / ``build_model`` with ``--num-layers``."""

    @pytest.fixture(scope="class")
    def deep_checkpoint(self, tmp_path_factory) -> str:
        return write_checkpoint(
            tmp_path_factory.mktemp("qwen3_5_deep"),
            {**TEXT_CONFIG, "num_hidden_layers": DEEP_LAYERS},
        )

    def test_truncated_export_contract_matches_the_built_model(self, deep_checkpoint) -> None:
        resolved = export_pipeline.resolve_model(
            ios_export_config(deep_checkpoint, compression="none")
        )
        assert resolved.model_class is Qwen3_5ForCausalLMForiOS
        assert resolved.hf_config.num_hidden_layers == NUM_LAYERS
        assert len(resolved.hf_config.layer_types) == DEEP_LAYERS

        model = export_pipeline.build_model(resolved).model
        assert len(model.extend.model.layers) == NUM_LAYERS
        # The exporter reads every shape off the resolved config, not the model's own.
        reference, shapes = build_contract(model, resolved.hf_config)
        model.validate_export_contract(reference, shapes)
        assert state_rows(reference[EXTEND_FUNCTION_NAME]) == (1, 1, 3, 3)

    def test_palettizer_gets_forward_arguments(self, deep_checkpoint, monkeypatch) -> None:
        captured: dict[str, tuple] = {}

        def record(model, example_inputs, palettization_config, export_backend=None):
            captured["inputs"] = example_inputs
            return model

        monkeypatch.setattr(export_pipeline, "palettize_pytorch_model", record)
        resolved = export_pipeline.resolve_model(
            ios_export_config(deep_checkpoint, compression="4bit_weight_palettized_group32")
        )
        model = export_pipeline.build_model(resolved).model

        names = list(inspect.signature(model.forward).parameters)
        bound = dict(zip(names, captured["inputs"], strict=True))
        assert state_rows(bound) == (1, 1, 3, 3)
        with torch.no_grad():
            logits = model(*captured["inputs"])
        assert logits.shape[-1] == TEXT_CONFIG["vocab_size"]
        assert logits.numel() == QUERY_LEN * TEXT_CONFIG["vocab_size"]
