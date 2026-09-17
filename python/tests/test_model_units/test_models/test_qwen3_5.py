# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Qwen3.5 numerics, loading, compression and export-contract checks.

A tiny random checkpoint in the Hugging Face layout -- text weights under
``model.language_model.``, beside a vision tower -- stands in for the real one. The
numerical tests run it at fp32 through ``transformers`` and through both exported model
classes and compare; none of these need the Core AI runtime, they stop before the
converter.
"""

from typing import NamedTuple

import pytest
import torch
from safetensors.torch import save_file
from torch.nn.utils import parametrize
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from coreai_models._constants import (
    EXTEND_FUNCTION_NAME,
    MAIN_GRAPH_NAME,
    PROMPT_OPT_FUNCTION_NAME,
)
from coreai_models._hf import resolve_rope_theta
from coreai_models.export.compression import quantize_for_export
from coreai_models.export.presets import get_preset
from coreai_models.models import base
from coreai_models.models.base import TraceSpec
from coreai_models.models.ios.qwen3_5 import (
    InterleavedMRoPE,
    Qwen3_5ForCausalLMForiOS,
    rotary_dims,
)
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM
from coreai_models.models.registry import get_model_entry
from coreai_models.primitives.ios.ssm_cache import SSMStateHandler
from coreai_models.primitives.macos.cache import SSMState

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


def text_config(cls, num_layers: int | None = None, **overrides):
    return cls._get_reauthored_config(
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


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    return write_checkpoint(tmp_path_factory.mktemp("qwen3_5"), TEXT_CONFIG)


class Trio(NamedTuple):
    """One checkpoint through the reference implementation and both exported classes."""

    reference: Qwen3_5ForConditionalGeneration
    macos: Qwen3_5ForCausalLM
    ios: Qwen3_5ForCausalLMForiOS


@pytest.fixture(scope="module")
def trio(tmp_path_factory):
    """Loads and caches a ``Trio`` per linear-attention key/value head count."""
    loaded: dict[tuple[int, int], Trio] = {}

    def load(n_k_heads: int, n_v_heads: int) -> Trio:
        if (n_k_heads, n_v_heads) not in loaded:
            path = write_checkpoint(
                tmp_path_factory.mktemp(f"qwen3_5_{n_k_heads}_{n_v_heads}"),
                {
                    **TEXT_CONFIG,
                    "linear_num_key_heads": n_k_heads,
                    "linear_num_value_heads": n_v_heads,
                },
            )
            common = {"max_context_length": MAX_CTX, "target_dtype": torch.float32}
            loaded[n_k_heads, n_v_heads] = Trio(
                Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval(),
                Qwen3_5ForCausalLM.from_hf(path, **common).eval(),
                Qwen3_5ForCausalLMForiOS.from_hf(
                    path, disable_embedding_quantization=True, **common
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


def macos_linear_attn(model, hidden: torch.Tensor) -> torch.Tensor:
    """The first linear-attention layer of the macOS decoder, from zeroed states."""
    layer = model.model.layers[0].linear_attn
    conv = SSMState(torch.zeros(1, 1, layer.conv_dim, layer.conv_state_len))
    recurrent = SSMState(torch.zeros(1, 1, layer.n_v_heads, layer.head_k_dim, layer.head_v_dim))
    with torch.no_grad():
        return layer(hidden, conv, recurrent)


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


def macos_states(model) -> dict:
    """Zeroed key/value cache and linear-attention states, sized for ``MAX_CTX``."""
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX, query_len=1)
    states = model.build_reference_inputs(model.config, torch.float32, spec)[MAIN_GRAPH_NAME]
    del states["input_ids"], states["position_ids"]
    return states


def macos_logits(model, states: dict, input_ids: torch.Tensor, offset: int) -> torch.Tensor:
    """One decoder call; ``states`` is rewritten in place, so calls chain."""
    query_len = input_ids.shape[1]
    with torch.no_grad():
        return model(
            input_ids,
            torch.arange(offset + query_len, dtype=torch.int32).unsqueeze(0),
            **states,
        )


def ios_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """``IOS_QUERY_LEN`` tokens per call, every state carried across the calls."""
    query_len = model.IOS_QUERY_LEN
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX, query_len=query_len)
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
        for start in range(0, input_ids.shape[1], query_len):
            positions = torch.arange(start, start + query_len, dtype=torch.int32).unsqueeze(0)
            # A text prompt sends the same row as time, height and width.
            cos, sin = rope(positions, positions, positions)
            inputs["transformer_input"] = model.gather_embeddings(
                input_ids[:, start : start + query_len], table
            )
            inputs["rope_cos"], inputs["rope_sin"] = cos.float(), sin.float()
            inputs["in_step"] = torch.tensor([start], dtype=torch.int32)
            inputs["causal_mask"] = torch.where(
                cache_positions <= positions.reshape(1, 1, 1, query_len), 0.0, float("-inf")
            )
            outputs.append(model.extend(**inputs).reshape(1, query_len, -1))
    return torch.cat(outputs, dim=1)


class TestMemoryEfficientLoading:
    """The macOS exporter loads through ``from_hf_memory_efficient`` with the registry's
    prefix, which strips ``model.language_model.layers.N.*`` to ``model.layers.N.*``."""

    @pytest.fixture(autouse=True)
    def local_snapshot(self, monkeypatch) -> None:
        monkeypatch.setattr(base, "snapshot_download", lambda repo_id, **_: repo_id)

    @pytest.fixture
    def entry(self):
        return get_model_entry("qwen3_5")

    def test_index_puts_each_layer_in_its_own_slice(self, checkpoint, entry, tmp_path) -> None:
        # The released checkpoints also carry a multi-token-prediction head at the root.
        mtp = str(tmp_path / "mtp.safetensors")
        save_file({"mtp.layers.0.mlp.up_proj.weight": torch.zeros(1)}, mtp)

        per_layer, shared = base._build_safetensors_key_index(
            [f"{checkpoint}/model.safetensors", mtp],
            num_layers=2,
            hf_state_dict_prefix=entry.hf_state_dict_prefix,
        )

        assert sorted(per_layer) == [0, 1]
        for layer_idx, keys in per_layer.items():
            prefix = f"model.language_model.layers.{layer_idx}."
            assert keys and all(key.startswith(prefix) for key in keys), layer_idx
        assert sorted(shared) == [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
        ]

    @pytest.mark.parametrize("num_layers", [None, 2])
    def test_matches_the_full_ram_loader(self, checkpoint, entry, tmp_path, num_layers) -> None:
        streamed = Qwen3_5ForCausalLM.from_hf_memory_efficient(
            checkpoint,
            target_dtype=torch.float32,
            mmap_path=str(tmp_path / "layers"),
            num_layers=num_layers,
            hf_config_attr=entry.hf_config_attr,
            hf_state_dict_prefix=entry.hf_state_dict_prefix,
        )
        reference = Qwen3_5ForCausalLM.from_hf(
            checkpoint, target_dtype=torch.float32, num_layers=num_layers
        )

        assert len(streamed.model.layers) == (num_layers or NUM_LAYERS)
        expected = reference.state_dict()
        got = streamed.state_dict()
        assert got.keys() == expected.keys()
        for key, tensor in expected.items():
            torch.testing.assert_close(got[key], tensor, rtol=0, atol=0, msg=key)


class TestMacOSInt4Preset:
    def test_default_preset_quantizes_around_the_gated_norm(self) -> None:
        """``RMSNormGated`` holds a rank-1 scale the per-block spec has no axis 1 for."""
        config = text_config(Qwen3_5ForCausalLM)
        torch.manual_seed(0)
        model = Qwen3_5ForCausalLM(config).eval()
        preset = dict(get_preset("4bit")["torch_quantization_config"])

        quantized = quantize_for_export(model, config, torch.float32, preset)

        linear_attn = quantized.model.layers[0].linear_attn
        for proj in (linear_attn.in_proj_qkv, linear_attn.out_proj, quantized.lm_head):
            assert parametrize.is_parametrized(proj, "weight")
        assert not parametrize.is_parametrized(linear_attn.norm, "weight")


def _build_contract(model, config):
    spec = TraceSpec(max_context_length=MAX_CTX, cache_seq_len=MAX_CTX)
    return (
        model.build_reference_inputs(config, torch.float16, spec),
        model.build_dynamic_shapes(config, spec),
    )


class TestExportNeedsAFullAttentionLayer:
    """Only full-attention layers write the key/value cache, which both contracts declare."""

    @pytest.fixture(params=[Qwen3_5ForCausalLM, Qwen3_5ForCausalLMForiOS])
    def cls(self, request):
        return request.param

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_a_linear_attention_only_stack_is_rejected(self, cls, num_layers) -> None:
        config = text_config(cls, num_layers)
        model = cls(config, model_device="cpu").to(torch.float16).eval()
        # Eager use of the short stack still works; only export is refused.
        built = _build_contract(model, config)
        with pytest.raises(ValueError, match="no full-attention layer"):
            model.validate_export_contract(*built)

    def test_a_stack_through_the_first_full_attention_layer_validates(self, cls) -> None:
        config = text_config(cls, NUM_LAYERS)
        model = cls(config, model_device="cpu").to(torch.float16).eval()
        model.validate_export_contract(*_build_contract(model, config))

    def test_ios_traces_write_every_declared_state(self) -> None:
        from coreai_models.export.ios import _export_programs

        config = text_config(Qwen3_5ForCausalLMForiOS, NUM_LAYERS)
        model = Qwen3_5ForCausalLMForiOS(config, model_device="cpu").to(torch.float16).eval()
        programs = _export_programs(model, *_build_contract(model, config))

        declared = model.export_state_names()[EXTEND_FUNCTION_NAME]
        for entrypoint in (EXTEND_FUNCTION_NAME, PROMPT_OPT_FUNCTION_NAME):
            mutated = programs[entrypoint].graph_signature.user_inputs_to_mutate.values()
            assert len(set(mutated)) == len(declared), entrypoint


@pytest.mark.parametrize(("n_k_heads", "n_v_heads"), HEAD_COUNTS, ids=HEAD_COUNT_IDS)
class TestGatedDeltaNet:
    """The linear-attention layer against the reference one, same weights, fp32."""

    def test_macos_matches_the_reference(self, trio, n_k_heads, n_v_heads) -> None:
        models = trio(n_k_heads, n_v_heads)
        hidden = hidden_states(32)

        assert_matches(
            macos_linear_attn(models.macos, hidden),
            reference_linear_attn(models.reference, hidden),
            "macOS gated delta net",
        )

    def test_ios_matches_the_reference(self, trio, n_k_heads, n_v_heads) -> None:
        models = trio(n_k_heads, n_v_heads)
        query_len = Qwen3_5ForCausalLMForiOS.IOS_QUERY_LEN
        hidden = hidden_states(query_len)

        assert_matches(
            ios_linear_attn(models.ios, hidden, query_len),
            reference_linear_attn(models.reference, hidden),
            "iOS gated delta net",
        )

    def test_ios_carries_its_states_between_calls(self, trio, n_k_heads, n_v_heads) -> None:
        """Two calls of the fixed query length cover what the reference does in one pass.

        The conv state has to hand the second call its left context and the recurrent
        state the delta rule's matrix, or the second half drifts.
        """
        models = trio(n_k_heads, n_v_heads)
        query_len = Qwen3_5ForCausalLMForiOS.IOS_QUERY_LEN
        hidden = hidden_states(2 * query_len)

        assert_matches(
            ios_linear_attn(models.ios, hidden, query_len),
            reference_linear_attn(models.reference, hidden),
            "iOS gated delta net over two calls",
        )


@pytest.mark.parametrize(("n_k_heads", "n_v_heads"), HEAD_COUNTS, ids=HEAD_COUNT_IDS)
class TestDecoderLogits:
    """Whole-stack logits, through a layer stack that reaches the first full-attention one."""

    def test_macos_matches_the_reference(self, trio, n_k_heads, n_v_heads) -> None:
        models = trio(n_k_heads, n_v_heads)
        ids = token_ids(32)

        with torch.no_grad():
            reference = models.reference(input_ids=ids.long()).logits
        got = macos_logits(models.macos, macos_states(models.macos), ids, offset=0)

        assert_matches(got, reference, "macOS logits")

    def test_macos_continues_from_its_states(self, trio, n_k_heads, n_v_heads) -> None:
        """A prompt split across two calls gives the same tail as one pass over all of it.

        The key/value cache alone does not carry a linear-attention layer; the conv and
        recurrent states have to survive the call boundary too.
        """
        models = trio(n_k_heads, n_v_heads)
        ids = token_ids(32)
        half = ids.shape[1] // 2

        one_pass = macos_logits(models.macos, macos_states(models.macos), ids, offset=0)
        states = macos_states(models.macos)
        macos_logits(models.macos, states, ids[:, :half], offset=0)
        tail = macos_logits(models.macos, states, ids[:, half:], offset=half)

        assert_matches(tail, one_pass[:, half:], "macOS logits after a split prompt")

    def test_ios_matches_the_reference(self, trio, n_k_heads, n_v_heads) -> None:
        """The iOS variant sends a fixed number of tokens per call, so a 32-token prompt
        is two calls carrying the key/value cache and both linear-attention states."""
        models = trio(n_k_heads, n_v_heads)
        ids = token_ids(2 * Qwen3_5ForCausalLMForiOS.IOS_QUERY_LEN)

        with torch.no_grad():
            reference = models.reference(input_ids=ids.long()).logits

        assert_matches(ios_logits(models.ios, ids), reference, "iOS logits")
