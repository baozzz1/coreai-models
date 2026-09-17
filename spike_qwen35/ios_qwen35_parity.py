"""The iOS variant of Qwen3.5 against the macOS decoder, and its rope against HuggingFace.

The macOS decoder is the one checked against `transformers` (`SPIKE.md`), so it is the
reference here: same checkpoint, same token ids, fp32 on both sides. The rope is checked
separately because the interleaved three-row form is the piece with no counterpart on the
macOS side, and equal rows have to reproduce the one-dimensional embedding exactly.

    uv run python spike_qwen35/ios_qwen35_parity.py [--layers 1 4 8 24]
"""

from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import TraceSpec
from coreai_models.models.ios.qwen3_5 import (
    InterleavedMRoPE,
    Qwen3_5ForCausalLMForiOS,
    rotary_dims,
)
from coreai_models._hf import resolve_rope_theta as _theta
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM

HF_ID = "Qwen/Qwen3.5-0.8B"
QUERY_LEN = 16
CONTEXT = 256


def causal_mask(context: int, query_len: int) -> torch.Tensor:
    key = torch.arange(context)[None, :, None, None]
    query = torch.arange(query_len)[None, None, None, :]
    return torch.where(key <= query, 0.0, float("-inf")).float()


def decoder_parity(num_layers: int | None) -> str:
    ids = torch.randint(1, 100_000, (1, QUERY_LEN), dtype=torch.int32)
    positions = torch.arange(QUERY_LEN, dtype=torch.int32).unsqueeze(0)
    spec = TraceSpec(
        max_context_length=CONTEXT, cache_seq_len=CONTEXT, query_len=QUERY_LEN
    )
    common = {"max_context_length": CONTEXT, "target_dtype": torch.float32}

    reference_model = Qwen3_5ForCausalLM.from_hf(
        HF_ID, num_layers=num_layers, **common
    ).eval()
    inputs = reference_model.build_reference_inputs(
        reference_model.config, torch.float32, spec
    )["main"]
    inputs["input_ids"] = ids
    inputs["position_ids"] = positions
    with torch.no_grad():
        reference = reference_model(**inputs).reshape(1, QUERY_LEN, -1).float().clone()
    del reference_model, inputs
    gc.collect()

    model = Qwen3_5ForCausalLMForiOS.from_hf(
        HF_ID, num_layers=num_layers, disable_embedding_quantization=True, **common
    ).eval()
    inputs = model.build_reference_inputs(model.config, torch.float32, spec)["extend"]
    table = model.load_embeddings.embedding_table
    inputs["transformer_input"] = model.gather_embeddings(ids, table)
    inputs["embedding_table"] = table
    rope = InterleavedMRoPE(
        rotary_dims(model.config),
        model.config.max_position_embeddings,
        float(_theta(model.config)),
    ).eval()
    with torch.no_grad():
        cos, sin = rope(positions, positions, positions)
    inputs["rope_cos"], inputs["rope_sin"] = cos.float(), sin.float()
    inputs["causal_mask"] = causal_mask(CONTEXT, QUERY_LEN)
    with torch.no_grad():
        got = model.extend(**inputs).reshape(1, QUERY_LEN, -1).float()

    cosine = torch.nn.functional.cosine_similarity(
        got.flatten().double(), reference.flatten().double(), dim=0
    )
    layers = num_layers or model.config.num_hidden_layers
    return (
        f"  layers={layers:<3} max_abs={(got - reference).abs().max():.3e} "
        f"scale={reference.abs().max():.4f} cos={cosine:.12f} "
        f"argmax={(got.argmax(-1) == reference.argmax(-1)).float().mean():.4f}"
    )


def rope_parity() -> list[str]:
    config = AutoConfig.from_pretrained(HF_ID).text_config
    config.max_position_embeddings = 512
    reference = Qwen3_5TextRotaryEmbedding(config).eval()
    mine = InterleavedMRoPE(
        rotary_dims(config),
        config.max_position_embeddings,
        float(resolve_rope_theta(config)),
    ).eval()
    seq = 24
    steps = torch.arange(seq)
    lines = [
        f"  rotary_dims={rotary_dims(config)} mrope_section={reference.mrope_section}"
    ]
    for name, positions in (
        ("equal rows", steps.repeat(3).reshape(3, seq)),
        ("image grid", torch.stack([steps, steps // 4, steps % 4])),
    ):
        cos_ref, sin_ref = reference(torch.zeros(1, seq, 1), positions.unsqueeze(1))
        rows = [r.reshape(1, -1).to(torch.int32) for r in positions]
        cos, sin = mine(*rows)
        lines.append(
            f"  {name:<12} cos_max_abs={(cos - cos_ref).abs().max():.3e}  "
            f"sin_max_abs={(sin - sin_ref).abs().max():.3e}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--full", action="store_true", help="also run all 24 layers")
    args = parser.parse_args()

    torch.manual_seed(0)
    print("=== interleaved M-RoPE vs Qwen3_5TextRotaryEmbedding ===")
    for line in rope_parity():
        print(line)

    print("\n=== iOS decoder vs macOS decoder, fp32 ===")
    for num_layers in [*args.layers, *([None] if args.full else [])]:
        print(decoder_parity(num_layers))


if __name__ == "__main__":
    main()
