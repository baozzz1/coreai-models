"""Does the iOS LUT4 preset cover Qwen3.5's module set?

`IOS_PRESETS["4bit_weight_palettized_group8"]` is what puts weights in a form the ANE will
take — per-block affine is not scheduled there. Its module exclusions name the embedding
classes and nothing else, so every other module is offered to k-means, including the two
Qwen3.5 introduces and no shipped iOS model has: `RMSNormGated` and the depthwise `conv1d`
of the linear-attention block.

The probe palettizes a truncated Qwen3.5 decoder (layers 0-2 linear-attention, layer 3 full
attention) and reports, per module, what came back.

    uv run python spike_qwen35/ios_palettize_probe.py [--num-layers 4] [--preset NAME]
"""

from __future__ import annotations

import argparse

import torch
from coreai_models.export.compression import palettize_pytorch_model
from coreai_models.export.presets import IOS_PRESETS
from coreai_models.models.base import TraceSpec
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM
from transformers import AutoConfig

HF_ID = "Qwen/Qwen3.5-0.8B"


def config(num_layers: int):
    cfg = AutoConfig.from_pretrained(HF_ID).text_config
    cfg.num_hidden_layers = num_layers
    cfg.layer_types = cfg.layer_types[:num_layers]
    return cfg


def leaf_weights(model: torch.nn.Module) -> dict[str, tuple[str, torch.Tensor]]:
    out = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            out[name] = (type(module).__name__, weight.detach().float().clone())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--preset", default="4bit_weight_palettized_group8")
    args = parser.parse_args()

    cfg = config(args.num_layers)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).eval()
    before = leaf_weights(model)
    print(f"{args.preset}  layers={cfg.layer_types}")
    print(f"  modules carrying a weight: {len(before)}")

    preset = IOS_PRESETS[args.preset]
    spec = TraceSpec(max_context_length=128, cache_seq_len=128, query_len=8)
    example = tuple(
        model.build_reference_inputs(cfg, torch.float32, spec)["main"].values()
    )
    try:
        palettized = palettize_pytorch_model(
            model, example, preset["torch_palettization_config"]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  palettize FAILED  {type(exc).__name__}: {exc}")
        return

    after = leaf_weights(palettized)
    by_kind: dict[tuple[str, tuple[int, ...]], list[float]] = {}
    for name, (kind, w0) in before.items():
        w1 = after[name][1]
        rel = float((w1 - w0).norm() / w0.norm()) if float(w0.norm()) > 0 else 0.0
        by_kind.setdefault((kind, tuple(w0.shape)), []).append(rel)

    print(
        "  rel_rmse per module type (random init, so this sizes applicability, not quality)"
    )
    for (kind, shape), rels in sorted(by_kind.items()):
        print(
            f"    {kind:<16} {str(shape):<22} n={len(rels):<3} "
            f"rel_rmse min={min(rels):.4f} max={max(rels):.4f}"
        )


if __name__ == "__main__":
    main()
