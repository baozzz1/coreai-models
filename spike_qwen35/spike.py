"""Driver for the Qwen3.5 VLM text decoder, the inputs_embeds variant of the macOS one.

Stages:

    parity-vlm  an image+text prompt through the embeddings decoder against HF
    export      torch.export + TorchConverter, `main` and `prefill` entrypoints,
                --compression 4bit for int4 weight-only

The token-id decoder is exported by `coreai.llm.export`.

Usage:
    uv run python spike_qwen35/spike.py <stage> [--dtype fp32|fp16]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from coreai_models.export.compression import quantize_for_export
from coreai_models.export.macos import export_macos_model
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.pipeline import ExportConfig
from coreai_models.export.presets import get_preset
from coreai_models.models.macos.qwen3_5_embeddings import Qwen3_5ForCausalLMEmbeddings
from coreai_models.models.registry import get_model_entry
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFModel,
)

HF_ID = "Qwen/Qwen3.5-0.8B"
IMAGE_PROMPT = "Describe this image in detail."
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
PRECISIONS = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spike")


def report(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> float:
    # float64: cosine similarity over a few million logits loses its last digits in fp32.
    ours_f, theirs_f = ours.double().flatten(), theirs.double().flatten()
    cos = F.cosine_similarity(ours_f, theirs_f, dim=0).item()
    max_abs = (ours_f - theirs_f).abs().max().item()
    rel = max_abs / (theirs_f.abs().max().item() + 1e-12)
    print(
        f"  {name:<28} cos={cos:.8f}  max_abs={max_abs:.3e}  max_rel={rel:.3e}  "
        f"ref_rms={theirs_f.pow(2).mean().sqrt():.4f}"
    )
    return cos


def load_decoder(
    dtype: torch.dtype, max_context_length: int | None = None
) -> Qwen3_5ForCausalLMEmbeddings:
    """The embeddings decoder, loaded the way `coreai.llm.export` loads the token one."""
    entry = get_model_entry("qwen3_5")
    model = Qwen3_5ForCausalLMEmbeddings.from_hf_memory_efficient(
        HF_ID,
        max_context_length=max_context_length,
        target_dtype=dtype,
        hf_config_attr=entry.hf_config_attr,
        hf_state_dict_prefix=entry.hf_state_dict_prefix,
    )
    return model.eval()


def zero_states(model: Qwen3_5ForCausalLMEmbeddings, seq_len: int, dtype: torch.dtype):
    """Key/value cache rows per full-attention layer, conv/recurrent per linear one."""
    k_cache, v_cache = model.create_kv_cache_tensors(
        model.config, dtype=dtype, seq_len=seq_len
    )
    conv_states, recurrent_states = model.create_delta_cache_tensors(
        model.config, dtype=dtype
    )
    return k_cache, v_cache, conv_states, recurrent_states


# ---------------------------------------------------------------------------
# Stage: parity-vlm
# ---------------------------------------------------------------------------


def _chat_prompt_ids(tokenizer, prompt: str, image_token_count: int) -> torch.Tensor:
    """The ChatML image prompt of the on-device runs, without added special tokens."""
    placeholder = "<|image_pad|>" * image_token_count
    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>\n"
        f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def _greedy_ours(
    model, logits, states, seq_len: int, n_tokens: int
) -> tuple[list[int], list]:
    """The prefill's argmax, then decode steps fed with their own argmax."""
    tokens = [int(logits[0, -1].argmax())]
    steps = []
    for step in range(n_tokens - 1):
        token = torch.tensor([[tokens[-1]]], dtype=torch.int32)
        position_ids = torch.arange(seq_len + step + 1, dtype=torch.int32).unsqueeze(0)
        logits = model(model.model.embed_tokens(token), position_ids, *states)
        steps.append(logits[0, -1].float())
        tokens.append(int(logits[0, -1].argmax()))
    return tokens, steps


def _greedy_hf(hf, logits, cache, n_tokens: int) -> tuple[list[int], list]:
    tokens = [int(logits[0, -1].argmax())]
    steps = []
    for _ in range(n_tokens - 1):
        embed = hf.get_input_embeddings()(torch.tensor([[tokens[-1]]]))
        out = hf(inputs_embeds=embed, past_key_values=cache, use_cache=True)
        steps.append(out.logits[0, -1].float())
        tokens.append(int(out.logits[0, -1].argmax()))
    return tokens, steps


def stage_parity_vlm(args) -> None:
    """An image+text prompt through the embeddings decoder, against HF."""
    if args.gen_tokens < 2:
        raise ValueError(
            "--gen-tokens must be at least 2: one from the prefill, one decode step"
        )
    dtype = DTYPES[args.dtype]
    features = torch.from_numpy(np.load(args.features)).to(dtype)
    image_token_id = AutoConfig.from_pretrained(HF_ID).image_token_id
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    ids = _chat_prompt_ids(tokenizer, args.prompt, features.shape[0])
    image_mask = (ids == image_token_id).unsqueeze(-1)
    if int(image_mask.sum()) != features.shape[0]:
        raise ValueError(
            f"prompt carries {int(image_mask.sum())} image tokens but the features hold "
            f"{features.shape[0]}"
        )
    seq_len = ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    ours = load_decoder(dtype)
    with torch.no_grad():
        embeds = ours.model.embed_tokens(ids).masked_scatter(image_mask, features)
        states = zero_states(ours, seq_len + args.gen_tokens + 8, dtype)
        got = ours(embeds, position_ids, *states)
        ours_tokens, ours_steps = _greedy_ours(
            ours, got, states, seq_len, args.gen_tokens
        )
    del ours
    gc.collect()

    ref_dtype = DTYPES[args.ref_dtype or args.dtype]
    hf = HFModel.from_pretrained(HF_ID, dtype=ref_dtype).eval()
    with torch.no_grad():
        cache = DynamicCache(config=hf.config.text_config)
        want = hf(
            inputs_embeds=embeds.to(ref_dtype), past_key_values=cache, use_cache=True
        ).logits
        hf_tokens, hf_steps = _greedy_hf(hf, want, cache, args.gen_tokens)

    print(
        f"parity-vlm (prompt={seq_len} tokens, image={features.shape[0]}, "
        f"ours={args.dtype}, ref={args.ref_dtype or args.dtype})"
    )
    cos = report("prefill logits", got, want)
    decode_cos = report(
        f"decode logits ({len(ours_steps)} steps)",
        torch.stack(ours_steps),
        torch.stack(hf_steps),
    )
    agreement = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    print(f"  argmax agreement            {agreement:.4f}")
    print(f"  ours tokens                 {ours_tokens}")
    print(f"  hf tokens                   {hf_tokens}")
    # Decode steps are the only place position_ids carry an offset, and a wrong
    # offset there still leaves the token chain intact: hold them to the prefill's
    # own agreement, with a floor for when the prefill lands on cos 1.0.
    decode_ok = (1 - decode_cos) <= max(1e-9, 100 * (1 - cos))
    passed = (
        cos >= 0.999 and decode_ok and agreement >= 0.99 and ours_tokens == hf_tokens
    )
    print(f"parity-vlm {'OK' if passed else 'FAILED'}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": HF_ID,
                    "prompt": args.prompt,
                    "features": args.features,
                    "prompt_token_count": seq_len,
                    "image_token_count": features.shape[0],
                    "prompt_token_ids": ids[0].tolist(),
                    "tokens_ours": ours_tokens,
                    "tokens_hf": hf_tokens,
                    "text_hf": tokenizer.decode(hf_tokens),
                    "logits_cosine": cos,
                    "decode_logits_cosine": decode_cos,
                    "argmax_agreement": agreement,
                },
                indent=2,
            )
        )


# ---------------------------------------------------------------------------
# Stage: export
# ---------------------------------------------------------------------------


def _prefill_logits(model, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """One fixed prefill through ``model``, on freshly zeroed states."""
    torch.manual_seed(0)
    input_ids = torch.randint(1, 100_000, (1, seq_len), dtype=torch.int32)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    states = zero_states(model, seq_len, dtype)
    with torch.no_grad():
        return model(model.model.embed_tokens(input_ids), position_ids, *states)


def stage_export(args) -> None:
    dtype = DTYPES[args.dtype]
    model = load_decoder(dtype, args.max_context_length)
    config = model.config

    dense_logits = None
    if args.compression == "4bit":
        # Capture the dense answer first: the quantizer works on the model in place.
        dense_logits = _prefill_logits(model, args.seq_len, dtype)
        log.info("applying int4 weight-only quantization")
        model = quantize_for_export(
            model, config, dtype, dict(get_preset("4bit")["torch_quantization_config"])
        )

    log.info("exporting %d layers to Core AI", config.num_hidden_layers)
    program = export_macos_model(
        model,
        config,
        ExportConfig(
            hf_model_id=HF_ID,
            compute_precision=PRECISIONS[args.dtype],
            max_context_length=args.max_context_length,
        ),
    )
    if args.save:
        path = Path(args.save)
        if path.exists():
            shutil.rmtree(path)
        program.save_asset(path, build_aimodel_metadata(HF_ID))
        log.info("saved %s", path)
    print(
        f"export OK ({config.num_hidden_layers} layers, {args.dtype}, {args.compression})"
    )

    if dense_logits is not None:
        print(f"quantized vs dense (prefill seq_len={args.seq_len})")
        got = _prefill_logits(model, args.seq_len, dtype)
        report("logits", got, dense_logits)
        print(
            f"  argmax agreement            "
            f"{(got.argmax(-1) == dense_logits.argmax(-1)).float().mean():.4f}"
        )


STAGES = {
    "parity-vlm": stage_parity_vlm,
    "export": stage_export,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--ref-dtype", choices=sorted(DTYPES), default=None)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument(
        "--features", default=None, help="reference vision features .npy"
    )
    parser.add_argument("--prompt", default=IMAGE_PROMPT)
    parser.add_argument("--gen-tokens", type=int, default=2)
    parser.add_argument("--out", default=None, help="write the parity record here")
    parser.add_argument("--compression", choices=["none", "4bit"], default="none")
    parser.add_argument("--save", default=None, help="write the .aimodel here")
    args = parser.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
