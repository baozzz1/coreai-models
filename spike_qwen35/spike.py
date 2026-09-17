"""Spike driver for the Qwen3.5 macOS decoder export.

Stages, in the order they were used:

    probe-gdu      coreai_torch GatedDeltaUpdate alone through export + TorchConverter
    parity-gdn     re-authored GatedDeltaNet vs HF Qwen3_5GatedDeltaNet (random weights)
    parity-attn    re-authored gated Attention vs HF Qwen3_5Attention (random weights)
    parity-full    full model logits vs HF, real weights, --num-layers to truncate
    parity-decode  prefill-then-step against a single prefill (state protocol self-check)
    parity-vlm     the probe's image+text prompt through the embeddings decoder vs HF
    check-graph    every layer's state write reaches the graph's mutated-input output
    export         torch.export + TorchConverter, --num-layers to truncate,
                   --compression 4bit for int4 weight-only

Usage:
    uv run python spike_qwen35/spike.py <stage> [--num-layers N] [--dtype fp32|fp16]
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import shutil
from pathlib import Path

import coreai_torch
import numpy as np
import torch
import torch.nn.functional as F
from coreai_models._constants import MAIN_GRAPH_NAME
from coreai_models.export.compression import quantize_pytorch_model
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.mlir_ops import remove_functionalization
from coreai_models.export.presets import MACOS_PRESETS
from coreai_models.models.base import TraceSpec
from coreai_models.models.macos.qwen3_5 import (
    Attention,
    GatedDeltaNet,
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMEmbeddings,
)
from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_torch.composite_ops import GatedDeltaUpdate
from torch.export.graph_signature import OutputKind
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention as HFAttention,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFModel,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet as HFGatedDeltaNet,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding as HFRotary,
)

HF_ID = "Qwen/Qwen3.5-0.8B"
PROBE_PROMPT = "Describe this image in detail."
STATE_KEYS = ("k_cache", "v_cache", "conv_state", "recurrent_state")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spike")


def text_config(num_layers: int | None = None):
    config = AutoConfig.from_pretrained(HF_ID).text_config
    if num_layers is not None:
        config.num_hidden_layers = num_layers
        config.layer_types = config.layer_types[:num_layers]
    return config


def randomize(module: torch.nn.Module) -> torch.nn.Module:
    """Give every parameter a nonzero value; several norms initialize to zero."""
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(0.0, 0.1)
    return module


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


# ---------------------------------------------------------------------------
# Stage: probe-gdu
# ---------------------------------------------------------------------------


class _GatedDeltaUpdateOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gated_delta_update = GatedDeltaUpdate()

    def forward(self, query, key, value, g, beta, state):
        return self.gated_delta_update(query, key, value, g, beta, state)


def stage_probe_gdu(args) -> None:
    """Smallest reproducer for the composite op, decoupled from the model."""
    batch, heads, dk, dv = 1, 4, 8, 8
    seq_len = args.seq_len
    module = _GatedDeltaUpdateOnly().eval()
    reference_inputs = {
        "query": torch.randn(batch, heads, seq_len, dk),
        "key": torch.randn(batch, heads, seq_len, dk),
        "value": torch.randn(batch, heads, seq_len, dv),
        "g": torch.randn(batch, heads, seq_len),
        "beta": torch.rand(batch, heads, seq_len),
        "state": torch.zeros(batch, heads, dk, dv),
    }
    seq = torch.export.Dim("seq", min=2, max=512)
    dynamic_shapes = {
        "query": {2: seq},
        "key": {2: seq},
        "value": {2: seq},
        "g": {2: seq},
        "beta": {2: seq},
        "state": None,
    }

    for label, shapes in (("static", None), ("dynamic seq", dynamic_shapes)):
        program = export_to_coreai(
            module,
            reference_inputs,
            dynamic_shapes=shapes,
            output_names=("out", "new_state"),
        )
        program.optimize()
        print(f"  probe-gdu {label:<12} OK")


# ---------------------------------------------------------------------------
# Stage: parity-gdn
# ---------------------------------------------------------------------------


def stage_parity_gdn(args) -> None:
    config = text_config()
    torch.manual_seed(0)

    ours = randomize(GatedDeltaNet(config, cache_idx=0).to(torch.float32)).eval()
    theirs = HFGatedDeltaNet(config, layer_idx=0).to(torch.float32).eval()

    # HF's conv1d is left-padded and its output truncated; ours reads left context
    # from the state instead, so only the weight carries over.
    sd = {k: v for k, v in ours.state_dict().items()}
    theirs.load_state_dict({k: sd[k] for k in theirs.state_dict()}, strict=True)

    seq_len = args.seq_len
    x = torch.randn(1, seq_len, config.hidden_size)

    conv = SSMState(torch.zeros(1, 1, ours.conv_dim, ours.conv_state_len))
    rec = SSMState(torch.zeros(1, 1, ours.n_v_heads, ours.head_k_dim, ours.head_v_dim))
    with torch.no_grad():
        got = ours(x, conv, rec)
        want = theirs(x, cache_params=None)

    print(f"parity-gdn (seq_len={seq_len}, fp32)")
    report("gated_delta_net.out", got, want)


# ---------------------------------------------------------------------------
# Stage: parity-attn
# ---------------------------------------------------------------------------


def stage_parity_attn(args) -> None:
    config = text_config()
    torch.manual_seed(0)

    ours = randomize(Attention(config, cache_idx=0).to(torch.float32)).eval()
    theirs = HFAttention(config, layer_idx=0).to(torch.float32).eval()
    ours_sd = ours.state_dict()
    theirs.load_state_dict(
        {
            "q_proj.weight": ours_sd["q_proj.weight"],
            "k_proj.weight": ours_sd["k_proj.weight"],
            "v_proj.weight": ours_sd["v_proj.weight"],
            "o_proj.weight": ours_sd["o_proj.weight"],
            "q_norm.weight": ours_sd["q_norm.weight"],
            "k_norm.weight": ours_sd["k_norm.weight"],
        },
        strict=True,
    )

    seq_len = args.seq_len
    x = torch.randn(1, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    k_cache = torch.zeros(1, 1, config.num_key_value_heads, seq_len, config.head_dim)
    v_cache = torch.zeros_like(k_cache)

    rotary = HFRotary(config)
    with torch.no_grad():
        got = ours(x, position_ids, KVCache(k_cache, v_cache))
        cos_sin = rotary(x, position_ids.long())
        mask = (
            torch.full((seq_len, seq_len), float("-inf"))
            .triu(1)
            .reshape(1, 1, seq_len, seq_len)
        )
        want, _ = theirs(x, position_embeddings=cos_sin, attention_mask=mask)

    print(f"parity-attn (seq_len={seq_len}, fp32)")
    report("attention.out", got, want)


# ---------------------------------------------------------------------------
# Real-weight stages
# ---------------------------------------------------------------------------


def load_ours(
    num_layers: int | None, dtype: torch.dtype, embeddings: bool = False
) -> Qwen3_5ForCausalLM:
    cls = Qwen3_5ForCausalLMEmbeddings if embeddings else Qwen3_5ForCausalLM
    model = cls.from_hf(HF_ID, target_dtype=dtype, num_layers=num_layers)
    return model.eval()


def truncate_hf(hf, num_layers: int | None) -> None:
    """Drop the HF model to `--num-layers`; ours truncates when it loads."""
    if num_layers is None:
        return
    hf.model.language_model.layers = hf.model.language_model.layers[:num_layers]
    hf.config.text_config.num_hidden_layers = num_layers
    hf.model.language_model.config.num_hidden_layers = num_layers


def as_model_input(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Token ids in the form this model's forward takes them."""
    if isinstance(model, Qwen3_5ForCausalLMEmbeddings):
        with torch.no_grad():
            return model.model.embed_tokens(input_ids)
    return input_ids


def zero_states(model: Qwen3_5ForCausalLM, seq_len: int, dtype: torch.dtype):
    config = model.config
    n_full = sum(1 for t in config.layer_types if t != "linear_attention")
    n_linear = sum(1 for t in config.layer_types if t == "linear_attention")
    linear = next(layer.linear_attn for layer in model.model.layers if layer.is_linear)
    shape = (n_full, 1, config.num_key_value_heads, seq_len, config.head_dim)
    return (
        torch.zeros(shape, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
        torch.zeros(n_linear, 1, linear.conv_dim, linear.conv_state_len, dtype=dtype),
        torch.zeros(
            n_linear,
            1,
            linear.n_v_heads,
            linear.head_k_dim,
            linear.head_v_dim,
            dtype=dtype,
        ),
    )


def stage_parity_full(args) -> None:
    dtype = DTYPES[args.dtype]
    prompt = torch.randint(1, 100_000, (1, args.seq_len), dtype=torch.int32)

    ours = load_ours(args.num_layers, dtype, embeddings=args.embeddings)
    states = zero_states(ours, args.seq_len + 8, dtype)
    position_ids = torch.arange(args.seq_len, dtype=torch.int32).unsqueeze(0)
    with torch.no_grad():
        got = ours(as_model_input(ours, prompt), position_ids, *states)
    del ours
    gc.collect()

    ref_dtype = DTYPES[args.ref_dtype or args.dtype]
    hf = HFModel.from_pretrained(HF_ID, dtype=ref_dtype).eval()
    truncate_hf(hf, args.num_layers)
    with torch.no_grad():
        want = hf(input_ids=prompt.long(), use_cache=False).logits

    print(
        f"parity-full (layers={args.num_layers or 'all'}, seq_len={args.seq_len}, "
        f"ours={args.dtype}, ref={args.ref_dtype or args.dtype})"
    )
    report("logits", got, want)
    print(
        f"  argmax agreement            {(got.argmax(-1) == want.argmax(-1)).float().mean():.4f}"
    )


def stage_parity_decode(args) -> None:
    """Prefill n-1 tokens then step the last one; compare against one full prefill."""
    dtype = DTYPES[args.dtype]
    prompt = torch.randint(1, 100_000, (1, args.seq_len), dtype=torch.int32)
    ours = load_ours(args.num_layers, dtype, embeddings=args.embeddings)

    cache_len = args.seq_len + 8
    with torch.no_grad():
        states = zero_states(ours, cache_len, dtype)
        full = ours(
            as_model_input(ours, prompt),
            torch.arange(args.seq_len, dtype=torch.int32).unsqueeze(0),
            *states,
        )

        states = zero_states(ours, cache_len, dtype)
        split = args.seq_len - 1
        ours(
            as_model_input(ours, prompt.narrow(1, 0, split)),
            torch.arange(split, dtype=torch.int32).unsqueeze(0),
            *states,
        )
        stepped = ours(
            as_model_input(ours, prompt.narrow(1, split, 1)),
            torch.arange(args.seq_len, dtype=torch.int32).unsqueeze(0),
            *states,
        )

    print(
        f"parity-decode (layers={args.num_layers or 'all'}, seq_len={args.seq_len}, {args.dtype})"
    )
    report("last-token logits", stepped[:, -1], full[:, -1])


# ---------------------------------------------------------------------------
# Stage: parity-vlm
# ---------------------------------------------------------------------------


def _probe_prompt_ids(tokenizer, prompt: str, image_token_count: int) -> torch.Tensor:
    """The iOS probe's ChatML prompt, tokenized the way the probe tokenizes it."""
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
    """The probe's image+text prompt through the embeddings decoder, against HF."""
    if args.gen_tokens < 2:
        raise ValueError(
            "--gen-tokens must be at least 2: one from the prefill, one decode step"
        )
    dtype = DTYPES[args.dtype]
    features = torch.from_numpy(np.load(args.features)).to(dtype)
    image_token_id = AutoConfig.from_pretrained(HF_ID).image_token_id
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    ids = _probe_prompt_ids(tokenizer, args.prompt, features.shape[0])
    image_mask = (ids == image_token_id).unsqueeze(-1)
    if int(image_mask.sum()) != features.shape[0]:
        raise ValueError(
            f"prompt carries {int(image_mask.sum())} image tokens but the features hold "
            f"{features.shape[0]}"
        )
    seq_len = ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)

    ours = load_ours(args.num_layers, dtype, embeddings=True)
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
    truncate_hf(hf, args.num_layers)
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


def _prefill_logits(model, states: dict, seq_len: int) -> torch.Tensor:
    """One fixed prefill through ``model``, on freshly zeroed copies of the states."""
    torch.manual_seed(0)
    input_ids = torch.randint(1, 100_000, (1, seq_len), dtype=torch.int32)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    fresh = [torch.zeros_like(states[k]) for k in STATE_KEYS]
    with torch.no_grad():
        return model(as_model_input(model, input_ids), position_ids, *fresh)


def stage_export(args) -> None:
    dtype = DTYPES[args.dtype]
    model = load_ours(args.num_layers, dtype, embeddings=args.embeddings)
    config = model.config

    spec = TraceSpec(max_context_length=args.max_context_length, cache_seq_len=256)
    reference_inputs = model.build_reference_inputs(config, dtype, spec)
    dynamic_shapes = model.build_dynamic_shapes(config, spec)
    model.validate_export_contract(reference_inputs, dynamic_shapes)
    graph_inputs = reference_inputs[MAIN_GRAPH_NAME]
    graph_shapes = dynamic_shapes[MAIN_GRAPH_NAME]

    dense_logits = None
    if args.compression == "4bit":
        # Capture the dense answer first: the quantizer works on the model in place.
        dense_logits = _prefill_logits(model, graph_inputs, args.seq_len)
        quant_config = copy.deepcopy(MACOS_PRESETS["4bit"]["torch_quantization_config"])
        log.info("applying int4 weight-only quantization")
        model = quantize_pytorch_model(
            model,
            model.reference_inputs_as_args(graph_inputs),
            graph_shapes,
            quant_config,
            cache_seq_len=spec.cache_seq_len,
            # The two non-state graph inputs come first; everything from k_cache
            # on is state.
            state_indices=tuple(range(2, len(graph_inputs))),
        )

    log.info("exporting %d layers to Core AI", config.num_hidden_layers)
    program = export_to_coreai(
        model,
        graph_inputs,
        dynamic_shapes=graph_shapes,
        input_names=model.export_input_names()[MAIN_GRAPH_NAME],
        output_names=model.export_output_names()[MAIN_GRAPH_NAME],
        state_names=model.export_state_names()[MAIN_GRAPH_NAME],
        export_prefill_graph=args.prefill,
    )
    log.info("converted; optimizing")
    program.optimize()
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
        got = _prefill_logits(model, graph_inputs, args.seq_len)
        report("logits", got, dense_logits)
        print(
            f"  argmax agreement            "
            f"{(got.argmax(-1) == dense_logits.argmax(-1)).float().mean():.4f}"
        )


# ---------------------------------------------------------------------------
# Stage: check-graph
# ---------------------------------------------------------------------------


def stage_check_graph(args) -> None:
    """Confirm every layer's state write reaches the graph's mutated-input output.

    The linear-attention states are written by a side-effecting custom op whose
    return value is discarded, so nothing in eager mode would notice if a write
    dropped out of the exported graph. Inspects the graph the converter is
    handed: decomposed, then defunctionalized.
    """
    dtype = DTYPES[args.dtype]
    model = load_ours(args.num_layers, dtype, embeddings=args.embeddings)
    config = model.config

    spec = TraceSpec(max_context_length=args.max_context_length, cache_seq_len=256)
    reference_inputs = model.build_reference_inputs(config, dtype, spec)[
        MAIN_GRAPH_NAME
    ]
    dynamic_shapes = model.build_dynamic_shapes(config, spec)[MAIN_GRAPH_NAME]
    with torch.no_grad():
        ep = torch.export.export(
            model, args=(), kwargs=reference_inputs, dynamic_shapes=dynamic_shapes
        )
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    remove_functionalization(ep)

    mutations = {
        s.target: s.arg.name
        for s in ep.graph_signature.output_specs
        if s.kind is OutputKind.USER_INPUT_MUTATION
    }
    placeholders = {n.name: n for n in ep.graph.nodes if n.op == "placeholder"}

    n_linear = sum(1 for t in config.layer_types if t == "linear_attention")
    n_full = config.num_hidden_layers - n_linear
    expected = {
        "k_cache": n_full,
        "v_cache": n_full,
        "conv_state": n_linear,
        "recurrent_state": n_linear,
    }

    print(f"check-graph (layers={config.num_hidden_layers})")
    ok = True
    for name, want_writes in expected.items():
        node = placeholders[name]
        writes = 0
        while True:
            successor = next(
                (u for u in node.users if "slice_update" in str(u.target)),
                None,
            )
            if successor is None:
                break
            node, writes = successor, writes + 1
        reaches_output = node.name == mutations.get(name)
        good = writes == want_writes and reaches_output
        ok &= good
        print(
            f"  {'OK ' if good else 'BAD'} {name:<16} chained writes={writes} "
            f"(expected {want_writes}), reaches mutated output={reaches_output}"
        )
    print(f"check-graph {'OK' if ok else 'FAILED'}")


STAGES = {
    "probe-gdu": stage_probe_gdu,
    "check-graph": stage_check_graph,
    "parity-gdn": stage_parity_gdn,
    "parity-attn": stage_parity_attn,
    "parity-full": stage_parity_full,
    "parity-decode": stage_parity_decode,
    "parity-vlm": stage_parity_vlm,
    "export": stage_export,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--ref-dtype", choices=sorted(DTYPES), default=None)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument(
        "--features", default=None, help="reference vision features .npy"
    )
    parser.add_argument("--prompt", default=PROBE_PROMPT)
    parser.add_argument("--gen-tokens", type=int, default=2)
    parser.add_argument("--out", default=None, help="write the parity record here")
    parser.add_argument("--compression", choices=["none", "4bit"], default="none")
    parser.add_argument("--save", default=None, help="write the .aimodel here")
    args = parser.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
