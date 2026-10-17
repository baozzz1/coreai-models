# Qwen3.5-0.8B: iOS static-shape variant and VLM text decoder

The macOS text decoder is `Qwen3_5ForCausalLM` in
`python/src/coreai_models/models/macos/qwen3_5.py` (registry entry `qwen3_5`, exported by
`coreai.llm.export`, tested in
`python/tests/test_model_units/test_models/test_macos_layers/test_qwen3_5.py`). This document
covers the two pieces built next to it:

- **iOS static-shape variant** (`models/ios/qwen3_5.py`): the same decoder written for the iOS
  static-shape export path, registered as the `ios_class` of the `qwen3_5` entry.
- **VLM entry** (`models/macos/qwen3_5_embeddings.py`): `Qwen3_5ForCausalLMEmbeddings`, which
  enters `Qwen3_5ForCausalLM` at the merged embedding sequence.

## Reproducing

| Script | What it does | Needs real weights |
|---|---|---|
| `spike.py parity-vlm` | An image+text prompt through the embeddings decoder, compared with HF element by element | Yes |
| `spike.py export` | torch.export + TorchConverter of the embeddings decoder; `--save` writes the `.aimodel` | Yes |
| `ios_qwen35_parity.py` | The whole iOS decoder against the macOS decoder, and interleaved M-RoPE against HF | Yes |
| `ios_palettize_probe.py` | Which of Qwen3.5's modules the iOS LUT4 preset picks up | No (reads the config only) |
| `chunked_gdn_parity.py` | The chunked recurrence against `mlx_lm`'s recurrence kernel and a whole-layer fixture, element by element | No (reads fixtures) |
| `ios_gdu_probe.py` | What is left in the MLIR after `GatedDeltaUpdate` goes through the static-shape path | No |

```
uv run python spike_qwen35/<script>.py ...
```

Unit tests guard the same properties on a small random checkpoint, without real weights:

- `test_ios_layers/test_qwen3_5.py`: the iOS variant's `GatedDeltaNet` and whole-model logits
  against HF, a run continued from carried states across calls, the refusal to export a stack with
  no full-attention layer, state shapes under `--num-layers`, and the palettizer's inputs.
- `test_macos_layers/test_qwen3_5_embeddings.py`: fed `embed_tokens(input_ids)`, the embeddings
  decoder is bit-identical to `Qwen3_5ForCausalLM` in logits, all four states and the prefill
  entrypoint; its export contract and the state writes after tracing.

## iOS static-shape variant

The recurrence uses the chunked form in `primitives/ios/gated_delta.py`: one call is one
16-token chunk, and the graph has no loop. The two linear-attention states go through
`primitives/ios/ssm_cache.py`. What this establishes is **matching numerics and a
self-consistent export contract**:

- `ios_qwen35_parity.py`: at fp32, the whole decoder against the macOS decoder (which has its
  own tests against HF), and interleaved M-RoPE against `Qwen3_5TextRotaryEmbedding`.
- `chunked_gdn_parity.py`: the chunked recurrence against `mlx_lm`'s recurrence kernel and a
  whole-layer fixture, element by element; `--export-check` checks that the exported MLIR
  is straight-line and fp16 throughout.
- `test_traces_write_every_declared_state` checks that the traced `extend` and `prompt_opt`
  write every declared state; `test_palettizer_gets_forward_arguments` checks that the
  arguments the palettizer gets bind to `forward` and run.
- The unit-test checkpoint takes two linear-attention head layouts: as many key heads as value
  heads (the 0.8B and 2B case), and value heads sharing a key head (key 2 / value 4, the 4B
  case). The latter expands q and k to the value head count with `repeat_interleave` before the
  recurrence. The checkpoint's `A_log` is set to a slow decay, so the recurrent state survives
  across calls and the stateful cases can see it.

### `--num-layers` and counts by layer type

The export pipeline's `--num-layers` changes only `num_hidden_layers` and leaves `layer_types`
whole. Every count the iOS variant takes by layer type (the KV cache's layer count, the conv /
recurrent states' layer count, each layer's row within its own kind of state) reads only the
first `num_hidden_layers` entries of `layer_types`, the same convention as the macOS decoder's
`_count_layer_type`. The config `resolve_model` hands the exporter and the model's own config
therefore give the same shapes.

Only full-attention layers write the key / value cache. With `--num-layers` at 1–3 every layer left
is linear attention: in the traced graph the KV cache is not a written state, the converter rejects
the declared state names, and the prefill entrypoint would fail first, indexing an empty cache.
`validate_export_contract` raises before tracing and asks for a cut at or past the first
full-attention layer (`--num-layers 4` for Qwen3.5); eager use of a short stack is unaffected.

### Palettizer inputs

The default iOS preset is k-means palettization, and coreai-opt first runs the model once with a
tuple of positional arguments. `BaseForCausalLMForiOS.build_compression_inputs` takes that tuple
from the model's own export contract: token ids stand in for the embedded input, and the
embedding table is not a parameter of `forward`. Qwen3.5's `forward` adds `rope_cos` /
`rope_sin` and two states, and the arguments still come out in `forward`'s order.

### `CoreAIStaticShapeEngine` cannot drive it

The exported `extend` places three requirements on its caller, and the engine meets none of
them today:

1. **The position inputs are `rope_cos` / `rope_sin` (each `[1, 16, 64]`), not `position_ids`.**
   The rope table lookup in the graph is a composite that the runtime resolves with a pass of its
   own, and that pass supports exactly one lookup, which must be fed by the graph's position input
   itself. Qwen3.5 needs one lookup for each of the time, height and width position rows: three
   rows as three inputs, one input sliced three ways, and a single `[3, seq]` tensor all crash with
   SIGSEGV during delegate specialization, on the GPU and the ANE alike. So the table lookup lives
   on the host, which has to compute the position rows anyway (they depend on the image layout);
   the algorithm is `InterleavedMRoPE`. The engine looks for a position input by a name containing
   `pos`, fills it with 1-D positions, and throws `has no position_ids input` when there is none.
2. **Two extra states: `conv_cache` `(18, 1, 6144, 1, 3)` and `recurrent_cache`
   `(18, 1, 16, 128, 128)`.** They are zeroed at the start of a session, passed in on every
   call, and `new_conv_cache` / `new_recurrent_cache` are fed back. The engine allocates and
   passes only `key_cache` / `value_cache`.
3. **Every call must carry 16 real tokens.** The linear-attention layers have no mask: every
   position in the query advances both states, a padded one included. When fewer than 16 tokens
   remain, the engine fills only the first slots; the padding in the rest throws the states off,
   and every later token is wrong without any error. Only the 16-token shape is traced, and there
   is no single-token decode graph; generation needs one graph per query length, or an input
   telling the linear-attention layers how many tokens are real.

### Not verified

1. On device: the iOS variant's numerics are checked only eagerly on a Mac and have never run
   on device (the engine lacks the three points above).
2. Grouped-head models (fewer key heads than value heads) are checked against HF only on small
   random configs; real weights, export and on-device runs have not been tried.
3. A whole-model export with real weights (including the default palettization preset) has not
   been run; the unit tests stop at tracing.

## VLM entry: the inputs_embeds decoder

`Qwen3_5ForCausalLMEmbeddings` enters the model at the merged embedding sequence. Its graph
inputs are `(inputs_embeds [1,q,hidden], position_ids int32 [1,p])`; the layers, the four
states (`keyCache` / `valueCache` / `convStates` / `recurrentStates`), the `prefill`
entrypoint, the `logits` and the weight loading are all `Qwen3_5ForCausalLM`'s. The runtime
splices the vision features into the image-token positions itself. The model carries
`embed_tokens`: `forward` never reads it, but a tied checkpoint's `lm_head` is that table, and
weight loading uses the token decoder's parameter names.

`coreai.vlm.export qwen3.5-0.8b --vision-only` exports only the vision tower; this decoder is
exported by `spike.py export`:

```
uv run python spike_qwen35/spike.py parity-vlm --dtype fp32 --gen-tokens 12 \
    --features <reference_features.npy> \
    --out <parity.json>
uv run python spike_qwen35/spike.py export --dtype fp16 \
    --max-context-length 4096 --save spike_qwen35/qwen3_5_0p8b_embeds.aimodel
```

`parity-vlm` is its parity harness. It builds a 224-token prompt in the ChatML template of
`_chat_prompt_ids` (the one the on-device runs below used), replaces the 196 `<|image_pad|>`
positions with reference vision features (the HF fp32 vision tower's output for a 448×448 test
image, 196×1024, passed with `--features`), compares against HF's `inputs_embeds` path element by
element, then decodes greedily on both sides. The criteria: prefill logits cosine ≥ 0.999, argmax
agreement ≥ 0.99, decode-step logits on the same scale as the prefill's, and greedy tokens
identical to HF one by one. Decode-step logits are reported separately because the offset semantics
of `position_ids` only take effect in decode steps, and the token chain is insensitive to them:
with every decode position off by one, the tokens can still match HF one by one while only the
decode cosine drops.

HF reference (M-RoPE), 12 greedy tokens:

```
[248068, 198, 760, 1156, 6587, 264, 11346, 3874, 314, 279, 3766, 2099]
'<think>\nThe user wants a detailed description of the provided image'
```

The bundle's `main` is this embeddings graph. A text-only run needs a bundle whose metadata
points `main` at the token-id decoder's `.aimodel` instead: feeding `input_ids` [1,N] to the
rank-3 `inputs_embeds` hits a fatalError in `NDArrayDescriptor.resolvingDynamicDimensions`.

### Before running on device

1. **The runtime's state factory owns the states.** `CoreAISequentialVLMEngine` allocates the
   fixed-size states beyond the KV pair, zeroes them, binds them on every call, zeroes them
   again on `reset()`, and refuses a partial rollback (`reset(to:)` to an intermediate position)
   for models that carry such states.
2. **The states must start at zero.** Both linear-attention states assume a zero start; a new
   session that does not zero them is simply wrong — no error, just wrong output.

### Acceptance gates

| Gate | Criterion |
|---|---|
| G1 | Mac-side eager parity (`parity-vlm`): cosine ≥0.999, argmax agreement ≥0.99, 12 tokens identical to HF |
| G2 | On-device fp16 GPU full chain: first 12 tokens == HF reference |
| G3 | 8-bit palettized on the ANE: first 12 tokens == the fp16 GPU run from the same batch |
| G4 | On-device fp16 ANE full chain: first 12 tokens == the fp16 GPU run from the same batch |

The records below are from 2026-09 runs, in an on-device test app, of the embeddings decoder assets
exported at the time. The app logs generated text rather than token ids, so the first 12 tokens are
that text re-encoded with the Qwen3.5-0.8B tokenizer. The `Qwen3_5ForCausalLMEmbeddings` described
in this section is built on the fused-projection `Qwen3_5ForCausalLM`, so its graph differs from
those assets, and the records describe only those assets:

- **G2 not passed**: the fp16 GPU full chain (vision tower on the GPU, decoder with no
  compute-unit override, 64 tokens) produced
  `[248068, 271, 248069, 271, 1919, 2099, 369, 264, 7309, 59076, 11, 12583]`
  (`<think>\n\n</think>\n\nThis image is a highly distorted…`), diverging at the second
  token; the 8-bit palettized GPU full chain's first 12 tokens match it token for token, wrong
  in the same way.
- **G3 not passed**: the 8-bit palettized ANE full chain's first 12 tokens match the HF reference
  one by one, the only correct on-device full-chain evidence; but G3 asks for it to equal the
  fp16 GPU run, which is itself wrong.
- **G4 not passed (did not run)**: both fp16 ANE full-chain runs failed with the runtime's
  `nilError`, with zero prompt and zero generated tokens; the log stops after the main decoder is
  prepared and before the engine initializes. Pinning the decoder to the GPU fails the same way.
  The minimal repro is one full-pipeline run with the vision tower on the Neural Engine and the
  fp16 bundle.

The G2 difference sits on the vision side, not in the decoder's specialized assets: between the
wrong fp16 GPU run and the correct 8-bit ANE run, two of the three specialization cache ids are
identical, and only the one that varies with the vision asset differs (2d5857… / 1c7dde…). With the
vision tower run on its own, its GPU fp16 features fed into HF match the reference; the GPU vision
features produced inside the full pipeline have not been checked separately.

### Not verified

1. `Qwen3_5ForCausalLMEmbeddings` has never run on device, and `parity-vlm` and `export` have
   not been run on it either (they need real weights); the unit tests only guard that it is
   bit-identical to the token decoder.
