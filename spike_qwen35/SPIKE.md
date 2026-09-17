# Qwen3.5-0.8B 解码器导出

macOS 路径上 Qwen3.5 的解码器可以直接导出，没有需要绕过的算子。`coreai-torch 0.4.2` 带
`GatedDeltaUpdate` 复合算子，`coreai_models` 的 `EXTERNALIZE_SPECS` 里登记了它
（`composite_op_name="gated_delta_update"`），linear attention 的核心递推直接用它，
不拆成基础算子。iOS 静态图路径有两处不同：递推是分块的基础算子写法，rope 的表查询在
宿主侧，见「iOS 静态图变体」。

## 复现

```
uv run python spike_qwen35/spike.py <stage>
```

| stage | 干什么 |
|---|---|
| `probe-gdu` | 只把 `GatedDeltaUpdate` 单独过 export + TorchConverter |
| `parity-gdn` | 重写的 `GatedDeltaNet` 对 HF `Qwen3_5GatedDeltaNet`（随机权重） |
| `parity-attn` | 重写的门控 `Attention` 对 HF `Qwen3_5Attention`（随机权重） |
| `parity-full` | 真权重整模型 logits 对 HF |
| `parity-decode` | prefill 一段再走一步，对齐一次性 prefill |
| `check-graph` | 每层的状态写入是否都串进图的 mutated-input 输出 |
| `export` | torch.export + TorchConverter，`--save` 落 `.aimodel` |

## GatedDeltaUpdate 的接法

`coreai_torch/composite_ops/_gated_delta_update.py` 里的实现，和 HF 的
`torch_recurrent_gated_delta_rule(use_qk_l2norm_in_kernel=True)` 是同一个递推，
只是输入布局固定为 `(b, h, s, d)`、时间维用 `torch.ops.higher_order.while_loop`
展开。所以对接方式就是直接实例化 `GatedDeltaUpdate()` 当子模块，把 q/k/v 转成
`(b, h, s, d)`、g/beta 转成 `(b, h, s)` 喂进去，externalize 会自动把它抓成
composite。

- `while_loop` 在 converter 里有 lowering（`_aten_to_core.py` 的
  `_higher_order_resolver`），静态 seq 和动态 seq 都过。
- 导出的 `.aimodel` 里能查到 `gated_delta_update` 声明和
  `model.layers.0.linear_attn.gated_delta_update_<uuid>` 这样的 per-call-site 图，
  说明 composite 确实被单独切出来了。

## 状态协议

四个命名状态，由 `export_state_names()` 声明；导出管线（`export/macos.py`）对状态名是
通用的。

| 状态 | 形状（0.8B） | 走哪个 primitive |
|---|---|---|
| `keyCache` / `valueCache` | `(6, 1, 2, seq, 256)` | `KVCache`（6 = full attention 层数） |
| `convState` | `(18, 1, 6144, 3)` | `SSMState` |
| `recurrentState` | `(18, 1, 16, 128, 128)` | `SSMState` |

Swift 运行时按名字和形状给状态分类：带动态维的是 KV cache，静态形状且名字含
`cache` / `kv` 的当作可截断的滑窗缓存，其余是定长状态。两个线性注意力状态随每个
token 前进、回退不到中间位置，所以名字里不能有 `cache`，否则 `reset(to:)` 的部分
回退不会被拒绝。

两类层各自独立编号：`TransformerBlock` 拿到的是 `cache_idx`（同类层内的序号），
不是全局 `layer_idx`。

conv 状态存 `kernel_size - 1 = 3` 帧左上下文，前向里 `cat([state, x], -1)` 再做
`padding=0` 的 depthwise conv，prefill 和 decode 走同一条路径，不需要按 `seq_len`
分支——HF 那边是 `seq_len == 1` 走 `causal_conv1d_update`、否则走带 padding 的
conv 再截断，两条路径在导出里都得展开，合成一条反而更干净。

### `SSMState.update_states` 的 `begin` 与 `end` 各有 `cache.dim()` 个元素

`primitives/macos/cache.py`。`coreai.slice_update` 要求两者同形；`end` 少一维时
`AIProgram.optimize()` 报：

```
'coreai.slice_update' op failed to verify that Operands should have same shape.
(tensor<3x1x6144x3xf16>, tensor<4xsi32>, tensor<3xsi32>, tensor<4xsi32>, ...)
```

eager 下这个错不显形：`mutable_slice_update` 里 `zip(begin, end, strict=False)`
把多出来的 `begin` 截掉，最后一维被当成整维切片，结果碰巧是对的。所以 `end` 取
`range(1, cache.dim())`。Qwen3.5 是 `SSMState` 在这个仓里的第一个调用方。

### `check-graph`：状态写入的串联只在 decomposition 之后才成立

`SSMState.update_states` 把 `mutable_slice_update` 的返回值丢掉了，纯副作用。
直接看 `torch.export.export` 的结果会发现 18 个写入全都以原始 placeholder 为
第 0 个参数、彼此没有先后关系，`graph_signature` 里也没有 `USER_INPUT_MUTATION`。
串联是在 `run_decompositions(coreai_torch.get_decomp_table())` 里由
auto-functionalization 建起来的，`remove_functionalization` 之后才变成
`immutable_slice_update` 链。

所以验证必须落在 converter 实际拿到的那张图上。24 层实测：

```
OK  k_cache          chained writes=6  (expected 6),  reaches mutated output=True
OK  v_cache          chained writes=6  (expected 6),  reaches mutated output=True
OK  conv_state       chained writes=18 (expected 18), reaches mutated output=True
OK  recurrent_state  chained writes=18 (expected 18), reaches mutated output=True
```

## 加载、量化与导出校验

### 1. 注册表前缀让 `from_hf_memory_efficient` 直接产出 decoder 的参数名

流式加载器按剥完前缀后的 key 分层和截断：`_build_safetensors_key_index` 用
`model\.layers\.(\d+)\.` 把 key 分到各层，`_is_layer_key_beyond` 用 `\.layers\.(\d+)\.`
做 `--num-layers` 截断。分到层的 key 赋值前还会经过 `_mutate_state_dict`；分不到层的进
`shared_dict`，这条路径不调用 `_mutate_state_dict`，名字必须直接就是模型的参数名。

Qwen3.5 的文本权重在 `model.language_model.` 下，所以注册表里 `qwen3_5_text` 的
`hf_state_dict_prefix` 是 `model.language_`：剥完正好是 `model.layers.N.*`、
`model.embed_tokens.weight`、`model.norm.weight`，与模型参数同名，`base.py` 的分层、截断、
赋值直接用这些名字。Qwen3.5-0.8B / 2B / 4B 的 checkpoint 里别的 key 都不以 `model.language_` 开头（视觉塔在
`model.visual.`，MTP 头在 `mtp.`），也没有单独的 `lm_head.weight`（词表权重与 embedding 共享）。
回归测试是 `python/tests/test_model_units/test_models/test_qwen3_5.py` 的
`TestMemoryEfficientLoading`，流式加载与 `from_hf` 逐张量相等，含截到 2 层。`spike.py` 走 `from_hf`。

### 2. `MACOS_PRESETS["4bit"]` 的模块排除表包含 `RMSNormGated`

`presets.py` 的 `_TORCH_MODULE_EXCLUSIONS` 排掉 `SDPA` / `RoPE` / `RMSNorm` /
`RMSNormPlusOne` / `RMSNormGated`。`RMSNormGated` 定义在 `rms_norm.py`，Qwen3.5 的
`linear_attn.norm` 是它的第一个用户；不排除的话量化在它上面失败：

```
ValueError: axis 1 is out of bounds for tensor of rank 1
  ... rms_norm.py line 51, in forward: x = self._rmsnorm_impl(x, self.weight)
```

`RMSNormGated` 把 rank-1 的 `weight` 当作可量化 state 交给 `RMSNormImpl`，
而 preset 的 `per_block(axis=1, block_size=32)` 在 rank-1 张量上没有 axis 1。

排除它对其它模型是空操作（没有别的用户）。回归测试是同一文件的 `TestMacOSInt4Preset`。

`linear_attn.conv1d.weight` 形状 `(6144, 1, 4)`，axis 1 长度是 1，量化器打 warning 后
跳过（`Tensor size 1 along axis 1 is not divisible by block size 32. Skipping
quantization.`）。深度可分离卷积核一共 24KB，留在 fp16。

### 3. 没有 full attention 层的栈不能导出

key / value cache 只由 full attention 层写。`--num-layers` 取 1–3 时留下的全是
linear attention 层，trace 出来的图里 KV cache 不是被写的状态，converter 会拒掉
声明的四个状态名（iOS 的 prefill 入口还会先在空 cache 上取下标失败）。两个平台的
`validate_export_contract` 在 trace 之前就报错，要求至少截到第一个 full attention
层（Qwen3.5 是 `--num-layers 4`）；eager 用短栈不受影响。回归测试是
`TestExportNeedsAFullAttentionLayer`。

## 数值

`report()` 的 cos 用 float64 算，logits 有 ~400 万个元素，fp32 累加会把末几位吃掉。

| 对比 | seq | cos | max_abs | argmax |
|---|---|---|---|---|
| `GatedDeltaNet` vs HF（fp32，随机权重） | 8 | 1.00000000 | 1.3e-06 | — |
| `Attention` vs HF（fp32，随机权重） | 8 | 1.00000000 | 0.0 | — |
| 4 层 logits vs HF（fp32） | 8 | 1.00000000 | 2.6e-05 | 100% |
| 24 层 logits vs HF（fp32） | 16 | 1.00000000 | 4.9e-05 | 100% |
| 24 层 logits vs HF（fp32） | 512 | 1.00000000 | 5.9e-05 | 100% |
| 24 层 logits fp16 vs HF fp32 | 512 | 0.99999881 | 4.8e-02 | 99.22% |
| **对照**：HF fp16 vs HF fp32 | 512 | 0.99999876 | 4.6e-02 | 99.61% |
| prefill+step vs 一次 prefill（fp32） | 16 | 1.00000000 | 2.5e-05 | — |
| prefill+step vs 一次 prefill（fp16） | 16 | 0.99999780 | 2.1e-02 | — |

fp32 下 cos 全部 ≥ 0.99999999，远超 0.999 的目标。

单元测试（`python/tests/test_model_units/test_models/test_qwen3_5.py`）在一个小的随机
checkpoint 上守着同一件事：两个平台的 `GatedDeltaNet` 与整模型 logits 对 HF（fp32，
相对误差 2e-7 量级，门限 5e-6），以及带着 conv / recurrent 状态续跑（16 + 16 个 token
对一次 32 个）。线性注意力的头配置取两组：key 头数等于 value 头数（0.8B 的情形），
和一组 value 头共用一个 key 头（key 2 / value 4）。后一种在递推前把 q、k 按
`repeat_interleave` 扩到 value 头数，与 HF 逐元素一致。测试用的 checkpoint 把
`A_log` 设在慢衰减区间，递推状态跨调用留得住，带状态的用例才看得见它。

fp16 下 512 token argmax 掉到 99.22%（512 个位置里差 4 个）。同样条件下 HF 自己的
fp16 对 fp32 是 99.61%（差 2 个），cos 比我们还低一点点。也就是说 fp16 的漂移基本
是 fp16 本身的，delta rule 的递推状态没有额外放大它——至少在 512 token 这个长度上。

**没验证的**：更长序列（几千 token）上 fp16 recurrent state 的累积漂移。18 层
每层一个 `(16,128,128)` 的递推矩阵，误差理论上会随步数累积，512 token 看不出来
不代表 4k 看不出来。真要上长上下文，得先测这个。

`GatedDeltaUpdate` 返回的 state 是 `state.to(query.dtype)`，所以 fp16 模型的
`recurrentState` 就是 fp16（HF config 里 `mamba_ssm_dtype: float32`）。状态要用 fp32
的话，把喂给 composite 的 q/k/v 提成 fp32 即可（内部本来就转 fp32），代价是状态从
9.4MB 变 18.9MB，以及可能影响端上算子选择。这里保持 fp16。

## 导出结果

24 层 fp16，`main` + `prefill` 两个入口，`AIProgram.optimize()` 和
`save_asset()` 都过，产物 1.4GB（未量化）。asset 里能查到
`main` / `prefill` / `keyCache` / `valueCache` / `convState` / `recurrentState` /
`gated_delta_update`。

```
uv run python spike_qwen35/spike.py export --dtype fp16 --prefill --save out.aimodel
```

## int4 量化档

`--compression 4bit` 走 `export/compression.py` 的 `quantize_pytorch_model` +
`MACOS_PRESETS["4bit"]`（weight-only，eager 模式，per-block int4，block 32，
不做激活校准），量化完再走同一条导出与保存。默认关闭。

`state_indices` 对应 forward 里 `k_cache` 起的四个实参，即 `(2, 3, 4, 5)`。

```
uv run python spike_qwen35/spike.py export --dtype fp16 --prefill \
    --compression 4bit --save spike_qwen35/qwen3_5_0p8b_q4.aimodel
```

导出、`optimize()`、`save_asset()` 全过。体积
1,505,321,024 B（1.40 GiB）→ 424,524,817 B（405 MiB），**3.55×**。

int4 对 fp16 dense，eager 前向：

| 输入 | seq | cos | argmax |
|---|---|---|---|
| 随机 token id | 16 | 0.93141934 | 56.25% |
| 真实文本（tokenizer 编码） | 66 | 0.97333100 | 74.24% |

随机 id 严重偏离训练分布，那一行会把结论带偏，以真实文本为准。

### 0.97 是 preset 的量级，不是 Qwen3.5 的问题

同一段文本、同一个 `report()`、同一套 `MACOS_PRESETS["4bit"]`，
拿仓里已经在 `LLM_PRESETS` 里出 4bit 档的 qwen3-0.6b 做基线：

| 模型 | 层 / vocab | cos | argmax |
|---|---|---|---|
| Qwen3-0.6B（基线，已出 4bit 档） | 28 / 151936 | 0.96477300 | 69.70% |
| Qwen3.5-0.8B | 24 / 248320 | 0.97333100 | 74.24% |

**Qwen3.5 两个指标都比基线略好。** delta rule 递推与 linear attention 的投影并不比
普通投影对权重量化更敏感。0.96–0.97 就是这套 weight-only int4 group-32 preset
在这个尺寸上的正常量级；换句话说，
**「16–66 token 前向的 logits cos / argmax 一致率」这个指标本身对 int4 太苛刻、
分辨率太低**，不适合用来判定量化是否可用——一个已经出货 4bit 档的模型在它上面
也只有 0.965 / 69.7%。

结论：q4 产物没有已知的 Qwen3.5 特有问题，不需要对 `in_proj_qkv` / `out_proj`
做特殊 block 或 8bit。质量结论仍然要靠真实评测（perplexity / 下游任务），
那是整套 preset 层面的评测。

附带数据：排除掉 tied embedding / `lm_head`（共享一个 248320×1024 张量）再量化，
cos 0.931 → 0.940、argmax 56% → 75%（随机 id，16 token）。tied head 有贡献
但不是大头。

这份 int4 产物没有实跑过，证据只到静态检查与 eager 数值这一层。

## 没验证的

按重要性排：

1. **端上的算子选择与性能。** composite 在设备上会不会落到融合 kernel、
   `while_loop` 运行时怎么执行、prefill 图性能如何，都没有测量。
2. **长上下文下 fp16 recurrent state 的累积漂移**（只测到 512 token）。
3. **量化质量**：int4 档能导出能存，对 qwen3-0.6b 基线也没有异常，
   但只有 eager 前向的 logits 对比——没有 perplexity，没有下游任务。
   mixed 4bit/8bit 预设一次没试。
4. **prefill 的性能形状**：`GatedDeltaUpdate` 用 `while_loop` 逐 token 展开，
   prefill 是 O(prompt_len) 次串行迭代；HF 用的是 `chunk_size=64` 的 chunked kernel。
   端上若没被替换成 chunked 实现，长 prompt 的 prefill 会很慢。这是最可能出问题的地方。
5. batch > 1、非零起始 offset 的 prefill、超出 `--max-context-length` 的行为。
6. `--num-layers` 截断路径的数值只在 4 层测过一次。
7. **分组头的型号**（key 头少于 value 头）只在上面那个随机小配置上对过 HF；真权重、导出和端上都没跑。

## 上机前置条件

1. **状态由运行时的状态工厂接管。** `CoreAISequentialEngine` 与
   `CoreAISequentialVLMEngine` 把 KV 之外的定长状态分配、清零、每次调用绑定、随
   `reset()` 一起清零，并拒绝带这类状态的模型做部分回退（`reset(to:)` 到中间位置）。
2. **状态初值必须全零。** 两个状态都假设从零开始，新会话不清零就是错的——
   不会报错，只是输出错。
3. **`LLM_PRESETS` 没有 Qwen3.5 条目。** `coreai.llm.export` 按 HF id 导出要带
   `--experimental --compute-precision float16`；macOS 默认 4bit 与 iOS 默认调色板
   两条路径都能导出。
4. **`FORCE_DECOMPOSED_ROPE=1`。** `primitives/macos/rope.py` 注释提到 partial rotary
   的 MLIR lowering 有过 bug（类似 FLUX.2 的 rdar://178555985），Qwen3.5 正好是
   partial rotary（256 维转 64 维）。fp32 eager 对齐走的是 torch 实现，不是 lowering。
   上机第一轮输出如果是乱的，先试这个环境变量。

## iOS 静态图变体

`models/ios/qwen3_5.py` 是同一个解码器在 iOS 静态 shape 导出路径上的写法：递推换成
`primitives/ios/gated_delta.py` 的分块形式，一次调用就是一个 16 token 的块，图里没有循环；
两个线性注意力状态走 `primitives/ios/ssm_cache.py`。这里做到的是**能导出、数值对得上**：

- `ios_qwen35_parity.py`：fp32 下整个解码器对 macOS 解码器（后者对 HF 对过），交错 M-RoPE 对
  `Qwen3_5TextRotaryEmbedding`。
- `chunked_gdn_parity.py`：分块递推对 `mlx_lm` 的递推 kernel 与整层 fixture 逐元素比；
  `--export-check` 看导出的 MLIR 是直线、全程 fp16。
- 默认调色板预设下 24 层与 4 层都能导出；`test_qwen3_5.py` 的
  `test_ios_traces_write_every_declared_state` 守着四个状态都被图写到。

**仓里的 `CoreAIStaticShapeEngine` 跑不了它。** 导出的 `extend` 对调用方有三条要求，引擎现在
一条都不满足：

1. **位置输入是 `rope_cos` / `rope_sin`（各 `[1, 16, 64]`），不是 `position_ids`。** 图里的 rope
   表查询是个 composite，运行时用自己的一个 pass 去解析它，这个 pass 只撑得住一次查询、且喂它的
   必须就是图的位置输入本身。Qwen3.5 要对时间、高、宽三行位置各查一次：三行作三个输入、一个输入
   切三段、`[3, seq]` 一个张量，三种喂法在 delegate 特化时都是 SIGSEGV，GPU 与 ANE 一样。所以表
   查询放在宿主，位置行本来也得宿主算（要知道图像布局），算法就是 `InterleavedMRoPE`。引擎按名字里
   带 `pos` 找位置输入、往里填一维位置，找不到就抛 `has no position_ids input`。
2. **多两个状态：`conv_cache` `(18, 1, 6144, 1, 3)` 与 `recurrent_cache` `(18, 1, 16, 128, 128)`。**
   会话开始清零，每次调用传入，再把 `new_conv_cache` / `new_recurrent_cache` 接回去。引擎只分配、
   只传 `key_cache` / `value_cache`。
3. **每次调用必须是 16 个真 token。** 线性注意力层没有 mask，query 里每个位置都会推进两个状态，
   补位的也一样。引擎在 token 不足 16 个时只填前几格，剩下几格的补位会把状态带偏，之后每个 token
   都错，且不报错。现在只 trace 了 16 这一档，单 token 的 decode 图没有；要跑生成，得按查询长度各
   trace 一张图，或者给线性注意力层加一个真 token 数的输入。

## VLM 入口：inputs_embeds 解码器

`Qwen3_5ForCausalLMEmbeddings` 从合并好的 embedding 序列进模型，图输入是
`(inputs_embeds f16 [1,q,1024], position_ids int32 [1,p])`，四个状态、`prefill`
入口、`logits` 与 token-id 解码器同一张图；视觉特征由运行时自己 splice 进
image token 位置。`spike.py parity-vlm` 是它的对齐台：按 iOS probe 的 ChatML
模板拼 224 token 提示词、把 196 个 `<|image_pad|>` 位置换成参考视觉特征，与 HF
`inputs_embeds` 路径逐位比。

fp32 CPU，参考特征是 HF fp32 视觉塔对 448×448 测试图的输出（196×1024，由 `--features` 传入）：

| 指标 | 值 |
|---|---|
| prefill logits cos（全部 224 个位置） | 0.9999999999930574 |
| decode logits cos（11 步） | 0.9999999999983991 |
| argmax 一致率 | 1.0000 |
| max_abs / max_rel | 2.791e-04 / 8.168e-06 |

decode 步的 logits 单列一行，因为 `position_ids` 的 offset 语义只在解码步起作用，
而 token 链对它不敏感：把解码步的 position 整体偏 1，12 个 token 仍与 HF 逐个相同，
只有 decode cos 掉下来。

贪心 12 token，我们和 HF 完全一致：

```
[248068, 198, 760, 1156, 6587, 264, 11346, 3874, 314, 279, 3766, 2099]
'<think>\nThe user wants a detailed description of the provided image'
```

```
uv run python spike_qwen35/spike.py parity-vlm --dtype fp32 --gen-tokens 12 \
    --features <reference_features.npy> \
    --out <parity.json>
uv run python spike_qwen35/spike.py export --embeddings --dtype fp16 --prefill \
    --max-context-length 4096 --save spike_qwen35/qwen3_5_0p8b_embeds.aimodel
```

导出 101 s，产物 `main.mlirb` 1,505,320,590 B，比 token-id 解码器的 1,505,320,887 B
小 297 B：tied embedding 表经 `lm_head` 计入一次。

编译后的 fp16 产物在 Mac 上用 `coreai.runtime.AIModel` 实跑过：
`function_names ['main','prefill']`，`main` 的
`input_names ['inputs_embeds','position_ids']`、`output_names ['logits']`、
`state_names ['keyCache','valueCache','convState','recurrentState']`，224 token
prefill 出 `(1,224,248320)` float16 无 NaN，之后 11 步贪心解码，12 个 token 与上表
逐个相同。注意 macOS 26.5.2 下 `coreai.runtime` 绑的是 wheel 自带的后端，不是
iOS 27 的 lowering——这一跑证明图、契约和 fp16 数值，不证明端上算子选择。

bundle 的 `main` 是这张 embeddings 图，纯文本模式（`VLMPROBE_MODE=llm`）要把
`metadata.json` 换成 `metadata_textonly.json`（`main` 指
`qwen3_5_0p8b_textonly.aimodel`）：`input_ids` [1,N] 喂 rank-3 的 `inputs_embeds`
会在 `NDArrayDescriptor.resolvingDynamicDimensions` 上 fatalError。

### 验收门

| 门 | 判据 | 结果 |
|---|---|---|
| G1 | Mac 逻辑 parity：cos ≥0.999、argmax 一致率 ≥0.99、12 token 与 HF 一致 | 过（上表） |
| G2 | 真机 fp16-GPU 全链前 12 token == HF 参考 | **未过** |
| G3 | 8bit 调色板 ANE 前 12 token == 同批 fp16-GPU 运行 | **未过** |
| G4 | 真机 fp16-ANE 全链前 12 token == 同批 fp16-GPU 运行 | **未过**，跑不起来 |

数字取自 iOS 探针的端上运行记录。probe 不落 token id，前 12 token 是把
`outputText` 用 Qwen3.5-0.8B tokenizer 重编码得到的；HF 参考（M-RoPE）是
`[248068, 198, 760, 1156, 6587, 264, 11346, 3874, 314, 279, 3766, 2099]`。

- **G2**：fp16-GPU 全链（vision=gpu、llm=auto、64 token）出
  `[248068, 271, 248069, 271, 1919, 2099, 369, 264, 7309, 59076, 11, 12583]`
  （`<think>\n\n</think>\n\nThis image is a highly distorted…`），第 2 个 token 就
  分叉；8bit 调色板的 GPU 全链前 12 token 与它逐个相同，错得一样。没有哪一次
  fp16-GPU 全链运行与参考一致。
- **G3**：8bit 调色板 ANE 全链的前 12 token 与 HF 参考逐个相同，是唯一一条
  端上正确的全链证据；但 G3 要它等于 fp16-GPU 那条，而那条自己是错的。
- **G4**：fp16-ANE 全链的两次运行
  都是 `error=nilError`、`generatedTokenCount=0`、`promptTokenCount=0`，日志
  停在 `stage=prepare_main_llm` 之后、
  `stage=engine_init` 之前；把 `llm` 固定成 gpu
  同样 nilError。最小复现是 vision=neuralEngine + fp16 `metadata.json` 跑一次
  `VLMPROBE_MODE=full`。

G2 的差别落在视觉侧，不在解码器的特化产物上：错的 fp16-GPU 那条和对的 8bit-ANE 那条，
三个 specialization cache id 里两个逐字相同，只有随 vision 资产变的那个不同
（2d5857… / 1c7dde…）。vision-only 模式下 GPU fp16 特征喂 HF 与参考一致；full 模式下的
GPU ViT 特征没有单独验证过。

## 之后可以做

- **q/k/v 投影融合**：现在三个独立 Linear + 两个独立 RMSNorm + 两次 rope。
  qwen3.py 的做法是融成 `qkv_proj` + `qk_norm(n_heads=n_q+n_kv)` + 一次 rope，
  composite 调用数减半。Qwen3.5 的 `q_proj` 输出里 query 和 gate 按 head 交错，
  融合要多一步 state dict 重排；现在的写法与 HF 逐模块对应。
- **视觉塔**：这里只有 text decoder，`model.visual.*` / `mtp.*` 在加载时丢弃。
- **recurrent state 提到 fp32**、**`g` 保持 fp32 进 composite**：取决于长序列漂移的测量结果。

