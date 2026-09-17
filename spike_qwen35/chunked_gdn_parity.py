"""Element-wise check of the chunked gated delta recurrence against the MLX oracle.

Two fixtures, both frozen:

``qwen35-gated-delta`` is the recurrence alone, from ``mlx_lm``'s own Metal kernel, on
random ``q``/``k``/``v``. ``qwen35-gdn-layer`` is a whole linear-attention layer — the four
projections, the depthwise conv and its rolling state, both gates, the two q/k scales, the
gated RMSNorm and the output projection — from ``mlx_lm``'s ``GatedDeltaNet``.

For each fixture the run sweeps chunk size and Horner order, in fp32 and in fp16, and
prints the error next to the floor that dtype imposes: the fixture's own reference rounded
to that dtype and back. A row at the floor is the same model; a row above it is not.

``--export-check`` additionally traces the module at every shape the iOS static ladder
uses and reports what the emitted MLIR contains, which is the other half of the claim:
straight-line and fp16 throughout.

    uv run python spike_qwen35/chunked_gdn_parity.py --fixtures DIR [--json OUT]
    uv run python spike_qwen35/chunked_gdn_parity.py --export-check
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from coreai_models.primitives.ios.gated_delta import ChunkedGatedDelta
from safetensors.torch import load_file

CONTROL_FLOW = re.compile(
    r"\b((?:coreai|scf)\.\w*(?:while|condition|yield)\w*)\b", re.I
)
LADDER = ((1, 1), (8, 8), (16, 16), (16, 64), (64, 64))
KERNEL_CASES = ("prefill_f32", "prefill_bf16", "decode_f32", "decode_bf16")
HEAD_K_DIM = HEAD_V_DIM = 128
N_HEADS = 16
CONV_KERNEL = 4


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class SequentialGatedDelta(torch.nn.Module):
    """The recurrence written as a loop, in whatever dtype it is handed.

    The control arm: it is the same arithmetic in the same precision, so any error the
    chunked form does not share with it belongs to the chunked form.
    """

    def forward(  # noqa: PLR0913
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.exp(log_g)
        outputs = []
        for t in range(query.shape[-2]):
            k = key[..., t, :]
            state = state * decay[..., t, None, None]
            delta = (value[..., t, :] - (state * k.unsqueeze(-2)).sum(-1)) * beta[
                ..., t, None
            ]
            state = state + k.unsqueeze(-2) * delta.unsqueeze(-1)
            outputs.append((state * query[..., t, :].unsqueeze(-2)).sum(-1))
        return torch.stack(outputs, dim=-2), state


def error(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Absolute and relative distance, next to what the compared dtype can represent."""
    ref32 = ref.float()
    got32 = got.float()
    scale = ref32.abs().max().item()
    floor = (ref32.to(got.dtype).float() - ref32).abs().max().item()
    max_abs = (got32 - ref32).abs().max().item()
    return {
        "max_abs": max_abs,
        "rel": max_abs / scale if scale else math.inf,
        "floor": floor,
        "scale": scale,
        "over_floor": max_abs / floor if floor else math.inf,
    }


def run_kernel_case(fixture: Path, chunk: int, order: int | None, dtype: torch.dtype):
    d = load_file(str(fixture))
    q, k, v = (d[n].to(dtype).transpose(1, 2) for n in ("q", "k", "v"))
    g, beta = (d[n].float().transpose(1, 2) for n in ("g", "beta"))
    log_g = g.log().to(dtype)
    state = d["state_in"].to(dtype)
    kernel = (
        SequentialGatedDelta() if order is None else ChunkedGatedDelta(chunk, order)
    )
    y, state_out = kernel(q, k, v, log_g, beta.to(dtype), state)
    return error(y.transpose(1, 2), d["y"]), error(state_out, d["state_out"])


class GatedDeltaLayer(torch.nn.Module):
    """``mlx_lm``'s ``GatedDeltaNet`` with the recurrence replaced by the chunked form."""

    def __init__(self, weights: dict[str, torch.Tensor], chunk: int, order: int | None):
        super().__init__()
        self.w = weights
        self.gdn = (
            SequentialGatedDelta() if order is None else ChunkedGatedDelta(chunk, order)
        )
        self.key_dim = N_HEADS * HEAD_K_DIM

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor, state: torch.Tensor):
        w, dtype = self.w, x.dtype
        batch, seq, _ = x.shape

        qkv = x @ w["in_proj_qkv.weight"].T
        z = (x @ w["in_proj_z.weight"].T).reshape(batch, seq, N_HEADS, HEAD_V_DIM)
        b = x @ w["in_proj_b.weight"].T
        a = x @ w["in_proj_a.weight"].T

        conv_in = torch.cat([conv_state, qkv], dim=1)
        # mlx holds the depthwise kernel as [channels, taps, 1] over an NLC tensor.
        kernel = w["conv1d.weight"].permute(0, 2, 1)
        conv_out = F.silu(
            F.conv1d(conv_in.transpose(1, 2), kernel, groups=kernel.shape[0]).transpose(
                1, 2
            )
        )
        conv_state_out = conv_in[:, -(CONV_KERNEL - 1) :, :]

        q, k, v = (
            t.reshape(batch, seq, N_HEADS, -1).transpose(1, 2)
            for t in conv_out.split([self.key_dim, self.key_dim, self.key_dim], dim=-1)
        )
        inv_scale = HEAD_K_DIM**-0.5
        q = (inv_scale**2) * rms_norm(q)
        k = inv_scale * rms_norm(k)

        log_g = -torch.exp(w["A_log"].float()) * F.softplus((a + w["dt_bias"]).float())
        beta = torch.sigmoid(b)
        y, state_out = self.gdn(
            q, k, v, log_g.transpose(1, 2).to(dtype), beta.transpose(1, 2), state
        )

        gated = (
            F.silu(z.float())
            * rms_norm(y.transpose(1, 2).float())
            * w["norm.weight"].float()
        )
        out = gated.to(dtype).reshape(batch, seq, -1) @ w["out_proj.weight"].T
        return out, conv_state_out, state_out


def _layer_regime_inputs(
    d: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` and log decay the layer hands the recurrence, from its own weights."""
    layer = GatedDeltaLayer({k: v for k, v in d.items()}, 1, 0)
    x = d["x"]
    conv_state = d.get("conv_state_in", torch.zeros(1, CONV_KERNEL - 1, 6144))
    qkv = x @ d["in_proj_qkv.weight"].T
    conv_in = torch.cat([conv_state, qkv], dim=1)
    kernel = d["conv1d.weight"].permute(0, 2, 1)
    conv_out = F.silu(
        F.conv1d(conv_in.transpose(1, 2), kernel, groups=kernel.shape[0]).transpose(
            1, 2
        )
    )
    k = conv_out[..., layer.key_dim : 2 * layer.key_dim].reshape(
        x.shape[0], x.shape[1], N_HEADS, -1
    )
    a = x @ d["in_proj_a.weight"].T
    log_g = -torch.exp(d["A_log"]) * F.softplus(a + d["dt_bias"])
    return HEAD_K_DIM**-0.5 * rms_norm(k), log_g


def run_layer_case(fixture: Path, chunk: int, order: int | None, dtype: torch.dtype):
    d = load_file(str(fixture))
    w = {
        k: v.to(dtype)
        for k, v in d.items()
        if k not in ("x", "y", "state_in", "state_out")
    }
    x = d["x"].to(dtype)
    conv_state = d.get("conv_state_in", torch.zeros(1, CONV_KERNEL - 1, 6144)).to(dtype)
    state = d.get("state_in", torch.zeros(1, N_HEADS, HEAD_V_DIM, HEAD_K_DIM)).to(dtype)
    y, conv_out, state_out = GatedDeltaLayer(w, chunk, order)(x, conv_state, state)
    return (
        error(y, d["y"]),
        error(state_out, d["state_out"]),
        error(conv_out, d["conv_state_out"]),
    )


def _orders(chunk: int) -> list[int | None]:
    """Horner steps worth reporting: a few truncations, the exact one, then the loop."""
    return [
        *sorted({o for o in (0, 1, 2, 4, 8, chunk - 1) if 0 <= o <= chunk - 1}),
        None,
    ]


def regime(fixture: Path, key_from) -> str:
    """What the recurrence is actually fed: the key norm and the decay it decays by.

    The model normalises ``k`` to unit rows before the recurrence, so a fixture whose keys
    are longer than that drives the state, and every intermediate of the chunked form,
    to a magnitude the model never reaches.
    """
    k, log_g = key_from(load_file(str(fixture)))
    norms = k.float().pow(2).sum(-1).sqrt()
    return (
        f"|k| rows {norms.min():.2f}-{norms.max():.2f}   "
        f"log decay per step {log_g.float().min():.2f}-{log_g.float().max():.2f}"
    )


def _label(order: int | None) -> str:
    return "loop" if order is None else str(order)


def fmt(e: dict[str, float]) -> str:
    return f"{e['max_abs']:.3e} (rel {e['rel']:.2e}, floor {e['floor']:.3e}, x{e['over_floor']:.1f})"


def export_check() -> None:
    """Trace the module at the ladder's shapes and report the graph it emits."""
    from coreai_models.export.macos import export_to_coreai

    print("=== exported graph ===")
    print(
        f"{'C':>3}{'T':>5}{'ord':>5}{'mlir lines':>12}{'control flow':>16}{'f32 tensors':>13}"
    )
    for chunk, seq in LADDER:
        module = ChunkedGatedDelta(chunk).eval().half()
        reference = {
            "query": torch.randn(1, N_HEADS, seq, HEAD_K_DIM, dtype=torch.float16),
            "key": torch.randn(1, N_HEADS, seq, HEAD_K_DIM, dtype=torch.float16),
            "value": torch.randn(1, N_HEADS, seq, HEAD_V_DIM, dtype=torch.float16),
            "log_g": -torch.rand(1, N_HEADS, seq, dtype=torch.float16),
            "beta": torch.rand(1, N_HEADS, seq, dtype=torch.float16),
            "state": torch.zeros(
                1, N_HEADS, HEAD_V_DIM, HEAD_K_DIM, dtype=torch.float16
            ),
        }
        program = export_to_coreai(
            module, reference, dynamic_shapes=None, output_names=("y", "state_out")
        )
        program.optimize()
        text = str(program)
        found = sorted(set(CONTROL_FLOW.findall(text))) or ["none"]
        print(
            f"{chunk:>3}{seq:>5}{module.order:>5}{len(text.splitlines()):>12}"
            f"{','.join(found):>16}{len(re.findall('xf32>', text)):>13}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixtures",
        type=Path,
        help="directory holding qwen35-gated-delta/ and qwen35-gdn-layer/",
    )
    ap.add_argument("--json", type=Path)
    ap.add_argument("--export-check", action="store_true")
    args = ap.parse_args()

    if args.export_check:
        export_check()
        return
    if args.fixtures is None:
        ap.error("--fixtures is required unless --export-check is given")

    rows = []
    dtypes = (("fp32", torch.float32), ("fp16", torch.float16))

    print("=== kernel fixture: qwen35-gated-delta ===")
    for case in KERNEL_CASES:
        path = args.fixtures / "qwen35-gated-delta" / f"{case}.safetensors"
        print(f"  {case:<14}{regime(path, lambda d: (d['k'], d['g'].float().log()))}")
    print(f"{'case':<14}{'dtype':<7}{'C':>3}{'ord':>6}   {'y':<48}state")
    for case in KERNEL_CASES:
        path = args.fixtures / "qwen35-gated-delta" / f"{case}.safetensors"
        chunks = (1,) if case.startswith("decode") else (16, 8, 4)
        for chunk in chunks:
            orders = _orders(chunk)
            for order in orders:
                for name, dtype in dtypes:
                    ey, es = run_kernel_case(path, chunk, order, dtype)
                    print(
                        f"{case:<14}{name:<7}{chunk:>3}{_label(order):>6}   {fmt(ey):<48}{fmt(es)}"
                    )
                    rows.append(
                        {
                            "fixture": "kernel",
                            "case": case,
                            "dtype": name,
                            "chunk": chunk,
                            "order": order,
                            "y": ey,
                            "state": es,
                        }
                    )

    print("\n=== layer fixture: qwen35-gdn-layer ===")
    for case in ("layer_prefill", "layer_decode"):
        path = args.fixtures / "qwen35-gdn-layer" / f"{case}.safetensors"
        print(f"  {case:<14}{regime(path, _layer_regime_inputs)}")
    print(f"{'case':<14}{'dtype':<7}{'C':>3}{'ord':>6}   {'y':<48}state")
    for case, chunks in (("layer_prefill", (16, 8, 4)), ("layer_decode", (1,))):
        path = args.fixtures / "qwen35-gdn-layer" / f"{case}.safetensors"
        for chunk in chunks:
            orders = _orders(chunk)
            for order in orders:
                for name, dtype in dtypes:
                    ey, es, ec = run_layer_case(path, chunk, order, dtype)
                    print(
                        f"{case:<14}{name:<7}{chunk:>3}{_label(order):>6}   {fmt(ey):<48}{fmt(es)}"
                    )
                    rows.append(
                        {
                            "fixture": "layer",
                            "case": case,
                            "dtype": name,
                            "chunk": chunk,
                            "order": order,
                            "y": ey,
                            "state": es,
                            "conv_state": ec,
                        }
                    )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
