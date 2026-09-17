"""What survives when Qwen3.5's linear-attention recurrence meets the iOS static-shape path.

The iOS export lane compiles for fixed shapes and hands the result to the Neural Engine.
`GatedDeltaUpdate` reaches Core AI as a composite whose time axis is
`torch.ops.higher_order.while_loop`, so the first question is whether static shapes turn
that loop into straight-line arithmetic — which the ANE takes — or leave a loop region in
the program, which it does not.

The probe carries the composite alone, at Qwen3.5-0.8B's real per-layer shape, through
`TorchConverter` → `set_static_shape_config` → `optimize()` → `save_asset()`, and reports
what the emitted MLIR contains.

    uv run python spike_qwen35/ios_gdu_probe.py [--seq 64] [--out DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from coreai_models.export.macos import export_to_coreai
from coreai_torch.composite_ops import GatedDeltaUpdate

HEADS, DK, DV = 16, 128, 128
LOOP_OPS = re.compile(
    r"\b((?:coreai|scf|mlprogram)\.\w*(?:while|loop|cond)\w*)\b", re.I
)


class GatedDeltaUpdateOnly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gated_delta_update = GatedDeltaUpdate()

    def forward(self, query, key, value, g, beta, state):
        return self.gated_delta_update(query, key, value, g, beta, state)


def inputs(seq: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "query": torch.randn(1, HEADS, seq, DK, dtype=dtype),
        "key": torch.randn(1, HEADS, seq, DK, dtype=dtype),
        "value": torch.randn(1, HEADS, seq, DV, dtype=dtype),
        "g": torch.randn(1, HEADS, seq, dtype=dtype),
        "beta": torch.rand(1, HEADS, seq, dtype=dtype),
        "state": torch.zeros(1, HEADS, DK, DV, dtype=dtype),
    }


def report(tag: str, program) -> None:
    text = str(program)
    ops = sorted(set(LOOP_OPS.findall(text)))
    lines = len(text.splitlines())
    print(f"  {tag:<24} mlir_lines={lines:<7} control_flow_ops={ops or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("spike_qwen35"))
    args = parser.parse_args()

    ref = inputs(args.seq, torch.float16)
    module = GatedDeltaUpdateOnly().eval().half()

    print(f"GatedDeltaUpdate  heads={HEADS} dk={DK} dv={DV} seq={args.seq} fp16")

    program = export_to_coreai(
        module, ref, dynamic_shapes=None, output_names=("out", "new_state")
    )
    report("after conversion", program)

    program.set_static_shape_config(
        "main",
        {
            f'"{args.seq}"': {
                "query": (1, HEADS, args.seq, DK),
                "key": (1, HEADS, args.seq, DK),
                "value": (1, HEADS, args.seq, DV),
                "g": (1, HEADS, args.seq),
                "beta": (1, HEADS, args.seq),
                "state": (1, HEADS, DK, DV),
            }
        },
    )
    print("  set_static_shape_config    OK")

    program.optimize()
    report("after optimize", program)

    args.out.mkdir(parents=True, exist_ok=True)
    asset = args.out / f"gdu_static_s{args.seq}.aimodel"
    program.save_asset(asset)
    blob = asset / "main.mlirb"
    print(f"  saved                     {blob} {blob.stat().st_size} B")


if __name__ == "__main__":
    main()
