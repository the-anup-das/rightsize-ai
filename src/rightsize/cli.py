"""Command-line interface. Standard-library argparse only (lightweight rule).

Every subcommand mirrors an SDK function (docs/plans/F06-surfaces.md). Until
the feature lands, a subcommand prints where its plan lives and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys

from rightsize import __version__

_PLANS = {
    "recommend": ("F4 rules and ranking", "docs/plans/F04-rules-engine.md"),
    "estimate": ("F3 fit engine", "docs/plans/F03-fit-engine.md"),
    "detect": ("F2 hardware detection", "docs/plans/F02-hardware.md"),
    "frameworks": ("F5 framework registry", "docs/plans/F05-framework-registry.md"),
    "plan": ("F5 recipe rendering", "docs/plans/F05-framework-registry.md"),
    "data": ("data package updates", "data/README.md"),
}

_ROADMAP = "https://github.com/the-anup-das/rightsize-ai/tree/main/"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rightsize",
        description="Quantize and fit any model to your hardware.",
    )
    p.add_argument("--version", action="version", version=f"rightsize {__version__}")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("recommend", help="hardware-first: rank models that fit your devices")
    r.add_argument("--task", help="e.g. coding, chat, agentic, stt, image")
    r.add_argument("--family", help="llm | diffusion | audio | vision | embedding")
    r.add_argument("--finetune-device", help="preset name or 'detect'")
    r.add_argument("--target-device", help="preset name or 'detect'")
    r.add_argument("--framework", help="pin a framework, e.g. unsloth, mlx, llama.cpp")

    e = sub.add_parser("estimate", help="memory and speed for one model on one device")
    e.add_argument("model", nargs="?", help="Hub id or GGUF path")
    e.add_argument("--device", help="preset name or 'detect'")
    e.add_argument("--quant", help="e.g. Q4_K_M, nf4, fp8")
    e.add_argument("--runtime", help="llama.cpp, ollama, vllm, mlx_lm, ...")
    e.add_argument("--ctx", type=int, help="context length")
    e.add_argument("--mode", choices=["infer", "lora", "qlora", "full"], default="infer")

    sub.add_parser("detect", help="detect this machine as a device")

    f = sub.add_parser("frameworks", help="list supported frameworks and toolkits")
    f.add_argument("name", nargs="?")

    pl = sub.add_parser("plan", help="work with saved plans")
    pl_sub = pl.add_subparsers(dest="plan_command")
    pr = pl_sub.add_parser("render", help="render a plan's recipe for a framework")
    pr.add_argument("plan_file")
    pr.add_argument("--framework")

    d = sub.add_parser("data", help="manage the rightsize-data package")
    d_sub = d.add_subparsers(dest="data_command")
    d_sub.add_parser("update", help="fetch the latest data release into the local cache")

    return p


def _not_yet(command: str, as_json: bool) -> int:
    feature, plan = _PLANS[command]
    if as_json:
        print(json.dumps({"status": "not_implemented", "feature": feature, "plan": plan}))
    else:
        print(f"rightsize {command}: not implemented yet in this placeholder release.")
        print(f"  feature: {feature}")
        print(f"  plan:    {_ROADMAP}{plan}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return _not_yet(args.command, args.json)


if __name__ == "__main__":
    sys.exit(main())
