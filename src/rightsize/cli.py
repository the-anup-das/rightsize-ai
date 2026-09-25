"""Command-line interface. Standard-library argparse only (lightweight rule).

Every subcommand mirrors an SDK function (docs/plans/F06-surfaces.md). Commands whose
feature has not landed print where its plan lives and exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys

from rightsize import __version__

_PLANS = {
    "recommend": ("F4 rules and ranking", "docs/plans/F04-rules-engine.md"),
    "plan": ("F5 recipe rendering", "docs/plans/F05-framework-registry.md"),
    "data": ("data package updates", "data/README.md"),
}
_ROADMAP = "https://github.com/the-anup-das/rightsize-ai/tree/main/"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rightsize", description="Quantize and fit any model to your hardware."
    )
    p.add_argument("--version", action="version", version=f"rightsize {__version__}")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("recommend", help="hardware-first: rank models that fit your devices")
    r.add_argument("--task")
    r.add_argument("--family")
    r.add_argument("--finetune-device")
    r.add_argument("--target-device")
    r.add_argument("--framework")

    e = sub.add_parser("estimate", help="memory and speed for one model on one device")
    e.add_argument("model", help="Hub id, e.g. Qwen/Qwen3-4B")
    e.add_argument("--device", default="detect", help="preset name or 'detect' (default)")
    e.add_argument("--quant", action="append", help="GGUF type, repeatable (default Q4_K_M)")
    e.add_argument("--runtime", default="llama.cpp")
    e.add_argument("--ctx", type=int, default=8192)
    e.add_argument("--revision", default="main")

    sub.add_parser("detect", help="detect this machine as a device")

    f = sub.add_parser("frameworks", help="list frameworks and recipes in the registry")
    f.add_argument("name", nargs="?")

    q = sub.add_parser("quantize", help="predict, convert, quantize, evaluate and record a model")
    q.add_argument("model", help="Hub id, e.g. Qwen/Qwen3-4B")
    q.add_argument("--to", default="gguf", choices=["gguf"], help="output format (gguf for now)")
    q.add_argument("--quant", action="append", help="GGUF type, repeatable (default Q4_K_M)")
    q.add_argument("--device", default="detect")
    q.add_argument("--ctx", type=int, default=8192, help="context used for the VRAM prediction")
    q.add_argument("--imatrix", action="store_true", help="compute an importance matrix first")
    q.add_argument("--eval", action="store_true", help="KL-divergence gate vs the 16-bit reference")
    q.add_argument("--eval-chunks", type=int, default=100, help="perplexity chunks (0 = all)")
    q.add_argument("--imatrix-chunks", type=int, default=None)
    q.add_argument("--outtype", default="auto", choices=["auto", "f16", "bf16", "f32"])
    q.add_argument("--out", default="runs", help="run directory root")
    q.add_argument("--models-dir", default="models", help="snapshots and 16-bit GGUFs")
    q.add_argument("--tools", default=None, help="llama.cpp folder (default: .tools/llama.cpp)")
    q.add_argument("--gpu-layers", default="all")
    q.add_argument("--revision", default="main")
    q.add_argument("--dry-run", action="store_true", help="render every step, run nothing")

    pl = sub.add_parser("plan", help="work with saved plans")
    pl_sub = pl.add_subparsers(dest="plan_command")
    pr = pl_sub.add_parser("render")
    pr.add_argument("plan_file")
    pr.add_argument("--framework")

    d = sub.add_parser("data", help="manage the rightsize-data package")
    d_sub = d.add_subparsers(dest="data_command")
    d_sub.add_parser("update")
    return p


def _not_yet(command: str, as_json: bool) -> int:
    feature, plan = _PLANS[command]
    if as_json:
        print(json.dumps({"status": "not_implemented", "feature": feature, "plan": plan}))
    else:
        print(f"rightsize {command}: not implemented yet in this release.")
        print(f"  feature: {feature}")
        print(f"  plan:    {_ROADMAP}{plan}")
    return 0


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*header), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*[str(c) for c in r]) for r in rows]
    return "\n".join(lines)


def cmd_detect(args: argparse.Namespace) -> int:
    from rightsize.hardware import detect

    dev = detect()
    if args.json:
        print(dev.model_dump_json(indent=2))
    else:
        print(
            f"{dev.name}  vendor={dev.vendor}  memory={dev.memory_gb} GB  "
            f"bandwidth={dev.bandwidth_gbps} GB/s  arch={dev.compute_arch}  "
            f"ram={dev.system_ram_gb} GB  os={dev.os}"
        )
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from rightsize.catalog import facts
    from rightsize.fit import estimate, predicted_file_gb
    from rightsize.hardware import resolve

    fx = facts(args.model, args.revision)
    dev = resolve(args.device)
    quants = [q.upper() for q in (args.quant or ["Q4_K_M"])]
    results = {q: estimate(fx, q, dev, runtime=args.runtime, ctx=args.ctx) for q in quants}
    if args.json:
        print(
            json.dumps(
                {
                    "model": fx.model_dump(mode="json"),
                    "device": dev.model_dump(mode="json"),
                    "ctx": args.ctx,
                    "results": {q: r.model_dump(mode="json") for q, r in results.items()},
                },
                indent=2,
            )
        )
        return 0
    print(
        f"{args.model}: {fx.params_total / 1e9:.2f}B params, {fx.num_layers} layers, "
        f"kv_heads {fx.num_kv_heads}, head_dim {fx.head_dim}  |  "
        f"{dev.name} {dev.memory_gb} GB, ctx {args.ctx}"
    )
    rows = []
    for q, r in results.items():
        rows.append(
            [
                q,
                f"{predicted_file_gb(fx, q):.2f}",
                f"{r.breakdown['weights']:.2f}",
                f"{r.breakdown['kv_cache']:.2f}",
                f"{r.vram_gb:.2f}",
                r.verdict.value,
                f"{r.speed:.0f} {r.speed_unit}" if r.speed else "?",
                f"{r.confidence:.1f}",
            ]
        )
    print(
        _table(rows, ["quant", "file GB", "weights", "kv", "vram GB", "verdict", "speed", "conf"])
    )
    first = next(iter(results.values()))
    print(f"formula {first.formula_id}; notes: {'; '.join(first.notes)}")
    return 0


def cmd_frameworks(args: argparse.Namespace) -> int:
    from rightsize.registry import all_recipes

    recipes = all_recipes()
    if args.name:
        recipes = {k: v for k, v in recipes.items() if v.framework == args.name}
    if args.json:
        print(json.dumps({k: v.model_dump(mode="json") for k, v in recipes.items()}, indent=2))
        return 0
    rows = [
        [r.framework, r.stage, r.id, r.version_tested or "", ", ".join(r.families)]
        for r in recipes.values()
    ]
    print(_table(rows, ["framework", "stage", "recipe", "tested", "families"]))
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    from rightsize.execution import quantize_model

    manifest = quantize_model(
        args.model,
        args.quant or ["Q4_K_M"],
        device=args.device,
        ctx=args.ctx,
        imatrix=args.imatrix,
        evaluate=args.eval,
        out_dir=args.out,
        models_dir=args.models_dir,
        tools=args.tools,
        outtype=args.outtype,
        imatrix_chunks=args.imatrix_chunks,
        eval_chunks=(args.eval_chunks or None),
        gpu_layers=args.gpu_layers,
        revision=args.revision,
        dry_run=args.dry_run,
        log=(lambda s: None) if args.json else (lambda s: print(s, flush=True)),
    )
    if args.json:
        print(manifest.model_dump_json(indent=2))
        return 0
    rows = []
    for step in manifest.steps:
        for m in step.measurements:
            if m.predicted is not None or m.kind in ("kld_mean", "top1_agreement", "ppl"):
                rows.append(
                    [
                        step.recipe_id,
                        m.kind,
                        m.note or "",
                        f"{m.predicted:.3f}" if m.predicted is not None else "",
                        f"{m.value:.3f}",
                        f"{(m.value - m.predicted) / m.predicted * 100:+.1f}%"
                        if m.predicted
                        else "",
                    ]
                )
    print()
    print(_table(rows, ["step", "measure", "what", "predicted", "measured", "error"]))
    for q, g in manifest.gate.items():
        print(f"gate {q}: {g['verdict']}")
    print(
        f"status {manifest.status}; artifacts: "
        + ", ".join(f"{k}={v}" for k, v in manifest.artifacts.items())
    )
    return 0 if manifest.status == "succeeded" else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handlers = {
        "detect": cmd_detect,
        "estimate": cmd_estimate,
        "frameworks": cmd_frameworks,
        "quantize": cmd_quantize,
    }
    if args.command in handlers:
        return handlers[args.command](args)
    return _not_yet(args.command, args.json)


if __name__ == "__main__":
    sys.exit(main())
