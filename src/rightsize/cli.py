"""Command-line interface. Standard-library argparse only (lightweight rule).

Every subcommand mirrors an SDK function (docs/plans/F06-surfaces.md). Commands whose
feature has not landed print where its plan lives and exit 0. Colour and progress bars come
from the optional ``rich`` package (extras ``cli`` / ``llamacpp``); without it, plain text.
"""

from __future__ import annotations

import argparse
import json
import sys

from rightsize import __version__

_PLANS = {
    "plan": ("F5 recipe rendering", "docs/plans/F05-framework-registry.md"),
    "data": ("data package updates", "data/README.md"),
}
_ROADMAP = "https://github.com/the-anup-das/rightsize-ai/tree/main/"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rightsize", description="Quantize and fit any model to your hardware."
    )
    p.add_argument("--version", action="version", version=f"rightsize {__version__}")
    p.add_argument("--json", action="store_true", help="machine-readable output, no colour")
    p.add_argument("--no-color", action="store_true", help="plain text (also: NO_COLOR=1)")
    p.add_argument("-v", "--verbose", action="store_true", help="echo tool output as it runs")
    p.add_argument("-q", "--quiet", action="store_true", help="only failures and final tables")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("recommend", help="rank the models that fit your hardware, with plans")
    r.add_argument("--task", default="chat", choices=["chat", "coding", "agentic"])
    r.add_argument("--target-device", "--device", dest="target_device", default="detect",
                   help="where it will run: preset or catalogue name, @hf-user, or detect")
    r.add_argument("--finetune-device", default=None, help="where it is fine-tuned, if at all")
    r.add_argument("--mode", default="infer", choices=["infer", "lora", "qlora", "full"])
    r.add_argument("--model", default=None, help="model-first: every quantization of this model")
    r.add_argument("--quality", default=None,
                   choices=["near-lossless", "good", "noticeable", "any"],
                   help="quantization loss to accept (default noticeable; any with --model)")
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--allow-slow", action="store_true", help="include plans under 5 tok/s")
    r.add_argument("--commands", action="store_true", help="print the top plan's commands")

    e = sub.add_parser("estimate", help="memory and speed for one model on one device")
    e.add_argument("model", help="Hub id, e.g. Qwen/Qwen3-4B")
    e.add_argument("--device", default="detect", help="preset name or 'detect' (default)")
    e.add_argument("--quant", action="append", help="GGUF type, repeatable (default Q4_K_M)")
    e.add_argument("--runtime", default="llama.cpp")
    e.add_argument(
        "--mode",
        default="infer",
        choices=["infer", "lora", "qlora", "full"],
        help="infer (default), or the memory to fine-tune it",
    )
    e.add_argument("--seq-len", type=int, default=2048, help="training sequence length")
    e.add_argument("--batch", type=int, default=1, help="training batch size")
    e.add_argument("--ctx", type=int, default=8192)
    e.add_argument("--revision", default="main")
    e.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="memory bandwidth in GB/s, when we have none for your device",
    )

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
    q.add_argument("--eval", action="store_true", help="KL-divergence gate vs the 16-bit file")
    q.add_argument("--eval-chunks", type=int, default=100, help="perplexity chunks (0 = all)")
    q.add_argument("--imatrix-chunks", type=int, default=None)
    q.add_argument("--outtype", default="auto", choices=["auto", "f16", "bf16", "f32"])
    q.add_argument("--out", default="runs", help="run directory root")
    q.add_argument("--models-dir", default="models", help="snapshots and 16-bit GGUFs")
    q.add_argument("--tools", default=None, help="llama.cpp folder (default: .tools/llama.cpp)")
    q.add_argument("--gpu-layers", default="all")
    q.add_argument("--revision", default="main")
    q.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="memory bandwidth in GB/s, when we have none for your device",
    )
    q.add_argument("--dry-run", action="store_true", help="render every step, run nothing")

    b = sub.add_parser("bench", help="measure this machine's real memory bandwidth")
    b.add_argument("model", help="Hub id of the model in the GGUF, e.g. Qwen/Qwen3-0.6B")
    b.add_argument("--gguf", default=None, help="the .gguf to run (default: search runs/, models/)")
    b.add_argument("--quant", default="Q4_K_M", help="quantization of that file")
    b.add_argument("--device", default="detect")
    b.add_argument("--gpu-layers", default="all")
    b.add_argument("--tools", default=None)
    b.add_argument("--revision", default="main")
    b.add_argument("--no-save", action="store_true", help="print it but do not remember it")

    t = sub.add_parser("tools", help="install pinned toolchains into .tools/")
    t_sub = t.add_subparsers(dest="tools_command")
    ti = t_sub.add_parser("install", help="download a toolchain (binaries + converter)")
    ti.add_argument("name", choices=["llama.cpp"])
    ti.add_argument(
        "--backend",
        default=None,
        help="cuda-12.4, cuda-13.4, cpu, vulkan, rocm-10.0, sycl (default: auto)",
    )
    ti.add_argument(
        "--version", dest="tool_version", default=None, help="release tag (default: pinned)"
    )
    ti.add_argument("--dest", default=".tools/llama.cpp")

    sub.add_parser("mcp", help="run the MCP server on stdio (needs the mcp extra)")

    pl = sub.add_parser("plan", help="work with saved plans")
    pl_sub = pl.add_subparsers(dest="plan_command")
    pr = pl_sub.add_parser("render")
    pr.add_argument("plan_file")
    pr.add_argument("--framework")

    d = sub.add_parser("data", help="manage the rightsize-data package")
    d_sub = d.add_subparsers(dest="data_command")
    d_sub.add_parser("update")
    return p


def _console(args: argparse.Namespace):
    from rightsize._console import Console

    color = False if (args.no_color or args.json) else None
    return Console(color=color, verbose=args.verbose, quiet=args.quiet or args.json)


def _not_yet(command: str, as_json: bool) -> int:
    feature, plan = _PLANS[command]
    if as_json:
        print(json.dumps({"status": "not_implemented", "feature": feature, "plan": plan}))
    else:
        print(f"rightsize {command}: not implemented yet in this release.")
        print(f"  feature: {feature}")
        print(f"  plan:    {_ROADMAP}{plan}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from rightsize.hardware import detect

    dev = detect()
    if args.json:
        print(dev.model_dump_json(indent=2))
        return 0
    con = _console(args)
    con.table(
        ["device", "vendor", "memory", "bandwidth", "arch", "system RAM", "os"],
        [
            [
                dev.name,
                dev.vendor,
                f"{dev.memory_gib} GiB",
                f"{dev.bandwidth_gbps or '?'} GB/s",
                dev.compute_arch or "?",
                f"{dev.system_ram_gib or '?'} GiB",
                dev.os,
            ]
        ],
    )
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from rightsize._console import verdict_style
    from rightsize.catalog import facts
    from rightsize.fit import estimate, predicted_file_gb

    fx = facts(args.model, args.revision)
    dev = _device(args)
    if args.mode != "infer":
        return _estimate_training(args, fx, dev)
    quants = [q.upper() for q in (args.quant or ["Q4_K_M"])]
    results = {q: estimate(fx, q, dev, runtime=args.runtime, ctx=args.ctx) for q in quants}
    if args.json:
        payload = {
            "model": fx.model_dump(mode="json"),
            "device": dev.model_dump(mode="json"),
            "ctx": args.ctx,
            "results": {q: r.model_dump(mode="json") for q, r in results.items()},
        }
        print(json.dumps(payload, indent=2))
        return 0
    con = _console(args)
    con.title(f"{args.model} on {dev.name}")
    con.info(
        f"{fx.params_total / 1e9:.2f}B params, {fx.num_layers} layers, "
        f"kv_heads {fx.num_kv_heads}, head_dim {fx.head_dim}  |  "
        f"{dev.memory_gib} GiB, {dev.bandwidth_gbps or '?'} GB/s, ctx {args.ctx}"
    )
    rows, styles = [], []
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
        styles.append(verdict_style(r.verdict.value))
    con.table(
        ["quant", "file GB", "weights", "kv", "vram GB", "verdict", "speed", "conf"],
        rows,
        styles=styles,
    )
    first = next(iter(results.values()))
    con.debug(f"formula {first.formula_id}; notes: {'; '.join(first.notes)}")
    return 0


def _device(args: argparse.Namespace):
    """The resolved device, with an explicit bandwidth applied if one was given.

    Some hardware has no published bandwidth to find - NVIDIA gives laptop GPUs a bus
    width but no bandwidth - so someone holding the spec sheet can supply it directly
    rather than going without a speed estimate.
    """
    from rightsize.hardware import resolve

    dev = resolve(args.device)
    if getattr(args, "bandwidth", None):
        dev = dev.model_copy(update={"bandwidth_gbps": args.bandwidth})
    return dev


def _estimate_training(args: argparse.Namespace, fx, dev) -> int:
    """Fine-tuning memory: one row per mode, so LoRA against QLoRA is one glance."""
    from rightsize._console import verdict_style
    from rightsize.fit import estimate_finetune

    modes = ["qlora", "lora", "full"] if args.mode == "full" else [args.mode]
    results = {
        m: estimate_finetune(fx, dev, m, seq_len=args.seq_len, batch=args.batch) for m in modes
    }
    if args.json:
        print(json.dumps({m: r.model_dump(mode="json") for m, r in results.items()}, indent=2))
        return 0
    con = _console(args)
    con.title(f"fine-tune {args.model} on {dev.name}")
    rows, styles = [], []
    for m, r in results.items():
        b = r.breakdown
        rows.append(
            [
                m,
                f"{b['weights']:.2f}",
                f"{b['trainable_state']:.2f}",
                f"{b['activations']:.2f}",
                f"{b.get('published_minimum', 0) or '':}",
                f"{r.vram_gb:.2f}",
                r.verdict.value,
            ]
        )
        styles.append(verdict_style(r.verdict.value))
    con.table(
        ["mode", "weights", "trainable", "activations", "published min", "vram GB", "verdict"],
        rows,
        styles=styles,
    )
    for note in next(iter(results.values())).notes:
        con.debug(note)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from rightsize._console import verdict_style
    from rightsize.rules.recommend import recommend_for_model, recommend_result

    common = dict(
        finetune_device=args.finetune_device, mode=args.mode, ctx=args.ctx,
        allow_slow=args.allow_slow, top_k=args.top,
    )
    if args.model:
        result = recommend_for_model(
            args.model, args.target_device, task=args.task, quality=args.quality or "any",
            **common,
        )
    else:
        result = recommend_result(
            args.task, args.target_device, quality=args.quality or "noticeable", **common
        )
    if args.json:
        print(json.dumps([p.model_dump(mode="json") for p in result.plans], indent=2))
        return 0 if result.plans else 1
    con = _console(args)
    target = result.plans[0].steps[-1].device.name if result.plans else args.target_device
    con.title(f"{args.model or args.task} on {target}")
    if not result.plans:
        con.fail("nothing fits under these constraints")
        for why in _closest_misses(result.rejected):
            con.info("  " + why)
        return 1
    rows, styles = [], []
    for p in result.plans:
        serve = p.steps[-1]
        f = serve.fit
        rows.append([
            str(p.rank),
            p.model.ref.repo,
            serve.quant.variant if serve.quant else "",
            f"{f.vram_gb:.2f}",
            f"{f.speed:.0f}" if f.speed else "?",
            f.verdict.value,
            f"{p.quality_penalty:.3f}" if p.quality_penalty is not None else "",
            f"{p.score:.2f}",
        ])
        styles.append(verdict_style(f.verdict.value))
    con.table(["#", "model", "quant", "vram GB", "tok/s", "verdict", "+ppl", "score"], rows,
              styles=styles)
    for line in result.plans[0].trace:
        con.debug(line)
    if args.commands:
        con.out()
        for step in result.plans[0].render():
            con.out("$ " + step.text)
    return 0


def _closest_misses(rejected) -> list[str]:
    """One line per reason, most common first: why nothing made it."""
    from collections import Counter

    counts = Counter(r.reason.split(":")[0] for r in rejected)
    examples = {}
    for r in rejected:
        examples.setdefault(r.reason.split(":")[0], r)
    return [
        f"{n} x {kind}: e.g. {examples[kind].repo} {examples[kind].quant or ''} "
        f"- {examples[kind].reason}"
        for kind, n in counts.most_common(4)
    ]


def cmd_mcp(args: argparse.Namespace) -> int:
    from rightsize.mcp_server import main as serve

    return serve()


def cmd_frameworks(args: argparse.Namespace) -> int:
    from rightsize.registry import all_recipes

    recipes = all_recipes()
    if args.name:
        recipes = {k: v for k, v in recipes.items() if v.framework == args.name}
    if args.json:
        print(json.dumps({k: v.model_dump(mode="json") for k, v in recipes.items()}, indent=2))
        return 0
    con = _console(args)
    rows = [
        [r.framework, r.stage, r.id, r.version_tested or "", ", ".join(r.families)]
        for r in recipes.values()
    ]
    con.table(["framework", "stage", "recipe", "tested", "families"], rows)
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from rightsize.execution.install import LLAMA_CPP_VERSION, install_llama_cpp

    con = _console(args)
    if args.tools_command != "install":
        print("usage: rightsize tools install llama.cpp [--backend ...]")
        return 2
    dest = install_llama_cpp(
        args.dest,
        version=args.tool_version or LLAMA_CPP_VERSION,
        backend=args.backend,
        log=con.info,
    )
    con.ok(f"installed into {dest}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    from rightsize._console import error_style
    from rightsize.execution import quantize_model

    con = _console(args)
    con.title(f"rightsize quantize {args.model}")
    manifest = quantize_model(
        args.model,
        args.quant or ["Q4_K_M"],
        device=_device(args),
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
        log=(lambda s: None) if args.json else con.info,
        console=None if args.json else con,
    )
    if args.json:
        print(manifest.model_dump_json(indent=2))
        return 0
    rows, styles = [], []
    for step in manifest.steps:
        for m in step.measurements:
            if m.predicted is None and m.kind not in ("kld_mean", "top1_agreement", "ppl"):
                continue
            err = (m.value - m.predicted) / m.predicted * 100 if m.predicted else None
            rows.append(
                [
                    step.recipe_id,
                    m.kind,
                    m.note or "",
                    f"{m.predicted:.3f}" if m.predicted is not None else "",
                    f"{m.value:.3f}",
                    f"{err:+.1f}%" if err is not None else "",
                ]
            )
            styles.append(error_style(err))
    if rows:
        con.out()
        con.table(
            ["step", "measure", "what", "predicted", "measured", "error"], rows, styles=styles
        )
    for q, g in manifest.gate.items():
        line = (
            f"gate {q}: {g['verdict']}  (mean KLD {g.get('kld_mean', '?')}, "
            f"top-1 agreement {g.get('top1_agreement', '?')})"
        )
        {"pass": con.ok, "warn": con.warn}.get(g["verdict"], con.fail)(line)
    (con.ok if manifest.status == "succeeded" else con.fail)(
        f"{manifest.status}: " + ", ".join(f"{k}={v}" for k, v in manifest.artifacts.items())
    )
    return 0 if manifest.status == "succeeded" else 1


def cmd_bench(args: argparse.Namespace) -> int:
    from rightsize.catalog import facts as hub_facts
    from rightsize.execution.bench import BenchError, measure
    from rightsize.hardware import resolve

    con = _console(args)
    fx = hub_facts(args.model, args.revision)
    dev = resolve(args.device)
    gguf = _find_gguf(args)
    if gguf is None:
        con.fail(
            f"no GGUF found for {args.model}. Pass --gguf, or make one with "
            f"'rightsize quantize {args.model} --quant {args.quant}'."
        )
        return 2
    try:
        record = measure(
            dev, fx, args.quant.upper(), gguf,
            tools=None if args.tools is None else __import__(
                "rightsize.execution.llamacpp", fromlist=["find_tools"]
            ).find_tools(args.tools),
            gpu_layers=args.gpu_layers,
            save=not args.no_save,
            log=(lambda s: None) if args.json else con.info,
        )
    except BenchError as exc:
        con.fail(str(exc))
        return 1
    if args.json:
        print(json.dumps({dev.name: record}, indent=2))
        return 0
    con.ok(f"{dev.name}: {record['bandwidth_gbps']} GB/s effective")
    return 0


def _find_gguf(args: argparse.Namespace):
    """The file named on the command line, else the newest match under runs/ or models/."""
    from pathlib import Path

    if args.gguf:
        p = Path(args.gguf)
        return p if p.exists() else None
    slug = args.model.replace("/", "__")
    found = [
        p
        for root in (Path("runs"), Path("models"))
        if root.is_dir()
        for p in root.rglob(f"{slug}*{args.quant.upper()}*.gguf")
    ]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handlers = {
        "mcp": cmd_mcp,
        "recommend": cmd_recommend,
        "detect": cmd_detect,
        "estimate": cmd_estimate,
        "frameworks": cmd_frameworks,
        "quantize": cmd_quantize,
        "tools": cmd_tools,
        "bench": cmd_bench,
    }
    if args.command in handlers:
        return handlers[args.command](args)
    return _not_yet(args.command, args.json)


if __name__ == "__main__":
    sys.exit(main())
