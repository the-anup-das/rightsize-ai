"""Command-line interface. Standard-library argparse only (lightweight rule).

Every subcommand mirrors an SDK function (docs/plans/F06-surfaces.md). Commands whose
feature has not landed print where its plan lives and exit 0. Colour and progress bars come
from the optional ``rich`` package (extras ``cli`` / ``llamacpp``); without it, plain text.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from rightsize import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rightsize", description="Quantize and fit any model to your hardware."
    )
    p.add_argument("--version", action="version", version=f"rightsize {__version__}")
    p.add_argument("--json", action="store_true", help="machine-readable output, no colour")
    p.add_argument("--no-color", action="store_true", help="plain text (also: NO_COLOR=1)")
    p.add_argument("-v", "--verbose", action="store_true", help="echo tool output as it runs")
    p.add_argument("-q", "--quiet", action="store_true", help="only failures and final tables")
    p.add_argument(
        "--offline", dest="offline_all", action="store_true",
        help="no network: model facts and prices from the cache only (also RIGHTSIZE_OFFLINE=1)",
    )
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
    r.add_argument(
        "--cloud", action="store_true",
        help="plan a fine-tune that does not fit on the cheapest rental GPU it fits",
    )
    r.add_argument(
        "--framework", default=None,
        help="use this toolkit where it has a role: a trainer (unsloth, trl, axolotl, mlx-lm) "
        "for the fine-tune, a server (ollama, llama.cpp) for the GGUF",
    )

    ca = sub.add_parser(
        "calibrate",
        help="compare predictions with what Ollama or LM Studio models hold on the GPU",
    )
    ca.add_argument("--device", default="detect")
    ca.add_argument("--no-record", action="store_true", help="compare only; never write")

    te = sub.add_parser("telemetry", help="local calibration records: status, on, off, export")
    te.add_argument("action", choices=["status", "on", "off", "show", "export"])
    te.add_argument("file", nargs="?", help="export: where to write the records")
    te.add_argument("--yes", action="store_true", help="on: accept without the prompt")

    cl = sub.add_parser("cloud", help="the cheapest rental GPUs for a memory need or a fine-tune")
    cl.add_argument("--vram", type=float, default=None, help="GB the job needs")
    cl.add_argument("--model", default=None, help="size the need from this model's fine-tune")
    cl.add_argument("--mode", default="qlora", choices=["lora", "qlora", "full"])
    cl.add_argument("--tokens", default="10M", help="training tokens per epoch (10M, 250k, ...)")
    cl.add_argument("--epochs", type=int, default=1)
    cl.add_argument("--provider", action="append", help="limit to these (runpod, lambda, aws, ...)")
    cl.add_argument("--spot", action="store_true", help="spot / interruptible prices")
    cl.add_argument("--top", type=int, default=10)
    cl.add_argument("--offline", action="store_true", help="use cached prices only")

    e = sub.add_parser(
        "estimate",
        help="memory and speed for one model on one device: LLM, diffusion, audio, "
        "vision or embedding",
    )
    e.add_argument("model", help="Hub id, e.g. Qwen/Qwen3-4B or black-forest-labs/FLUX.1-dev")
    e.add_argument(
        "--file", default=None,
        help="one GGUF in the repo; a GGUF repo is otherwise sized at its Q4_K_M, or the "
        "nearest file it has",
    )
    e.add_argument("--device", default="detect", help="preset name or 'detect' (default)")
    e.add_argument(
        "--quant",
        action="append",
        help="GGUF type or format (bf16, fp8, nf4, int8, int4, awq, mlx-4bit, ...), "
        "repeatable; default Q4_K_M for LLMs, bf16 for diffusion",
    )
    e.add_argument(
        "--runtime",
        default=None,
        help="llama.cpp (LLM default), ollama; whisper.cpp, faster-whisper, transformers (audio)",
    )
    e.add_argument(
        "--mode",
        default="infer",
        choices=["infer", "lora", "qlora", "full"],
        help="infer (default), or the memory to fine-tune it",
    )
    e.add_argument(
        "--seq-len", type=int, default=None,
        help="training sequence length (default 2048), or embedding input length (512)",
    )
    e.add_argument(
        "--batch", type=int, default=None,
        help="training batch (default 1), images per prompt, or embedding batch (32)",
    )
    e.add_argument(
        "--offload",
        action="append",
        choices=["none", "model", "sequential"],
        help="diffusion: which weights stay on the GPU, repeatable to compare (default none)",
    )
    e.add_argument("--resolution", default="1024x1024", help="diffusion: WIDTHxHEIGHT")
    e.add_argument("--frames", type=int, default=None, help="diffusion: frames, for video")
    e.add_argument(
        "--te-quant", default=None, help="diffusion: format for the text encoders (e.g. nf4)"
    )
    e.add_argument(
        "--vae-slicing", action="store_true", help="diffusion: decode one image at a time"
    )
    e.add_argument("--ctx", type=int, default=8192)
    e.add_argument("--revision", default="main")
    e.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="memory bandwidth in GB/s, when we have none for your device",
    )

    va = sub.add_parser(
        "variants", help="quantized copies of a model already on the Hub: GGUF, MLX, AWQ, FP8, ..."
    )
    va.add_argument("model", help="Hub id of the model, or of any quantized copy of it")
    va.add_argument("--format", action="append", help="only this format (gguf, mlx, awq, ...)")
    va.add_argument("--limit", type=int, default=20, help="repos to list (default 20)")
    va.add_argument("--no-files", action="store_true", help="one line per repo, no sizes")
    va.add_argument("--all", action="store_true",
                    help="include repos whose name is not the base model's (fine-tunes, drafts)")

    se = sub.add_parser(
        "search", help="curated models by family, task and size: LLM, diffusion, audio, ..."
    )
    se.add_argument("--family", choices=["llm", "diffusion", "audio", "vision", "embedding"])
    se.add_argument("--task", help="chat, coding, text-to-image, text-to-speech, ...; "
                    "'--task list' shows them all")
    se.add_argument("--max-b", type=float, default=None, help="at most this many billion params")
    se.add_argument("--min-b", type=float, default=None, help="at least this many billion params")
    se.add_argument("--publisher", default=None)
    se.add_argument("--license", default=None, help="e.g. apache-2.0, mit")
    se.add_argument("--no-gated", action="store_true", help="leave out repos with terms to accept")
    se.add_argument("--limit", type=int, default=20)

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

    t = sub.add_parser("tools", help="install, list and remove toolkits (one per framework)")
    t_sub = t.add_subparsers(dest="tools_command")
    ti = t_sub.add_parser(
        "install", help="llama.cpp's binaries, or a Python toolkit in its own environment")
    ti.add_argument("name", help="a framework: llama.cpp, unsloth, llm-compressor, ...")
    ti.add_argument(
        "--backend",
        default=None,
        help="llama.cpp: cuda-12.4, cuda-13.4, cpu, vulkan, rocm-10.0, sycl (default: auto)",
    )
    ti.add_argument(
        "--version", dest="tool_version", default=None,
        help="llama.cpp: release tag (default: pinned)",
    )
    ti.add_argument("--dest", default=None, help="llama.cpp: where (default .tools/llama.cpp)")
    ti.add_argument("--python", default="3.12", help="Python toolkits: the version to use")
    t_sub.add_parser("list", help="each framework and where it is installed, if anywhere")
    tr = t_sub.add_parser("remove", help="delete a toolkit environment rightsize made")
    tr.add_argument("name")

    ru = sub.add_parser("run", help="carry out a saved plan on this machine, step by step")
    ru.add_argument("plan_file", help="a Plan as JSON, or a list of them (recommend --json)")
    ru.add_argument("--rank", type=int, default=1, help="which plan of a list (default 1)")
    ru.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a recipe input, e.g. --set max_steps=30")
    ru.add_argument("--workdir", default=None, help="run directory (default runs/<model>-<time>)")
    ru.add_argument("--models-dir", default="models", help="where Hub snapshots are downloaded")
    ru.add_argument("--install", action="store_true",
                    help="install a toolkit the plan needs when it is missing")
    ru.add_argument("--serve", action="store_true", help="also start the serve step")
    ru.add_argument("--dry-run", action="store_true", help="prepare every step, run none")

    sub.add_parser("mcp", help="run the MCP server on stdio (needs the mcp extra)")

    pl = sub.add_parser("plan", help="work with saved plans (recommend --json writes them)")
    pl_sub = pl.add_subparsers(dest="plan_command")
    pr = pl_sub.add_parser("render", help="the commands of a saved plan")
    pr.add_argument("plan_file", help="a Plan as JSON, or a list of them (recommend --json)")
    pr.add_argument("--rank", type=int, default=1, help="which plan of a list (default 1)")
    pr.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a recipe input, e.g. --set quantize_bin=/opt/llama-quantize")

    d = sub.add_parser("data", help="newer hardware, quantization, rules and recipe data")
    d_sub = d.add_subparsers(dest="data_command")
    du = d_sub.add_parser("update", help="download, validate and switch to newer data")
    du.add_argument("--ref", default="main", help="git ref of the project: main or a tag")
    d_sub.add_parser("status", help="which data is in use")
    d_sub.add_parser("reset", help="back to the data this release shipped with")
    return p


def _console(args: argparse.Namespace):
    from rightsize._console import Console

    color = False if (args.no_color or args.json) else None
    return Console(color=color, verbose=args.verbose, quiet=args.quiet or args.json)


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
    from rightsize.fit.llm import repo_file

    fx = facts(args.model, args.revision, file=args.file)
    dev = _device(args)
    if fx.family.value != "llm":
        return _estimate_other(args, fx, dev)
    if args.mode != "infer":
        return _estimate_training(args, fx, dev)
    own = (fx.extra or {}).get("gguf") or {}
    quants = [q.upper() for q in (args.quant or [own.get("quant") or "Q4_K_M"])]
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
    files = (fx.extra or {}).get("gguf_files") or {}
    if own:
        con.info(f"read from {fx.ref.file}; the repo has " + ", ".join(
            f"{q} {g['bytes'] / 1e9:.2f} GB" for q, g in sorted(
                files.items(), key=lambda kv: kv[1]["bytes"])))
    rows, styles = [], []
    for q, r in results.items():
        mine = repo_file(fx, q)
        rows.append(
            [
                q,
                f"{(mine['bytes'] / 1e9 if mine else predicted_file_gb(fx, q)):.2f}",
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
        m: estimate_finetune(fx, dev, m, seq_len=args.seq_len or 2048, batch=args.batch or 1)
        for m in modes
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


def _estimate_other(args: argparse.Namespace, fx, dev) -> int:
    """Diffusion, audio, vision and embedding models: one row per quant (and per offload
    strategy for diffusion), with the columns that family's estimate has."""
    from rightsize import _resolution
    from rightsize._console import verdict_style
    from rightsize.fit import estimate

    if args.mode != "infer":
        print(f"rightsize estimate: fine-tuning memory for {fx.family.value} models is not "
              "modelled yet (docs/plans/F03-fit-engine.md)", file=sys.stderr)
        return 1
    diffusion = fx.family.value == "diffusion"
    offloads = args.offload or ["none"] if diffusion else [None]
    quants = args.quant or [None]
    results = {}
    for q in quants:
        for off in offloads:
            r = estimate(
                fx, q, dev, runtime=args.runtime, batch=args.batch, offload=off,
                resolution=_resolution(args.resolution), frames=args.frames,
                text_encoder_quant=args.te_quant, vae_slicing=args.vae_slicing or None,
                seq_len=args.seq_len,
            )
            label = (q or "default") + (f"/{off}" if diffusion else "")
            results[label] = r
    if args.json:
        payload = {
            "model": fx.model_dump(mode="json"),
            "device": dev.model_dump(mode="json"),
            "results": {k: r.model_dump(mode="json") for k, r in results.items()},
        }
        print(json.dumps(payload, indent=2))
        return 0
    con = _console(args)
    con.title(f"{args.model} on {dev.name}")
    con.info(
        f"{fx.family.value}, {fx.params_total / 1e9:.2f}B params  |  {dev.memory_gib} GiB"
        + (f"  |  {args.resolution}" if diffusion else "")
    )
    rows, styles = [], []
    for label, r in results.items():
        b = r.breakdown
        if diffusion:
            rows.append([label, f"{b['weights']:.2f}", f"{b['activations']:.2f}",
                         f"{r.vram_gb:.2f}", f"{r.ram_gb:.1f}", r.verdict.value,
                         f"{r.confidence:.1f}"])
        else:
            rows.append([label, f"{b['weights']:.2f}",
                         f"{b.get('activations', b.get('overhead', 0)):.2f}",
                         f"{r.vram_gb:.2f}", r.verdict.value, f"{r.confidence:.1f}"])
        styles.append(verdict_style(r.verdict.value))
    headers = (["quant/offload", "weights", "activations", "vram GB", "ram GB", "verdict",
                "conf"] if diffusion else
               ["quant", "weights", "overhead", "vram GB", "verdict", "conf"])
    con.table(headers, rows, styles=styles)
    first = next(iter(results.values()))
    con.debug(f"formula {first.formula_id}")
    for r in results.values():
        for note in r.notes:
            con.debug(note)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from rightsize._console import verdict_style
    from rightsize.registry import get as get_recipe
    from rightsize.rules.recommend import recommend_for_model, recommend_result

    common = dict(
        finetune_device=args.finetune_device, mode=args.mode, ctx=args.ctx,
        allow_slow=args.allow_slow, top_k=args.top, cloud=args.cloud,
        framework=args.framework,
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
    headers = ["#", "model", "quant", "vram GB", "tok/s", "verdict", "+ppl", "score"]
    if args.mode != "infer":
        headers.append("fine-tune on")
        for row, p in zip(rows, result.plans, strict=True):
            fb = p.cloud_fallback
            row.append(f"rent {fb['offer']['gpu']} ${fb['offer']['usd_per_hour']:.2f}/h" if fb
                       else p.steps[0].device.name)
    con.table(headers, rows, styles=styles)
    for line in result.plans[0].trace:
        con.debug(line)
    rented = next((p for p in result.plans if p.cloud_fallback), None)
    if rented:
        con.info(rented.trace[-1])
    if args.commands:
        con.out()
        for step in result.plans[0].render():
            if step.kind == "command":
                con.out("$ " + step.text)
            else:  # a config: say where it goes, then its contents
                con.out(f"# {get_recipe(step.recipe_id).file_name()}:")
                con.out(step.text.rstrip())
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


def _count(text: str) -> int:
    """10M, 250k, 1.5B or a plain number."""
    t = text.strip().lower().replace("_", "")
    scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get(t[-1:], 1)
    return int(float(t[:-1] if scale != 1 else t) * scale)


def cmd_cloud(args: argparse.Namespace) -> int:
    from rightsize.cloud import cheapest, estimate_job, offers

    need, params = args.vram, None
    if args.model:
        from rightsize.catalog import facts
        from rightsize.fit import estimate_finetune
        from rightsize.types import Device

        fx = facts(args.model)
        params = fx.params_active or fx.params_total
        big = Device(name="sizing", vendor="nvidia", memory_gib=10_000)
        need = estimate_finetune(fx, big, args.mode).vram_gb
    if not need:
        print("rightsize cloud: give --vram GB or --model", file=sys.stderr)
        return 2
    pool, missing = offers(args.provider, offline=args.offline)
    found = cheapest(need, pool=pool, spot=args.spot, top=args.top)
    tokens = _count(args.tokens)
    jobs = {}
    if params:
        jobs = {id(o): estimate_job(params, tokens, o, epochs=args.epochs) for o in found}
    if args.json:
        print(json.dumps({
            "need_gb": need, "missing_providers": missing,
            "offers": [{**o.model_dump(mode="json"),
                        "job": jobs[id(o)].model_dump(mode="json") if params else None}
                       for o in found],
        }, indent=2))
        return 0 if found else 1
    con = _console(args)
    what = f"{args.model} ({args.mode})" if args.model else "a job"
    con.title(f"rental GPUs for {what} needing {need:.1f} GB")
    if missing:
        con.warn("no prices for " + ", ".join(missing) + (" (offline)" if args.offline else ""))
    if not found:
        con.fail("no single rental GPU has that much memory")
        return 1
    rows = []
    for o in found:
        row = [o.provider, o.gpu, f"{o.vram_gib:.0f}", f"{o.usd_per_hour:.2f}", o.region or ""]
        if params:
            j = jobs[id(o)]
            row += [f"{j.hours:g}" if j.hours is not None else "?",
                    f"{j.usd:,.2f}" if j.usd is not None else "?"]
        rows.append(row)
    headers = ["provider", "gpu", "GiB", "$/h", "region"]
    if params:
        headers += ["hours", f"$ for {args.tokens} tok"]
    con.table(headers, rows)
    if params and not any(jobs[id(o)].usd is not None for o in found):
        timed = [(estimate_job(params, tokens, o, epochs=args.epochs), o)
                 for o in cheapest(need, pool=pool, spot=args.spot, top=200)]
        timed = [(j, o) for j, o in timed if j.usd is not None]
        if timed:
            j, o = min(timed, key=lambda t: t[0].usd)
            con.info(f"cheapest with a time estimate: {o.gpu} on {o.provider} at "
                     f"${o.usd_per_hour:.2f}/h, about {j.hours:g} h and ${j.usd:,.2f}")
    con.info("prices: SkyPilot's open catalog (github.com/skypilot-org/skypilot-catalog), "
             "cached for a day; job time assumes 35% of datasheet throughput, confidence 0.3")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from rightsize.hardware import resolve
    from rightsize.telemetry import enabled
    from rightsize.telemetry.calibrate import calibrate

    dev = resolve(args.device)
    rows = calibrate(dev, write=not args.no_record)
    if args.json:
        print(json.dumps([{
            "runtime": c.runtime, "model": c.model, "quant": c.quant, "ctx": c.ctx,
            "predicted_gb": c.predicted_gb, "measured_gb": c.measured_gb, "error": c.error,
            "note": c.note,
        } for c in rows], indent=2))
        return 0 if rows else 1
    con = _console(args)
    con.title(f"predicted vs measured on {dev.name}")
    if not rows:
        con.fail("nothing loaded: start a model in Ollama or LM Studio, then run this again")
        return 1
    con.table(
        ["runtime", "model", "quant", "ctx", "predicted GB", "measured GB", "error", "note"],
        [[c.runtime, c.model, c.quant or "", str(c.ctx or ""),
          f"{c.predicted_gb:.2f}" if c.predicted_gb else "?",
          f"{c.measured_gb:.2f}" if c.measured_gb else "?",
          f"{c.error:+.1%}" if c.error is not None else "", c.note] for c in rows],
    )
    recorded = sum(1 for c in rows if c.record)
    if enabled() and not args.no_record:
        con.info(f"recorded {recorded} comparison(s) locally (rightsize telemetry show)")
    elif recorded:
        con.info("not recorded: 'rightsize telemetry on' keeps these, locally, to refit the "
                 "constants; nothing is sent")
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    from rightsize import telemetry
    from rightsize.telemetry import record

    if args.action == "on":
        print(telemetry.WHAT_IS_RECORDED)
        if not args.yes:
            if not sys.stdin.isatty():
                print()
                print("run again with --yes to turn recording on", file=sys.stderr)
                return 2
            print()
            try:  # Windows reports NUL as a terminal, so stdin can still be empty here
                answer = input("Turn recording on? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                print("left off")
                return 1
        telemetry.enable()
        print(f"recording on; records go to {telemetry.store_path()}")
        return 0
    if args.action == "off":
        telemetry.disable()
        print("recording off; existing records are kept until you delete the file")
        return 0
    if args.action == "export":
        if not args.file:
            print("rightsize telemetry export FILE", file=sys.stderr)
            return 2
        n = record.export(args.file)
        print(f"wrote {n} record(s) to {args.file}; read it before sharing it")
        return 0
    if args.action == "show":
        for r in record.read():
            print(r.model_dump_json())
        return 0
    st = telemetry.status()
    if args.json:
        print(json.dumps(st, indent=2))
    else:
        state = f"on since {st['since']}" if st["enabled"] else "off (the default)"
        print(f"recording: {state}")
        print(f"store: {st['store']} ({st['records']} records)")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from rightsize import data_update

    if args.data_command == "update":
        try:
            marker = data_update.update(args.ref)
        except data_update.DataUpdateError as exc:
            print(f"rightsize data update: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(marker, indent=2))
        else:
            print(f"data updated to {marker['repo']}@{marker['ref']} "
                  f"({marker['files']} files, validated); 'rightsize data reset' undoes it")
        return 0
    if args.data_command == "reset":
        removed = data_update.reset()
        print("back to the bundled data" if removed else "already on the bundled data")
        return 0
    st = data_update.status()
    if args.json:
        print(json.dumps(st, indent=2))
    else:
        print(f"in use: {st['in_use']}")
        if st["override"]:
            print("  (RIGHTSIZE_DATA_DIR overrides everything)")
        elif st["updated"]:
            u = st["updated"]
            print(f"  updated from {u['repo']}@{u['ref']} on {u['fetched_at']}")
        else:
            print("  the data this release shipped with")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from rightsize.types import Plan

    if args.plan_command != "render":
        print("rightsize plan render PLAN.json", file=sys.stderr)
        return 2
    raw = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
    plans = [Plan.model_validate(p) for p in (raw if isinstance(raw, list) else [raw])]
    plan = next((p for p in plans if p.rank == args.rank), None)
    if plan is None:
        print(f"no plan with rank {args.rank} in {args.plan_file}", file=sys.stderr)
        return 1
    inputs = dict(kv.split("=", 1) for kv in args.set)
    steps = plan.render(**inputs)
    if args.json:
        print(json.dumps([s.model_dump(mode="json") for s in steps], indent=2))
        return 0
    for step in steps:
        print(f"# {step.recipe_id} ({step.verified})")
        print(("$ " if step.kind == "command" else "") + step.text)
        print()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from rightsize.mcp_server import main as serve

    return serve()


def cmd_search(args: argparse.Namespace) -> int:
    from rightsize.catalog.curated import search, tasks

    if args.task == "list":
        known = tasks()
        if args.json:
            print(json.dumps(known, indent=2))
        else:
            for family, names in known.items():
                print(f"{family:10s} {', '.join(names)}")
        return 0
    found = search(
        args.family, args.task,
        max_params=args.max_b * 1e9 if args.max_b is not None else None,
        min_params=args.min_b * 1e9 if args.min_b is not None else None,
        publisher=args.publisher, license=args.license, include_gated=not args.no_gated,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([e.model_dump(mode="json") for e in found], indent=2))
        return 0
    con = _console(args)
    rows = [[e.family.value, e.repo, ", ".join(e.tasks), f"{e.params_total / 1e9:.2f}",
             str(e.license or "?"), "yes" if e.gated else "", f"{e.downloads_30d or 0:,}"]
            for e in found]
    con.table(["family", "repo", "tasks", "B params", "license", "gated", "downloads"], rows)
    if not found:
        con.info("nothing on the curated lists matches; any Hub id can still be estimated")
    return 0


def cmd_variants(args: argparse.Namespace) -> int:
    from rightsize.catalog import variants

    found = variants(args.model, formats=args.format, limit=args.limit, files=not args.no_files)
    hidden = [v for v in found if not v.name_matches_base]
    shown = found if args.all else [v for v in found if v.name_matches_base]
    if args.json:
        print(json.dumps([v.model_dump(mode="json") for v in shown], indent=2))
        return 0
    con = _console(args)
    con.title(f"published quantizations of {args.model}")
    rows = []
    for v in shown:
        who = "official" if v.official else ("known" if v.known_publisher else "")
        rows.append([
            v.format, v.ref.repo, v.quant or "",
            f"{v.size_bytes / 1e9:.2f}" if v.size_bytes else "?",
            f"{v.bits_per_weight:.2f}" if v.bits_per_weight else "?",
            f"{v.downloads:,}" if v.downloads is not None else "?", who,
        ])
    con.table(["format", "repo", "quant", "GB", "bpw", "downloads", "publisher"], rows)
    if hidden and not args.all:
        repos = sorted({v.ref.repo for v in hidden})
        con.info(f"{len(repos)} more whose names are not the base model's (fine-tunes or "
                 f"drafts published as quants): {', '.join(repos)}; --all shows them")
    return 0


def cmd_frameworks(args: argparse.Namespace) -> int:
    from rightsize.registry import all_recipes, framework, framework_infos
    from rightsize.registry.docs import _runs_on

    infos = framework_infos()
    if args.name:
        infos = {args.name: framework(args.name)}
    recipes = all_recipes()
    if args.json:
        print(json.dumps({
            name: {**info.model_dump(mode="json"),
                   "recipes": {k: r.model_dump(mode="json") for k, r in recipes.items()
                               if r.framework == name}}
            for name, info in infos.items()
        }, indent=2))
        return 0
    con = _console(args)
    if args.name:
        info = infos[args.name]
        con.title(info.title)
        con.info(info.summary)
        con.info(f"runs on {_runs_on(info)}; install: {info.install.line}")
        rows = [[r.stage, r.id, r.verified, r.version_tested or "", ", ".join(r.families)]
                for r in recipes.values() if r.framework == args.name]
        con.table(["stage", "recipe", "checked", "tested", "families"], rows)
        return 0
    rows = []
    for name, info in sorted(infos.items()):
        trains = ""
        if info.finetune and info.finetune.default_for:
            trains = ", ".join(v if v != "*" else "others" for v in info.finetune.default_for)
        rows.append([name, ", ".join(info.stages), _runs_on(info), trains,
                     str(sum(r.framework == name for r in recipes.values()))])
    con.table(["framework", "stages", "runs on", "default trainer on", "recipes"], rows)
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from rightsize.execution import envs
    from rightsize.registry import framework, framework_infos

    con = _console(args)
    if args.tools_command == "list":
        rows = []
        for name, info in sorted(framework_infos().items()):
            st = envs.status(info)
            rows.append([name, info.install.kind, st["where"] or "not installed",
                         st["version"] or "", st["path"] or info.install.line])
        if args.json:
            print(json.dumps({r[0]: dict(zip(["kind", "where", "version", "path"], r[1:],
                                             strict=True)) for r in rows}, indent=2))
            return 0
        con.table(["framework", "kind", "installed", "version", "where / how"], rows)
        return 0
    if args.tools_command == "remove":
        framework(args.name)
        if envs.remove(args.name):
            con.ok(f"removed {envs.tools_root() / args.name}")
            return 0
        con.info(f"no environment rightsize made for {args.name} under {envs.tools_root()}")
        return 1
    if args.tools_command != "install":
        print("usage: rightsize tools install NAME | list | remove NAME", file=sys.stderr)
        return 2
    info = framework(args.name)
    if info.name == "llama.cpp":
        from rightsize.execution.install import LLAMA_CPP_VERSION, install_llama_cpp

        dest = install_llama_cpp(
            args.dest or envs.tools_root() / "llama.cpp",
            version=args.tool_version or LLAMA_CPP_VERSION,
            backend=args.backend,
            log=con.info,
        )
        con.ok(f"installed into {dest}")
        return 0
    env = envs.install(info, python=args.python, log=con.info)
    con.ok(f"{info.title} installed into {env.python.parent.parent}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from rightsize.execution.runner import run_plan
    from rightsize.types import Plan

    raw = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
    plans = [Plan.model_validate(p) for p in (raw if isinstance(raw, list) else [raw])]
    plan = next((p for p in plans if p.rank == args.rank), None)
    if plan is None:
        raise KeyError(f"no plan with rank {args.rank} in {args.plan_file}")
    con = _console(args)
    manifest = run_plan(
        plan, workdir=args.workdir, inputs=dict(kv.split("=", 1) for kv in args.set),
        models_dir=args.models_dir, install_missing=args.install, include_serve=args.serve,
        dry_run=args.dry_run, log=con.info, echo=con.debug,
    )
    if args.json:
        print(manifest.model_dump_json(indent=2))
        return 0
    rows = []
    for st in manifest.steps:
        wall = next((m.value for m in st.measurements if m.kind == "wall_s"), None)
        vram = next((m.value for m in st.measurements if m.kind == "peak_vram_gb"), None)
        sizes = ", ".join(f"{m.value:.2f} GB" for m in st.measurements
                          if m.kind == "file_size_gb")
        rows.append([st.recipe_id, "skipped" if st.skipped else f"exit {st.returncode}",
                     f"{wall:.0f} s" if wall else "", f"{vram:.2f}" if vram else "", sizes])
    con.table(["step", "result", "time", "VRAM used GB", "wrote"], rows)
    where = Path(args.workdir) if args.workdir else Path("runs") / manifest.id
    con.ok(f"{manifest.status}: {where}")
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
    # A tool's output (progress bars, a model card's emoji) can hold characters the console's
    # codepage cannot encode; print a replacement rather than fail the command over it.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.offline_all:
        os.environ["RIGHTSIZE_OFFLINE"] = "1"
    handlers = {
        "data": cmd_data,
        "plan": cmd_plan,
        "mcp": cmd_mcp,
        "cloud": cmd_cloud,
        "calibrate": cmd_calibrate,
        "telemetry": cmd_telemetry,
        "recommend": cmd_recommend,
        "detect": cmd_detect,
        "estimate": cmd_estimate,
        "variants": cmd_variants,
        "search": cmd_search,
        "frameworks": cmd_frameworks,
        "quantize": cmd_quantize,
        "tools": cmd_tools,
        "run": cmd_run,
        "bench": cmd_bench,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:
        message = _expected_error(exc)
        if message is None or args.verbose:
            raise
        print(f"rightsize {args.command}: {message}", file=sys.stderr)
        return 1


#: The Hub answers 401 for a repo that does not exist as well as for a private one, so as
#: not to say which private repos exist.
_UNAUTHORIZED = (" (no such repo, or a private or gated one: check the name, or accept its "
                 "terms on the Hub and set HF_TOKEN)")
_GATED = " (gated: accept its terms on the Hub and set HF_TOKEN)"


def _expected_error(exc: BaseException) -> str | None:
    """One line for the errors a user can act on: a name that is not known, a repo that is
    missing or gated, an offline cache miss, the network being down. Anything else keeps its
    traceback, and -v shows the traceback for these too."""
    from rightsize.errors import RightsizeError

    kind = type(exc).__name__
    if kind == "HTTPStatusError":
        status = exc.response.status_code
        hint = {401: _UNAUTHORIZED, 403: _GATED, 404: " (no such repo or file: check the name)"}
        return f"HTTP {status} from {exc.request.url}{hint.get(status, '')}"
    if kind in ("ConnectError", "ConnectTimeout", "ReadTimeout"):
        return f"cannot reach the network ({exc}); --offline answers from the cache"
    if isinstance(exc, KeyError):
        return str(exc.args[0]) if exc.args else "not found"
    if isinstance(exc, (RightsizeError, FileNotFoundError, ValueError)):
        return str(exc)
    return None


if __name__ == "__main__":
    sys.exit(main())
