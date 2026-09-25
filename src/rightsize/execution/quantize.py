"""The first end-to-end slice: predict, convert, (imatrix), quantize, evaluate, record.

    quantize_model("Qwen/Qwen3-4B", ["Q4_K_M", "Q8_0"], imatrix=True, evaluate=True)

Every step is a rendered recipe from the registry; every number the fit engine predicted is
written next to what was measured, so the run doubles as calibration data (F9 seed).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from rightsize import __version__
from rightsize.catalog import facts as hub_facts
from rightsize.execution.llamacpp import (
    LlamaCppTools,
    append_measurements,
    convert_step,
    ensure_snapshot,
    file_size_gb,
    find_tools,
    gate,
    imatrix_step,
    kld_base_step,
    kld_eval_step,
    parse_perplexity_output,
    quantize_step,
    run_step,
    write_manifest,
)
from rightsize.fit import estimate, predicted_file_gb
from rightsize.hardware import resolve as resolve_device
from rightsize.types import Device, Measurement, ModelRef, RunManifest, RunStep

Log = Callable[[str], None]

CALIBRATION_URL = "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt"
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
EVAL_CTX = 512
_IQ_TYPES = re.compile(r"^(IQ\d|TQ\d)", re.IGNORECASE)


def _slug(repo: str) -> str:
    return repo.replace("/", "__")


def _now_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def default_texts(tools: LlamaCppTools, log: Log) -> tuple[Path, Path]:
    """Calibration text for imatrix and evaluation text for the gate; downloaded once."""
    base = Path(os.environ.get("RIGHTSIZE_TEXTS_DIR", tools.root.parent / "data"))
    base.mkdir(parents=True, exist_ok=True)
    calib = base / "calibration_datav3.txt"
    wiki = base / "wikitext-2-raw" / "wiki.test.raw"
    if not calib.exists():
        log(f"downloading calibration text -> {calib}")
        calib.write_bytes(httpx.get(CALIBRATION_URL, follow_redirects=True, timeout=60).content)
    if not wiki.exists():
        log(f"downloading wikitext-2 -> {wiki.parent}")
        z = base / "wikitext-2-raw-v1.zip"
        z.write_bytes(httpx.get(WIKITEXT_URL, follow_redirects=True, timeout=120).content)
        zipfile.ZipFile(z).extractall(base)
        z.unlink()
    return calib, wiki


def _skipped(recipe_id: str, argv: list[str], note: str) -> RunStep:
    ts = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    return RunStep(
        recipe_id=recipe_id,
        argv=argv,
        started=ts,
        finished=ts,
        skipped=True,
        measurements=[Measurement(kind="wall_s", value=0.0, note=note)],
    )


def quantize_model(
    repo: str,
    quants: list[str],
    *,
    device: Device | str | None = "detect",
    ctx: int = 8192,
    imatrix: bool = False,
    evaluate: bool = False,
    out_dir: str | os.PathLike = "runs",
    models_dir: str | os.PathLike = "models",
    tools: str | os.PathLike | None = None,
    outtype: str = "auto",
    imatrix_chunks: int | None = None,
    eval_chunks: int | None = 100,
    gpu_layers: str = "all",
    revision: str = "main",
    token: str | None = None,
    dry_run: bool = False,
    log: Log = print,
) -> RunManifest:
    quants = [q.upper() for q in quants]
    facts = hub_facts(repo, revision, token=token)
    dev = resolve_device(device)
    tc = find_tools(tools)
    run_id = f"{_slug(repo)}-{_now_id()}"
    run_dir = Path(out_dir) / run_id
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    slug = _slug(repo)

    manifest = RunManifest(
        id=run_id,
        created=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        rightsize_version=__version__,
        model=ModelRef(repo=repo, revision=revision),
        facts=facts,
        device=dev,
        toolchain=tc.as_dict(),
    )

    # ---- predictions first, so the run records what we believed before running anything
    log(
        f"model {repo}: {facts.params_total / 1e9:.2f}B params, {facts.num_layers} layers, "
        f"kv_heads {facts.num_kv_heads}, head_dim {facts.head_dim}"
    )
    log(f"device {dev.name}: {dev.memory_gb} GB, bandwidth {dev.bandwidth_gbps} GB/s")
    for q in quants:
        manifest.predicted[q] = estimate(facts, q, dev, ctx=ctx)
        fr = manifest.predicted[q]
        log(
            f"predict {q:8s} file {predicted_file_gb(facts, q):.2f} GB  "
            f"vram@{ctx} {fr.vram_gb:.2f} GB "
            f"{fr.verdict.value:7s} {fr.speed or '?'} {fr.speed_unit or ''}"
        )

    def run(step, *, sample_vram=False):
        rs, text, peak = run_step(
            step,
            log_path=run_dir
            / "logs"
            / f"{len(manifest.steps):02d}-{step.stage}-{Path(step.argv[-1]).stem[:40]}.log",
            log=lambda line: log("  " + line),
            extra_path=tc.root,
            sample_vram=sample_vram,
        )
        manifest.steps.append(rs)
        if rs.returncode != 0:
            manifest.status = "failed"
            write_manifest(manifest, run_dir)
            raise RuntimeError(
                f"{step.recipe_id} failed with exit code {rs.returncode}; see {rs.log_path}"
            )
        return rs, text, peak

    # ---- convert
    base_gguf = models / f"{slug}-{outtype}.gguf"
    if dry_run:
        conv = convert_step(tc, models / slug, base_gguf, outtype)
        manifest.steps.append(_skipped(conv.recipe_id, conv.argv or [], "dry run"))
    elif base_gguf.exists():
        log(f"convert: {base_gguf.name} exists, skipping")
        manifest.steps.append(_skipped("llama.cpp/convert", [], "already converted"))
    else:
        snapshot = ensure_snapshot(repo, models, revision, token)
        conv = convert_step(tc, snapshot, base_gguf, outtype)
        log(f"convert: {conv.text}")
        rs, _, _ = run(conv)
        rs.measurements.append(
            Measurement(
                kind="file_size_gb",
                value=file_size_gb(base_gguf),
                predicted=round(predicted_file_gb(facts, "BF16"), 3),
                note="16-bit base",
            )
        )
    manifest.artifacts["base_gguf"] = str(base_gguf)

    # ---- imatrix
    imat: Path | None = None
    needs_imatrix = imatrix or any(_IQ_TYPES.match(q) for q in quants)
    if needs_imatrix:
        imat = models / f"{slug}-imatrix.gguf"
        if dry_run or imat.exists():
            note = "dry run" if dry_run else "imatrix exists"
            log(f"imatrix: {note}")
            manifest.steps.append(_skipped("llama.cpp/imatrix", [], note))
        else:
            calib, _ = default_texts(tc, log)
            st = imatrix_step(
                tc, base_gguf, calib, imat, gpu_layers=gpu_layers, chunks=imatrix_chunks
            )
            log(f"imatrix: {st.text}")
            run(st, sample_vram=True)
        manifest.artifacts["imatrix"] = str(imat)

    # ---- quantize
    outputs: dict[str, Path] = {}
    for q in quants:
        out = run_dir / f"{slug}-{q}.gguf"
        st = quantize_step(tc, base_gguf, out, q, imatrix=imat)
        log(f"quantize {q}: {st.text}")
        if dry_run:
            manifest.steps.append(_skipped(st.recipe_id, st.argv or [], "dry run"))
            continue
        rs, _, _ = run(st)
        rs.measurements.append(
            Measurement(
                kind="file_size_gb",
                value=file_size_gb(out),
                predicted=round(predicted_file_gb(facts, q), 3),
                note=q,
            )
        )
        outputs[q] = out
        manifest.artifacts[q] = str(out)
        log(
            f"  {q}: {file_size_gb(out):.3f} GB measured vs "
            f"{predicted_file_gb(facts, q):.3f} GB predicted"
        )

    # ---- evaluate: KL divergence vs the 16-bit reference, peak VRAM vs prediction
    if evaluate and not dry_run:
        _, wiki = default_texts(tc, log)
        logits = models / f"{slug}-{outtype}-wiki{eval_chunks or 'all'}.kld"
        if logits.exists():
            log("kld-base: reference logits exist, skipping")
            manifest.steps.append(_skipped("llama.cpp/kld-base", [], "logits exist"))
        else:
            st = kld_base_step(
                tc, base_gguf, wiki, logits, gpu_layers=gpu_layers, chunks=eval_chunks
            )
            log(f"kld-base: {st.text}")
            rs, text, peak = run(st, sample_vram=True)
            pred = estimate(facts, "BF16", dev, ctx=EVAL_CTX)
            if peak:
                rs.measurements.append(
                    Measurement(
                        kind="peak_vram_gb",
                        value=peak,
                        predicted=pred.vram_gb,
                        note=f"16-bit base, ctx {EVAL_CTX}",
                    )
                )
            if (m := parse_perplexity_output(text)).get("ppl"):
                rs.measurements.append(Measurement(kind="ppl", value=m["ppl"], note="16-bit base"))
        for q, out in outputs.items():
            st = kld_eval_step(tc, out, wiki, logits, gpu_layers=gpu_layers, chunks=eval_chunks)
            log(f"kld-eval {q}: {st.text}")
            rs, text, peak = run(st, sample_vram=True)
            metrics = parse_perplexity_output(text)
            pred = estimate(facts, q, dev, ctx=EVAL_CTX)
            if peak:
                rs.measurements.append(
                    Measurement(
                        kind="peak_vram_gb",
                        value=peak,
                        predicted=pred.vram_gb,
                        note=f"{q}, ctx {EVAL_CTX}",
                    )
                )
            for k in ("kld_mean", "top1_agreement", "ppl"):
                if k in metrics:
                    rs.measurements.append(Measurement(kind=k, value=metrics[k], note=q))
            manifest.gate[q] = gate(metrics)
            log(
                f"  {q}: gate {manifest.gate[q]['verdict']}  KLD {metrics.get('kld_mean')}  "
                f"top1 {metrics.get('top1_agreement')}  ppl {metrics.get('ppl')}"
            )

    manifest.status = "succeeded"
    path = write_manifest(manifest, run_dir)
    if not dry_run:
        append_measurements(manifest, Path(out_dir) / "measurements.jsonl")
    log(f"manifest: {path}")
    return manifest
