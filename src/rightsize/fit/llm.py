"""LLM fit estimator, GGUF path (F3, first slice).

formula_id ``llm.gguf.analytic.v1``:
  weights_gb   = params x effective_bpw / 8, or the tensor bytes of the repo's own GGUF
                 for that quantization when the facts came from a GGUF repo
  embedding_gb = the input embedding at the type llama-quantize gives it; llama.cpp keeps it
                 in system RAM (data/runtimes/llama.cpp/embedding.yaml). v0 counted it in VRAM
  gpu_weights  = weights - embedding, unless the output head is tied to it: llama.cpp then
                 keeps a copy on the GPU as the head, and only RAM grows
  kv_gb        = per-layer KV as llama.cpp allocates it (fit/kv.py): plain GQA is
                 2 x layers x kv_heads x head_dim x ctx x batch x kv_bytes; sliding-window,
                 MLA and hybrid models follow data/runtimes/llama.cpp/kv_cache.yaml
  overhead_gb  = runtime constant + fraction x gpu_weights  (llama.cpp ~0.75 GB + 2%)
  vram_gb      = gpu_weights + kv + overhead;  ram_gb = embedding_gb
  tok/s        ~ bandwidth / (weights + kv) bytes read per decoded token, x efficiency
Confidence 0.6 (analytic, uncalibrated). Every number is explainable in ``breakdown``.
"""

from __future__ import annotations

from rightsize._data import load_yaml
from rightsize.fit import kv as _kv
from rightsize.types import Device, FitResult, Mode, ModelFacts, QuantSpec, RuntimeSpec, Verdict

FORMULA_ID = "llm.gguf.analytic.v1"


def overhead_constants(runtime: str) -> tuple[float, float]:
    """(fixed GB, fraction of weights) a runtime adds, from data/runtimes/overheads.yaml,
    which scripts/refit_constants.py refits from measurements. Unknown runtimes get
    llama.cpp's, since most local runtimes wrap it."""
    rows = {r["name"]: r for r in load_yaml("runtimes/overheads.yaml")["runtimes"]}
    row = rows.get(runtime.lower()) or rows["llama.cpp"]
    return float(row["fixed_gb"]), float(row["fraction"])


_HEADROOM = 0.10  # verdict "tight" inside this fraction of usable memory
_DECODE_EFFICIENCY = 0.70  # fraction of peak bandwidth a llama.cpp decode loop reaches
_SLOW_TOK_S = 5.0


def gguf_bpw(quant: str) -> tuple[float, str]:
    """Effective bits-per-weight for a GGUF file type or tensor type, and where it came from.

    llama.cpp's own measurement comes first (data/quality/gguf_types.yaml, effective bits on
    Llama-3.1-8B for every type). Before it was consulted, the i-quants were sized by their
    nominal bits - IQ4_XS at 4.25 against a real 4.46, five percent small - while the
    quality ranking used the effective figure, so two tables disagreed about one number.
    """
    q = quant.upper()
    measured = (load_yaml("quality/gguf_types.yaml").get("file_types") or {}).get(q) or {}
    if measured.get("bpw"):
        return float(measured["bpw"]), "measured by llama.cpp"
    table = load_yaml("quants/gguf_bpw.yaml")
    ft = table.get("file_types") or {}
    if q in ft:
        return float(ft[q]), "file_types"
    tt = table["tensor_types"]
    if q in tt:
        return float(tt[q]), "tensor_types"
    # Q4_K_M / Q4_K_S style names fall back to their base tensor type when unmeasured.
    base = q.rsplit("_", 1)[0] if q.endswith(("_M", "_S", "_L", "_XL", "_XS", "_XXS")) else q
    if base in tt:
        return float(tt[base]), "tensor_types(base)"
    raise KeyError(f"unknown GGUF quant type {quant!r}")


def predicted_file_gb(facts: ModelFacts, quant: str) -> float:
    if not facts.params_total:
        raise ValueError("params_total unknown; cannot predict size")
    return _table_weights(facts, quant)[0]


def _routed_experts(facts: ModelFacts) -> int | None:
    """Parameters in routed expert tensors. The catalog counts active parameters as the total
    less the idle share of the routed experts, so the routed count follows back from it."""
    extra = facts.extra or {}
    n, k = extra.get("num_experts"), extra.get("num_experts_per_tok")
    if not (facts.params_total and facts.params_active and n and k and n > k):
        return None
    return round((facts.params_total - facts.params_active) * n / (n - k))


def _table_weights(facts: ModelFacts, quant: str) -> tuple[float, float, str]:
    """(GB, bits per weight, where from) from the tables. MXFP4_MOE - "MXFP4" in ggml-org's
    and LM Studio's file names - is MXFP4 only for the routed experts: llama-quantize writes
    every other tensor as Q8_0, which params x 4.25 bits left out (gpt-oss-20b: 11.11 GB
    against the 12.10 GB of tensors its GGUF holds)."""
    total = facts.params_total or 0
    experts = _routed_experts(facts) if quant.upper() in ("MXFP4", "MXFP4_MOE") else None
    if experts:
        sizes = _type_sizes()
        mx, q8 = (sizes[t][1] * 8 / sizes[t][0] for t in ("MXFP4", "Q8_0"))
        bits = experts * mx + (total - experts) * q8
        return (
            bits / 8 / 1e9,
            bits / total,
            "MXFP4 routed experts, Q8_0 elsewhere (llama-quantize's MXFP4_MOE)",
        )
    try:
        bpw, source = gguf_bpw(quant)
    except KeyError:
        from rightsize.fit.formats import lookup

        fmt = lookup(quant)  # a format outside GGUF, by id or alias
        bpw, source = fmt.bpw, f"the {fmt.id} format (formats.yaml)"
    return total * bpw / 8 / 1e9, bpw, source


def repo_file(facts: ModelFacts, quant: str) -> dict | None:
    """The repo's own GGUF for this quantization, when the facts came from a GGUF repo."""
    files = (facts.extra or {}).get("gguf_files") or {}
    return {k.upper(): v for k, v in files.items()}.get(quant.upper())


def format_weights_gb(
    facts: ModelFacts, bpw: float, embedding_bits: float | None, head_bits: float | None
) -> float | None:
    """The weights a toolkit writes: the body at the format's bits, the input embedding and
    an output head of its own at the bits the toolkit leaves them (llm-compressor and
    bitsandbytes quantize Linear layers only). A head tied to the embedding is saved once,
    even where the source checkpoint stores it twice (F1's tied_head_stored)."""
    if not facts.params_total:
        return None
    extra = facts.extra or {}
    body = facts.params_total - int(extra.get("tied_head_stored") or 0)
    vocab, hidden = extra.get("vocab_size"), extra.get("hidden_size")
    if not (vocab and hidden):
        return body * bpw / 8 / 1e9
    table = vocab * hidden
    head = table if extra.get("tie_word_embeddings") is False else 0
    bits = (body - table - head) * bpw + table * (embedding_bits or bpw) + head * (head_bits or bpw)
    return bits / 8 / 1e9


def weights(facts: ModelFacts, quant: QuantSpec | str) -> tuple[float, float, str]:
    """(GB, bits per weight, where from) for the weights at ``quant``.

    A GGUF repo's own file beats any table: its tensors are what llama.cpp loads, and
    publishers' own mixes (Unsloth's UD-Q4_K_XL) have no table entry at all. Otherwise
    parameters x the bits per weight llama.cpp measured for the type, or, for a format a
    toolkit makes (a QuantSpec with its bits), what that toolkit writes."""
    if not facts.params_total:
        raise ValueError(f"{facts.ref.repo}: parameter count unknown; cannot size the weights")
    if isinstance(quant, QuantSpec):
        if quant.bits_per_weight:
            gb = format_weights_gb(
                facts, quant.bits_per_weight, quant.embedding_bits, quant.head_bits
            )
            kept = f", embedding at {quant.embedding_bits:g} bits" if quant.embedding_bits else ""
            return (
                gb or 0.0,
                gb * 8e9 / facts.params_total if gb else 0.0,
                (f"the {quant.variant or quant.method} format (formats.yaml){kept}"),
            )
        quant = quant.variant or quant.method
    own = repo_file(facts, quant)
    if own and own.get("weights_bytes"):
        nbytes = own["weights_bytes"]
        how = "its header" if own.get("exact") else "its file size"
        return (
            nbytes / 1e9,
            nbytes * 8 / facts.params_total,
            f"the repo's {quant.upper()} file ({how})",
        )
    return _table_weights(facts, quant)


def _type_sizes() -> dict[str, tuple[int, int]]:
    """(block size, bytes per block) for every ggml tensor type."""
    return {
        t["name"]: (t["block_size"], t["type_size"])
        for t in load_yaml("quants/ggml_types.yaml")["tensor_types"]
    }


def embedding_type(
    file_type: str, hidden: int, *, tied: bool, architecture: str | None = None
) -> str | None:
    """The tensor type llama-quantize stores ``token_embd.weight`` in for ``file_type``, from
    data/runtimes/llama.cpp/embedding.yaml; None for a name llama-quantize does not make,
    such as a publisher's own mix."""
    rules = load_yaml("runtimes/llama.cpp/embedding.yaml")
    ft = rules.get("aliases", {}).get(file_type.upper(), file_type.upper())
    new = rules["default_type"].get(ft)
    if new is None:
        return None
    sizes = _type_sizes()
    if sizes[new][0] == 1:  # F32, F16, BF16 are not quantized, so no rule applies
        return new
    if tied:
        t = rules["tied"]
        if hidden % sizes[new][0] or (architecture or "").lower() in t["architectures"]:
            new = t["uneven_rows"]
        elif ft in t["file_types"]:
            new = t["file_types"][ft]
        elif new not in t["keep"]:
            new = t["default"]
    else:
        new = rules["own"]["file_types"].get(ft, new)
    if hidden % sizes[new][0]:
        new = rules["fallback"].get(new, new)
        if hidden % sizes[new][0]:
            new = "F16"
    return new


def input_embedding(facts: ModelFacts, quant: str) -> tuple[float, bool, str] | None:
    """The input embedding table of the model's GGUF at ``quant``: (GB, whether the model has
    an output head of its own, where the figure came from), or None when the facts cannot say.

    Our converter writes an output head whenever the checkpoint stores one, tied or not; a
    GGUF repo's own tensors say whether it has one. A config that does not say is taken as
    tied, which leaves the VRAM estimate where it was."""
    extra = facts.extra or {}
    g = extra.get("gguf") or {}
    if g:
        own_head = g.get("output_bytes") is not None
        read = {str(g.get("quant", "")).upper(), str(g.get("file_type", "")).upper()}
        if g.get("input_embedding_bytes") and quant.upper() in read:
            return g["input_embedding_bytes"] / 1e9, own_head, "its GGUF header"
    else:
        own_head = extra.get("tie_word_embeddings") is False or bool(extra.get("tied_head_stored"))
    vocab, hidden = extra.get("vocab_size"), extra.get("hidden_size")
    if not (vocab and hidden):
        return None
    typ = embedding_type(quant, hidden, tied=not own_head, architecture=extra.get("model_type"))
    if typ is None:
        return None
    block, size = _type_sizes()[typ]
    return (
        vocab * hidden * size / block / 1e9,
        own_head,
        f"{typ}, the type llama-quantize gives it in {quant.upper()}",
    )


def embedding_in_ram(runtime: str, device: Device) -> bool:
    """llama.cpp and the runtimes built on it keep the input embedding on the CPU; where GPU
    and CPU share one memory, that is no saving."""
    runtimes = load_yaml("runtimes/llama.cpp/embedding.yaml")["runtimes"]
    return runtime.lower() in runtimes and not device.unified_memory and device.vendor != "cpu"


def kv_cache_gb(facts: ModelFacts, ctx: int, batch: int = 1, kv_bytes: float = 2.0) -> float:
    """Context-dependent memory as llama.cpp allocates it. Delegates to fit.kv, which knows
    sliding-window, MLA and hybrid layouts; plain GQA models come out as before."""
    return _kv.kv_cache_gb(facts, ctx, batch, kv_bytes)


def read_per_token(facts: ModelFacts) -> int:
    """Weights a decode step reads: active parameters, less the input embedding table when
    it is separate from the output head. Looking a token up reads one row of the table; the
    output head is read in full. With tied embeddings they are one matrix, read in full."""
    params = facts.params_active or facts.params_total or 0
    extra = facts.extra or {}
    vocab, hidden = extra.get("vocab_size"), extra.get("hidden_size")
    if vocab and hidden and extra.get("tie_word_embeddings") is False:
        params -= vocab * hidden
    return max(params, 0)


def estimate(
    facts: ModelFacts,
    quant: QuantSpec | str,
    device: Device,
    *,
    runtime: RuntimeSpec | str = "llama.cpp",
    mode: Mode = Mode.infer,
    ctx: int = 8192,
    batch: int = 1,
    kv_bytes: float = 2.0,
    all_logits: bool = False,
) -> FitResult:
    """``all_logits`` sizes a pass that keeps every token's logits - llama-perplexity and
    llama-imatrix - rather than a server, which keeps them only for the tokens it samples."""
    if mode is not Mode.infer:
        # training memory does not depend on the serving quantization; see fit/finetune.py
        from rightsize.fit.finetune import estimate_finetune

        return estimate_finetune(facts, device, mode, batch=batch)
    qname = quant.variant or quant.method if isinstance(quant, QuantSpec) else str(quant)
    rname = runtime.name if isinstance(runtime, RuntimeSpec) else str(runtime)
    fixed, frac = overhead_constants(rname)

    if not (facts.num_layers and facts.num_kv_heads and facts.head_dim):
        hint = (
            " The repo is gated: accept its terms on the Hub and set HF_TOKEN."
            if (facts.extra or {}).get("gated")
            else ""
        )
        raise ValueError(
            f"{facts.ref.repo}: its layer count, KV heads and head size are unknown, so the "
            f"KV cache cannot be sized.{hint}"
        )
    sized = quant if isinstance(quant, QuantSpec) and quant.bits_per_weight else qname
    weights_gb, bpw, bpw_source = weights(facts, sized)
    notes = [f"bpw {bpw:.3f} from {bpw_source}", f"ctx {ctx}, kv {kv_bytes:g} B/elem"]
    logits = 0.0
    if all_logits:
        # n_batch / ctx sequences at once, each with its own KV, and a micro-batch of logits
        d = load_yaml("runtimes/llama.cpp/kv_cache.yaml")["defaults"]
        batch *= max(1, d["n_batch"] // ctx)
        logits = d["n_ubatch"] * ((facts.extra or {}).get("vocab_size") or 0) * 4 / 1e9
        notes.append(
            f"every token's logits kept: {batch} sequences in parallel, "
            f"{logits:.3f} GB of logits for a {d['n_ubatch']}-token micro-batch"
        )
    kv = kv_cache_gb(facts, ctx, batch, kv_bytes)
    # llama.cpp keeps the input embedding in system RAM; a tied output head is a copy of it
    # that stays on the GPU, so only a model with a head of its own sheds it from VRAM
    ram = on_gpu = 0.0
    embedding = input_embedding(facts, qname) if embedding_in_ram(rname, device) else None
    if embedding:
        ram, own_head, where = embedding
        on_gpu = ram if not own_head else 0.0
        notes.append(
            f"input embedding {ram:.3f} GB in system RAM ({where})"
            + ("" if own_head else "; its tied copy as the output head stays in VRAM")
        )
    gpu_weights = weights_gb - ram + on_gpu
    overhead = fixed + frac * gpu_weights + logits
    vram = gpu_weights + kv + overhead
    usable = device.memory_gb * device.usable_fraction

    if repo_file(facts, qname) is None and (facts.extra or {}).get("gguf_files"):
        notes.append(f"the repo has no {qname.upper()} file; sized from the bits-per-weight table")
    lay = _kv.layout(facts)
    if lay.kind != "gqa":
        notes.append(lay.rule)
    notes.extend(lay.notes)
    if vram <= usable * (1 - _HEADROOM):
        verdict = Verdict.fits
    elif vram <= usable:
        verdict = Verdict.tight
    elif gpu_weights * 0.5 + kv + overhead <= usable and device.system_ram_gb:
        verdict = Verdict.offload
        notes.append("does not fit fully; partial offload to system RAM possible")
    else:
        verdict = Verdict.no_fit

    speed = None
    if device.bandwidth_gbps:
        active_gb = read_per_token(facts) * bpw / 8 / 1e9
        speed = _DECODE_EFFICIENCY * device.bandwidth_gbps / (active_gb + kv)
        if verdict is Verdict.offload:
            speed = speed * 0.15  # CPU-bound once layers spill; refit by F9
            notes.append("speed assumes partial offload (large penalty)")
        if speed < _SLOW_TOK_S:
            notes.append(f"below {_SLOW_TOK_S:g} tok/s; fits but will feel slow")
        if facts.params_active:
            # The efficiency constant was measured on a dense model. Routing tokens to
            # experts costs llama.cpp bandwidth efficiency, so for MoE this is a ceiling.
            notes.append(
                "mixture of experts: speed is an upper bound; expert routing is "
                "usually less bandwidth-efficient than the dense model it was fit on"
            )
    else:
        # Say so rather than showing a blank column. Some devices have no published
        # bandwidth at all - NVIDIA gives laptop GPUs a bus width but no bandwidth,
        # because the memory speed is the laptop maker's choice - so measuring beats
        # hunting for a number that does not exist.
        notes.append(
            "no speed estimate: memory bandwidth unknown for this device. "
            "Measure it with 'rightsize bench', or pass one you trust."
        )

    return FitResult(
        verdict=verdict,
        vram_gb=round(vram, 2),
        ram_gb=round(ram, 2),
        breakdown={
            "weights": round(gpu_weights, 3),
            **({"input_embedding_ram": round(ram, 3)} if ram else {}),
            "kv_cache": round(kv, 3),
            **{k: v for k, v in _kv.kv_breakdown(facts, ctx, batch, kv_bytes).items() if v},
            "overhead": round(overhead, 3),
            "usable_memory": round(usable, 2),
        },
        speed=round(speed, 1) if speed else None,
        speed_unit="tok/s" if speed else None,
        confidence=0.6,
        formula_id=FORMULA_ID,
        notes=notes,
    )
