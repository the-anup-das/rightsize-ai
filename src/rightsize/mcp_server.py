"""MCP server: the SDK as tools an agent can call (F6). Needs the ``mcp`` extra.

    rightsize mcp            # stdio, for desktop assistants, editors and other MCP clients

Every tool wraps the same function the CLI and the Python API call, and returns JSON built
from the same Pydantic types, so the three surfaces cannot drift apart. The recommendation
tools include each plan's rendered commands, so an agent can act on a plan without a second
round trip.

Works with the MCP Python SDK 2.x (``MCPServer``) and 1.x (``FastMCP``), which the ``mcp``
extra's lower bound still allows.
"""

from __future__ import annotations

import inspect
from typing import Any

INSTRUCTIONS = """\
Rightsize answers "which model, at which quantization, fits this hardware, and how fast
will it run". Start with recommend (hardware-first) or recommend_for_model (model-first);
each plan comes with a trace explaining its fit, its score and every rule that fired, plus
the commands to carry it out. estimate_memory sizes one model on one device. Devices are
preset or catalogue names ("RTX 4090", "M4 Max 64GB"), "@hf-username" for the hardware on a
Hugging Face profile, or "detect" for the machine this server runs on. Numbers are
estimates: each result says which formula produced it and how confident it is."""


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # the 1.x name

        return FastMCP


def _plan_payload(plan: Any) -> dict[str, Any]:
    out = plan.model_dump(mode="json")
    out["commands"] = [
        {"text": step.text, "verified": step.verified, "source": step.source_doc_url}
        for step in plan.render()
    ]
    return out


def _result_payload(result: Any) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for r in result.rejected:
        key = r.reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "plans": [_plan_payload(p) for p in result.plans],
        "rejected_by_reason": reasons,
    }


def build_server():
    """The server with its tools registered; separate from main() so tests can call tools."""
    from rightsize import __version__

    cls = _server_class()
    # Pass only what this SDK's constructor takes: MCPServer (2.x) takes both, older
    # FastMCP releases fewer.
    params = inspect.signature(cls.__init__).parameters
    optional = {"instructions": INSTRUCTIONS, "version": __version__}
    server = cls(name="rightsize", **{k: v for k, v in optional.items() if k in params})

    @server.tool(description="Rank the best models for a device, one plan each, best first.")
    def recommend(
        task: str = "chat",
        device: str = "detect",
        finetune_device: str | None = None,
        mode: str = "infer",
        quality: str = "noticeable",
        ctx: int = 8192,
        top_k: int = 5,
        allow_slow: bool = False,
        cloud: bool = False,
    ) -> dict[str, Any]:
        """task: chat, coding or agentic. mode: infer, or lora / qlora / full to plan a
        fine-tune on finetune_device first. quality: near-lossless, good, noticeable, any.
        cloud: plan a fine-tune that does not fit on the cheapest rental GPU it fits; the
        plan's cloud_fallback then names the offer and a cost per 10M training tokens."""
        from rightsize.rules.recommend import recommend_result

        return _result_payload(
            recommend_result(
                task, device, finetune_device=finetune_device, mode=mode, ctx=ctx,
                quality=quality, allow_slow=allow_slow, top_k=top_k, cloud=cloud,
            )
        )

    @server.tool(description="Every quantization of one model that works on a device.")
    def recommend_for_model(
        model: str,
        device: str = "detect",
        finetune_device: str | None = None,
        mode: str = "infer",
        quality: str = "any",
        ctx: int = 8192,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """model: a Hugging Face repo id such as Qwen/Qwen3-14B."""
        from rightsize.rules.recommend import recommend_for_model as _for_model

        return _result_payload(
            _for_model(model, device, finetune_device=finetune_device, mode=mode, ctx=ctx,
                       quality=quality, top_k=top_k)
        )

    @server.tool(
        description="Memory, and speed where modelled, for one model on a device: LLM, "
        "diffusion, audio, vision or embedding."
    )
    def estimate_memory(
        model: str,
        quant: str | None = None,
        device: str = "detect",
        ctx: int = 8192,
        mode: str = "infer",
        bandwidth_gbps: float | None = None,
        runtime: str | None = None,
        batch: int | None = None,
        offload: str = "none",
        resolution: str = "1024x1024",
        frames: int | None = None,
        text_encoder_quant: str | None = None,
    ) -> dict[str, Any]:
        """quant: a GGUF type (Q4_K_M) or a format (bf16, fp8, nf4, int8, int4, awq,
        mlx-4bit); left out, the model family's default. mode: infer, or lora / qlora /
        full for an LLM's fine-tuning memory. bandwidth_gbps fills in a device with no
        known bandwidth. Diffusion: offload none / model / sequential, resolution
        WIDTHxHEIGHT, frames for video, text_encoder_quant. Audio: runtime whisper.cpp /
        faster-whisper / transformers. Arguments for another family are ignored."""
        import rightsize

        r = rightsize.estimate(
            model, quant, device, ctx=ctx, mode=mode, bandwidth_gbps=bandwidth_gbps,
            runtime=runtime, batch=batch, offload=offload, resolution=resolution,
            frames=frames, text_encoder_quant=text_encoder_quant,
        )
        return r.model_dump(mode="json")

    @server.tool(
        description="The cheapest rental GPUs with enough memory for a job or a model's "
        "fine-tune, with an estimated time and cost."
    )
    def cloud_offers(
        min_vram_gb: float | None = None,
        model: str | None = None,
        mode: str = "qlora",
        tokens: int = 10_000_000,
        providers: list[str] | None = None,
        spot: bool = False,
        top: int = 5,
    ) -> dict[str, Any]:
        """Give min_vram_gb, or a model (Hub id) and mode (lora, qlora, full) to size the
        fine-tune. Prices from SkyPilot's open catalog; times assume 35% of datasheet
        tensor throughput (confidence 0.3). Nothing is rented or launched."""
        from rightsize.cloud import cheapest, estimate_job

        params = None
        need = min_vram_gb
        if model:
            from rightsize.catalog import facts
            from rightsize.fit import estimate_finetune
            from rightsize.types import Device

            fx = facts(model)
            params = fx.params_active or fx.params_total
            big = Device(name="sizing", vendor="nvidia", memory_gib=10_000)
            need = estimate_finetune(fx, big, mode).vram_gb
        if not need:
            raise ValueError("give min_vram_gb, or a model to size the fine-tune")
        found = cheapest(need, providers=providers, spot=spot, top=top)
        return {
            "need_gb": need,
            "offers": [
                {**o.model_dump(mode="json"),
                 "job": estimate_job(params, tokens, o).model_dump(mode="json") if params else None}
                for o in found
            ],
        }

    @server.tool(
        description="Quantized copies of a model already published on the Hub (GGUF files, "
        "MLX, AWQ, FP8, ...), with sizes and effective bits per weight."
    )
    def list_variants(
        model: str, format: str | None = None, limit: int = 20, include_unmatched: bool = False
    ) -> list[dict[str, Any]]:
        """model: a Hub id, or any quantized copy of it. format: gguf, mlx, awq, gptq, bnb,
        fp8, compressed-tensors, ... include_unmatched also returns repos whose name is not
        the base model's (fine-tunes or drafts that call themselves quantizations)."""
        from rightsize.catalog import variants

        found = variants(model, formats=[format] if format else None, limit=limit)
        return [v.model_dump(mode="json") for v in found
                if include_unmatched or v.name_matches_base]

    @server.tool(
        description="Models on rightsize's curated lists, by family, task and size: LLMs "
        "(chat, coding), diffusion, audio, vision and embedding models."
    )
    def search_models(
        family: str | None = None,
        task: str | None = None,
        max_params_b: float | None = None,
        min_params_b: float | None = None,
        license: str | None = None,
        include_gated: bool = True,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """family: llm, diffusion, audio, vision, embedding. task: chat or coding for LLMs,
        else a Hub task (text-to-image, text-to-video, automatic-speech-recognition,
        text-to-speech, object-detection, sentence-similarity, text-ranking, ...). Sizes in
        billions of parameters."""
        from rightsize.catalog import search

        found = search(
            family, task,
            max_params=max_params_b * 1e9 if max_params_b is not None else None,
            min_params=min_params_b * 1e9 if min_params_b is not None else None,
            license=license, include_gated=include_gated, limit=limit,
        )
        return [e.model_dump(mode="json") for e in found]

    @server.tool(description="Devices rightsize knows, filtered by a name fragment.")
    def list_hardware(query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        from rightsize.hardware import catalog, presets

        q = query.lower()
        seen, out = set(), []
        for table in (presets(), catalog()):
            for name, dev in table.items():
                if q in name.lower() and name not in seen:
                    seen.add(name)
                    out.append({
                        "name": name,
                        "vendor": dev.vendor,
                        "memory_gib": dev.memory_gib,
                        "bandwidth_gbps": dev.bandwidth_gbps,
                        "compute_capability": dev.compute_capability,
                        "unified_memory": dev.unified_memory,
                    })
        return out[:limit]

    @server.tool(description="The machine this server runs on, as a device.")
    def detect_hardware() -> dict[str, Any]:
        import rightsize

        return rightsize.detect().model_dump(mode="json")

    @server.tool(
        description="Frameworks (fine-tuning, quantization, export and serving toolkits): what "
        "each is for, where it runs, how to install it, and its recipes with how far each was "
        "verified."
    )
    def list_frameworks(framework: str = "") -> list[dict[str, Any]]:
        from rightsize.registry import all_recipes, framework_infos

        recipes = all_recipes().values()
        return [
            {"name": info.name, "summary": info.summary, "stages": info.stages,
             "hardware": info.hardware, "install": info.install.line,
             "default_trainer_on": info.finetune.default_for if info.finetune else [],
             "recipes": [
                 {"id": r.id, "stage": r.stage, "families": r.families, "verified": r.verified,
                  "version": r.version_tested, "source": r.source_doc_url}
                 for r in recipes if r.framework == info.name
             ]}
            for info in framework_infos().values()
            if not framework or info.name == framework
        ]

    @server.tool(description="Render one recipe into a command with the given inputs.")
    def render_recipe(recipe_id: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        from rightsize.registry import get, render

        return render(get(recipe_id), **(inputs or {})).model_dump(mode="json")

    return server


def main() -> int:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        from rightsize.errors import MissingExtraError

        raise MissingExtraError("mcp", "The Rightsize MCP server") from exc
    build_server().run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
