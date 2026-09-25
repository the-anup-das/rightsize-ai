"""The KV cache prediction against llama.cpp's own dry-run allocation.

Slow and opt-in (``uv run pytest -m slow``): needs llama.cpp installed and the small GGUFs
from scripts/validate_kv.py under models/validation. Facts come from the recorded fixtures,
so no network. When this was first run it found LFM2 over-stated by 166% and a cache that
kept serving the wrong answer after the fix; both are covered in the unit tests now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rightsize.execution.llamacpp import ToolchainError, find_tools
from rightsize.fit import kv
from rightsize.types import GB, ModelFacts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_kv import CASES, llama_context_mib  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("gguf_name,repo,what", CASES)
@pytest.mark.parametrize("ctx", [4096, 32768])
def test_matches_llama_cpp(gguf_name: str, repo: str, what: str, ctx: int) -> None:
    try:
        tools = find_tools()
    except ToolchainError:
        pytest.skip("llama.cpp not installed")
    exe = "llama-fit-params.exe" if sys.platform == "win32" else "llama-fit-params"
    fit_params = tools.root / exe
    gguf = ROOT / "models" / "validation" / gguf_name
    if not (fit_params.exists() and gguf.exists()):
        pytest.skip("llama-fit-params or the validation GGUF is missing")
    fixture = ROOT / "tests" / "fixtures" / "facts" / (repo.replace("/", "__") + ".json")
    fx = ModelFacts.model_validate_json(fixture.read_text(encoding="utf-8"))

    ours = kv.kv_cache_gb(fx, ctx) * GB / 1024**2
    theirs = llama_context_mib(fit_params, gguf, ctx)
    assert ours == pytest.approx(theirs, rel=0.01), (
        f"{what}: ours {ours:.1f} MiB, llama.cpp {theirs:.1f} MiB"
    )
