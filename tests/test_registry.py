"""Recipe loading and rendering."""

from __future__ import annotations

import pytest

from rightsize.registry import TemplateError, all_recipes, frameworks, get, render
from rightsize.registry.render import render_text, validate_template
from rightsize.registry.schema import Recipe


def test_bundled_recipes_load_and_have_provenance() -> None:
    recipes = all_recipes()
    assert {
        "llama.cpp/convert",
        "llama.cpp/quantize",
        "llama.cpp/imatrix",
        "llama.cpp/kld-base",
        "llama.cpp/kld-eval",
    } <= set(recipes)
    for r in recipes.values():
        assert r.source_doc_url.startswith("https://")
        assert r.version_tested
    assert frameworks() == ["llama.cpp"]


def test_render_quantize_with_and_without_imatrix() -> None:
    r = get("llama.cpp/quantize")
    s = render(
        r,
        quantize_bin="llama-quantize",
        input_gguf="in.gguf",
        output_gguf="out.gguf",
        quant="Q4_K_M",
    )
    assert s.argv == ["llama-quantize", "in.gguf", "out.gguf", "Q4_K_M"]
    s2 = render(
        r,
        quantize_bin="llama-quantize",
        input_gguf="in.gguf",
        output_gguf="out.gguf",
        quant="IQ4_XS",
        imatrix="im.gguf",
        threads=8,
    )
    assert s2.argv == [
        "llama-quantize",
        "--imatrix",
        "im.gguf",
        "in.gguf",
        "out.gguf",
        "IQ4_XS",
        "8",
    ]


def test_render_keeps_paths_with_spaces_and_backslashes_intact() -> None:
    r = get("llama.cpp/quantize")
    s = render(
        r,
        quantize_bin=r"C:\Program Files\llama\llama-quantize.exe",
        input_gguf="in.gguf",
        output_gguf="out.gguf",
        quant="Q8_0",
    )
    assert s.argv[0] == r"C:\Program Files\llama\llama-quantize.exe"
    assert "in.gguf" in s.argv and s.text


def test_missing_required_and_bad_enum_raise() -> None:
    r = get("llama.cpp/quantize")
    with pytest.raises(TemplateError):
        render(r, quantize_bin="q", input_gguf="a", output_gguf="b")
    with pytest.raises(TemplateError):
        render(r, quantize_bin="q", input_gguf="a", output_gguf="b", quant="Q4_K_ULTRA")
    with pytest.raises(TemplateError):
        render(r, quantize_bin="q", input_gguf="a", output_gguf="b", quant="Q8_0", bogus=1)


def test_template_syntax_checks() -> None:
    validate_template("{{ a }} {% if b %}--b {{ b }}{% endif %}")
    with pytest.raises(TemplateError):
        validate_template("{% if a %}{% if b %}x{% endif %}{% endif %}")
    assert render_text("x {% if not flag %}--no-flag{% endif %}", {"flag": False}) == "x --no-flag"
    assert render_text("x {% if not flag %}--no-flag{% endif %}", {"flag": True}) == "x"


def test_recipe_model_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError):
        Recipe(id="x", framework="y", stage="teleport", template="a", source_doc_url="https://x")
