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
    assert {
        "llama.cpp",
        "unsloth",
        "trl",
        "axolotl",
        "mlx-lm",
        "vllm",
        "ollama",
        "optimum-intel",
        "llm-compressor",
        "transformers",
        "diffusers",
        "whisper.cpp",
        "ctranslate2",
        "sentence-transformers",
        "modelopt",
    } == set(frameworks())


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


def _example_values(recipe: Recipe) -> dict:
    """Every input filled: its default, else its first allowed value, else a stand-in."""
    values = {}
    for name, spec in recipe.inputs.items():
        if spec.default is not None:
            values[name] = spec.default
        elif spec.values:
            values[name] = spec.values[0]
        elif spec.required:
            values[name] = "org/model" if name == "model" else f"{name}.bin"
    return values


@pytest.mark.parametrize("recipe_id", sorted(all_recipes()))
def test_every_recipe_renders_and_config_recipes_parse(recipe_id: str) -> None:
    """A config recipe that renders broken Python or YAML would only fail on the user's
    machine; parsing the render here catches it first."""
    import ast

    import yaml

    recipe = get(recipe_id)
    step = render(recipe, **_example_values(recipe))
    assert step.text and "{{" not in step.text and "{%" not in step.text
    if recipe.language == "python":
        ast.parse(step.text)
    elif recipe.language == "yaml":
        assert isinstance(yaml.safe_load(step.text), dict)


def test_python_recipes_spell_booleans_the_python_way() -> None:
    text = render(get("unsloth/sft"), model="org/model", load_in_4bit="False").text
    assert "load_in_4bit=False" in text
    text = render(get("axolotl/qlora"), model="org/model", load_in_4bit=True).text
    assert "load_in_4bit: true" in text


def test_every_toolkit_recipe_says_how_far_it_was_checked() -> None:
    """Recipes transcribed from documentation say so. The ones marked run were run end to end
    here, and adding one to that list is a deliberate edit to this test."""
    ran_outside_llama_cpp = set()
    for recipe in all_recipes().values():
        assert recipe.version_tested, recipe.id
        if recipe.framework != "llama.cpp" and recipe.verified != "docs":
            ran_outside_llama_cpp.add(recipe.id)
    assert ran_outside_llama_cpp == {
        "unsloth/sft",
        "llm-compressor/fp8-dynamic",
        "llm-compressor/gptq-w4a16",
        "transformers/bnb-nf4",
        "optimum-intel/export-openvino",
        "sentence-transformers/onnx-int8",
        "ctranslate2/convert-whisper",
        "transformers/kld-eval",
        "sentence-transformers/cosine-eval",
        "llm-compressor/awq-w4a16",
    }
