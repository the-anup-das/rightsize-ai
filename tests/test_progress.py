"""Progress parsers against the tools' real output shapes, and the console's plain fallback."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from rightsize._console import Console, error_style, verdict_style
from rightsize.execution.progress import parser_for

QUANT = (
    "[   1/ 311]                    token_embd.weight - [ 2048, 151936,     1,     1], "
    "type =   bf16, converting to q6_K .. size =   593.50 MiB ->   243.47 MiB\n"
    "[   2/ 311]                blk.0.attn_k.weight - [ 2048,  1024,     1,     1], "
    "type =   bf16, converting to q4_K .. size =     4.00 MiB ->     1.12 MiB\n"
    "[  12/ 311]         blk.1.ffn_down.weight - [ 6144,  2048"
)
PPL = (
    "perplexity: calculating perplexity over 100 chunks, n_ctx=512, batch_size=2048, n_seq=4\n"
    "perplexity: 1.23 seconds per pass - ETA 0.52 minutes\n"
    "[1]4.9797,[2]5.8801,[3]6.2345,[4]"
)
IMATRIX = (
    "compute_imatrix: computing over 140 chunks, n_ctx=512, batch_size=2048, n_seq=4\n[1]5.1,[2]"
)
CONVERT = (
    "INFO:hf-to-gguf:Loading model: Qwen__Qwen3-1.7B\n"
    "INFO:hf-to-gguf:token_embd.weight,       torch.bfloat16 --> BF16, shape = {2048, 151936}\n"
    "INFO:hf-to-gguf:blk.0.attn_k.weight,     torch.bfloat16 --> BF16, shape = {2048, 1024}\n"
    "INFO:hf-to-gguf:blk.0.attn_norm.weight,  torch.bfloat16 --> F32, shape = {2048}\n"
)


def test_quantize_parser_reads_latest_tensor_index() -> None:
    assert parser_for("llama.cpp/quantize")(QUANT) == (12, 311)
    assert parser_for("llama.cpp/quantize")("nothing yet") is None


def test_chunk_parsers_take_total_from_announcement_and_progress_from_partial_line() -> None:
    p = parser_for("llama.cpp/kld-eval")
    assert p("perplexity: calculating perplexity over 100 chunks, n_ctx=512") == (0, 100)
    assert p(PPL) == (3, 100)
    q = parser_for("llama.cpp/imatrix")
    assert q(IMATRIX) == (1, 140)


def test_convert_parser_counts_tensors() -> None:
    assert parser_for("llama.cpp/convert", total_tensors=310)(CONVERT) == (3, 310)
    assert parser_for("llama.cpp/convert", total_tensors=None)("INFO: Loading") is None


def test_unknown_recipe_has_no_parser() -> None:
    assert parser_for("unsloth/finetune") is None


def test_plain_console_prints_progress_at_ten_percent_steps() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        c = Console(color=False)
        with c.progress("quantize Q4_K_M", total=100) as p:
            for i in range(1, 101):
                p.update(i)
        c.table(["a", "b"], [[1, "x"]])
        c.ok("fine")
    out = buf.getvalue()
    assert "quantize Q4_K_M ..." in out
    assert out.count("%") >= 9 and "done in" in out
    assert "a  b" in out and "fine" in out
    assert not c.use_rich


def test_quiet_console_only_prints_failures(capsys) -> None:
    c = Console(color=False, quiet=True)
    c.info("hidden")
    c.ok("hidden too")
    c.fail("boom")
    captured = capsys.readouterr()
    assert captured.out == "" and "boom" in captured.err


def test_styles() -> None:
    assert verdict_style("pass") == "green" and verdict_style("fail") == "red"
    assert error_style(3.0) == "green" and error_style(-40.0) == "red" and error_style(None) is None
