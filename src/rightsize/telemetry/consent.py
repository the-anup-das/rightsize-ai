"""Consent for calibration records: off until the user turns it on (F9).

Nothing is recorded while this is off, and nothing is ever sent: records stay in a local
file the user can read, and leave the machine only when the user exports them and attaches
them somewhere themselves.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

WHAT_IS_RECORDED = """\
rightsize calibrate compares what rightsize predicted with what your GPU actually holds,
for the models Ollama or LM Studio have loaded. With recording on, each comparison is
appended to a local file, one JSON line per model:

  device    vendor, catalog name (e.g. "RTX 4070 Ti SUPER 16GB"), memory, compute capability, OS
  runtime   ollama, lm studio or llama.cpp, and its version when it reports one
  model     the Hugging Face repo id only when the repo is public; parameter count,
            architecture, quantization, context length
  numbers   predicted memory (weights, KV cache, overhead), measured memory, speed if known
  meta      rightsize version, the date (no time)

Never recorded: file paths, user or host names, IP addresses, prompts or outputs, the
names of private models. Nothing is sent anywhere. 'rightsize telemetry export FILE' writes
the records to a file you can read first and attach to an issue if you choose to."""


def _config_dir() -> Path:
    return Path(os.environ.get("RIGHTSIZE_CONFIG_DIR", Path.home() / ".config" / "rightsize"))


def data_home() -> Path:
    return Path(
        os.environ.get("RIGHTSIZE_DATA_HOME", Path.home() / ".local" / "share" / "rightsize")
    )


def store_path() -> Path:
    return data_home() / "measurements.jsonl"


def _consent_file() -> Path:
    return _config_dir() / "telemetry.json"


def enabled() -> bool:
    try:
        return bool(json.loads(_consent_file().read_text(encoding="utf-8")).get("enabled"))
    except (OSError, ValueError):
        return False


def status() -> dict:
    try:
        state = json.loads(_consent_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    path = store_path()
    count = 0
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())
    return {
        "enabled": bool(state.get("enabled")),
        "since": state.get("since"),
        "store": str(path),
        "records": count,
    }


def _write(state: dict) -> None:
    path = _consent_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def enable() -> None:
    _write({"enabled": True, "since": dt.date.today().isoformat()})


def disable() -> None:
    _write({"enabled": False})
