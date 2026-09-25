"""Locate and load the bundled data seed (hardware, quants, rules, recipes).

Installed wheels carry the seed at ``rightsize/_data`` (hatch force-include). Editable
installs and source checkouts fall back to ``<repo>/data``. No I/O at import time.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    """Where the data is read from: RIGHTSIZE_DATA_DIR, else the data a ``rightsize data
    update`` installed (validated before it was switched to), else the bundled seed."""
    override = os.environ.get("RIGHTSIZE_DATA_DIR")
    if override:
        return Path(override)
    base = Path(os.environ.get("RIGHTSIZE_CACHE_DIR", Path.home() / ".cache" / "rightsize"))
    updated = base / "data" / "current"
    if (updated / ".rightsize-data.json").is_file():
        return updated
    return bundled_dir()


def bundled_dir() -> Path:
    """The data this release shipped with: the wheel's copy, or data/ in a checkout."""
    here = Path(__file__).resolve().parent
    bundled = here / "_data"
    if bundled.is_dir():
        return bundled
    repo = here.parent.parent / "data"
    if repo.is_dir():
        return repo
    raise FileNotFoundError(
        "rightsize data seed not found; set RIGHTSIZE_DATA_DIR or reinstall the package"
    )


@functools.lru_cache(maxsize=64)
def load_yaml(relative: str) -> Any:
    import yaml

    path = data_dir() / relative
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
