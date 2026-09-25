"""Search the curated model lists by family, task and size (F1).

Two lists, both generated from what people download on the Hub and shipped as data, so a
search is offline and instant: data/models/candidates.yaml for LLMs (tasks ``chat`` and
``coding``, the pool recommend ranks) and data/models/families.yaml for diffusion, audio,
vision and embedding models (tasks named as the Hub names them: text-to-image,
automatic-speech-recognition, sentence-similarity, ...). Anything not on a list can still be
estimated by its Hub id.
"""

from __future__ import annotations

from typing import Any

from rightsize._data import load_yaml
from rightsize.types import CatalogEntry, Family


def _entries() -> list[CatalogEntry]:
    out = []
    for m in load_yaml("models/candidates.yaml")["models"]:
        f = m["facts"]
        out.append(CatalogEntry(
            repo=m["repo"], family=Family.llm, tasks=list(m["tasks"]), publisher=m["publisher"],
            params_total=f["params_total"], params_active=f.get("params_active"),
            license=m.get("license"), gated=bool(m.get("gated")),
            created_at=m.get("created_at"), downloads_30d=m.get("downloads_30d"),
        ))
    for m in load_yaml("models/families.yaml")["models"]:
        out.append(CatalogEntry(
            repo=m["repo"], family=Family(m["family"]), tasks=list(m["tasks"]),
            publisher=m["publisher"], params_total=m["params_total"],
            license=m.get("license"), gated=bool(m.get("gated")),
            created_at=m.get("created_at"), downloads_30d=m.get("downloads_30d"),
        ))
    return out


def tasks() -> dict[str, list[str]]:
    """Every task search() knows, by family."""
    known: dict[str, Any] = dict(load_yaml("models/families.yaml")["tasks"])
    return {"llm": ["chat", "coding"], **known}


def search(
    family: str | Family | None = None,
    task: str | None = None,
    *,
    max_params: float | None = None,
    min_params: float | None = None,
    publisher: str | None = None,
    license: str | None = None,
    include_gated: bool = True,
    limit: int = 20,
) -> list[CatalogEntry]:
    """Curated models that match, most downloaded first.

    ``max_params`` and ``min_params`` bound the total parameter count (``13e9``); a
    mixture-of-experts model counts all its experts, since all of them take memory.
    ``license`` matches the card's licence id (apache-2.0, mit, ...)."""
    fam = Family(family) if family else None
    if task and fam is None:
        fam = next((Family(f) for f, ts in tasks().items() if task in ts), None)
    found = []
    for e in _entries():
        if fam is not None and e.family is not fam:
            continue
        if task and task not in e.tasks:
            continue
        if max_params is not None and e.params_total > max_params:
            continue
        if min_params is not None and e.params_total < min_params:
            continue
        if publisher and e.publisher.lower() != publisher.lower():
            continue
        licences = e.license if isinstance(e.license, list) else [e.license]
        if license and license.lower() not in {str(x).lower() for x in licences}:
            continue
        if e.gated and not include_gated:
            continue
        found.append(e)
    found.sort(key=lambda e: -(e.downloads_30d or 0))
    return found[:limit]
