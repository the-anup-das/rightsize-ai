"""Newer data without a new release: rightsize data update (F6, Run 1).

The hardware table, quantization tables, rules, recipes and measurements change more often
than the code. ``update()`` downloads the ``data/`` directory of a git ref of this project
(main by default, or a tag), validates every file against the schemas of the rightsize
that is installed, and only then makes it the data rightsize reads, kept in
~/.cache/rightsize/data/current. Data that does not validate - written for a newer
rightsize, or broken - is refused and the old data stays. ``reset()`` goes back to the
data the package shipped with.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

REPO = "the-anup-das/rightsize-ai"
TARBALL = "https://codeload.github.com/{repo}/tar.gz/{ref}"
MARKER = ".rightsize-data.json"
#: File types a data directory may hold; anything else in the tarball is ignored.
_ALLOWED = (".yaml", ".yml", ".json", ".md", ".csv", ".txt")


class DataUpdateError(RuntimeError):
    pass


def cache_root() -> Path:
    base = Path(os.environ.get("RIGHTSIZE_CACHE_DIR", Path.home() / ".cache" / "rightsize"))
    return base / "data"


def current_dir() -> Path | None:
    """The updated data directory, when one has been installed and not reset."""
    path = cache_root() / "current"
    return path if (path / MARKER).is_file() else None


def _safe_members(tf: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    """(member, path inside data/) for regular files under <root>/data/, and nothing that
    could land outside the destination: no links, no absolute paths, no '..'."""
    out = []
    for m in tf.getmembers():
        parts = PurePosixPath(m.name).parts
        if len(parts) < 3 or parts[1] != "data" or not m.isfile():
            continue
        rel = PurePosixPath(*parts[2:])
        if rel.is_absolute() or ".." in rel.parts or not rel.name.endswith(_ALLOWED):
            continue
        out.append((m, rel))
    return out


def _extract(blob: bytes, dest: Path) -> tuple[int, str]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        members = _safe_members(tf)
        if not members:
            raise DataUpdateError("the download has no data/ directory")
        for m, rel in members:
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is not None:
                target.write_bytes(src.read())
        root = PurePosixPath(tf.getmembers()[0].name).parts[0]
    return len(members), root


def update(ref: str = "main", *, repo: str = REPO, client=None) -> dict:
    """Download, validate and switch to the data at ``ref``. Returns the marker written."""
    import httpx

    from rightsize._data import bundled_dir, load_yaml
    from rightsize.data_models import validate_all

    own = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=True)
    try:
        r = client.get(TARBALL.format(repo=repo, ref=ref))
    except httpx.HTTPError as exc:
        raise DataUpdateError(f"download failed: {exc}") from exc
    finally:
        if own:
            client.close()
    if r.status_code != 200:
        raise DataUpdateError(f"download failed: HTTP {r.status_code} for {ref!r}")

    root = cache_root()
    staging = root / f"staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        count, tar_root = _extract(r.content, staging)
        # Every file this release reads must be there: data from an older or partial ref
        # can validate file by file and still lack one, and that breaks at run time.
        wanted = {f.relative_to(bundled_dir()).as_posix() for f in bundled_dir().rglob("*.yaml")}
        have = {f.relative_to(staging).as_posix() for f in staging.rglob("*.yaml")}
        missing = sorted(wanted - have)
        if missing:
            raise DataUpdateError(
                f"the new data lacks {len(missing)} file(s) this rightsize reads, so it is "
                "older or partial; keeping the current data. Missing: " + ", ".join(missing[:5])
            )
        problems = validate_all(staging)
        if problems:
            raise DataUpdateError(
                "the new data does not validate against this rightsize (it may need a newer "
                "release); keeping the current data. First problems:\n  "
                + "\n  ".join(problems[:5])
            )
        marker = {
            "repo": repo,
            "ref": ref,
            "archive_root": tar_root,
            "files": count,
            "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        (staging / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        current, previous = root / "current", root / "previous"
        shutil.rmtree(previous, ignore_errors=True)
        if current.exists():
            current.rename(previous)
        staging.rename(current)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    load_yaml.cache_clear()
    return marker


def status() -> dict:
    from rightsize._data import data_dir

    cur = current_dir()
    marker = json.loads((cur / MARKER).read_text(encoding="utf-8")) if cur else None
    return {
        "in_use": str(data_dir()),
        "updated": marker,
        "override": os.environ.get("RIGHTSIZE_DATA_DIR"),
    }


def reset() -> bool:
    """Back to the data the package shipped with. True when there was an update to remove."""
    from rightsize._data import load_yaml

    cur = cache_root() / "current"
    existed = cur.exists()
    shutil.rmtree(cur, ignore_errors=True)
    load_yaml.cache_clear()
    return existed
