"""rightsize data update: newer data, validated before it is used (F6, Run 1)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import httpx
import pytest
import yaml

from rightsize import data_update
from rightsize._data import data_dir, load_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGHTSIZE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("RIGHTSIZE_DATA_DIR", raising=False)
    load_yaml.cache_clear()
    yield
    load_yaml.cache_clear()


def _tarball(edit=None, extra: dict[str, bytes] | None = None) -> bytes:
    """The repo's data/ as GitHub's codeload serves it: under one <repo>-<ref>/ folder."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted((ROOT / "data").rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT / "data").as_posix()
            body = path.read_bytes()
            if edit:
                body = edit(rel, body)
            info = tarfile.TarInfo(f"rightsize-ai-main/data/{rel}")
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
        for name, body in (extra or {}).items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
        link = tarfile.TarInfo("rightsize-ai-main/data/hardware/link.yaml")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    return buf.getvalue()


def _client(blob: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=blob))
    )


def _more_presets(rel: str, body: bytes) -> bytes:
    if rel != "hardware/presets.yaml":
        return body
    doc = yaml.safe_load(body)
    doc["presets"][0]["bandwidth_gbps"] = 123.0
    return yaml.safe_dump(doc).encode()


def test_an_update_is_validated_then_used() -> None:
    bundled = data_dir()
    marker = data_update.update("main", client=_client(_tarball(_more_presets)))
    assert marker["ref"] == "main" and marker["files"] > 20
    assert data_dir() == data_update.current_dir() != bundled
    assert load_yaml("hardware/presets.yaml")["presets"][0]["bandwidth_gbps"] == 123.0
    assert data_update.status()["updated"]["archive_root"] == "rightsize-ai-main"


def test_data_that_does_not_validate_is_refused_and_the_old_data_stays() -> None:
    data_update.update("main", client=_client(_tarball()))
    good = data_update.current_dir()

    def break_it(rel: str, body: bytes) -> bytes:
        return b"presets: [{name: x}]\n" if rel == "hardware/presets.yaml" else body

    with pytest.raises(data_update.DataUpdateError, match="does not validate"):
        data_update.update("v9", client=_client(_tarball(break_it)))
    assert data_update.current_dir() == good
    assert data_update.status()["updated"]["ref"] == "main"


def test_nothing_lands_outside_the_data_directory() -> None:
    hostile = {
        "rightsize-ai-main/data/../../escape.yaml": b"x: 1\n",
        "rightsize-ai-main/README.md": b"not data\n",
        "rightsize-ai-main/data/hardware/tool.exe": b"MZ",
    }
    data_update.update("main", client=_client(_tarball(extra=hostile)))
    cur = data_update.current_dir()
    assert not (cur.parent.parent / "escape.yaml").exists()
    assert not (cur / "hardware" / "tool.exe").exists()
    assert not (cur / "hardware" / "link.yaml").exists(), "links are not extracted"


def test_a_failed_download_changes_nothing() -> None:
    with pytest.raises(data_update.DataUpdateError, match="HTTP 404"):
        data_update.update("no-such-ref", client=_client(b"", status=404))
    assert data_update.current_dir() is None


def test_reset_returns_to_the_bundled_data() -> None:
    bundled = data_dir()
    data_update.update("main", client=_client(_tarball()))
    assert data_update.reset() is True
    assert data_dir() == bundled and data_update.reset() is False


def test_offline_answers_from_the_cache_only(monkeypatch) -> None:
    from rightsize.catalog import facts

    monkeypatch.setenv("RIGHTSIZE_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline"):
        facts("someone/never-cached")


def test_partial_data_is_refused_even_when_what_is_there_validates() -> None:
    """A branch from before the formats table and the rules existed validates file by
    file; switching to it would break everything that reads the files it lacks."""
    blob = _tarball()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        kept = [m for m in tf.getmembers() if not m.name.endswith("quants/formats.yaml")]
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as out:
            for m in kept:
                out.addfile(m, tf.extractfile(m) if m.isfile() else None)
    with pytest.raises(data_update.DataUpdateError, match="quants/formats.yaml"):
        data_update.update("old", client=_client(buf.getvalue()))
    assert data_update.current_dir() is None
