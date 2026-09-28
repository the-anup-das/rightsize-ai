"""Read a Hugging Face profile's saved hardware (F2).

Signed-in Hub users can record what they own, and the Hub serves it on the public profile:

    GET https://huggingface.co/api/users/<name>/overview
    {"hardwareItems": [{"sku": ["GPU", "NVIDIA", "RTX 3090"], "mem": 24, "num": 2}, ...]}

The SKU triple is keyed to the same table data/hardware/gpus.yaml is ingested from, so an
entry resolves straight to a catalogue Device, bandwidth and all. It means someone can say
``rightsize estimate <model> --device @their-username`` instead of typing specs.
"""

from __future__ import annotations

import httpx

from rightsize.errors import RightsizeError
from rightsize.hardware.db import catalog
from rightsize.types import Device, Provenance

API = "https://huggingface.co/api/users/{username}/overview"


class HubHardwareError(RightsizeError):
    pass


def from_hf(
    username: str,
    *,
    token: str | None = None,
    timeout: float = 15.0,
    transport: httpx.BaseTransport | None = None,
) -> list[Device]:
    """Devices from a Hub profile, the user's primary machine first.

    Raises ``HubHardwareError`` when the profile does not exist or lists no hardware; an
    empty list is never returned, so callers do not silently estimate against nothing.
    """
    username = username.lstrip("@")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as c:
        r = c.get(API.format(username=username), headers=headers)
    if r.status_code == 404:
        raise HubHardwareError(f"no Hugging Face profile for {username!r}")
    r.raise_for_status()
    items = r.json().get("hardwareItems") or []
    if not items:
        raise HubHardwareError(
            f"{username!r} has no hardware saved on their profile "
            "(https://huggingface.co/settings/local-apps)"
        )
    items.sort(key=lambda i: not i.get("isPrimary"))
    return [d for item in items for d in _devices(item, username)]


def _devices(item: dict, username: str) -> list[Device]:
    """One item may be several identical cards (``num``); each becomes its own Device."""
    sku = item.get("sku") or []
    model = sku[-1] if sku else "unknown"
    mem = item.get("mem")
    base = _lookup(model, mem)
    prov = Provenance(
        source_url=API.format(username=username),
        fetched_at="",
        note=f"saved on the {username} profile as {sku}",
    )
    out = []
    for n in range(max(1, int(item.get("num", 1)))):
        name = base.name if item.get("num", 1) == 1 else f"{base.name} #{n + 1}"
        out.append(base.model_copy(update={"name": name, "provenance": prov}))
    return out


def _lookup(model: str, mem: float | None) -> Device:
    """Match the SKU against the ingested catalogue, honouring the memory the user chose."""
    cat = catalog()
    if mem is not None and (exact := cat.get(f"{model} {mem:g}GB")):
        return exact
    same_model = [d for name, d in cat.items() if name.rsplit(" ", 1)[0] == model]
    if same_model:
        # A memory size we do not list (odd configs, eGPU boxes): keep the rest of the
        # record and take the user's word for the size.
        best = max(same_model, key=lambda d: d.memory_gib)
        return best.model_copy(update={"memory_gib": mem}) if mem else best
    if mem is None:
        raise HubHardwareError(f"{model!r} is not in the catalogue and the profile gives no size")
    return Device(name=f"{model} {mem:g}GB", memory_gib=mem)
