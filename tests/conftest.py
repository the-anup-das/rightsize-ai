"""Test isolation: nothing a developer's own rightsize has stored reaches a test.

Model facts, measured bandwidth, cloud prices and updated data live under the cache
directory; consent and calibration records under the config and data homes. Every test
session gets empty ones, so a warm cache or a 'rightsize data update' on this machine
cannot change what a test sees.
"""

from __future__ import annotations

import os

import pytest

_ISOLATED = ("RIGHTSIZE_CACHE_DIR", "RIGHTSIZE_CONFIG_DIR", "RIGHTSIZE_DATA_HOME")


@pytest.fixture(autouse=True, scope="session")
def _isolated_homes(tmp_path_factory):
    saved = {k: os.environ.get(k) for k in (*_ISOLATED, "RIGHTSIZE_OFFLINE")}
    for key in _ISOLATED:
        os.environ[key] = str(tmp_path_factory.mktemp(key.lower()))
    os.environ.pop("RIGHTSIZE_OFFLINE", None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
