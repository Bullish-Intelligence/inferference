"""Locate the V620 ``pp_table`` blobs that the tests use.

The vendor blobs are **not redistributed in this repository**. A ``pp_table`` is
AMD board-power firmware data read from ``/sys/class/drm/card<N>/device/pp_table``:
it is vendor-supplied configuration with no redistribution grant attached, so it
lives in the private project log and is supplied to the tests through the
environment.

Two flavours, deliberately separate:

* :func:`real_stock_blob` returns the genuine bytes when the blob is present and
  SKIPS (loudly) when it is not. Only the evidentiary tests use it — the ones
  that assert a pinned SHA-256 or a byte-for-byte round trip of real firmware.
* :func:`synthetic_stock_blob` builds a structurally valid 2470-byte table from
  the public header constants. Every non-evidentiary test uses it, so the codec,
  preflight, manifest and analysis tests run on any machine.

Neither path is a silent pass: an absent blob skips with its reason, and a blob
whose hash does not match the pin FAILS.
"""

from __future__ import annotations

import hashlib
import os
import struct
from functools import lru_cache
from pathlib import Path

import pytest

from inferference.pptable import (
    FORMAT_REVISION,
    STRUCT_SIZE,
    TABLE_REVISION,
    PptThrottler,
    TempSensor,
    decode,
)

STOCK_NAME = "stock-card0.pp_table"
STOCK_SHA256 = "a6fc019fdada096422629293bee778e8857af3330fd2dc2de42dd9d9d921b1c8"

# The documented stock limits (also asserted by the fake hosts in
# tests/test_experiment_{preflight,runner}.py).
STOCK_PPT0_W = 250
STOCK_TEMPERATURE_LIMIT_C = {
    TempSensor.EDGE: 100,
    TempSensor.HOTSPOT: 100,
    TempSensor.MEM: 98,
}


def blob_dir() -> Path:
    """The directory holding the non-redistributed blobs."""
    return Path(os.environ.get("INFERFERENCE_PP_TABLE_DIR", "~/.local/share/inferference/pp-tables")).expanduser()


def stock_path() -> Path:
    return blob_dir() / STOCK_NAME


def real_stock_blob() -> bytes:
    """The genuine stock blob, or a skip naming the path that was looked at."""
    path = stock_path()
    if not path.is_file():
        pytest.skip(
            f"vendor pp_table blob not present at {path}: the blobs are not redistributed "
            "with this repository; set INFERFERENCE_PP_TABLE_DIR to the directory that holds them"
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != STOCK_SHA256:
        pytest.fail(f"{path} sha256 is {digest}; expected the pinned {STOCK_SHA256}")
    return data


@lru_cache(maxsize=1)
def synthetic_stock_blob() -> bytes:
    """A decodable 2470-byte table carrying the documented stock values.

    Built from the public header constants and the four stock limits the project
    documents in prose (PPT0 250 W; EDGE/HOTSPOT 100 C; MEM 98 C). It contains
    no vendor firmware bytes — only values already written down in the tests and
    READMEs — but it decodes and round-trips, so the codec, preflight, runner and
    analysis tests exercise real logic on any machine.
    """
    buf = bytearray(STRUCT_SIZE)
    struct.pack_into("<H", buf, 0, STRUCT_SIZE)
    buf[2] = FORMAT_REVISION
    buf[4] = TABLE_REVISION
    table = decode(bytes(buf))
    table = table.set_socket_power_limit_ac(PptThrottler.PPT0, STOCK_PPT0_W)
    for sensor, celsius in STOCK_TEMPERATURE_LIMIT_C.items():
        table = table.set_temperature_limit(sensor, celsius)
    return table.encode()


__all__ = [
    "STOCK_NAME",
    "STOCK_PPT0_W",
    "STOCK_SHA256",
    "STOCK_TEMPERATURE_LIMIT_C",
    "blob_dir",
    "real_stock_blob",
    "stock_path",
    "synthetic_stock_blob",
]
