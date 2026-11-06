"""Resolve machine-inventory paths that the GPU suites and gates need.

Nothing here carries a default path. Which GGUF a machine holds, and where it
keeps its LoRA probes and tokenizer, is INVENTORY: it is declared per machine in
the untracked ``ci/runner/rig-local.sh`` (see ``rig-local.example.sh``) and
exported by ``ci/runner/rig-env.sh`` as ``MODEL_<ROLE>`` / ``TOKENIZER_<ROLE>``.

An unset variable therefore yields a path that cannot exist. That is deliberate:
the caller's readiness check returns False, the suite SKIPS with its reason, and
the probe exits non-zero. A missing capability is a skip or a loud failure,
never a silent pass against a model nobody chose.
"""

from __future__ import annotations

import os
from pathlib import Path

# A path under a directory no real machine has. ``Path.exists()`` is False, so a
# caller that checks `.exists()` skips; nothing is ever read from here.
_UNSET_ROOT = Path("/nonexistent/inferference-inventory")


def asset_path(env_name: str) -> Path:
    """Return the path named by ``env_name``, or an unset marker when it is empty."""
    value = os.environ.get(env_name, "").strip()
    return Path(value) if value else _UNSET_ROOT / env_name


__all__ = ["asset_path"]
