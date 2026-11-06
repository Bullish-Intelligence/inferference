"""Baseline store + regression gate.

Baselines are keyed by workload + fingerprint (model sha + lib manifest + GPU +
driver), never by date. This module stores the
aggregate summary of a run under
``artifacts/baselines/<workload>@<anchor>-<featureset>.json`` and provides the
regression gate: a p50 decode-tps regression beyond a threshold vs the baseline
fails (for the same model + GPU + driver fingerprint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BaselineStore:
    """Filesystem baseline store under an artifacts root."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "baselines"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def save(self, key: str, summary: dict[str, Any]) -> Path:
        target = self._path(key)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return target

    def load(self, key: str) -> dict[str, Any] | None:
        target = self._path(key)
        if not target.is_file():
            return None
        return json.loads(target.read_text())

    def check_regression(
        self,
        key: str,
        summary: dict[str, Any],
        *,
        metric: str = "generation_tokens_per_second",
        max_regression_percent: float = 5.0,
    ) -> tuple[bool, dict[str, Any]]:
        """Compare a run's aggregate vs the stored baseline; True = regression.

        Fail-closed: no stored baseline is NOT a regression (it is a new
        baseline — the caller decides whether to save it); a missing metric is a
        regression (cannot verify == fail).
        """
        baseline = self.load(key)
        if baseline is None:
            return False, {"reason": "no_baseline"}
        got = _nested(summary, ("end_to_end", metric))
        want = _nested(baseline, ("end_to_end", metric))
        if got is None or want is None or want == 0:
            return True, {"reason": "missing_metric", "metric": metric, "got": got, "baseline": want}
        delta_percent = (got - want) * 100 / want
        return delta_percent < -max_regression_percent, {
            "metric": metric,
            "got": got,
            "baseline": want,
            "delta_percent": delta_percent,
        }


def _nested(mapping: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cursor: object = mapping
    for part in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return float(cursor) if isinstance(cursor, (int, float)) else None
