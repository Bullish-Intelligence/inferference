"""Baseline save/compare step for the bench job.

Wraps :class:`baselines.BaselineStore` for CI: given a run dir (``summary.json``
+ ``envelope.json``) and the build manifest, derive the baseline key
``<workload>@<built_commit>-<featureset>`` (never a date), save the run's
aggregate as the baseline when absent, and compare (regression gate on the p50
decode-tps metric) when present.

The A/B diff applies the false-positive rule: a perf delta is reported, and
only a regression beyond the configured threshold fails. The
threshold default is 5%; the key embeds the fingerprint so baselines
are only comparable for the same model + lib + GPU.

Usage:
    ci/library/.venv/bin/python src/inferference/benchkit/baseline_gate.py \
        --run-dir <run dir> --manifest <build-manifest.json> \
        [--workload json_workload] [--max-regression-percent 5.0]
        [--save]   # record the run as the new baseline (default: compare only)
        [--artifacts <baselines root>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from inferference.gpu_env import UNKNOWN_BACKEND_LABEL, lib_backend_label

from .baselines import BaselineStore
from .env import visible_gpu_uuid


def legacy_baseline_key(manifest: dict[str, object], workload: str) -> str:
    """The key format before the backend joined it: workload@<commit>-<featureset>.

    A baseline stored under this key came from a build whose backend nobody
    recorded. The recorded ones come from an earlier CUDA rig. Nothing reads
    them as a match. The gate reports them and leaves them on disk.
    """
    built = str(manifest.get("llama_cpp_commit") or manifest.get("ref_built") or "unknown")
    raw_features = manifest.get("features") or []
    assert isinstance(raw_features, list)
    featureset = ",".join(sorted(str(f) for f in raw_features)) or "stock"
    return f"{workload}@{built}-{featureset}"


def baseline_key(manifest: dict[str, object], workload: str, backend: str) -> str:
    """workload@<built-commit>-<featureset>@<backend> — deterministic, never a date.

    The backend is part of the key. A CUDA run and a Vulkan run of one commit
    are different quantities and must never compare.
    """
    return f"{legacy_baseline_key(manifest, workload)}@{backend}"


def cross_device_error(run_uuid: str | None, stored_uuid: str | None) -> str | None:
    """Error text when a baseline was recorded on a different GPU, else None.

    Fail-closed: the device index is a function of the device-visibility variable
    and reads 0 for whichever card was exposed, so only the GPU UUID names a
    card. A baseline recorded on one GPU must never be compared against a run on
    another — both cards are the same model, so the drift would be small and
    would never look wrong.
    """
    if stored_uuid is None:
        return None
    if run_uuid is None:
        return f"cannot read the run's GPU UUID to compare against baseline GPU {stored_uuid}"
    if run_uuid != stored_uuid:
        return f"cross-device compare refused: run GPU={run_uuid}, baseline GPU={stored_uuid}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="a bench run dir (summary.json + envelope.json)")
    ap.add_argument("--manifest", required=True, type=Path, help="the build-manifest.json of the lib that ran")
    ap.add_argument("--workload", default="json_workload")
    ap.add_argument("--max-regression-percent", type=float, default=5.0)
    ap.add_argument("--save", action="store_true", help="save the run as the baseline when absent")
    ap.add_argument("--artifacts", type=Path, default=None, help="baselines root (default: repo ci/.ci-artifacts)")
    args = ap.parse_args()

    summary_path = args.run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no summary.json in {args.run_dir}")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(args.manifest.read_text())

    # Persistent baselines root — resolution order: --artifacts, then
    # $BASELINES_ROOT (rig cache, survives Act workdir trims), then the repo
    # ci/.ci-artifacts fallback. The Act workdir's own ci/.ci-artifacts is
    # EPHEMERAL (trimmed with the workdir — a first baseline died that
    # way); rig-env.sh exports BASELINES_ROOT outside the workdir so the store
    # persists. Re-save-on-first-run is the documented semantics: a key with no
    # stored baseline saves (--save) instead of comparing.
    root = args.artifacts or Path(os.environ.get("BASELINES_ROOT") or "")
    if not root.is_dir():
        root = Path(__file__).resolve().parents[3] / "ci/.ci-artifacts"
    store = BaselineStore(root)
    backend = lib_backend_label(args.manifest.parent / "lib", manifest)
    if backend == UNKNOWN_BACKEND_LABEL:
        # An unknown backend would store or compare under one shared key.
        print(f"[baseline] FAIL: the library set in {args.manifest.parent / 'lib'} gives no evidence of its backend")
        return 1
    key = baseline_key(manifest, args.workload, backend)
    legacy = legacy_baseline_key(manifest, args.workload)
    gpu_uuid = visible_gpu_uuid()
    print(f"[baseline] key={key} (workload@commit-featureset@backend) backend={backend} gpu={gpu_uuid}")
    if store.load(legacy) is not None:
        print(
            f"[baseline] NOTE legacy baseline '{legacy}' exists in {store.dir}. It has no backend "
            f"and is NOT compared with this {backend} run. It stays on disk, untouched."
        )

    # Cross-device guard: a baseline recorded on one GPU must never be
    # compared against a run on another. The GPU UUID is the identity, not the
    # index (the index is a function of the device-visibility variable).
    existing = store.load(key)
    guard_error = cross_device_error(gpu_uuid, existing.get("_baseline_gpu_uuid") if existing else None)
    if guard_error is not None:
        print(f"[baseline] FAIL — {guard_error}")
        return 1

    regression, info = store.check_regression(
        key,
        summary,
        metric="generation_tokens_per_second",
        max_regression_percent=args.max_regression_percent,
    )
    print(f"[baseline] check: {info}")
    if info.get("reason") == "no_baseline":
        if args.save:
            summary["_baseline_gpu_uuid"] = gpu_uuid
            summary["_baseline_backend"] = backend
            path = store.save(key, summary)
            print(f"[baseline] NO COMPARISON MADE: no baseline for key {key}. Saved this run as the baseline -> {path}")
            return 0
        print(f"[baseline] FAIL — no baseline for key {key} in {store.dir}. Nothing was compared. Use --save.")
        return 1
    if regression:
        print(
            f"[baseline] FAIL — {info.get('metric')} regressed {info.get('delta_percent'):.2f}% "
            f"vs baseline (got={info.get('got')} baseline={info.get('baseline')})"
        )
        return 1
    print(
        f"[baseline] PASS — {info.get('metric')} delta {info.get('delta_percent'):.2f}% "
        f"(got={info.get('got')} baseline={info.get('baseline')}); FP rule: reported, not failed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
