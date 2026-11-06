"""benchkit — the bench harness package.

Public surface: :func:`run` (one workload run), :func:`ab` (A/B across two
builds), :func:`gate` (the token-exact validation gates). Imports are
capability-guarded where a fork capability is involved (the seq-routing
surface probes at bind time, never at import — the library's own rule).
"""

from __future__ import annotations

from .gates import aggregate_outcomes, run_once
from .workload import Workload, list_workloads, load

__all__ = ["Workload", "aggregate_outcomes", "list_workloads", "load", "run_once"]
