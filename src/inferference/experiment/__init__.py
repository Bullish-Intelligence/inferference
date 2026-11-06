"""Declare, run, analyze, and compare hardware experiments.

The package validates experiment specs, checks host state before a run, and
stores write-once artifacts. Its analysis modules derive results from those
artifacts without hardware access. Run the command interface with
``python -m inferference.experiment``.
"""

from __future__ import annotations
