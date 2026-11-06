"""Thermal pacing: hold a workload inside a temperature band.

WHY THIS EXISTS. The MI25s are passive server cards: they carry no fan of their
own BY DESIGN and are cooled by the chassis/server fans. So nothing on the card
responds to a rising temperature, and the hardware throttles that would normally
solve this (``power1_cap``, ``power_dpm_force_performance_level``) are root-owned
and not writable by the user this harness runs as.

So the only lever left is HOW OFTEN WE ASK THE CARD TO WORK, and this is it.

THE CURVE THAT MAKES IT CHEAP (measured 2026-09-05, card0 HBM,
``live-2026-09-05/thermal-curve.tsv``):

    heating   59 -> 79 C    55 s   (~0.36 C/s under sustained decode)
    cooling   79 -> 63 C    10 s   (idle draw is near zero, so it dumps fast)
    cooling   63 -> 50 C   100 s   (long tail, and never needed)

Cooling is strongly asymmetric: the card sheds the top of its range in seconds
and then crawls toward a floor of about 48 C.

THE GRANULARITY IS ONE REQUEST, AND THAT IS WHAT DECIDES THE COST. A pause can
only be taken BETWEEN requests, so the safe band is set by how much ONE request
heats the card — measured here as roughly **+30 C over about 100 s**. A first
attempt paced at 78/66 looked cheap on the cooling curve and was wrong: a request
that began at 66 C ended at **95 C**, HBM's rated critical, because nothing could
intervene once it had started.

So the resume mark has to satisfy ``resume_c + per_request_rise < abort_c``,
which on this rig means resuming near the 50 C floor and waiting out the slow
tail. That is roughly **100 s of cooling per 100 s of work — about 2x wall-clock
overhead**, not the 25% the cooling curve alone suggests. Pacing makes a long run
POSSIBLE here; it does not make it cheap.

WHAT PACING DOES TO THE NUMBERS. A paced run's wall clock includes time the card
was deliberately idle, so throughput over wall clock would measure the cooling
system rather than the build. Every rate this harness reports for a paced run is
therefore computed over ACTIVE time, with the paused time recorded separately
and visibly. A paced run is honest about being paced or it is not comparable
with anything.

Pacing happens BETWEEN requests, never inside one: a pause mid-generation would
land in that request's decode timing and corrupt exactly what is being measured.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ThermalAbort", "ThermalPacer", "hottest_c", "read_card_temps"]


class ThermalAbort(RuntimeError):
    """A hard temperature ceiling was crossed while a request was in flight."""


def read_card_temps(card: int) -> dict[str, int]:
    """Every amdgpu temperature sensor for a DRM card, in whole degrees C."""
    base = Path(f"/sys/class/drm/card{card}/device")
    out: dict[str, int] = {}
    for hwmon in sorted(base.glob("hwmon/hwmon*")):
        try:
            if (hwmon / "name").read_text().strip() != "amdgpu":
                continue
        except OSError:
            continue
        for sensor in sorted(hwmon.glob("temp*_input")):
            label_file = sensor.with_name(sensor.name.replace("_input", "_label"))
            try:
                value = int(sensor.read_text().strip()) // 1000
            except (OSError, ValueError):
                continue
            try:
                label = label_file.read_text().strip()
            except OSError:
                label = sensor.name
            out[label] = value
    return out


def hottest_c(card: int) -> int | None:
    """The highest reading on a card.

    The HOTTEST sensor, not the edge one. Measured on this rig: at the moment a
    run had to be killed, edge was 71 C and junction 79 C while memory was 88 C.
    Memory is what binds here, always, and gating on edge would have missed
    every abort.
    """
    temps = read_card_temps(card)
    return max(temps.values()) if temps else None


@dataclass
class ThermalPacer:
    """Hold a card inside ``[resume_c, pause_c]`` by delaying new requests.

    A pacer with ``card=None`` is disabled and costs nothing, so the drive loop
    can call it unconditionally.
    """

    card: int | None = None
    pace: bool = True
    """Whether to PAUSE between requests. The abort watchdog is independent: a
    run that is not paced still must not be allowed to cook the card."""
    pause_c: int = 55
    resume_c: int = 50
    abort_c: int = 85
    poll_s: float = 2.0

    paused_s: float = 0.0
    pauses: int = 0
    peak_c: int = 0
    _samples: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.card is None:
            return
        if self.pace and self.resume_c >= self.pause_c:
            raise ValueError(
                f"resume_c ({self.resume_c}) must be below pause_c ({self.pause_c}); "
                f"otherwise the pacer can never release."
            )
        if self.pace and self.pause_c >= self.abort_c:
            raise ValueError(
                f"pause_c ({self.pause_c}) must be below abort_c ({self.abort_c}); "
                f"a pacer that only acts at the abort ceiling protects nothing."
            )

    @property
    def enabled(self) -> bool:
        """Whether anything thermal is active at all — pacing or the watchdog."""
        return self.card is not None

    @property
    def pacing(self) -> bool:
        return self.card is not None and self.pace

    def observe(self) -> int | None:
        if self.card is None:
            return None
        now = hottest_c(self.card)
        if now is not None:
            self._samples.append(now)
            self.peak_c = max(self.peak_c, now)
        return now

    async def wait_if_hot(self) -> None:
        """Block until the card is below ``resume_c``. No-op when cool.

        Called before a request is ISSUED. It never interrupts one in flight.
        """
        if not self.pacing:
            return
        now = self.observe()
        if now is None or now < self.pause_c:
            return
        started = time.perf_counter()
        self.pauses += 1
        while True:
            await asyncio.sleep(self.poll_s)
            now = self.observe()
            if now is None or now <= self.resume_c:
                break
        self.paused_s += time.perf_counter() - started

    async def guard(self) -> None:
        """Watch the card WHILE requests are in flight, and abort if it cooks.

        Pacing alone is not enough and this rig proves it. The pacer can only act
        BETWEEN requests, but a single request here runs for around 100 s and
        raises HBM by roughly 30 C. Measured 2026-09-05: a run paced at 78/66
        completed, but a request that began at 66 C drove the card to **95 C** —
        HBM's rated critical — with the pacer unable to intervene because it was
        waiting for that request to finish.

        So the ceiling is enforced by this watchdog instead, which samples
        throughout and aborts the run. A voided run is the correct outcome: the
        alternative is a number obtained by cooking the card, and a thermally
        throttled card reads as a slow build anyway.

        Runs forever until cancelled or until the ceiling is crossed.
        """
        if self.card is None:
            await asyncio.Event().wait()
        while True:
            await asyncio.sleep(self.poll_s)
            now = self.observe()
            if now is not None and now >= self.abort_c:
                raise ThermalAbort(
                    f"card {self.card} reached {now} C, at or over the {self.abort_c} C ceiling, "
                    f"while a request was in flight. The run is VOID: its timings describe a "
                    f"card that was throttling. Lower the per-request output budget, or improve "
                    f"chassis airflow."
                )

    def record(self) -> dict[str, object]:
        """What the envelope needs to say the run was paced, and by how much."""
        return {
            "enabled": self.enabled,
            "pacing": self.pacing,
            "card": self.card,
            "pause_c": self.pause_c,
            "resume_c": self.resume_c,
            "abort_c": self.abort_c,
            "pauses": self.pauses,
            "paused_s": round(self.paused_s, 2),
            "peak_c": self.peak_c or None,
            "mean_c": (round(sum(self._samples) / len(self._samples), 1) if self._samples else None),
        }
