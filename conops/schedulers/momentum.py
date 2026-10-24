"""The stored momentum a plan builds up, forecast along its timeline.

A plan holds attitudes and flies slews, and the environment torques the
spacecraft throughout, so the momentum its wheels store follows from the
plan itself. :class:`MomentumForecast` integrates the configured torque
sources along a timeline of blocks, holds between them included, with the
same tracker the simulations use, and finds where the stored momentum first
reaches a level. The planners reserve a thruster dump there and plan again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..common import ObsType
from ..common.ephemeris import position_vectors
from ..common.vector import attitude_to_quat
from ..ditl.plan_validator import entry_obstype
from ..simulation.momentum import (
    MomentumState,
    StoredMomentumTracker,
    slew_sample_interval_s,
    tracker_from_config,
)

if TYPE_CHECKING:
    from ..config import MissionConfig
    from .context import Attitude, SchedulingContext
    from .priority_scheduler import _Block


@dataclass(frozen=True)
class _Span:
    """An interval of the timeline at one attitude or along one slew."""

    begin: float
    end: float
    attitude: Attitude | None
    """The attitude held, or None along a slew."""
    slew: object | None = None
    """The slew flown, with an ``attitude(utime)`` of its own."""
    unload_at_end: float | None = None
    """The residual fraction a thruster dump leaves when the span ends."""

    @property
    def key(self) -> tuple[object, ...]:
        """What the momentum stored over the span depends on, besides the
        momentum it starts from: its times, its attitude or its slew's
        path, and the unloading at its end."""
        slew = self.slew
        path = (
            (
                float(slew.slewstart),  # type: ignore[attr-defined]
                float(slew.slewend),  # type: ignore[attr-defined]
                float(slew.slewdist),  # type: ignore[attr-defined]
                float(slew.startra),  # type: ignore[attr-defined]
                float(slew.startdec),  # type: ignore[attr-defined]
                float(slew.startroll),  # type: ignore[attr-defined]
                float(slew.endra),  # type: ignore[attr-defined]
                float(slew.enddec),  # type: ignore[attr-defined]
                float(slew.endroll),  # type: ignore[attr-defined]
            )
            if slew is not None
            else None
        )
        return (self.begin, self.end, self.attitude, path, self.unload_at_end)


_Integrated = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], MomentumState]
"""A span integrated: its sample times, the norm at each and the state at its end."""

SpanCache = dict[tuple[object, ...], _Integrated]
"""Spans integrated, by the span's key and the state they started from.
A plan's passes share one: each reserves a dump and plans again from the
start, so the timeline up to the new dump is the one before and its spans
are read from the cache; only the spans after it are integrated again."""


class MomentumForecast:
    """The stored momentum along a planned timeline."""

    def __init__(
        self,
        config: MissionConfig,
        ctx: SchedulingContext,
        *,
        start_momentum_eci_n_m_s: npt.ArrayLike | None = None,
        start_attitude: Attitude | None = None,
        cache: SpanCache | None = None,
    ) -> None:
        self.config = config
        self.ctx = ctx
        self.cache: SpanCache = cache if cache is not None else {}
        self.start_attitude = start_attitude
        """The attitude held from the horizon's start until the first planned
        slew, where the plan takes over part-way through a run; None where
        the run starts fresh and the spacecraft idles until first commanded."""
        self.tracker: StoredMomentumTracker | None = tracker_from_config(config)
        # As the simulations sample: a slew at the configured attitude step,
        # a held attitude at the ephemeris resolution.
        self.slew_sample_s = (
            slew_sample_interval_s(config) if self.tracker is not None else 0.0
        )
        self.hold_sample_s = float(ctx.ephem.step_size)
        self.start_momentum_eci_n_m_s = (
            np.asarray(start_momentum_eci_n_m_s, dtype=np.float64)
            if start_momentum_eci_n_m_s is not None
            else None
        )
        ephem = ctx.ephem
        self._times = np.array(
            [t.timestamp() for t in ephem.timestamp], dtype=np.float64
        )
        self._positions = np.asarray(position_vectors(ephem, "gcrs"), dtype=np.float64)
        self._sun = np.asarray(position_vectors(ephem, "sun"), dtype=np.float64)
        ecef = getattr(getattr(ephem, "itrs_pv", None), "position", None)
        self._ecef = np.asarray(ecef, dtype=np.float64) if ecef is not None else None

    @property
    def active(self) -> bool:
        return self.tracker is not None

    @property
    def level_n_m_s(self) -> float | None:
        """The stored momentum a thruster dump is called for at."""
        return (
            self.config.spacecraft_bus.attitude_control.stored_momentum.dump_level_n_m_s
        )

    @property
    def capacity_n_m_s(self) -> float | None:
        return (
            self.config.spacecraft_bus.attitude_control.stored_momentum.capacity_n_m_s
        )

    # ── The timeline as attitudes ─────────────────────────────────────────

    def _spans(self, timeline: Sequence[_Block]) -> list[_Span]:
        """The timeline as spans at one attitude or along one slew, from the
        horizon's start to its end, holding the attitude it is at between
        blocks; a visit's steps are spans of their own."""
        spans: list[_Span] = []
        cursor = self.ctx.ustart
        attitude: Attitude | None = None
        """What is held between blocks; None before the first command."""
        dump = self.config.spacecraft_bus.attitude_control.stored_momentum.thruster_dump
        for block in sorted(timeline, key=lambda b: float(b.entry.begin)):
            steps = block.steps or [block.entry]
            slews = block.step_slews or [block.slew]
            for step, slew in zip(steps, slews, strict=False):
                begin = float(step.begin)
                if begin > cursor:
                    spans.append(_Span(cursor, begin, attitude))
                    cursor = begin
                target = (
                    step.spacecraft_attitude
                    if step.spacecraft_attitude is not None
                    else (float(step.ra), float(step.dec), float(step.roll))
                )
                if slew is not None and float(slew.slewtime) > 0.0:
                    slew_end = min(float(slew.slewend), float(step.end))
                    if slew_end > cursor:
                        spans.append(_Span(cursor, slew_end, None, slew=slew))
                        cursor = slew_end
                end = float(step.end)
                unload = None
                if (
                    entry_obstype(block.entry) == ObsType.ACTIVITY
                    and block.entry.desaturation
                    and dump is not None
                ):
                    unload = dump.residual_fraction
                if end > cursor:
                    spans.append(_Span(cursor, end, target, unload_at_end=unload))
                    cursor = end
                attitude = target
        if self.ctx.uend > cursor:
            spans.append(_Span(cursor, self.ctx.uend, attitude))
        return spans

    def _attitude_at(self, span: _Span, utime: float) -> Attitude:
        if span.slew is not None:
            sampled = span.slew.attitude(utime)  # type: ignore[attr-defined]
            return (float(sampled[0]), float(sampled[1]), float(sampled[2]))
        if span.attitude is not None:
            return span.attitude
        if self.start_attitude is not None:
            return self.start_attitude
        return self.ctx.initial_attitude(utime)

    # ── Integration ───────────────────────────────────────────────────────

    def _interpolate(
        self, table: npt.NDArray[np.float64], utime: float
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [np.interp(utime, self._times, table[:, k]) for k in range(3)],
            dtype=np.float64,
        )

    def _sunlit(self, utime: float) -> bool:
        index = int(
            np.clip(np.searchsorted(self._times, utime), 0, len(self._times) - 1)
        )
        if index > 0 and abs(self._times[index - 1] - utime) < abs(
            self._times[index] - utime
        ):
            index -= 1
        return not self.config.constraint.in_eclipse(
            0.0, 0.0, float(self._times[index])
        )

    def profile(
        self, timeline: Sequence[_Block]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sample times and the stored momentum's norm at each, along
        ``timeline``, with each thruster dump unloading the wheels as it ends."""
        tracker = self.tracker
        if tracker is None:
            return np.empty(0), np.empty(0)
        if self.start_momentum_eci_n_m_s is not None:
            tracker.start_from_eci(self.start_momentum_eci_n_m_s)
        else:
            tracker.reset()
        return self._integrate(tracker, self._spans(timeline))

    def loading(self, block: _Block) -> float:
        """The most the block alone stores, from empty wheels: its slew and
        its hold, with nothing before or after it."""
        tracker = self.tracker
        if tracker is None:
            return 0.0
        begin = float(block.entry.begin)
        spans = [
            span
            for span in self._spans([block])
            if span.begin >= begin and span.end <= float(block.end)
        ]
        tracker.reset()
        _, norms = self._integrate(tracker, spans)
        return float(norms.max()) if len(norms) else 0.0

    def _integrate(
        self, tracker: StoredMomentumTracker, spans: Sequence[_Span]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Advance ``tracker`` through ``spans``: the sample times and the
        norm at each. A span integrated before from the same state is read
        from the cache, and the tracker continues from its end."""
        times: list[float] = []
        norms: list[float] = []
        for span in spans:
            key = (span.key, tracker.state)
            cached = self.cache.get(key)
            if cached is not None:
                times.extend(cached[0])
                norms.extend(cached[1])
                tracker.restore(cached[2])
                continue
            first = len(times)
            interval = (
                self.slew_sample_s if span.slew is not None else self.hold_sample_s
            )
            samples = np.arange(span.begin, span.end, interval)
            if len(samples) == 0 or samples[-1] < span.end:
                samples = np.append(samples, span.end)
            for utime in samples:
                position = self._interpolate(self._positions, float(utime))
                attitude = self._attitude_at(span, float(utime))
                sample = tracker.update(
                    utime=float(utime),
                    position_eci_km=position,
                    attitude_quaternion_eci_to_body=attitude_to_quat(*attitude),
                    sun_eci_km=self._interpolate(self._sun, float(utime)) - position,
                    sunlit=self._sunlit(float(utime)),
                    position_ecef_km=(
                        self._interpolate(self._ecef, float(utime))
                        if self._ecef is not None
                        else None
                    ),
                )
                times.append(float(utime))
                norms.append(sample.stored_momentum_norm_n_m_s)
            if span.unload_at_end is not None:
                tracker.unload(span.unload_at_end)
                times.append(span.end)
                norms.append(tracker.stored_momentum_norm_n_m_s)
            self.cache[key] = (
                np.array(times[first:]),
                np.array(norms[first:]),
                tracker.state,
            )
        return np.array(times), np.array(norms)

    def first_crossing(
        self, timeline: Sequence[_Block], level: float, after: float | None = None
    ) -> float | None:
        """When the stored momentum first reaches ``level`` along the
        timeline, after ``after`` if given; None if it never does."""
        times, norms = self.profile(timeline)
        for utime, norm in zip(times, norms, strict=True):
            if after is not None and utime <= after:
                continue
            if norm >= level:
                return float(utime)
        return None
