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
from ..common.ephemeris import position_vectors, velocity_vectors
from ..common.vector import attitude_to_quat
from ..ditl.plan_validator import entry_obstype
from ..simulation.attitude import Attitude, slew_angular_velocity
from ..simulation.momentum import (
    MomentumState,
    StoredMomentumTracker,
    slew_sample_interval_s,
    tracker_from_config,
)

if TYPE_CHECKING:
    from ..config import MissionConfig
    from ..simulation.slew import Slew
    from .context import SchedulingContext
    from .priority_scheduler import _Block


_SpanKey = tuple[float, float, Attitude | None, tuple[float, ...] | None, float | None]


@dataclass(frozen=True)
class _Span:
    """An interval of the timeline at one attitude or along one slew."""

    begin: float
    end: float
    attitude: Attitude | None
    """The attitude held, or None along a slew."""
    slew: Slew | None = None
    """The slew flown, with an ``attitude(utime)`` of its own."""
    unload_at_end: float | None = None
    """The residual fraction a thruster dump leaves when the span ends."""

    @property
    def key(self) -> _SpanKey:
        """What the momentum stored over the span depends on, besides the
        momentum it starts from: its times, its attitude or its slew's
        path and rate, and the unloading at its end."""
        slew = self.slew
        path = (
            (
                float(slew.slewstart),
                float(slew.slewend),
                float(slew.slewdist),
                float(slew.rate_fraction),
                float(slew.startra),
                float(slew.startdec),
                float(slew.startroll),
                float(slew.endra),
                float(slew.enddec),
                float(slew.endroll),
            )
            if slew is not None
            else None
        )
        return (self.begin, self.end, self.attitude, path, self.unload_at_end)


_Integrated = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    MomentumState,
]
"""A span integrated: its sample times, the norm at each, what the wheels
hold at each and the state at its end."""

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
        self.slew_momentum = bool(
            config.spacecraft_bus.attitude_control.stored_momentum.slew_momentum_enabled
        )
        """Whether the momentum the body carries along a slew counts against
        the capacity: what the wheels hold is then the stored momentum less it."""
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
        self._velocities = velocity_vectors(ephem)

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
            sampled = span.slew.attitude(utime)
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
        self, timeline: Sequence[_Block], reset_at: Sequence[float] = ()
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sample times and the stored momentum's norm at each, along
        ``timeline``, with each thruster dump unloading the wheels as it ends."""
        times, norms, _ = self._profile(timeline, reset_at)
        return times, norms

    def demand(
        self, timeline: Sequence[_Block], reset_at: Sequence[float] = ()
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sample times and what the wheels hold at each along
        ``timeline``: the stored momentum less the body's own along each
        slew where slew momentum counts, the stored momentum otherwise."""
        times, _, demands = self._profile(timeline, reset_at)
        return times, demands

    def _profile(
        self, timeline: Sequence[_Block], reset_at: Sequence[float] = ()
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        tracker = self.tracker
        if tracker is None:
            return np.empty(0), np.empty(0), np.empty(0)
        if self.start_momentum_eci_n_m_s is not None:
            tracker.start_from_eci(self.start_momentum_eci_n_m_s)
        else:
            tracker.reset()
        return self._integrate(tracker, self._spans(timeline), reset_at)

    def loading(self, block: _Block) -> float:
        """The most the block alone asks of the wheels, from empty: its slew
        and its hold, with nothing before or after it."""
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
        _, _, demands = self._integrate(tracker, spans)
        return float(demands.max()) if len(demands) else 0.0

    def _integrate(
        self,
        tracker: StoredMomentumTracker,
        spans: Sequence[_Span],
        reset_at: Sequence[float] = (),
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Advance ``tracker`` through ``spans``: the sample times, the norm
        at each and what the wheels hold at each. A span integrated before
        from the same state is read from the cache, and the tracker continues
        from its end."""
        times: list[float] = []
        norms: list[float] = []
        demands: list[float] = []
        dump = self.config.spacecraft_bus.attitude_control.stored_momentum.thruster_dump
        residual = dump.residual_fraction if dump is not None else 0.0
        pending = sorted(reset_at)
        for span in spans:
            while pending and span.begin >= pending[0]:
                tracker.unload(residual)
                pending.pop(0)
            # Resets inside the span apply at the first sample at or after
            # each; a span integrated with them is cached apart.
            inside = tuple(reset for reset in pending if reset < span.end)
            pending = pending[len(inside) :]
            key: tuple[object, ...] = (
                (span.key, tracker.state, inside)
                if inside
                else (span.key, tracker.state)
            )
            cached = self.cache.get(key)
            if cached is not None:
                times.extend(cached[0])
                norms.extend(cached[1])
                demands.extend(cached[2])
                tracker.restore(cached[3])
                continue
            first = len(times)
            interval = (
                self.slew_sample_s if span.slew is not None else self.hold_sample_s
            )
            samples = np.arange(span.begin, span.end, interval)
            if len(samples) == 0 or samples[-1] < span.end:
                samples = np.append(samples, span.end)
            rate_at = (
                slew_angular_velocity(span.slew)
                if span.slew is not None and self.slew_momentum
                else None
            )
            resets = list(inside)
            for utime in samples:
                while resets and float(utime) >= resets[0]:
                    tracker.unload(residual)
                    resets.pop(0)
                position = self._interpolate(self._positions, float(utime))
                attitude = self._attitude_at(span, float(utime))
                sample = tracker.update(
                    utime=float(utime),
                    position_eci_km=position,
                    attitude_quaternion_eci_to_body=attitude_to_quat(*attitude),
                    sun_eci_km=self._interpolate(self._sun, float(utime)) - position,
                    sunlit=self._sunlit(float(utime)),
                    velocity_eci_km_s=(
                        self._interpolate(self._velocities, float(utime))
                        if self._velocities is not None
                        else None
                    ),
                    position_ecef_km=(
                        self._interpolate(self._ecef, float(utime))
                        if self._ecef is not None
                        else None
                    ),
                    angular_velocity_body_rad_s=(
                        rate_at(float(utime)) if rate_at is not None else None
                    ),
                )
                times.append(float(utime))
                norms.append(sample.stored_momentum_norm_n_m_s)
                demands.append(
                    sample.actuator_momentum_norm_n_m_s
                    if sample.actuator_momentum_norm_n_m_s is not None
                    else sample.stored_momentum_norm_n_m_s
                )
            if span.unload_at_end is not None:
                tracker.unload(span.unload_at_end)
                times.append(span.end)
                norms.append(tracker.stored_momentum_norm_n_m_s)
                demands.append(tracker.stored_momentum_norm_n_m_s)
            self.cache[key] = (
                np.array(times[first:]),
                np.array(norms[first:]),
                np.array(demands[first:]),
                tracker.state,
            )
        return np.array(times), np.array(norms), np.array(demands)

    def first_crossing(
        self,
        timeline: Sequence[_Block],
        level: float,
        after: float | None = None,
        reset_at: Sequence[float] = (),
        *,
        demand: bool = False,
    ) -> float | None:
        """When the stored momentum, or with ``demand`` what the wheels hold,
        first reaches ``level`` along the timeline, after ``after`` if given;
        None if it never does."""
        times, norms, demands = self._profile(timeline, reset_at)
        for utime, norm in zip(times, demands if demand else norms, strict=True):
            if after is not None and utime <= after:
                continue
            if norm >= level:
                return float(utime)
        return None

    def first_slew_overload(
        self, timeline: Sequence[_Block], capacity: float
    ) -> tuple[float, Slew] | None:
        """The first slew along the timeline in which what the wheels hold
        reaches ``capacity``: when, and the slew; None where none does.
        Samples at a slew's ends carry no rate, so a reading there is the
        hold's, not the slew's."""
        if not self.slew_momentum:
            return None
        times, _, demands = self._profile(timeline)
        for span in self._spans(timeline):
            if span.slew is None:
                continue
            inside = (times > span.begin) & (times < span.end) & (demands >= capacity)
            if inside.any():
                return float(times[inside][0]), span.slew
        return None
