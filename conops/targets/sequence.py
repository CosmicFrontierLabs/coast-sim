"""Observation sequences: ordered steps observed as one visit.

Some observations are not one pointing held for a time but a fixed series of
steps: a reference star, then the target, then the reference again, each for
its own duration and perhaps with its own instrument, with calibration steps
that take time but collect no science. An :class:`ObservationSequence` on a
:class:`~conops.targets.Pointing` describes the steps; a visit runs them all
in order, with real slews between different pointings and no setup between
steps unless a step asks for one, and nothing is inserted inside a visit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..config.acs import AttitudeControlSystem
from ..config.observation_timing import ObservationTiming


class SequenceStep(BaseModel):
    """One step of a sequence."""

    model_config = ConfigDict(frozen=True)

    name: str
    duration_seconds: float = Field(gt=0.0)
    ra: float | None = Field(
        default=None, description="Pointing of the step; None: the target's own"
    )
    dec: float | None = None
    instrument_name: str | None = Field(
        default=None, description="Instrument observing the step; None: the target's"
    )
    science: bool = Field(
        default=True,
        description=(
            "Whether the step collects science. A calibration step takes its "
            "time, which is counted separately, and collects none."
        ),
    )
    setup: bool = Field(
        default=False,
        description=(
            "Whether the payload's setup budget is paid before this step. The "
            "first step of a visit always pays it; later steps by default do not."
        ),
    )


class ObservationSequence(BaseModel):
    """An ordered list of steps observed as one visit."""

    model_config = ConfigDict(frozen=True)

    steps: list[SequenceStep] = Field(min_length=1)

    @property
    def science_seconds(self) -> float:
        """Seconds of science a visit collects."""
        return sum(step.duration_seconds for step in self.steps if step.science)

    @property
    def calibration_seconds(self) -> float:
        """Seconds of calibration a visit takes."""
        return sum(step.duration_seconds for step in self.steps if not step.science)

    def pointing_of(self, index: int, ra: float, dec: float) -> tuple[float, float]:
        """Where step ``index`` points, given the target's own pointing."""
        step = self.steps[index]
        return (
            float(step.ra) if step.ra is not None else float(ra),
            float(step.dec) if step.dec is not None else float(dec),
        )

    def step_timing(self, base: ObservationTiming, index: int) -> ObservationTiming:
        """The timing budgets step ``index`` pays: setup for the first step and
        for a step that asks for it, cleanup and handoff for the last step."""
        first = index == 0
        last = index == len(self.steps) - 1
        return ObservationTiming(
            setup_seconds=base.setup_seconds
            if first or self.steps[index].setup
            else 0.0,
            cleanup_seconds=base.cleanup_seconds if last else 0.0,
            handoff_seconds=base.handoff_seconds if last else 0.0,
        )

    def overhead_seconds(
        self,
        acs: AttitudeControlSystem,
        timing: ObservationTiming,
        ra: float,
        dec: float,
    ) -> float:
        """Seconds a visit takes beyond its science and the visit's own setup
        and cleanup: calibration steps, the slews between steps that point
        elsewhere, and the setup of steps that ask for it."""
        seconds = self.calibration_seconds
        previous = self.pointing_of(0, ra, dec)
        for index, step in enumerate(self.steps):
            if index > 0:
                current = self.pointing_of(index, ra, dec)
                if current != previous:
                    seconds += float(acs.slew_time(_separation_deg(previous, current)))
                    previous = current
                if step.setup:
                    seconds += timing.setup_seconds
        return seconds


def sequence_steps(sequence: ObservationSequence | None) -> Sequence[SequenceStep]:
    return sequence.steps if sequence is not None else ()


def _separation_deg(one: tuple[float, float], two: tuple[float, float]) -> float:
    """The angle between two (RA, Dec) pointings in degrees."""
    ra1, dec1 = np.deg2rad(one)
    ra2, dec2 = np.deg2rad(two)
    cosine = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        ra1 - ra2
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
