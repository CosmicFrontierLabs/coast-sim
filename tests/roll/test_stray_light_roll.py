"""Stray light depends on the roll: it steers the roll optimiser and is left out
of the roll-free field of regard."""

from datetime import timedelta
from typing import Any, cast

import pytest
import rust_ephem
from rust_ephem import TLEEphemeris

from conops.config import Constraint, Telescope
from conops.schedulers import SchedulingContext
from conops.simulation.roll import optimum_body_roll
from conops.targets import Pointing
from tests.scheduler.planning_scenario import BEGIN as SCENARIO_BEGIN
from tests.scheduler.planning_scenario import make_config, make_target

from .test_science_roll_filter import UTIME, _constraint, _side_panel, _sun_north

needs_stray_light = pytest.mark.skipif(
    not hasattr(rust_ephem, "StrayLightConstraint"),
    reason="this rust_ephem has no StrayLightConstraint",
)

CONVERSION = 3.0e-9
OFF_AXIS = [0.0, 5.0, 40.0, 180.0]
AZIMUTH = [float(a) for a in range(0, 360, 10)]


def _stray_light(leaky_azimuth: float | None = 0.0, dark: float = -18.0) -> Any:
    """A telescope that leaks within 45 degrees of one azimuth (none: all round)."""
    row = [
        -2.0
        if leaky_azimuth is None
        or abs((a - leaky_azimuth + 180.0) % 360.0 - 180.0) <= 45.0
        else dark
        for a in AZIMUTH
    ]
    return cast(Any, rust_ephem).StrayLightConstraint(
        pst=cast(Any, rust_ephem).PointSourceTransmittance(
            off_axis_deg=OFF_AXIS,
            azimuth_deg=AZIMUTH,
            log10_pst=[row, row, row, [-18.0] * len(AZIMUTH)],
            azimuth_symmetric=False,
        ),
        conversion_factor=CONVERSION,
        earth_albedo=1.0e-12,  # the Sun alone, so the geometry is one body's
        moon_v_mag_full=60.0,
    )


class _StrayLightNode:
    """Stands in for the class where rust_ephem does not have it."""

    type = "stray_light"

    def __init__(self, blocked: Any = lambda roll: False) -> None:
        self.blocked = blocked

    def __or__(self, other: Any) -> Any:  # pragma: no cover
        raise AssertionError("stray light must not join the roll-free tree")

    def in_constraint(self, *, target_roll: float, **_: Any) -> bool:
        return bool(self.blocked(target_roll))


def _with_node(ephem: TLEEphemeris, node: _StrayLightNode) -> Constraint:
    constraint = _constraint(ephem)
    constraint.science_constraint = node  # type: ignore[assignment]
    return constraint


class TestAnyRustEphem:
    def test_roll_dependent(self, example_ephem: TLEEphemeris) -> None:
        constraint = _with_node(example_ephem, _StrayLightNode())
        assert constraint.science_constraint_roll_dependent
        assert constraint.roll_independent_constraint is None
        assert constraint.science_line_of_sight_constraint is None

    def test_the_other_rules_stay_in_the_field_of_regard(
        self, example_ephem: TLEEphemeris
    ) -> None:
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        constraint = Constraint(sun_constraint=sun)
        constraint.science_constraint = _StrayLightNode()  # type: ignore[assignment]
        assert constraint.roll_independent_constraint is sun
        assert constraint.science_line_of_sight_constraint is sun

    def test_best_clear_roll(self, example_ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(example_ephem)
        node = _StrayLightNode(lambda roll: 60.0 <= roll <= 110.0)
        roll = optimum_body_roll(
            ra,
            dec,
            UTIME,
            example_ephem,
            _side_panel(),
            _with_node(example_ephem, node),
        )
        # Power is best at 90 degrees; 111 is nearer it than 59.
        assert roll == pytest.approx(111.0)

    def test_no_clear_roll(self, example_ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(example_ephem)
        node = _StrayLightNode(lambda roll: True)
        roll = optimum_body_roll(
            ra,
            dec,
            UTIME,
            example_ephem,
            _side_panel(),
            _with_node(example_ephem, node),
        )
        assert roll == pytest.approx(90.0)

    def test_single_reachable_roll(self, example_ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(example_ephem)
        node = _StrayLightNode(lambda roll: True)
        roll = optimum_body_roll(
            ra,
            dec,
            UTIME,
            example_ephem,
            _side_panel(),
            _with_node(example_ephem, node),
            reference_roll=30.0,
            max_roll_delta=0.0,
        )
        assert roll == pytest.approx(30.0)


@needs_stray_light
class TestStrayLight:
    @pytest.mark.parametrize("leaky", [0.0, 90.0, 200.0])
    def test_steers_to_a_clear_roll(
        self, example_ephem: TLEEphemeris, leaky: float
    ) -> None:
        ra, dec = _sun_north(example_ephem)
        constraint = _constraint(example_ephem, _stray_light(leaky))
        clear = [
            not constraint.in_explicit_science(ra, dec, UTIME, target_roll=float(r))
            for r in range(360)
        ]
        assert any(clear) and not all(clear)
        roll = optimum_body_roll(
            ra, dec, UTIME, example_ephem, _side_panel(), constraint
        )
        assert clear[int(roll)]
        offsets = [abs((r - 90.0 + 180.0) % 360.0 - 180.0) for r in range(360)]
        best = min(o for o, ok in zip(offsets, clear, strict=True) if ok)
        assert abs(offsets[int(roll)] - best) <= 1.0

    def test_no_clear_roll(self, example_ephem: TLEEphemeris) -> None:
        ra, dec = _sun_north(example_ephem)
        constraint = _constraint(example_ephem, _stray_light(None))
        roll = optimum_body_roll(
            ra, dec, UTIME, example_ephem, _side_panel(), constraint
        )
        assert roll == pytest.approx(90.0)

    def test_roll_free_part(self) -> None:
        sun = rust_ephem.SunConstraint(min_angle=45.0)
        moon = rust_ephem.MoonConstraint(min_angle=10.0)
        stray = _stray_light()
        free = Constraint._roll_free_part
        assert free(sun | moon) == sun | moon
        assert free(sun | stray) == sun
        assert free(sun | (stray & moon)) == sun
        assert free(stray) is None
        assert free(stray & sun) is None
        assert free(~stray) is None
        assert free(sun.boresight_offset(yaw_deg=30.0) | stray) is None

    def test_windows_keep_a_target_some_roll_can_observe(
        self, example_ephem: TLEEphemeris
    ) -> None:
        ra, dec = _sun_north(example_ephem)
        stray = _stray_light(0.0)
        sun = rust_ephem.SunConstraint(min_angle=5.0)
        constraint = _constraint(example_ephem, sun | stray)
        roll = optimum_body_roll(
            ra, dec, UTIME, example_ephem, _side_panel(), constraint
        )
        assert not constraint.in_explicit_science(ra, dec, UTIME, target_roll=roll)
        first = example_ephem.timestamp[0]
        # With no roll stray light is judged at its nominal attitude alone.
        assert stray.evaluate(example_ephem, ra, dec, times=[first]).constraint_array[0]
        tree = constraint.roll_independent_constraint
        assert tree is not None
        assert not tree.evaluate(
            example_ephem, ra, dec, times=[first]
        ).constraint_array[0]
        # The Sun rule still applies.
        assert tree.evaluate(
            example_ephem, ra, dec + 18.0, times=[first]
        ).constraint_array[0]

    def test_a_targets_own_ratio_decides_its_roll(self) -> None:
        """Two targets at one position, under one mission stray-light
        constraint. The mission's limit blocks the power-optimal roll, so a
        target that keeps it is steered to a clear roll; a target given a
        looser limit of its own has nothing to avoid and takes the
        power-optimal roll."""
        config = make_config(2, open_sky=True)
        config.solar_panel = _side_panel()
        config.constraint.science_constraint = _stray_light(0.0)
        ephem = config.constraint.ephem
        assert ephem is not None
        ra, dec = _sun_north(ephem)
        utime = SCENARIO_BEGIN.timestamp()
        context = SchedulingContext(
            config, SCENARIO_BEGIN, SCENARIO_BEGIN + timedelta(hours=2)
        )
        held = make_target(config, 1, ra, dec)
        loosened = make_target(config, 2, ra, dec)
        loosened.stray_light_max_ratio = 1.0e30

        def blocked(target: Pointing, roll: float) -> bool:
            constraint = target.observing_constraint()
            assert constraint is not None
            return constraint.in_explicit_science(ra, dec, utime, target_roll=roll)

        assert blocked(held, 90.0) and not blocked(loosened, 90.0)
        steered = context.optimum_roll(held, utime)
        assert steered != pytest.approx(90.0)
        assert not blocked(held, steered)
        assert context.optimum_roll(loosened, utime) == pytest.approx(90.0)
        # The mission's own limit is as it was.
        assert config.constraint.science_constraint.max_ratio == 3.0

    def test_not_telescope_local(self) -> None:
        with pytest.raises(ValueError, match="roll-independent"):
            Telescope(constraint=Constraint(science_constraint=_stray_light()))
