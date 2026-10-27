"""A moving pointing: its track, its position at a time and its visibility
along the track."""

from datetime import datetime, timedelta, timezone

import pytest
import rust_ephem
from pydantic import ValidationError
from rust_ephem import TLEEphemeris

from conops.config import Constraint, MissionConfig
from conops.targets import MovingPointing, Pointing, TrackPoint, track_from_table
from conops.targets.track import position_along, rate_along

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
T0 = BEGIN.timestamp()
HOUR = 3600.0


def _track(hours: int = 4, rate_deg_per_h: float = 1.0) -> list[TrackPoint]:
    return track_from_table(
        [T0 + k * HOUR for k in range(hours + 1)],
        [10.0 + k * rate_deg_per_h for k in range(hours + 1)],
        [5.0 + 0.1 * k for k in range(hours + 1)],
    )


class TestTrack:
    def test_built_in_time_order(self) -> None:
        track = track_from_table([T0 + HOUR, T0], [11.0, 10.0], [5.0, 5.0])
        assert [p.utime for p in track] == [T0, T0 + HOUR]
        assert [p.ra for p in track] == [10.0, 11.0]

    def test_accepts_datetimes_and_strings(self) -> None:
        track = track_from_table([BEGIN, BEGIN.isoformat()], [10.0, 10.5], [5.0, 5.0])
        assert track[0].utime == T0 and track[1].utime == T0

    def test_interpolates_and_holds_outside(self) -> None:
        track = _track()
        assert position_along(track, T0 + 1.5 * HOUR) == pytest.approx((11.5, 5.15))
        assert position_along(track, T0 - HOUR) == (10.0, 5.0)
        assert position_along(track, T0 + 10 * HOUR) == (14.0, 5.4)

    def test_crosses_the_right_ascension_seam_the_short_way(self) -> None:
        track = track_from_table([T0, T0 + HOUR], [359.0, 1.0], [0.0, 0.0])
        ra, _ = position_along(track, T0 + HOUR / 2)
        assert ra == pytest.approx(0.0, abs=1e-9)

    def test_positions_together_match_each_position(self) -> None:
        """Positions along a track at many times at once are the positions
        one by one, across the seam and outside the track."""
        from conops.targets.track import positions_along

        track = _track(hours=4, rate_deg_per_h=100.0)
        first = track[0].utime
        times = [
            first - 600.0,
            *[first + 300.0 * k for k in range(50)],
            first + 10 * 3600.0,
        ]
        ras, decs = positions_along(track, times)
        for utime, ra, dec in zip(times, ras, decs, strict=True):
            assert (float(ra), float(dec)) == pytest.approx(
                position_along(track, utime)
            )

    def test_rate(self) -> None:
        track = _track(rate_deg_per_h=2.0)
        assert rate_along(track, T0 + HOUR / 2) * HOUR == pytest.approx(2.0, abs=0.01)
        assert rate_along(track, T0 - 1.0) == 0.0
        assert rate_along(track[:1], T0) == 0.0


class TestMovingPointing:
    def test_needs_a_track(self) -> None:
        with pytest.raises(ValidationError, match="needs a track"):
            MovingPointing(obsid=1, name="m", track=[])

    def test_starts_at_the_tracks_first_position(self) -> None:
        mover = MovingPointing(obsid=1, name="m", track=_track())
        assert (mover.ra, mover.dec) == (10.0, 5.0)
        assert mover.moves
        assert mover.position_at(T0 + 2 * HOUR) == pytest.approx((12.0, 5.2))
        mover.point_at(T0 + 3 * HOUR)
        assert (mover.ra, mover.dec) == pytest.approx((13.0, 5.3))
        assert mover.target_body_attitude(20.0, at=T0 + HOUR) == pytest.approx(
            (11.0, 5.1, 20.0)
        )

    def test_a_fixed_pointing_does_not_move(self) -> None:
        fixed = Pointing(obsid=2, name="f", ra=10.0, dec=5.0)
        assert not fixed.moves
        assert fixed.position_at(T0 + HOUR) == (10.0, 5.0)
        fixed.point_at(T0 + HOUR)
        assert (fixed.ra, fixed.dec) == (10.0, 5.0)
        assert "track" not in fixed.model_dump()

    def test_track_serialises_and_rejects_disorder(self) -> None:
        mover = MovingPointing(obsid=1, name="m", track=_track(1))
        dumped = mover.model_dump()
        assert [p["ra"] for p in dumped["track"]] == [10.0, 11.0]
        with pytest.raises(ValidationError, match="increase strictly"):
            Pointing(
                obsid=3,
                name="d",
                track=[
                    TrackPoint(utime=T0, ra=1, dec=1),
                    TrackPoint(utime=T0, ra=2, dec=1),
                ],
            )

    def test_visit_pointings_carry_the_track(self) -> None:
        from conops.targets import visit_windows

        mover = MovingPointing(
            obsid=1,
            name="m",
            track=_track(),
            visit_windows=visit_windows([T0, T0 + 2 * HOUR], HOUR),
        )
        visit = mover.visit_pointing(1)
        assert visit.moves and visit.track == mover.track


@pytest.fixture(scope="module")
def ephem() -> TLEEphemeris:
    return TLEEphemeris(
        tle="examples/example.tle",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=6),
        step_size=60,
    )


class TestVisibility:
    def test_windows_follow_the_track(self, ephem: TLEEphemeris) -> None:
        """A track that walks into the Sun constraint loses its visibility
        where a fixed pointing at its start keeps it."""
        config = MissionConfig()
        config.constraint = Constraint(
            sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
        )
        config.constraint.ephem = ephem
        sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
        # From 60 degrees east of the Sun to 20 degrees east over six hours.
        track = track_from_table(
            [T0 + k * HOUR for k in range(7)],
            [(sun_ra + 60.0 - 40.0 * k / 6.0) % 360.0 for k in range(7)],
            [sun_dec] * 7,
        )
        mover = MovingPointing(config=config, obsid=5, name="m", track=track)
        mover.constraint, mover.ephem = config.constraint, ephem
        mover.visibility()
        assert mover.windows
        assert mover.windows[-1][1] < T0 + 6 * HOUR - HOUR
        fixed = Pointing(
            config=config, obsid=6, name="f", ra=track[0].ra, dec=track[0].dec
        )
        fixed.constraint, fixed.ephem = config.constraint, ephem
        fixed.visibility()
        assert fixed.windows[-1][1] == pytest.approx(T0 + 6 * HOUR)


class _BodyEphemeris:
    """A stand-in for the ephemeris's body lookup: a body drifting east."""

    def __init__(self, ephem: TLEEphemeris) -> None:
        self.timestamp = ephem.timestamp
        self.calls: list[tuple[str, str | None, bool]] = []

    def get_body(
        self, body: str, spice_kernel: str | None = None, use_horizons: bool = False
    ):
        from types import SimpleNamespace

        import numpy as np

        self.calls.append((body, spice_kernel, use_horizons))
        n = len(self.timestamp)
        return SimpleNamespace(
            ra=SimpleNamespace(deg=np.linspace(10.0, 11.0, n)),
            dec=SimpleNamespace(deg=np.full(n, 5.0)),
        )


class TestBodyLookup:
    def test_needs_a_track_or_a_body(self) -> None:
        with pytest.raises(ValidationError, match="track or a body"):
            MovingPointing(obsid=1, name="m")
        mover = MovingPointing(obsid=1, name="m", body="Jupiter")
        assert not mover.moves and mover.body == "Jupiter"

    def test_track_looked_up_at_first_visibility(
        self, ephem: TLEEphemeris, monkeypatch
    ) -> None:
        from conops.targets import track as track_module

        stand_in = _BodyEphemeris(ephem)
        monkeypatch.setattr(
            track_module,
            "body_track",
            lambda e, body, **kw: track_module.track_from_table(
                [t.timestamp() for t in e.timestamp],
                stand_in.get_body(body, **kw).ra.deg,
                stand_in.get_body(body, **kw).dec.deg,
            ),
        )
        import conops.targets.pointing as pointing_module

        monkeypatch.setattr(pointing_module, "body_track", track_module.body_track)
        config = MissionConfig()
        config.constraint = Constraint()
        config.constraint.ephem = ephem
        mover = MovingPointing(
            config=config,
            obsid=7,
            name="j",
            body="Jupiter",
            use_horizons=False,
            spice_kernel="k.bsp",
        )
        mover.constraint, mover.ephem = config.constraint, ephem
        assert not mover.moves
        mover.visibility()
        assert mover.moves and len(mover.track) == len(ephem.timestamp)
        assert stand_in.calls[0] == ("Jupiter", "k.bsp", False)
        assert (mover.ra, mover.dec) == (10.0, 5.0)
        assert mover.position_at(T0 + 3 * HOUR) == pytest.approx((10.5, 5.0))
        assert mover.windows
        # Looked up once: a second visibility computation keeps the track.
        calls = len(stand_in.calls)
        mover.visibility()
        assert len(stand_in.calls) == calls

    def test_body_track_reads_the_lookup(self, ephem: TLEEphemeris) -> None:
        from conops.targets.track import body_track

        stand_in = _BodyEphemeris(ephem)
        track = body_track(stand_in, "Jupiter")  # type: ignore[arg-type]
        assert len(track) == len(ephem.timestamp)
        assert track[0].utime == T0 and track[0].ra == 10.0
        assert stand_in.calls == [("Jupiter", None, True)]

    @pytest.mark.skipif(
        not __import__("os").environ.get("COAST_NETWORK_TESTS"),
        reason="needs network access to JPL Horizons; set COAST_NETWORK_TESTS=1",
    )
    def test_horizons_lookup(self, ephem: TLEEphemeris) -> None:
        from conops.targets.track import body_track

        track = body_track(ephem, "Jupiter")
        assert len(track) == len(ephem.timestamp)
        assert all(0.0 <= p.ra < 360.0 for p in track)


class TestRefresh:
    def test_the_lookup_is_taken_and_the_given_track_stands_in_on_failure(
        self, ephem: TLEEphemeris
    ) -> None:
        from unittest.mock import patch

        from conops.targets import Program

        config = MissionConfig()
        config.constraint = Constraint()
        config.constraint.ephem = ephem
        given = _track()
        mover = MovingPointing(
            config=config, obsid=5, name="body", body="12345", track=given
        )
        looked_up = _track(rate_deg_per_h=2.0)
        with patch(
            "conops.targets.pointing.body_track", return_value=looked_up
        ) as lookup:
            assert mover.refresh_track() == "looked up"
            assert lookup.call_count == 1
        assert mover.track == looked_up
        assert (mover.ra, mover.dec) == (looked_up[0].ra, looked_up[0].dec)

        with patch(
            "conops.targets.pointing.body_track", side_effect=RuntimeError("down")
        ):
            assert mover.refresh_track() == "given"
        assert mover.track == looked_up

        bare = MovingPointing(config=config, obsid=6, name="bare", body="12345")
        with patch(
            "conops.targets.pointing.body_track", side_effect=RuntimeError("down")
        ):
            with pytest.raises(RuntimeError):
                bare.refresh_track()

        program = Program(requests=[mover], begin=BEGIN, end=BEGIN + timedelta(hours=2))
        with patch("conops.targets.pointing.body_track", return_value=given) as lookup:
            program.bind(config, ephem)
            assert lookup.call_count == 0
            program.bind(config, ephem, refresh_tracks=True)
            assert lookup.call_count == 1
        assert mover.track == given
