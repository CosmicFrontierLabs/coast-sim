"""Unit tests for telemetry.py."""

import math
from datetime import datetime, timezone

import pytest

from conops.common.enums import ACSMode
from conops.common.vector import attitude_to_quat, quat_to_attitude
from conops.ditl.ditl import DITL
from conops.ditl.telemetry import (
    Housekeeping,
    HousekeepingList,
    PayloadData,
    Telemetry,
)


class TestHousekeeping:
    """Test Housekeeping class."""

    def test_housekeeping_creation_with_none_values(self) -> None:
        """Test creating Housekeeping with None values."""
        from datetime import datetime, timezone

        timestamp = datetime.fromtimestamp(1000.0, tz=timezone.utc)
        hk = Housekeeping(timestamp=timestamp)
        assert hk.timestamp == timestamp
        assert hk.ra is None
        assert hk.dec is None
        assert hk.roll == 0.0  # Has default
        assert hk.acs_mode is None
        assert hk.for_solid_angle_sr is None
        assert hk.earth_angle_deg is None
        assert hk.moon_angle_deg is None
        assert hk.in_constraint is None

    def test_housekeeping_creation_with_values(self) -> None:
        """Test creating Housekeeping with specific values."""
        from datetime import datetime, timezone

        expected_dt = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        hk = Housekeeping(
            timestamp=expected_dt,
            ra=45.0,
            dec=30.0,
            roll=10.0,
            acs_mode=ACSMode.SCIENCE,
            panel_illumination=0.8,
            power_usage=100.0,
            battery_level=0.9,
            obsid=42,
        )
        assert hk.timestamp == expected_dt
        assert hk.ra == 45.0
        assert hk.dec == 30.0
        assert hk.roll == 10.0
        assert hk.acs_mode == ACSMode.SCIENCE
        assert hk.panel_illumination == 0.8
        assert hk.power_usage == 100.0
        assert hk.battery_level == 0.9
        assert hk.obsid == 42

    def test_extract_field_empty_list(self) -> None:
        """Test extract_field with empty list."""
        result = Housekeeping.extract_field([], "ra")
        assert result == []

    def test_extract_field_single_record(self) -> None:
        """Test extract_field with single record."""
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=45.0, dec=30.0
        )
        result = Housekeeping.extract_field([hk], "ra")
        assert result == [45.0]

    def test_extract_field_multiple_records(self) -> None:
        """Test extract_field with multiple records."""
        hk1 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=45.0, dec=30.0
        )
        hk2 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=90.0, dec=60.0
        )
        hk3 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=None, dec=45.0
        )
        result = Housekeeping.extract_field([hk1, hk2, hk3], "ra")
        assert result == [45.0, 90.0, None]

    def test_extract_field_invalid_attribute(self) -> None:
        """Test extract_field with invalid attribute."""
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=45.0
        )
        with pytest.raises(AttributeError):
            Housekeeping.extract_field([hk], "invalid_field")

    def test_extract_fields_empty_list(self) -> None:
        """Test extract_fields with empty list."""
        result = Housekeeping.extract_fields([], ["ra", "dec"])
        assert result == {"ra": [], "dec": []}

    def test_extract_fields_multiple_records(self) -> None:
        """Test extract_fields with multiple records."""
        hk1 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=45.0,
            dec=30.0,
            roll=10.0,
        )
        hk2 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=90.0,
            dec=60.0,
            roll=20.0,
        )
        hk3 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=None,
            dec=45.0,
            roll=None,
        )
        result = Housekeeping.extract_fields([hk1, hk2, hk3], ["ra", "dec", "roll"])
        expected = {
            "ra": [45.0, 90.0, None],
            "dec": [30.0, 60.0, 45.0],
            "roll": [10.0, 20.0, None],
        }
        assert result == expected

    def test_extract_fields_invalid_attribute(self) -> None:
        """Test extract_fields with invalid attribute."""
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=45.0
        )
        with pytest.raises(AttributeError):
            Housekeeping.extract_fields([hk], ["ra", "invalid_field"])


class TestPayloadData:
    """Test PayloadData class."""

    def test_payload_data_creation(self) -> None:
        """Test creating PayloadData."""
        from datetime import datetime, timezone

        expected_dt = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        pd = PayloadData(timestamp=expected_dt, data_size_gb=1.5)
        assert pd.timestamp == expected_dt
        assert pd.data_size_gb == 1.5


class TestTelemetry:
    """Test Telemetry class."""

    def test_init_empty(self) -> None:
        """Test initialization with no data."""
        tm = Telemetry()
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, list)
        assert len(tm.housekeeping) == 0
        assert len(tm.data) == 0

    def test_init_with_data(self) -> None:
        """Test initialization with data."""
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=45.0
        )
        pd = PayloadData(
            timestamp=datetime.fromtimestamp(1234567890.0, tz=timezone.utc),
            data_size_gb=1.5,
        )
        tm = Telemetry(housekeeping=HousekeepingList([hk]), data=[pd])
        assert len(tm.housekeeping) == 1
        assert len(tm.data) == 1
        assert tm.housekeeping[0].ra == 45.0
        assert tm.data[0].data_size_gb == 1.5

    def test_list_assignment(self) -> None:
        """Test that lists can be assigned."""
        tm = Telemetry()
        # Initially should be empty
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, list)

        # Setting new lists
        tm.housekeeping = HousekeepingList(
            [
                Housekeeping(
                    timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc), ra=45.0
                )
            ]
        )
        tm.data = [
            PayloadData(
                timestamp=datetime.fromtimestamp(1234567890.0, tz=timezone.utc),
                data_size_gb=1.5,
            )
        ]
        assert isinstance(tm.housekeeping, HousekeepingList)
        assert isinstance(tm.data, list)
        assert tm.housekeeping.ra == [45.0]
        # data is a regular list, so no attribute access
        assert tm.data[0].data_size_gb == 1.5

    def test_attribute_access_on_housekeeping(self) -> None:
        """Test attribute access on housekeeping list."""
        hk1 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=45.0,
            dec=30.0,
            acs_mode=ACSMode.SCIENCE,
            sun_angle_deg=10.0,
            for_solid_angle_sr=3.1,
            in_eclipse=False,
            radiator_sun_exposure=0.2,
            radiator_earth_exposure=0.4,
            radiator_heat_dissipation_w=120.0,
        )
        hk2 = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=90.0,
            dec=60.0,
            acs_mode=ACSMode.SAFE,
            sun_angle_deg=20.0,
            for_solid_angle_sr=2.8,
            in_eclipse=True,
            radiator_sun_exposure=0.1,
            radiator_earth_exposure=0.3,
            radiator_heat_dissipation_w=150.0,
        )
        tm = Telemetry(housekeeping=HousekeepingList([hk1, hk2]))

        assert tm.housekeeping.ra == [45.0, 90.0]
        assert tm.housekeeping.dec == [30.0, 60.0]
        assert tm.housekeeping.acs_mode == [ACSMode.SCIENCE, ACSMode.SAFE]
        assert tm.housekeeping.sun_angle_deg == [10.0, 20.0]
        assert tm.housekeeping.for_solid_angle_sr == [3.1, 2.8]
        assert tm.housekeeping.in_eclipse == [False, True]
        assert tm.housekeeping.radiator_sun_exposure == [0.2, 0.1]
        assert tm.housekeeping.radiator_earth_exposure == [0.4, 0.3]
        assert tm.housekeeping.radiator_heat_dissipation_w == [120.0, 150.0]


class TestHousekeepingNewFields:
    """Test earth_angle_deg, moon_angle_deg, and in_constraint fields."""

    def _make_ts(self, offset: float = 0.0) -> datetime:
        return datetime.fromtimestamp(1000.0 + offset, tz=timezone.utc)

    def test_new_fields_default_to_none(self) -> None:
        """New fields are None when not supplied."""
        hk = Housekeeping(timestamp=self._make_ts())
        assert hk.earth_angle_deg is None
        assert hk.moon_angle_deg is None
        assert hk.in_constraint is None

    def test_new_fields_round_trip(self) -> None:
        """New fields survive a full construction → access round-trip."""
        hk = Housekeeping(
            timestamp=self._make_ts(),
            earth_angle_deg=25.5,
            moon_angle_deg=42.0,
            in_constraint="Earth",
        )
        assert hk.earth_angle_deg == 25.5
        assert hk.moon_angle_deg == 42.0
        assert hk.in_constraint == "Earth"

    def test_extract_field_earth_angle(self) -> None:
        """extract_field works for earth_angle_deg."""
        hk1 = Housekeeping(timestamp=self._make_ts(0), earth_angle_deg=10.0)
        hk2 = Housekeeping(timestamp=self._make_ts(1), earth_angle_deg=None)
        hk3 = Housekeeping(timestamp=self._make_ts(2), earth_angle_deg=30.0)
        assert Housekeeping.extract_field([hk1, hk2, hk3], "earth_angle_deg") == [
            10.0,
            None,
            30.0,
        ]

    def test_extract_field_moon_angle(self) -> None:
        """extract_field works for moon_angle_deg."""
        hk1 = Housekeeping(timestamp=self._make_ts(0), moon_angle_deg=5.0)
        hk2 = Housekeeping(timestamp=self._make_ts(1), moon_angle_deg=15.0)
        assert Housekeeping.extract_field([hk1, hk2], "moon_angle_deg") == [5.0, 15.0]

    def test_extract_field_in_constraint(self) -> None:
        """extract_field works for in_constraint."""
        hk1 = Housekeeping(timestamp=self._make_ts(0), in_constraint="Sun")
        hk2 = Housekeeping(timestamp=self._make_ts(1), in_constraint=None)
        hk3 = Housekeeping(timestamp=self._make_ts(2), in_constraint="Moon")
        assert Housekeeping.extract_field([hk1, hk2, hk3], "in_constraint") == [
            "Sun",
            None,
            "Moon",
        ]

    def test_housekeeping_list_earth_angle_property(self) -> None:
        """HousekeepingList.earth_angle_deg returns values from all records."""
        hk1 = Housekeeping(timestamp=self._make_ts(0), earth_angle_deg=10.0)
        hk2 = Housekeeping(timestamp=self._make_ts(1), earth_angle_deg=None)
        hkl = HousekeepingList([hk1, hk2])
        assert hkl.earth_angle_deg == [10.0, None]

    def test_housekeeping_list_moon_angle_property(self) -> None:
        """HousekeepingList.moon_angle_deg returns values from all records."""
        hk1 = Housekeeping(timestamp=self._make_ts(0), moon_angle_deg=55.0)
        hk2 = Housekeeping(timestamp=self._make_ts(1), moon_angle_deg=22.3)
        hkl = HousekeepingList([hk1, hk2])
        assert hkl.moon_angle_deg == [55.0, 22.3]

    def test_housekeeping_list_in_constraint_property(self) -> None:
        """HousekeepingList.in_constraint returns values from all records."""
        hk1 = Housekeeping(timestamp=self._make_ts(0), in_constraint="Sun")
        hk2 = Housekeeping(timestamp=self._make_ts(1), in_constraint=None)
        hk3 = Housekeeping(timestamp=self._make_ts(2), in_constraint="Earth")
        hkl = HousekeepingList([hk1, hk2, hk3])
        assert hkl.in_constraint == ["Sun", None, "Earth"]

    def test_telemetry_attribute_access_new_fields(self) -> None:
        """New fields are accessible via Telemetry.housekeeping properties."""
        hk1 = Housekeeping(
            timestamp=self._make_ts(0),
            earth_angle_deg=12.0,
            moon_angle_deg=45.0,
            in_constraint=None,
        )
        hk2 = Housekeeping(
            timestamp=self._make_ts(1),
            earth_angle_deg=None,
            moon_angle_deg=55.0,
            in_constraint="Moon",
        )
        tm = Telemetry(housekeeping=HousekeepingList([hk1, hk2]))
        assert tm.housekeeping.earth_angle_deg == [12.0, None]
        assert tm.housekeeping.moon_angle_deg == [45.0, 55.0]
        assert tm.housekeeping.in_constraint == [None, "Moon"]

    def test_attribute_access_on_data(self) -> None:
        """Test attribute access on data list."""
        from datetime import datetime, timezone

        dt1 = datetime.fromtimestamp(1234567890.0, tz=timezone.utc)
        dt2 = datetime.fromtimestamp(1234567891.0, tz=timezone.utc)
        pd1 = PayloadData(timestamp=dt1, data_size_gb=1.5)
        pd2 = PayloadData(timestamp=dt2, data_size_gb=2.0)
        tm = Telemetry(data=[pd1, pd2])

        # data is a regular list, so access attributes directly
        assert [pd.timestamp for pd in tm.data] == [dt1, dt2]
        assert [pd.data_size_gb for pd in tm.data] == [1.5, 2.0]

    def test_model_config(self) -> None:
        """Test that model config is properly set."""
        tm = Telemetry()
        assert tm.model_config["arbitrary_types_allowed"] is True


class TestRollOffsetDeg:
    """Tests for the roll_offset_deg housekeeping field."""

    def _ts(self) -> datetime:
        return datetime.fromtimestamp(1000.0, tz=timezone.utc)

    def test_roll_offset_deg_defaults_to_none(self) -> None:
        hk = Housekeeping(timestamp=self._ts())
        assert hk.roll_offset_deg is None

    def test_roll_offset_deg_round_trip(self) -> None:
        hk = Housekeeping(timestamp=self._ts(), roll_offset_deg=15.0)
        assert hk.roll_offset_deg == pytest.approx(15.0)

    def test_roll_offset_deg_negative(self) -> None:
        hk = Housekeeping(timestamp=self._ts(), roll_offset_deg=-90.0)
        assert hk.roll_offset_deg == pytest.approx(-90.0)

    def test_roll_offset_deg_boundary_values(self) -> None:
        # The wrap formula produces values in [-180, 180)
        hk_lo = Housekeeping(timestamp=self._ts(), roll_offset_deg=-180.0)
        hk_hi = Housekeeping(timestamp=self._ts(), roll_offset_deg=179.9)
        assert hk_lo.roll_offset_deg == pytest.approx(-180.0)
        assert hk_hi.roll_offset_deg == pytest.approx(179.9)

    def test_housekeeping_list_roll_offset_deg_property(self) -> None:
        hk1 = Housekeeping(timestamp=self._ts(), roll_offset_deg=10.0)
        hk2 = Housekeeping(timestamp=self._ts(), roll_offset_deg=None)
        hk3 = Housekeeping(timestamp=self._ts(), roll_offset_deg=-45.0)
        hkl = HousekeepingList([hk1, hk2, hk3])
        assert hkl.roll_offset_deg == [10.0, None, -45.0]

    def test_roll_offset_wrap_formula_zero(self) -> None:
        # When actual roll == nominal roll the offset is 0
        roll, nominal = 45.0, 45.0
        offset = (roll - nominal + 180.0) % 360.0 - 180.0
        assert offset == pytest.approx(0.0)
        hk = Housekeeping(timestamp=self._ts(), roll_offset_deg=offset)
        assert hk.roll_offset_deg == pytest.approx(0.0)

    def test_roll_offset_wrap_formula_wraps_to_negative(self) -> None:
        # roll=350, nominal=10 → difference is -20 (not +340)
        roll, nominal = 350.0, 10.0
        offset = (roll - nominal + 180.0) % 360.0 - 180.0
        assert offset == pytest.approx(-20.0)
        hk = Housekeeping(timestamp=self._ts(), roll_offset_deg=offset)
        assert hk.roll_offset_deg == pytest.approx(-20.0)

    def test_roll_offset_wrap_formula_wraps_to_positive(self) -> None:
        # roll=10, nominal=350 → difference is +20 (not -340)
        roll, nominal = 10.0, 350.0
        offset = (roll - nominal + 180.0) % 360.0 - 180.0
        assert offset == pytest.approx(20.0)

    def test_extract_field_roll_offset_deg(self) -> None:
        hk1 = Housekeeping(timestamp=self._ts(), roll_offset_deg=5.0)
        hk2 = Housekeeping(timestamp=self._ts(), roll_offset_deg=-15.0)
        result = Housekeeping.extract_field([hk1, hk2], "roll_offset_deg")
        assert result == pytest.approx([5.0, -15.0])

    def test_telemetry_roll_offset_deg_accessible(self) -> None:
        hk = Housekeeping(timestamp=self._ts(), roll_offset_deg=30.0)
        tm = Telemetry(housekeeping=HousekeepingList([hk]))
        assert tm.housekeeping.roll_offset_deg == [pytest.approx(30.0)]


class TestBodyVectorFields:
    """Tests for sun_body_vector and earth_body_vector fields added in PR #107."""

    def _ts(self) -> datetime:
        return datetime.fromtimestamp(1000.0, tz=timezone.utc)

    def test_body_vectors_default_to_none(self) -> None:
        hk = Housekeeping(timestamp=self._ts())
        assert hk.sun_body_vector is None
        assert hk.earth_body_vector is None

    def test_body_vectors_round_trip(self) -> None:
        sbv = [0.707, 0.0, 0.707]
        ebv = [0.0, -1.0, 0.0]
        hk = Housekeeping(
            timestamp=self._ts(),
            sun_body_vector=sbv,
            earth_body_vector=ebv,
        )
        assert hk.sun_body_vector == pytest.approx(sbv)
        assert hk.earth_body_vector == pytest.approx(ebv)

    def test_housekeeping_list_sun_body_vector_property(self) -> None:
        sbv1 = [1.0, 0.0, 0.0]
        sbv2 = [0.0, 1.0, 0.0]
        hk1 = Housekeeping(timestamp=self._ts(), sun_body_vector=sbv1)
        hk2 = Housekeeping(timestamp=self._ts(), sun_body_vector=sbv2)
        hkl = HousekeepingList([hk1, hk2])
        assert hkl.sun_body_vector == [sbv1, sbv2]

    def test_housekeeping_list_earth_body_vector_property(self) -> None:
        ebv1 = [0.0, 0.0, -1.0]
        ebv2 = None
        hk1 = Housekeeping(timestamp=self._ts(), earth_body_vector=ebv1)
        hk2 = Housekeeping(timestamp=self._ts(), earth_body_vector=ebv2)
        hkl = HousekeepingList([hk1, hk2])
        assert hkl.earth_body_vector == [ebv1, None]

    def test_extract_field_sun_body_vector(self) -> None:
        sbv = [0.5, 0.5, 0.707]
        hk = Housekeeping(timestamp=self._ts(), sun_body_vector=sbv)
        result = Housekeeping.extract_field([hk], "sun_body_vector")
        assert result == [sbv]

    def test_telemetry_body_vectors_accessible(self) -> None:
        sbv = [1.0, 0.0, 0.0]
        ebv = [-1.0, 0.0, 0.0]
        hk = Housekeeping(
            timestamp=self._ts(),
            sun_body_vector=sbv,
            earth_body_vector=ebv,
        )
        tm = Telemetry(housekeeping=HousekeepingList([hk]))
        assert tm.housekeeping.sun_body_vector == [sbv]
        assert tm.housekeeping.earth_body_vector == [ebv]


class TestQuaternionFields:
    """Tests for quat_w/x/y/z attitude quaternion fields."""

    def _ts(self, offset: float = 0.0) -> datetime:
        return datetime.fromtimestamp(1000.0 + offset, tz=timezone.utc)

    # ------------------------------------------------------------------
    # Field defaults
    # ------------------------------------------------------------------

    def test_quaternion_fields_default_to_none(self) -> None:
        hk = Housekeeping(timestamp=self._ts())
        assert hk.quat_w is None
        assert hk.quat_x is None
        assert hk.quat_y is None
        assert hk.quat_z is None

    # ------------------------------------------------------------------
    # Round-trip: store known values, read them back
    # ------------------------------------------------------------------

    def test_quaternion_fields_round_trip(self) -> None:
        hk = Housekeeping(
            timestamp=self._ts(),
            quat_w=0.9239,
            quat_x=0.3827,
            quat_y=0.0,
            quat_z=0.0,
        )
        assert hk.quat_w == pytest.approx(0.9239)
        assert hk.quat_x == pytest.approx(0.3827)
        assert hk.quat_y == pytest.approx(0.0)
        assert hk.quat_z == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # Values agree with attitude_to_quat()
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "ra, dec, roll",
        [
            (0.0, 0.0, 0.0),
            (45.0, 30.0, 15.0),
            (270.0, -45.0, 90.0),
            (359.9, 89.0, 180.0),
        ],
    )
    def test_quaternion_matches_attitude_to_quat(
        self, ra: float, dec: float, roll: float
    ) -> None:
        q = attitude_to_quat(ra, dec, roll)
        hk = Housekeeping(
            timestamp=self._ts(),
            ra=ra,
            dec=dec,
            roll=roll,
            quat_w=float(q[0]),
            quat_x=float(q[1]),
            quat_y=float(q[2]),
            quat_z=float(q[3]),
        )
        assert hk.quat_w == pytest.approx(float(q[0]), abs=1e-9)
        assert hk.quat_x == pytest.approx(float(q[1]), abs=1e-9)
        assert hk.quat_y == pytest.approx(float(q[2]), abs=1e-9)
        assert hk.quat_z == pytest.approx(float(q[3]), abs=1e-9)

    @pytest.mark.parametrize(
        "ra, dec, roll",
        [
            (0.0, 0.0, 0.0),
            (45.0, 30.0, 15.0),
            (270.0, -45.0, 90.0),
        ],
    )
    def test_stored_quaternion_is_unit(
        self, ra: float, dec: float, roll: float
    ) -> None:
        q = attitude_to_quat(ra, dec, roll)
        hk = Housekeeping(
            timestamp=self._ts(),
            quat_w=float(q[0]),
            quat_x=float(q[1]),
            quat_y=float(q[2]),
            quat_z=float(q[3]),
        )
        norm = math.sqrt(
            hk.quat_w**2 + hk.quat_x**2 + hk.quat_y**2 + hk.quat_z**2  # type: ignore[operator]
        )
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_quaternion_round_trips_to_attitude(self) -> None:
        ra_in, dec_in, roll_in = 123.4, -22.5, 47.0
        q = attitude_to_quat(ra_in, dec_in, roll_in)
        hk = Housekeeping(
            timestamp=self._ts(),
            quat_w=float(q[0]),
            quat_x=float(q[1]),
            quat_y=float(q[2]),
            quat_z=float(q[3]),
        )
        import numpy as np

        q_back = np.array([hk.quat_w, hk.quat_x, hk.quat_y, hk.quat_z])
        ra_out, dec_out, roll_out = quat_to_attitude(q_back)
        assert ra_out == pytest.approx(ra_in, abs=1e-6)
        assert dec_out == pytest.approx(dec_in, abs=1e-6)
        assert roll_out == pytest.approx(roll_in, abs=1e-6)

    # ------------------------------------------------------------------
    # HousekeepingList property access
    # ------------------------------------------------------------------

    def test_housekeeping_list_quat_properties(self) -> None:
        q1 = attitude_to_quat(0.0, 0.0, 0.0)
        q2 = attitude_to_quat(90.0, 45.0, 0.0)
        hk1 = Housekeeping(
            timestamp=self._ts(0),
            quat_w=float(q1[0]),
            quat_x=float(q1[1]),
            quat_y=float(q1[2]),
            quat_z=float(q1[3]),
        )
        hk2 = Housekeeping(
            timestamp=self._ts(1),
            quat_w=float(q2[0]),
            quat_x=float(q2[1]),
            quat_y=float(q2[2]),
            quat_z=float(q2[3]),
        )
        hkl = HousekeepingList([hk1, hk2])
        assert hkl.quat_w == [pytest.approx(float(q1[0])), pytest.approx(float(q2[0]))]
        assert hkl.quat_x == [pytest.approx(float(q1[1])), pytest.approx(float(q2[1]))]
        assert hkl.quat_y == [pytest.approx(float(q1[2])), pytest.approx(float(q2[2]))]
        assert hkl.quat_z == [pytest.approx(float(q1[3])), pytest.approx(float(q2[3]))]

    def test_housekeeping_list_quat_none_passthrough(self) -> None:
        hk = Housekeeping(timestamp=self._ts())
        hkl = HousekeepingList([hk])
        assert hkl.quat_w == [None]
        assert hkl.quat_x == [None]
        assert hkl.quat_y == [None]
        assert hkl.quat_z == [None]

    # ------------------------------------------------------------------
    # extract_field
    # ------------------------------------------------------------------

    def test_extract_field_quat_w(self) -> None:
        q = attitude_to_quat(45.0, 0.0, 0.0)
        hk1 = Housekeeping(timestamp=self._ts(0), quat_w=float(q[0]))
        hk2 = Housekeeping(timestamp=self._ts(1), quat_w=None)
        result = Housekeeping.extract_field([hk1, hk2], "quat_w")
        assert result[0] == pytest.approx(float(q[0]))
        assert result[1] is None

    # ------------------------------------------------------------------
    # Telemetry container access
    # ------------------------------------------------------------------

    def test_telemetry_quat_accessible(self) -> None:
        q = attitude_to_quat(30.0, 10.0, 5.0)
        hk = Housekeeping(
            timestamp=self._ts(),
            quat_w=float(q[0]),
            quat_x=float(q[1]),
            quat_y=float(q[2]),
            quat_z=float(q[3]),
        )
        tm = Telemetry(housekeeping=HousekeepingList([hk]))
        assert tm.housekeeping.quat_w[0] == pytest.approx(float(q[0]))
        assert tm.housekeeping.quat_x[0] == pytest.approx(float(q[1]))
        assert tm.housekeeping.quat_y[0] == pytest.approx(float(q[2]))
        assert tm.housekeeping.quat_z[0] == pytest.approx(float(q[3]))

    # ------------------------------------------------------------------
    # Integration: DITL simulation loop writes quaternion values
    # ------------------------------------------------------------------

    def test_ditl_simulation_writes_quaternion(self, ditl: DITL) -> None:  # type: ignore[name-defined]
        """After calc(), every housekeeping record has a non-None unit quaternion."""
        ditl.acs.pointing.return_value = (45.0, 30.0, 15.0, 1)
        ditl.calc()
        hk = ditl.telemetry.housekeeping[0]
        assert hk.quat_w is not None
        assert hk.quat_x is not None
        assert hk.quat_y is not None
        assert hk.quat_z is not None
        norm = math.sqrt(hk.quat_w**2 + hk.quat_x**2 + hk.quat_y**2 + hk.quat_z**2)
        assert norm == pytest.approx(1.0, abs=1e-6)
        # Values must agree with attitude_to_quat for the same attitude
        q_expected = attitude_to_quat(45.0, 30.0, 15.0)
        assert hk.quat_w == pytest.approx(float(q_expected[0]), abs=1e-6)
        assert hk.quat_x == pytest.approx(float(q_expected[1]), abs=1e-6)
        assert hk.quat_y == pytest.approx(float(q_expected[2]), abs=1e-6)
        assert hk.quat_z == pytest.approx(float(q_expected[3]), abs=1e-6)
