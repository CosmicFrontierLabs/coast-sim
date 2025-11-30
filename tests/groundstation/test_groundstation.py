import pytest
from pydantic import ValidationError

from conops import GroundStation


class TestGroundStationAntennaFields:
    def test_default_bands_empty(self):
        gs = GroundStation(code="X", name="X", latitude_deg=0.0, longitude_deg=0.0)
        assert gs.bands == []

    def test_default_gain_db_none(self):
        gs = GroundStation(code="X", name="X", latitude_deg=0.0, longitude_deg=0.0)
        assert gs.gain_db is None

    def test_default_overall_max_downlink_none(self):
        gs = GroundStation(code="X", name="X", latitude_deg=0.0, longitude_deg=0.0)
        assert gs.get_overall_max_downlink() is None


class TestGroundStation:
    def test_code_is_uppercased(self):
        gs = GroundStation(
            code=" rwa ", name="Rwanda", latitude_deg=1.96, longitude_deg=30.39
        )
        assert gs.code == "RWA"

    @pytest.mark.parametrize("value", [-1, 91])
    def test_min_elevation_bounds_raises_validation_error(self, value):
        with pytest.raises(ValidationError):
            GroundStation(
                code="A",
                name="A",
                latitude_deg=0.0,
                longitude_deg=0.0,
                min_elevation_deg=value,
            )

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_schedule_probability_bounds_raises_validation_error(self, value):
        with pytest.raises(ValidationError):
            GroundStation(
                code="A",
                name="A",
                latitude_deg=0.0,
                longitude_deg=0.0,
                schedule_probability=value,
            )


class TestGroundStationRegistry:
    def test_add_and_contains(self, groundstation_registry, sample_groundstation):
        groundstation_registry.add(sample_groundstation)
        assert "GHA" in groundstation_registry

    def test_get_returns_groundstation(
        self, groundstation_registry, sample_groundstation
    ):
        groundstation_registry.add(sample_groundstation)
        assert groundstation_registry.get("GHA").name == "Ghana"

    def test_codes_contains_added_code(
        self, groundstation_registry, sample_groundstation
    ):
        groundstation_registry.add(sample_groundstation)
        assert "GHA" in groundstation_registry.codes()

    def test_iteration_returns_multiple_groundstations(self, default_registry):
        items = list(default_registry)
        assert len(items) >= 2

    def test_iteration_returns_groundstation_instances(self, default_registry):
        items = list(default_registry)
        for s in items:
            assert isinstance(s, GroundStation)

    def test_min_elevation_returns_float(self, default_registry):
        assert isinstance(default_registry.min_elevation("MAL"), float)

    def test_schedule_probability_for_returns_float(self, default_registry):
        assert isinstance(default_registry.schedule_probability_for("SGS"), float)

    def test_get_nonexistent_code_raises_keyerror(self, groundstation_registry):
        with pytest.raises(KeyError):
            groundstation_registry.get("NONEXISTENT")

    def test_add_new_station_increases_length(
        self, groundstation_registry, sample_groundstation
    ):
        initial_length = len(groundstation_registry.stations)
        groundstation_registry.add(sample_groundstation)
        assert len(groundstation_registry.stations) == initial_length + 1

    def test_add_existing_code_replaces_station(
        self, groundstation_registry, sample_groundstation
    ):
        groundstation_registry.add(sample_groundstation)
        original_name = sample_groundstation.name
        updated_station = GroundStation(
            code=sample_groundstation.code,
            name="Updated Name",
            latitude_deg=sample_groundstation.latitude_deg,
            longitude_deg=sample_groundstation.longitude_deg,
        )
        groundstation_registry.add(updated_station)
        retrieved = groundstation_registry.get(sample_groundstation.code)
        assert retrieved.name == "Updated Name"
        assert retrieved.name != original_name

    def test_add_existing_code_does_not_increase_length(
        self, groundstation_registry, sample_groundstation
    ):
        groundstation_registry.add(sample_groundstation)
        initial_length = len(groundstation_registry.stations)
        updated_station = GroundStation(
            code=sample_groundstation.code,
            name="Another Name",
            latitude_deg=sample_groundstation.latitude_deg,
            longitude_deg=sample_groundstation.longitude_deg,
        )
        groundstation_registry.add(updated_station)
        assert len(groundstation_registry.stations) == initial_length
