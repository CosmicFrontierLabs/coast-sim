import pytest
from pydantic import ValidationError

from conops.groundstation import Antenna, GroundStation


class TestAntenna:
    def test_default_bands_empty(self):
        a = Antenna()
        assert a.bands == []

    def test_default_gain_db_none(self):
        a = Antenna()
        assert a.gain_db is None

    def test_default_max_data_rate_mbps_none(self):
        a = Antenna()
        assert a.max_data_rate_mbps is None

    def test_default_simultaneous_links_one(self):
        a = Antenna()
        assert a.simultaneous_links == 1

    def test_simultaneous_links_zero_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Antenna(simultaneous_links=0)


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
