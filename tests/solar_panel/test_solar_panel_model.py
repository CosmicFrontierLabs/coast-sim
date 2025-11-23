import pytest

from conops.solar_panel import SolarPanel, SolarPanelSet


@pytest.fixture
def default_panel_set():
    return SolarPanelSet(name="Default Set")


@pytest.fixture
def multi_panel_set():
    return SolarPanelSet(
        name="Array",
        conversion_efficiency=0.95,
        panels=[
            SolarPanel(
                name="P1", sidemount=True, cant_x=5.0, cant_y=0.0, max_power=300.0
            ),
            SolarPanel(
                name="P2", sidemount=False, cant_x=0.0, cant_y=12.0, max_power=700.0
            ),
        ],
    )


@pytest.fixture
def efficiency_fallback_panel_set():
    return SolarPanelSet(
        conversion_efficiency=0.91,
        panels=[
            SolarPanel(name="P1", max_power=100.0, conversion_efficiency=None),
            SolarPanel(name="P2", max_power=100.0, conversion_efficiency=0.88),
        ],
    )


class TestDefaultSolarPanelSet:
    def test_contains_one_panel(self, default_panel_set):
        effective = default_panel_set._effective_panels()
        assert len(effective) == 1
        assert isinstance(effective[0], SolarPanel)


class TestMultiplePanelsConfiguration:
    def test_panel_count(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert len(effective) == 2

    def test_first_panel_name(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[0].name == "P1"

    def test_second_panel_name(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[1].name == "P2"

    def test_first_panel_sidemount(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[0].sidemount is True

    def test_second_panel_sidemount(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[1].sidemount is False

    def test_first_panel_cant_x(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[0].cant_x == pytest.approx(5.0)

    def test_second_panel_cant_y(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[1].cant_y == pytest.approx(12.0)

    def test_first_panel_max_power(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[0].max_power == pytest.approx(300.0)

    def test_second_panel_max_power(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        assert effective[1].max_power == pytest.approx(700.0)

    def test_total_max_power(self, multi_panel_set):
        total_power = sum(
            panel.max_power for panel in multi_panel_set._effective_panels()
        )
        assert total_power == pytest.approx(1000.0)

    def test_combined_power_output(self, multi_panel_set):
        effective = multi_panel_set._effective_panels()
        combined_power = effective[0].max_power + effective[1].max_power
        assert combined_power == pytest.approx(1000.0)


class TestPanelEfficiencyFallback:
    def test_first_panel_efficiency_is_none(self, efficiency_fallback_panel_set):
        effective = efficiency_fallback_panel_set._effective_panels()
        assert effective[0].conversion_efficiency is None

    def test_second_panel_efficiency(self, efficiency_fallback_panel_set):
        effective = efficiency_fallback_panel_set._effective_panels()
        assert effective[1].conversion_efficiency == pytest.approx(0.88)
