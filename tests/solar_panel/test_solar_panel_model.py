import pytest

from conops import SolarPanel


class TestDefaultSolarPanelSet:
    def test_contains_one_panel(self, default_panel_set):
        panels = default_panel_set.panels
        assert len(panels) == 1
        assert isinstance(panels[0], SolarPanel)


class TestMultiplePanelsConfiguration:
    def test_panel_count(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert len(panels) == 2

    def test_first_panel_name(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert panels[0].name == "P1"

    def test_second_panel_name(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert panels[1].name == "P2"

    def test_first_panel_normal(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert panels[0].normal == (0.0, 1.0, 0.0)

    def test_second_panel_normal(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert panels[1].normal == (0.0, 0.0, -1.0)

    def test_first_panel_max_power(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert panels[0].max_power == pytest.approx(300.0)

    def test_second_panel_max_power(self, multi_panel_set):
        panels = multi_panel_set.panels
        assert panels[1].max_power == pytest.approx(700.0)

    def test_total_max_power(self, multi_panel_set):
        total_power = sum(panel.max_power for panel in multi_panel_set.panels)
        assert total_power == pytest.approx(1000.0)

    def test_combined_power_output(self, multi_panel_set):
        panels = multi_panel_set.panels
        combined_power = panels[0].max_power + panels[1].max_power
        assert combined_power == pytest.approx(1000.0)


class TestPanelEfficiencyFallback:
    def test_first_panel_efficiency_is_none(self, efficiency_fallback_panel_set):
        panels = efficiency_fallback_panel_set.panels
        assert panels[0].conversion_efficiency is None

    def test_second_panel_efficiency(self, efficiency_fallback_panel_set):
        panels = efficiency_fallback_panel_set.panels
        assert panels[1].conversion_efficiency == pytest.approx(0.88)
