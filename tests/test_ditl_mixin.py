from datetime import datetime, timezone
from unittest.mock import Mock, patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

from conops.ditl_mixin import DITLMixin

"""Unit tests for DITLMixin."""

matplotlib.use("Agg")


@pytest.fixture
def mock_config():
    """Create a minimal mock config with required attributes for DITLMixin."""
    cfg = Mock()
    cfg.name = "test"
    cfg.constraint = Mock()
    cfg.constraint.ephem = Mock()  # DITLMixin asserts this is not None
    cfg.battery = Mock()
    cfg.battery.max_depth_of_discharge = 0.5
    return cfg


def test_init_sets_defaults_and_uses_passes_and_acs(mock_config):
    """DITLMixin.__init__ should set up default attributes and use PassTimes and ACS."""
    with (
        patch("conops.ditl_mixin.PassTimes") as mock_pass_class,
        patch("conops.ditl_mixin.ACS") as mock_acs_class,
        patch("conops.ditl_mixin.Plan") as mock_plan_class,
    ):
        # Set return values for patched classes
        mock_pass_inst = Mock()
        mock_pass_class.return_value = mock_pass_inst
        mock_acs_inst = Mock()
        mock_acs_class.return_value = mock_acs_inst
        mock_plan_class.return_value = Mock()

        ditl = DITLMixin(config=mock_config)

        # Basic attribute checks
        assert ditl.config is mock_config
        assert isinstance(ditl.ra, list) and ditl.ra == []
        assert isinstance(ditl.dec, list) and ditl.dec == []
        assert isinstance(ditl.utime, list) and ditl.utime == []
        assert ditl.ephem is None

        # Check PassTimes and ACS usage
        mock_pass_class.assert_called()
        assert ditl.passes is mock_pass_inst
        assert ditl.executed_passes is mock_pass_inst  # our mock returns same instance
        mock_acs_class.assert_called_once_with(
            constraint=mock_config.constraint, config=mock_config
        )
        assert ditl.acs is mock_acs_inst

        # Defaults set in __init__
        assert (
            ditl.begin - datetime(2018, 11, 27, 0, 0, 0, tzinfo=timezone.utc)
        ).total_seconds() == 0
        assert (
            ditl.end - datetime(2018, 11, 28, 0, 0, 0, tzinfo=timezone.utc)
        ).total_seconds() == 0
        assert ditl.step_size == 60
        assert ditl.ppt is None


def test_plot_creates_subplots_and_battery_line(mock_config):
    """Plot should create 7 subplots and include a dashed battery horizontal line."""
    with (
        patch("conops.ditl_mixin.PassTimes") as mock_pass_class,
        patch("conops.ditl_mixin.ACS") as mock_acs_class,
        patch("conops.ditl_mixin.Plan") as mock_plan_class,
    ):
        # Create mock PassTimes and ACS instances
        mock_pass_class.return_value = Mock()
        mock_acs_class.return_value = Mock()
        mock_plan_class.return_value = Mock()

        ditl = DITLMixin(config=mock_config)

        # Populate data arrays of same length
        base_time = 1514764800.0
        ditl.utime = [base_time + i * 60 for i in range(4)]
        ditl.ra = [1.0, 2.0, 3.0, 4.0]
        ditl.dec = [0.5, 0.6, 0.7, 0.8]
        ditl.mode = [0, 1, 2, 3]
        ditl.batterylevel = [0.2, 0.3, 0.4, 0.5]
        ditl.panel = [0.1, 0.2, 0.3, 0.4]
        ditl.power = [5.0, 6.0, 7.0, 8.0]
        ditl.obsid = [0, 1, 2, 3]

        # Ensure no figures are open
        plt.close("all")

        # Call plot(), should not raise
        ditl.plot()

        fig = plt.gcf()
        # There should be 7 axes (subplots)
        assert len(fig.axes) == 7

        # Title of first axis should include config.name
        assert (
            fig.axes[0].get_title()
            == f"Timeline for DITL Simulation: {mock_config.name}"
        )

        # Battery axis (4th subplot) should contain a dashed horizontal line
        batt_ax = fig.axes[3]
        has_dashed_hline = any(line.get_linestyle() == "--" for line in batt_ax.lines)
        assert has_dashed_hline

        # Last axis should have an x-label indicating time in hours
        assert fig.axes[-1].get_xlabel() == "Time (hour of day)"

        plt.close("all")
