"""Tests for conops.constraint module."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops.constraint import Constraint


class TestConstraintInit:
    """Test Constraint initialization."""

    def test_constraint_init_defaults(self):
        """Test Constraint initialization with default values."""
        constraint = Constraint()

        assert constraint.bestroll == 0.0
        assert np.array_equal(constraint.bestpointing, np.array([-1, -1, -1]))
        assert constraint.ephem is None

    def test_constraint_init_has_constraints(self):
        """Test Constraint has all constraint objects."""
        constraint = Constraint()

        assert constraint.sun_constraint is not None
        assert constraint.anti_sun_constraint is not None
        assert constraint.moon_constraint is not None
        assert constraint.earth_constraint is not None
        assert constraint.panel_constraint is not None

    def test_constraint_ephemeris_assertion(self):
        """Test constraint methods assert ephemeris is set."""
        constraint = Constraint(ephem=None)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_sun(45.0, 30.0, 1700000000.0)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_panel(45.0, 30.0, 1700000000.0)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_anti_sun(45.0, 30.0, 1700000000.0)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_earth(45.0, 30.0, 1700000000.0)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_moon(45.0, 30.0, 1700000000.0)


class TestConstraintProperties:
    """Test Constraint model properties."""

    def test_constraint_bestpointing_default(self):
        """Test bestpointing default value."""
        constraint = Constraint()
        expected = np.array([-1, -1, -1])
        assert np.array_equal(constraint.bestpointing, expected)

    def test_constraint_bestroll_default(self):
        """Test bestroll default value."""
        constraint = Constraint()
        assert constraint.bestroll == 0.0

    def test_constraint_exclusion_from_serialization(self):
        """Test that ephem is excluded from model serialization."""
        constraint = Constraint()
        # ephem is marked with exclude=True in Field definition
        # so it should not appear in model_dump
        dumped = constraint.model_dump(exclude_unset=False)
        assert "ephem" not in dumped


class TestInSunMethod:
    """Test in_sun method - requires actual Ephemeris, skip detailed tests."""

    def test_in_sun_requires_ephemeris(self):
        """Test in_sun raises assertion without ephemeris."""
        constraint = Constraint(ephem=None)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_sun(45.0, 30.0, 1700000000.0)


class TestInPanelMethod:
    """Test in_panel method - requires actual Ephemeris."""

    def test_in_panel_requires_ephemeris(self):
        """Test in_panel raises assertion without ephemeris."""
        constraint = Constraint(ephem=None)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_panel(45.0, 30.0, 1700000000.0)


class TestInAntiSunMethod:
    """Test in_anti_sun method - requires actual Ephemeris."""

    def test_in_anti_sun_requires_ephemeris(self):
        """Test in_anti_sun raises assertion without ephemeris."""
        constraint = Constraint(ephem=None)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_anti_sun(45.0, 30.0, 1700000000.0)


class TestInEarthMethod:
    """Test in_earth method - requires actual Ephemeris."""

    def test_in_earth_requires_ephemeris(self):
        """Test in_earth raises assertion without ephemeris."""
        constraint = Constraint(ephem=None)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_earth(45.0, 30.0, 1700000000.0)


class TestInMoonMethod:
    """Test in_moon method - requires actual Ephemeris."""

    def test_in_moon_requires_ephemeris(self):
        """Test in_moon raises assertion without ephemeris."""
        constraint = Constraint(ephem=None)

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            constraint.in_moon(45.0, 30.0, 1700000000.0)


class TestInOccultMethod:
    """Test inoccult method logic."""

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_no_violations(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with no violations."""
        mock_sun.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = False
        mock_moon.return_value = False
        mock_panel.return_value = False

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is False

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_sun_violation(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with sun constraint violation."""
        mock_sun.return_value = True
        mock_antisun.return_value = False
        mock_earth.return_value = False
        mock_moon.return_value = False
        mock_panel.return_value = False

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is True

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_multiple_violations(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with multiple violations."""
        mock_sun.return_value = True
        mock_antisun.return_value = False
        mock_earth.return_value = True
        mock_moon.return_value = False
        mock_panel.return_value = False

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is True


class TestInOccultCountMethod:
    """Test inoccult_count method logic."""

    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_count_no_violations(
        self, mock_sun, mock_moon, mock_antisun, mock_earth
    ):
        """Test inoccult_count with no violations."""
        mock_sun.return_value = False
        mock_moon.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = False

        constraint = Constraint()
        count = constraint.inoccult_count(45.0, 30.0, 1700000000.0)

        assert count == 0

    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_count_sun_only(
        self, mock_sun, mock_moon, mock_antisun, mock_earth
    ):
        """Test inoccult_count with only sun violation."""
        mock_sun.return_value = True
        mock_moon.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = False

        constraint = Constraint()
        count = constraint.inoccult_count(45.0, 30.0, 1700000000.0)

        assert count == 2


class TestInGalConsMethod:
    """Test ingalcons method - which appears to be missing from implementation."""

    def test_ingalcons_not_implemented(self):
        """Test that ingalcons method doesn't exist (likely legacy code reference)."""
        constraint = Constraint()
        # The method is called in inoccult_count with hardonly=False, but doesn't exist
        assert not hasattr(constraint, "ingalcons")


class TestConstraintFloatTimeReturnsScalar:
    """Test that float time returns scalar value, not array."""

    @patch("rust_ephem.SunConstraint.in_constraint")
    def test_in_sun_with_float_returns_scalar(self, mock_in_constraint):
        """Test in_sun with float time returns scalar."""
        constraint = Constraint()
        constraint.ephem = Mock()
        constraint.ephem._tle_ephem = Mock()

        # Mock the in_constraint method to return True
        mock_in_constraint.return_value = True

        result = constraint.in_sun(45.0, 30.0, 1700000000.0)

        # Should be scalar, not array
        assert isinstance(result, bool)
        assert result
        # Verify the constraint was called
        assert mock_in_constraint.called

    @patch("rust_ephem.AndConstraint.in_constraint")
    def test_in_panel_with_float_returns_scalar(self, mock_in_constraint):
        """Test in_panel with float time returns scalar."""
        constraint = Constraint()
        constraint.ephem = Mock()
        constraint.ephem._tle_ephem = Mock()

        # Mock the in_constraint method to return False
        mock_in_constraint.return_value = False

        result = constraint.in_panel(45.0, 30.0, 1700000000.0)

        # Should be scalar, not array
        assert isinstance(result, bool)
        assert not result
        # Verify the constraint was called
        assert mock_in_constraint.called

    @patch("rust_ephem.SunConstraint.in_constraint")
    def test_in_anti_sun_with_float_returns_scalar(self, mock_in_constraint):
        """Test in_anti_sun with float time returns scalar."""
        constraint = Constraint()
        constraint.ephem = Mock()
        constraint.ephem._tle_ephem = Mock()

        # Mock the in_constraint method to return True
        mock_in_constraint.return_value = True

        result = constraint.in_anti_sun(45.0, 30.0, 1700000000.0)

        # Should be scalar, not array
        assert isinstance(result, bool)
        assert result
        # Verify the constraint was called
        assert mock_in_constraint.called

    @patch("rust_ephem.EarthLimbConstraint.in_constraint")
    def test_in_earth_with_float_returns_scalar(self, mock_in_constraint):
        """Test in_earth with float time returns scalar."""
        constraint = Constraint()
        constraint.ephem = Mock()
        constraint.ephem._tle_ephem = Mock()

        # Mock the in_constraint method to return False
        mock_in_constraint.return_value = False

        result = constraint.in_earth(45.0, 30.0, 1700000000.0)

        # Should be scalar, not array
        assert isinstance(result, bool)
        assert not result
        # Verify the constraint was called
        assert mock_in_constraint.called

    @patch("rust_ephem.MoonConstraint.in_constraint")
    def test_in_moon_with_float_returns_scalar(self, mock_in_constraint):
        """Test in_moon with float time returns scalar."""
        constraint = Constraint()
        constraint.ephem = Mock()
        constraint.ephem._tle_ephem = Mock()

        # Mock the in_constraint method to return True
        mock_in_constraint.return_value = True

        result = constraint.in_moon(45.0, 30.0, 1700000000.0)

        # Should be scalar, not array
        assert isinstance(result, bool)
        assert result
        # Verify the constraint was called
        assert mock_in_constraint.called

    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_count_all_violations(
        self, mock_sun, mock_moon, mock_antisun, mock_earth
    ):
        """Test inoccult_count with all hard constraints violated."""
        mock_sun.return_value = True
        mock_moon.return_value = True
        mock_antisun.return_value = True
        mock_earth.return_value = True

        constraint = Constraint()
        count = constraint.inoccult_count(45.0, 30.0, 1700000000.0)

        assert count == 8


class TestConstraintWithTimeObjects:
    """Test constraint methods with Time objects instead of floats."""

    @patch("rust_ephem.SunConstraint.evaluate")
    def test_in_sun_with_time_object(self, mock_evaluate):
        """Test in_sun with Time object returns array."""
        constraint = Constraint()
        # Create a minimal mock ephemeris that passes validation
        constraint.ephem = Mock()
        constraint.ephem.timestamp = Time([1700000000.0, 1700000100.0], format="unix")
        constraint.ephem._tle_ephem = Mock()

        # Mock the evaluate method to return a constraint result
        mock_result = Mock()
        mock_result.constraint_array = np.array(
            [False, True]
        )  # False = violated, True = satisfied
        mock_evaluate.return_value = mock_result

        time_obj = [
            datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
            datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
        ]
        result = constraint.in_sun(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("rust_ephem.AndConstraint.evaluate")
    def test_in_panel_with_time_object(self, mock_evaluate):
        """Test in_panel with Time object returns array."""
        constraint = Constraint()
        constraint.ephem = Mock()
        constraint.ephem._tle_ephem = Mock()

        # Mock the evaluate method to return a result with constraint_array
        mock_result = Mock()
        mock_result.constraint_array = np.array([True, False])
        mock_evaluate.return_value = mock_result

        time_obj = [
            datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
            datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
        ]
        result = constraint.in_panel(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert mock_evaluate.called
        result = constraint.in_panel(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("rust_ephem.SunConstraint.evaluate")
    def test_in_anti_sun_with_time_object(self, mock_evaluate):
        """Test in_anti_sun with Time object returns array."""
        constraint = Constraint()
        # Create a minimal mock ephemeris that passes validation
        constraint.ephem = Mock()
        constraint.ephem.timestamp = Time([1700000000.0, 1700000100.0], format="unix")
        constraint.ephem._tle_ephem = Mock()

        # Mock the evaluate method to return a constraint result
        mock_result = Mock()
        mock_result.constraint_array = np.array(
            [False, True]
        )  # False = violated, True = satisfied
        mock_evaluate.return_value = mock_result

        time_obj = [
            datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
            datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
        ]
        result = constraint.in_anti_sun(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("rust_ephem.EarthLimbConstraint.evaluate")
    def test_in_earth_with_time_object(self, mock_evaluate):
        """Test in_earth with Time object returns array."""
        constraint = Constraint()
        # Create a minimal mock ephemeris that passes validation
        constraint.ephem = Mock()
        constraint.ephem.timestamp = Time([1700000000.0, 1700000100.0], format="unix")
        constraint.ephem._tle_ephem = Mock()

        # Mock the evaluate method to return a constraint result
        mock_result = Mock()
        mock_result.constraint_array = np.array(
            [False, True]
        )  # False = violated, True = satisfied
        mock_evaluate.return_value = mock_result

        time_obj = [
            datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
            datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
        ]
        result = constraint.in_earth(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("rust_ephem.MoonConstraint.evaluate")
    def test_in_moon_with_time_object(self, mock_evaluate):
        """Test in_moon with Time object returns array."""
        constraint = Constraint()
        # Create a minimal mock ephemeris that passes validation
        constraint.ephem = Mock()
        constraint.ephem.timestamp = Time([1700000000.0, 1700000100.0], format="unix")
        constraint.ephem._tle_ephem = Mock()

        # Mock the evaluate method to return a constraint result
        mock_result = Mock()
        mock_result.constraint_array = np.array(
            [False, True]
        )  # False = violated, True = satisfied
        mock_evaluate.return_value = mock_result

        time_obj = [
            datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
            datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
        ]
        result = constraint.in_moon(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_with_time_object(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with Time object."""
        mock_sun.return_value = np.array([True, False])
        mock_antisun.return_value = np.array([False, False])
        mock_earth.return_value = np.array([False, False])
        mock_moon.return_value = np.array([False, True])
        mock_panel.return_value = np.array([False, False])

        constraint = Constraint()
        time_obj = [
            datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
            datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
        ]
        result = constraint.inoccult(45.0, 30.0, time_obj)

        assert isinstance(result, np.ndarray)
        assert result[0]  # sun violation
        assert result[1]  # moon violation


class TestConstraintEdgeCases:
    """Test edge cases and additional paths."""

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_panel_violation(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with panel constraint violation."""
        mock_sun.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = False
        mock_moon.return_value = False
        mock_panel.return_value = True

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is True

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_antisun_violation(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with antisun constraint violation."""
        mock_sun.return_value = False
        mock_antisun.return_value = True
        mock_earth.return_value = False
        mock_moon.return_value = False
        mock_panel.return_value = False

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is True

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_moon_violation(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with moon constraint violation."""
        mock_sun.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = False
        mock_moon.return_value = True
        mock_panel.return_value = False

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is True

    @patch("conops.constraint.Constraint.in_panel")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_earth_violation(
        self, mock_sun, mock_antisun, mock_earth, mock_moon, mock_panel
    ):
        """Test inoccult with earth constraint violation."""
        mock_sun.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = True
        mock_moon.return_value = False
        mock_panel.return_value = False

        constraint = Constraint()
        result = constraint.inoccult(45.0, 30.0, 1700000000.0)

        assert result is True

    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_count_moon_only(
        self, mock_sun, mock_moon, mock_antisun, mock_earth
    ):
        """Test inoccult_count with only moon violation."""
        mock_sun.return_value = False
        mock_moon.return_value = True
        mock_antisun.return_value = False
        mock_earth.return_value = False

        constraint = Constraint()
        count = constraint.inoccult_count(45.0, 30.0, 1700000000.0)

        assert count == 2

    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_count_antisun_only(
        self, mock_sun, mock_moon, mock_antisun, mock_earth
    ):
        """Test inoccult_count with only antisun violation."""
        mock_sun.return_value = False
        mock_moon.return_value = False
        mock_antisun.return_value = True
        mock_earth.return_value = False

        constraint = Constraint()
        count = constraint.inoccult_count(45.0, 30.0, 1700000000.0)

        assert count == 2

    @patch("conops.constraint.Constraint.in_earth")
    @patch("conops.constraint.Constraint.in_anti_sun")
    @patch("conops.constraint.Constraint.in_moon")
    @patch("conops.constraint.Constraint.in_sun")
    def test_inoccult_count_earth_only(
        self, mock_sun, mock_moon, mock_antisun, mock_earth
    ):
        """Test inoccult_count with only earth violation."""
        mock_sun.return_value = False
        mock_moon.return_value = False
        mock_antisun.return_value = False
        mock_earth.return_value = True

        constraint = Constraint()
        count = constraint.inoccult_count(45.0, 30.0, 1700000000.0)

        assert count == 2
