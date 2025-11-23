import numpy as np
import pytest

from conops.vector import (
    great_circle,
    radec2vec,
    roll_over_angle,
    rotvec,
    scbodyvector,
    separation,
)


class TestRadec2vec:
    def test_radec2vec_zero(self):
        """Test conversion at RA=0, Dec=0."""
        result = radec2vec(0, 0)
        np.testing.assert_array_almost_equal(result, [1, 0, 0])

    def test_radec2vec_north_pole(self):
        """Test conversion at north celestial pole."""
        result = radec2vec(0, np.pi / 2)
        np.testing.assert_array_almost_equal(result, [0, 0, 1])

    def test_radec2vec_south_pole(self):
        """Test conversion at south celestial pole."""
        result = radec2vec(0, -np.pi / 2)
        np.testing.assert_array_almost_equal(result, [0, 0, -1])

    def test_radec2vec_various_angles(self):
        """Test conversion at various angles."""
        result = radec2vec(np.pi / 2, 0)
        np.testing.assert_array_almost_equal(result, [0, 1, 0])


class TestScbodyvector:
    def test_scbodyvector_zero_angles(self):
        """Test spacecraft body vector with zero angles."""
        ecivec = np.array([1, 0, 0])
        result = scbodyvector(0, 0, 0, ecivec)
        assert result.shape == (3,)

    def test_scbodyvector_with_roll(self):
        """Test spacecraft body vector with roll."""
        ecivec = np.array([0, 1, 0])
        result = scbodyvector(0, 0, np.pi / 2, ecivec)
        np.testing.assert_array_almost_equal(result[0], 0, decimal=10)

    def test_scbodyvector_various_angles(self):
        """Test spacecraft body vector with various angles."""
        ecivec = np.array([1, 1, 1])
        result = scbodyvector(np.pi / 4, np.pi / 4, np.pi / 4, ecivec)
        assert result.shape == (3,)


class TestRotvec:
    def test_rotvec_axis1(self):
        """Test rotation around axis 1."""
        v = np.array([1.0, 0.0, 0.0])
        result = rotvec(1, np.pi / 2, v)
        np.testing.assert_array_almost_equal(result, [1, 0, 0])

    def test_rotvec_axis2(self):
        """Test rotation around axis 2."""
        v = np.array([1.0, 0.0, 0.0])
        result = rotvec(2, np.pi / 2, v)
        np.testing.assert_array_almost_equal(result, [0, 0, 1])

    def test_rotvec_axis3(self):
        """Test rotation around axis 3."""
        v = np.array([1.0, 0.0, 0.0])
        result = rotvec(3, np.pi / 2, v)
        np.testing.assert_array_almost_equal(result, [0, -1, 0])

    def test_rotvec_full_rotation(self):
        """Test full rotation returns to original."""
        v = np.array([1.0, 2.0, 3.0])
        result = rotvec(1, 2 * np.pi, v.copy())
        np.testing.assert_array_almost_equal(result, v)


class TestSeparation:
    def test_separation_same_point(self):
        """Test separation between same point."""
        one = [0, 0]
        two = [0, 0]
        result = separation(one, two)
        assert result == pytest.approx(0, abs=1e-10)

    def test_separation_orthogonal(self):
        """Test separation between orthogonal points."""
        one = [0, 0]
        two = [np.pi / 2, 0]
        result = separation(one, two)
        assert result == pytest.approx(np.pi / 2, abs=1e-6)

    def test_separation_opposite(self):
        """Test separation between opposite points."""
        one = [0, 0]
        two = [np.pi, 0]
        result = separation(one, two)
        assert result == pytest.approx(np.pi, abs=1e-6)


class TestGreatCircle:
    def test_great_circle_same_point(self):
        """Test great circle with same start and end."""
        ras, decs = great_circle(0, 0, 0, 0, npts=10)
        assert len(ras) == 12  # npts + 2 (start and end)
        assert len(decs) == 12
        assert ras[0] == 0
        assert ras[-1] == 0

    def test_great_circle_different_points(self):
        """Test great circle between different points."""
        ras, decs = great_circle(0, 0, 90, 45, npts=50)
        assert len(ras) == 52
        assert len(decs) == 52
        assert ras[0] == 0
        assert ras[-1] == 90
        assert decs[0] == 0
        assert decs[-1] == 45

    def test_great_circle_varying_npts(self):
        """Test great circle with different npts."""
        ras1, decs1 = great_circle(10, 20, 30, 40, npts=10)
        ras2, decs2 = great_circle(10, 20, 30, 40, npts=100)
        assert len(ras1) < len(ras2)


class TestRollOverAngle:
    def test_roll_over_angle_no_rollover(self):
        """Test roll over angle with no rollover."""
        angles = [10, 20, 30, 40]
        result = roll_over_angle(angles)
        np.testing.assert_array_almost_equal(result, angles)

    def test_roll_over_angle_positive_rollover(self):
        """Test roll over angle with positive rollover."""
        angles = [350, 355, 5, 10]
        result = roll_over_angle(angles)
        # Should be smoothed out
        assert result[0] < result[1] < result[2] < result[3]

    def test_roll_over_angle_negative_rollover(self):
        """Test roll over angle with negative rollover."""
        angles = [10, 5, 355, 350]
        result = roll_over_angle(angles)
        # Should be smoothed out
        assert result[0] > result[1] > result[2] > result[3]

    def test_roll_over_angle_multiple_rollovers(self):
        """Test roll over angle with multiple rollovers."""
        angles = [350, 355, 5, 10, 350, 355]
        result = roll_over_angle(angles)
        # Result should be monotonic or have controlled flips
        assert len(result) == len(angles)

    def test_roll_over_angle_single_value(self):
        """Test roll over angle with single value."""
        angles = [180]
        result = roll_over_angle(angles)
        np.testing.assert_array_almost_equal(result, angles)
