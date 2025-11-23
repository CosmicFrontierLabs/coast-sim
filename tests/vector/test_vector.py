import numpy as np
import pytest

from conops import (
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
    def test_scbodyvector_zero_angles(self, ecivec_x):
        """Test spacecraft body vector with zero angles."""
        result = scbodyvector(0, 0, 0, ecivec_x)
        assert result.shape == (3,)

    def test_scbodyvector_with_roll(self, ecivec_y):
        """Test spacecraft body vector with roll."""
        result = scbodyvector(0, 0, np.pi / 2, ecivec_y)
        np.testing.assert_array_almost_equal(result[0], 0, decimal=10)

    def test_scbodyvector_various_angles(self, ecivec_xyz):
        """Test spacecraft body vector with various angles."""
        result = scbodyvector(np.pi / 4, np.pi / 4, np.pi / 4, ecivec_xyz)
        assert result.shape == (3,)


class TestRotvec:
    def test_rotvec_axis1(self, x_axis):
        """Test rotation around axis 1."""
        result = rotvec(1, np.pi / 2, x_axis)
        np.testing.assert_array_almost_equal(result, [1, 0, 0])

    def test_rotvec_axis2(self, x_axis):
        """Test rotation around axis 2."""
        result = rotvec(2, np.pi / 2, x_axis)
        np.testing.assert_array_almost_equal(result, [0, 0, 1])

    def test_rotvec_axis3(self, x_axis):
        """Test rotation around axis 3."""
        result = rotvec(3, np.pi / 2, x_axis)
        np.testing.assert_array_almost_equal(result, [0, -1, 0])

    def test_rotvec_full_rotation(self, test_vector):
        """Test full rotation returns to original."""
        result = rotvec(1, 2 * np.pi, test_vector.copy())
        np.testing.assert_array_almost_equal(result, test_vector)


class TestSeparation:
    def test_separation_same_point(self, origin):
        """Test separation between same point."""
        result = separation(origin, origin)
        assert result == pytest.approx(0, abs=1e-10)

    def test_separation_orthogonal(self, origin, equator_90):
        """Test separation between orthogonal points."""
        result = separation(origin, equator_90)
        assert result == pytest.approx(np.pi / 2, abs=1e-6)

    def test_separation_opposite(self, origin, opposite):
        """Test separation between opposite points."""
        result = separation(origin, opposite)
        assert result == pytest.approx(np.pi, abs=1e-6)


class TestGreatCircle:
    def test_great_circle_same_point(self, small_npts):
        """Test great circle with same start and end."""
        ras, decs = great_circle(0, 0, 0, 0, npts=small_npts)
        assert len(ras) == 12  # npts + 2 (start and end)
        assert len(decs) == 12
        assert ras[0] == 0
        assert ras[-1] == 0

    def test_great_circle_different_points(self, large_npts):
        """Test great circle between different points."""
        ras, decs = great_circle(0, 0, 90, 45, npts=large_npts)
        assert len(ras) == 52
        assert len(decs) == 52
        assert ras[0] == 0
        assert ras[-1] == 90
        assert decs[0] == 0
        assert decs[-1] == 45

    def test_great_circle_varying_npts(self, small_npts):
        """Test great circle with different npts."""
        ras1, decs1 = great_circle(10, 20, 30, 40, npts=small_npts)
        ras2, decs2 = great_circle(10, 20, 30, 40, npts=100)
        assert len(ras1) < len(ras2)


class TestRollOverAngle:
    def test_roll_over_angle_no_rollover(self, no_rollover_angles):
        """Test roll over angle with no rollover."""
        result = roll_over_angle(no_rollover_angles)
        np.testing.assert_array_almost_equal(result, no_rollover_angles)

    def test_roll_over_angle_positive_rollover(self, positive_rollover_angles):
        """Test roll over angle with positive rollover."""
        result = roll_over_angle(positive_rollover_angles)
        # Should be smoothed out
        assert result[0] < result[1] < result[2] < result[3]

    def test_roll_over_angle_negative_rollover(self, negative_rollover_angles):
        """Test roll over angle with negative rollover."""
        result = roll_over_angle(negative_rollover_angles)
        # Should be smoothed out
        assert result[0] > result[1] > result[2] > result[3]

    def test_roll_over_angle_multiple_rollovers(self, multiple_rollover_angles):
        """Test roll over angle with multiple rollovers."""
        result = roll_over_angle(multiple_rollover_angles)
        # Result should be monotonic or have controlled flips
        assert len(result) == len(multiple_rollover_angles)

    def test_roll_over_angle_single_value(self, single_angle):
        """Test roll over angle with single value."""
        result = roll_over_angle(single_angle)
        np.testing.assert_array_almost_equal(result, single_angle)
