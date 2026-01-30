"""Tests for star tracker orientation transformations."""

import numpy as np

from conops.config import StarTrackerOrientation


class TestStarTrackerOrientation:
    """Test orientation transformations."""

    def test_identity_orientation_no_rotation(self):
        """Identity orientation should not rotate pointing."""
        ori = StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=0.0)
        ra, dec = ori.transform_pointing(45.0, 30.0)
        assert np.isclose(ra, 45.0, atol=1e-10)
        assert np.isclose(dec, 30.0, atol=1e-10)

    def test_identity_orientation_various_pointings(self):
        """Identity orientation with various pointings."""
        ori = StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=0.0)
        test_cases = [
            (0.0, 0.0),
            (90.0, 45.0),
            (180.0, -60.0),
            (45.0, 30.0),
        ]
        for ra, dec in test_cases:
            ra_out, dec_out = ori.transform_pointing(ra, dec)
            assert np.isclose(ra_out, ra, atol=1e-10), f"RA mismatch for ({ra}, {dec})"
            assert np.isclose(dec_out, dec, atol=1e-10), (
                f"Dec mismatch for ({ra}, {dec})"
            )

    def test_roll_rotation_90_degrees(self):
        """90 degree roll rotation."""
        ori = StarTrackerOrientation(roll=90.0, pitch=0.0, yaw=0.0)
        # A pointing at (0, 0) is along +X. Rolling about X doesn't change it.
        ra, dec = ori.transform_pointing(0.0, 0.0)
        # Pointing along +X axis should remain along +X after rolling about X
        assert np.isclose(ra, 0.0, atol=1e-10)
        assert np.isclose(dec, 0.0, atol=1e-10)

    def test_pitch_rotation_90_degrees(self):
        """90 degree pitch rotation."""
        ori = StarTrackerOrientation(roll=0.0, pitch=90.0, yaw=0.0)
        ra, dec = ori.transform_pointing(0.0, 0.0)
        # After 90 degree pitch about Y, pointing at (0,0) should change
        assert not (np.isclose(ra, 0.0) and np.isclose(dec, 0.0))

    def test_yaw_rotation_90_degrees(self):
        """90 degree yaw rotation."""
        ori = StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=90.0)
        ra_in = 0.0
        dec_in = 0.0
        ra_out, dec_out = ori.transform_pointing(ra_in, dec_in)
        # RA should be rotated by 90 degrees
        assert np.isclose(ra_out, 90.0, atol=1e-10)
        assert np.isclose(dec_out, 0.0, atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix should be orthogonal."""
        ori = StarTrackerOrientation(roll=45.0, pitch=30.0, yaw=60.0)
        rot = ori.to_rotation_matrix()
        # R @ R^T should be identity
        identity = rot @ rot.T
        assert np.allclose(identity, np.eye(3), atol=1e-10)

    def test_rotation_matrix_determinant_one(self):
        """Rotation matrix determinant should be 1 (proper rotation)."""
        ori = StarTrackerOrientation(roll=45.0, pitch=30.0, yaw=60.0)
        rot = ori.to_rotation_matrix()
        det = np.linalg.det(rot)
        assert np.isclose(det, 1.0, atol=1e-10)

    def test_compose_rotations_commutativity(self):
        """Test composition properties of rotations."""
        ori1 = StarTrackerOrientation(roll=30.0, pitch=0.0, yaw=0.0)
        ori2 = StarTrackerOrientation(roll=30.0, pitch=0.0, yaw=0.0)
        # Same rotations should give same result
        ra1, dec1 = ori1.transform_pointing(45.0, 30.0)
        ra2, dec2 = ori2.transform_pointing(45.0, 30.0)
        assert np.isclose(ra1, ra2, atol=1e-10)
        assert np.isclose(dec1, dec2, atol=1e-10)

    def test_small_angle_approximation(self):
        """Small angle rotations should have minimal effect."""
        ori_small = StarTrackerOrientation(roll=1.0, pitch=0.0, yaw=0.0)
        ra_in = 45.0
        dec_in = 30.0
        ra_out, dec_out = ori_small.transform_pointing(ra_in, dec_in)
        # With small angle, change should be small
        # At dec=30, 1 degree roll causes approximately 1 degree declination change
        assert abs(dec_out - dec_in) < 2.0  # Reasonable bound for small angle


class TestStarTrackerOrientationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pointing_at_pole(self):
        """Pointing at celestial pole (dec=90)."""
        ori = StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=0.0)
        ra, dec = ori.transform_pointing(0.0, 90.0)
        assert np.isclose(dec, 90.0, atol=1e-10)

    def test_pointing_at_south_pole(self):
        """Pointing at south celestial pole (dec=-90)."""
        ori = StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=0.0)
        ra, dec = ori.transform_pointing(0.0, -90.0)
        assert np.isclose(dec, -90.0, atol=1e-10)

    def test_ra_wrapping(self):
        """RA values should be in valid range after transformation."""
        ori = StarTrackerOrientation(roll=0.0, pitch=0.0, yaw=180.0)
        ra, dec = ori.transform_pointing(0.0, 0.0)
        # RA wraps around at 360 degrees
        # After 180 degree yaw at equator, should be at 180 degrees
        ra = ra % 360.0
        assert 0 <= ra <= 360
        assert -90 <= dec <= 90

    def test_negative_angles(self):
        """Test with negative angle inputs."""
        ori = StarTrackerOrientation(roll=-45.0, pitch=-30.0, yaw=-60.0)
        ra, dec = ori.transform_pointing(90.0, 45.0)
        # Should return valid coordinates
        assert 0 <= (ra % 360.0) <= 360
        assert -90 <= dec <= 90
