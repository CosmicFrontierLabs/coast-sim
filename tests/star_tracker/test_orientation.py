"""Tests for star tracker orientation transformations."""

import numpy as np
import pytest

from conops.config import StarTrackerOrientation


class TestStarTrackerOrientation:
    """Test orientation transformations with boresight vectors."""

    def test_identity_orientation_no_rotation(self):
        """Identity orientation (boresight along +X) should not rotate pointing."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        ra, dec = ori.transform_pointing(45.0, 30.0)
        assert np.isclose(ra, 45.0, atol=1e-10)
        assert np.isclose(dec, 30.0, atol=1e-10)

    def test_identity_orientation_various_pointings(self):
        """Identity orientation with various pointings."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
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

    def test_boresight_along_y_axis(self):
        """Star tracker boresight pointing along +Y direction."""
        ori = StarTrackerOrientation(boresight=(0.0, 1.0, 0.0))
        ra, dec = ori.transform_pointing(0.0, 0.0)
        # A pointing at (0, 0) is along +X. With boresight at +Y, we get rotation
        assert not (np.isclose(ra, 0.0) and np.isclose(dec, 0.0))

    def test_boresight_along_z_axis(self):
        """Star tracker boresight pointing along +Z direction."""
        ori = StarTrackerOrientation(boresight=(0.0, 0.0, 1.0))
        ra_in = 0.0
        dec_in = 0.0
        ra_out, dec_out = ori.transform_pointing(ra_in, dec_in)
        # Should return valid coordinates
        assert 0 <= (ra_out % 360.0) <= 360
        assert -90 <= dec_out <= 90

    def test_boresight_negative_z(self):
        """Star tracker boresight pointing along -Z direction."""
        ori = StarTrackerOrientation(boresight=(0.0, 0.0, -1.0))
        ra, dec = ori.transform_pointing(45.0, 30.0)
        # Should return valid coordinates
        assert 0 <= (ra % 360.0) <= 360
        assert -90 <= dec <= 90

    def test_rotation_matrix_orthogonal(self):
        """Rotation matrix should be orthogonal."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        rot = ori.to_rotation_matrix()
        # R @ R^T should be identity
        identity = rot @ rot.T
        assert np.allclose(identity, np.eye(3), atol=1e-10)

    def test_rotation_matrix_determinant_one(self):
        """Rotation matrix determinant should be 1 (proper rotation)."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        rot = ori.to_rotation_matrix()
        det = np.linalg.det(rot)
        assert np.isclose(det, 1.0, atol=1e-10)

    def test_diagonal_boresight(self):
        """Test with diagonal boresight vector."""
        # Normalize diagonal vector
        norm = np.sqrt(3)
        ori = StarTrackerOrientation(boresight=(1.0 / norm, 1.0 / norm, 1.0 / norm))
        rot = ori.to_rotation_matrix()
        # Rotation matrix should still be orthogonal
        identity = rot @ rot.T
        assert np.allclose(identity, np.eye(3), atol=1e-10)
        # Determinant should be 1
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)

    def test_boresight_first_column_of_matrix(self):
        """Boresight should be the first column of rotation matrix."""
        boresight = (1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2))
        ori = StarTrackerOrientation(boresight=boresight)
        rot = ori.to_rotation_matrix()
        # First column should be the boresight vector
        assert np.allclose(rot[:, 0], boresight, atol=1e-10)

    def test_compose_rotations_consistency(self):
        """Same rotations should give same result."""
        ori1 = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        ori2 = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        # Same orientations should give same result
        ra1, dec1 = ori1.transform_pointing(45.0, 30.0)
        ra2, dec2 = ori2.transform_pointing(45.0, 30.0)
        assert np.isclose(ra1, ra2, atol=1e-10)
        assert np.isclose(dec1, dec2, atol=1e-10)

    def test_invalid_boresight_not_unit_vector(self):
        """Non-unit vectors should be rejected."""
        # This should fail because (2, 0, 0) is not a unit vector
        with pytest.raises(ValueError):
            StarTrackerOrientation(boresight=(2.0, 0.0, 0.0))

    def test_nearly_unit_vector_accepted(self):
        """Nearly unit vectors within tolerance should be accepted."""
        # Create a vector very close to unit (within 1%)
        magnitude = 1.005
        ori = StarTrackerOrientation(boresight=(1.0 / magnitude, 0.0, 0.0))
        assert ori is not None


class TestStarTrackerOrientationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pointing_at_pole(self):
        """Pointing at celestial pole (dec=90)."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        ra, dec = ori.transform_pointing(0.0, 90.0)
        assert np.isclose(dec, 90.0, atol=1e-10)

    def test_pointing_at_south_pole(self):
        """Pointing at south celestial pole (dec=-90)."""
        ori = StarTrackerOrientation(boresight=(1.0, 0.0, 0.0))
        ra, dec = ori.transform_pointing(0.0, -90.0)
        assert np.isclose(dec, -90.0, atol=1e-10)

    def test_ra_wrapping(self):
        """RA values should be in valid range after transformation."""
        # Boresight pointing along -Y should rotate RA by ~90 degrees
        ori = StarTrackerOrientation(boresight=(0.0, -1.0, 0.0))
        ra, dec = ori.transform_pointing(0.0, 0.0)
        # RA wraps around at 360 degrees
        ra = ra % 360.0
        assert 0 <= ra <= 360
        assert -90 <= dec <= 90

    def test_dec_range(self):
        """Declination should always be in [-90, 90] range."""
        test_boresights = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0),
            (1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)),
        ]
        for boresight in test_boresights:
            ori = StarTrackerOrientation(boresight=boresight)
            ra, dec = ori.transform_pointing(90.0, 45.0)
            assert -90 <= dec <= 90, (
                f"Dec out of range for boresight {boresight}: {dec}"
            )

    def test_multiple_pointings_same_orientation(self):
        """Multiple pointings with same orientation should be consistent."""
        ori = StarTrackerOrientation(
            boresight=(1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0)
        )

        results = []
        for i in range(5):
            ra, dec = ori.transform_pointing(i * 30.0, i * 10.0 - 30)
            results.append((ra, dec))

        # No NaNs or infinities
        for ra, dec in results:
            assert np.isfinite(ra)
            assert np.isfinite(dec)
