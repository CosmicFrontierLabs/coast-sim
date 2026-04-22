"""Tests for PanelGeometry and compute_shadow_fraction."""

from __future__ import annotations

import numpy as np
import pytest

from conops.config.geometry import PanelGeometry, compute_shadow_fraction

# ---------------------------------------------------------------------------
# Fixtures: two stacked identical panels in the XZ plane, both facing +Y.
# The occluder is at y=1, the receiver at y=0.
# When sun = +Y, the occluder completely shadows the receiver.
# ---------------------------------------------------------------------------


@pytest.fixture()
def receiver_xz() -> PanelGeometry:
    """2 m × 2 m receiver in the XZ plane at y=0, facing +Y."""
    return PanelGeometry(
        center_m=(0.0, 0.0, 0.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 0.0, 1.0),
        width_m=2.0,
        height_m=2.0,
    )


@pytest.fixture()
def occluder_xz() -> PanelGeometry:
    """2 m × 2 m occluder in the XZ plane at y=1, facing +Y (same as receiver)."""
    return PanelGeometry(
        center_m=(0.0, 1.0, 0.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 0.0, 1.0),
        width_m=2.0,
        height_m=2.0,
    )


# ---------------------------------------------------------------------------
# PanelGeometry unit tests
# ---------------------------------------------------------------------------


class TestPanelGeometry:
    def test_normal_is_u_cross_v(self) -> None:
        g = PanelGeometry(
            center_m=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(g.normal, [0.0, -1.0, 0.0], atol=1e-12)

    def test_corners_shape(self, receiver_xz: PanelGeometry) -> None:
        assert receiver_xz.corners.shape == (4, 3)

    def test_corners_centroid(self, receiver_xz: PanelGeometry) -> None:
        centroid = receiver_xz.corners.mean(axis=0)
        np.testing.assert_allclose(centroid, [0.0, 0.0, 0.0], atol=1e-12)

    def test_corners_span_width(self, receiver_xz: PanelGeometry) -> None:
        xs = receiver_xz.corners[:, 0]
        assert xs.max() == pytest.approx(1.0)
        assert xs.min() == pytest.approx(-1.0)

    def test_unit_vector_validation(self) -> None:
        with pytest.raises(ValueError, match="unit vector"):
            PanelGeometry(
                center_m=(0.0, 0.0, 0.0),
                u=(2.0, 0.0, 0.0),  # not unit
                v=(0.0, 1.0, 0.0),
            )


# ---------------------------------------------------------------------------
# compute_shadow_fraction tests
# ---------------------------------------------------------------------------


class TestComputeShadowFraction:
    def test_no_occluders_returns_zero(self, receiver_xz: PanelGeometry) -> None:
        sun = np.array([0.0, 1.0, 0.0])
        assert compute_shadow_fraction(sun, [], receiver_xz) == pytest.approx(0.0)

    def test_complete_shadow_sun_normal_to_receiver(
        self, occluder_xz: PanelGeometry, receiver_xz: PanelGeometry
    ) -> None:
        """Identical stacked panels with sun = +Y → full shadow on receiver."""
        sun = np.array([0.0, 1.0, 0.0])
        frac = compute_shadow_fraction(sun, [occluder_xz], receiver_xz)
        assert frac == pytest.approx(1.0, abs=1e-9)

    def test_zero_shadow_sun_antiparallel_to_receiver(
        self, occluder_xz: PanelGeometry, receiver_xz: PanelGeometry
    ) -> None:
        """Sun comes from -Y: receiver faces away, s·n_rec < 0 → no shadow."""
        sun = np.array([0.0, -1.0, 0.0])
        frac = compute_shadow_fraction(sun, [occluder_xz], receiver_xz)
        assert frac == pytest.approx(0.0)

    def test_zero_shadow_sun_parallel_to_receiver(
        self, occluder_xz: PanelGeometry, receiver_xz: PanelGeometry
    ) -> None:
        """Sun direction parallel to receiver face (s·n_rec ≈ 0) → no shadow."""
        sun = np.array([1.0, 0.0, 0.0])
        frac = compute_shadow_fraction(sun, [occluder_xz], receiver_xz)
        assert frac == pytest.approx(0.0)

    def test_partial_shadow_sun_offset_45_degrees(
        self, occluder_xz: PanelGeometry, receiver_xz: PanelGeometry
    ) -> None:
        """Sun at 45° from +Y in XY plane → shadow shifts by 1 m in -X.

        The identical 2 m × 2 m occluder at y=1 casts a shadow shifted 1 m in -X.
        Half the receiver (x ∈ [-1, 0]) is in shadow → fraction = 0.5.
        """
        sun = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])
        frac = compute_shadow_fraction(sun, [occluder_xz], receiver_xz)
        assert frac == pytest.approx(0.5, abs=1e-9)

    def test_smaller_occluder_partial_shadow(self, receiver_xz: PanelGeometry) -> None:
        """1 m × 1 m occluder directly above a 2 m × 2 m receiver with sun +Y → 25 % shadow."""
        occ = PanelGeometry(
            center_m=(0.0, 1.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
            width_m=1.0,
            height_m=1.0,
        )
        sun = np.array([0.0, 1.0, 0.0])
        frac = compute_shadow_fraction(sun, [occ], receiver_xz)
        assert frac == pytest.approx(0.25, abs=1e-9)

    def test_shadow_entirely_outside_receiver(self, receiver_xz: PanelGeometry) -> None:
        """Occluder far to the side → shadow does not overlap receiver."""
        occ = PanelGeometry(
            center_m=(10.0, 1.0, 0.0),  # far in +X
            u=(1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
            width_m=1.0,
            height_m=1.0,
        )
        sun = np.array([0.0, 1.0, 0.0])
        frac = compute_shadow_fraction(sun, [occ], receiver_xz)
        assert frac == pytest.approx(0.0)

    def test_two_occluders_union(self, receiver_xz: PanelGeometry) -> None:
        """Two non-overlapping 1 m × 2 m occluders each shadow 25 % → combined 50 %."""
        occ_left = PanelGeometry(
            center_m=(-0.5, 1.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
            width_m=1.0,
            height_m=2.0,
        )
        occ_right = PanelGeometry(
            center_m=(0.5, 1.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
            width_m=1.0,
            height_m=2.0,
        )
        sun = np.array([0.0, 1.0, 0.0])
        frac = compute_shadow_fraction(sun, [occ_left, occ_right], receiver_xz)
        assert frac == pytest.approx(1.0, abs=1e-9)

    def test_perpendicular_panel_shadowing(self) -> None:
        """Canonical perpendicular-mount scenario.

        Solar panel:  2 m × 2 m panel in the YZ plane (facing +X), located at x=0,
                      z ∈ [0, 2].  Represents a wing panel attached to the spacecraft.
        Radiator:     2 m × 2 m panel in the XY plane (facing +Z), located at z=0.

        The solar panel is perpendicular to the radiator.  With sun = (1/√2, 0, 1/√2)
        the panel is between the sun and the left half of the radiator:
        the two z=0 panel corners land exactly at the receiver edge (t=0), and the
        two z=2 corners project inward, giving a 50 % shadow.
        """
        solar_panel = PanelGeometry(
            # YZ plane, facing +X: u=(0,1,0), v=(0,0,1) → n = u×v = (1,0,0)
            center_m=(0.0, 0.0, 1.0),
            u=(0.0, 1.0, 0.0),
            v=(0.0, 0.0, 1.0),
            width_m=2.0,
            height_m=2.0,
        )
        radiator = PanelGeometry(
            # XY plane, facing +Z: u=(1,0,0), v=(0,1,0) → n = u×v = (0,0,1)
            center_m=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            width_m=2.0,
            height_m=2.0,
        )
        # Sun at 45° in the XZ plane: illuminates the radiator (facing +Z).
        # The solar panel (in YZ plane at x=0) is between the sun and the -X half
        # of the radiator, casting a 50 % shadow.
        sun = np.array([1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)])
        frac = compute_shadow_fraction(sun, [solar_panel], radiator)
        assert frac == pytest.approx(0.5, abs=1e-9)
