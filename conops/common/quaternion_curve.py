"""Rate-continuous quaternion Hermite curves with interval limit checks.

Normalize a cubic quaternion polynomial p(s). For the ECI-to-body convention,
body rate is -2 vec(p' conjugate(p)) / |p|². Differentiating this rational
expression gives body angular acceleration. Bernstein convex-hull bounds check
both envelopes over whole intervals, not just a grid of evaluation times.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import comb, isclose, isfinite

import numpy as np
from numpy.polynomial.polynomial import polyval

from .numerics import ROUNDOFF
from .vector import _quat_mul

_RELATIVE_TOLERANCE = 1e-10
"""The relative slack that absorbs floating-point roundoff in the
coefficients when a bound is certified."""
_MAX_SUBDIVISION_DEPTH = 14
"""How many times an interval is halved before a bound is given up on."""


@lru_cache(maxsize=16)
def _bernstein_matrix(degree: int) -> np.ndarray:
    return np.asarray(
        [
            [comb(k, i) / comb(degree, i) if i <= k else 0 for i in range(degree + 1)]
            for k in range(degree + 1)
        ]
    )


def _split(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide Bernstein control coefficients at the interval midpoint."""
    left, right = np.empty_like(coefficients), np.empty_like(coefficients)
    left[0], right[-1] = coefficients[0], coefficients[-1]
    for index in range(1, len(coefficients)):
        coefficients = (coefficients[:-1] + coefficients[1:]) / 2
        left[index], right[-index - 1] = coefficients[0], coefficients[-1]
    return left, right


def _certify(
    denominator: np.ndarray, numerator: np.ndarray, power: int, depth: int = 0
) -> bool:
    """Prove ||numerator|| <= denominator**power via interval subdivision.

    Failure to establish a bound is rejection, never permission to exceed it.
    The small relative tolerance absorbs floating-point coefficient roundoff.
    """
    lower = float(np.min(denominator))
    upper = float(np.max(np.linalg.norm(numerator, axis=1)))
    if lower > ROUNDOFF and upper <= lower**power * (1 + _RELATIVE_TOLERANCE):
        return True
    if depth == _MAX_SUBDIVISION_DEPTH:
        return False
    dl, dr = _split(denominator)
    nl, nr = _split(numerator)
    # An actual violation at the midpoint permits immediate rejection.
    if dl[-1] <= ROUNDOFF or np.linalg.norm(nl[-1]) > dl[-1] ** power * (
        1 + _RELATIVE_TOLERANCE
    ):
        return False
    return _certify(dl, nl, power, depth + 1) and _certify(dr, nr, power, depth + 1)


@dataclass(frozen=True)
class QuaternionHermite:
    first: tuple[float, float, float, float]
    last: tuple[float, float, float, float]
    first_rate: tuple[float, float, float]
    last_rate: tuple[float, float, float]
    duration: float

    def __post_init__(self) -> None:
        if not isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Curve duration must be finite and positive")
        for quaternion in (self.first, self.last):
            if not isclose(
                float(np.linalg.norm(quaternion)), 1.0, rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError("Curve endpoints must be finite unit quaternions")
        if not all(
            isfinite(value)
            for rate in (self.first_rate, self.last_rate)
            for value in rate
        ):
            raise ValueError("Curve endpoint rates must be finite")

    @cached_property
    def _coefficients(self) -> np.ndarray:
        q0, q1 = np.asarray(self.first), np.asarray(self.last)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        v0 = (
            -0.5
            * self.duration
            * _quat_mul(np.r_[0.0, np.deg2rad(self.first_rate)], q0)
        )
        v1 = (
            -0.5 * self.duration * _quat_mul(np.r_[0.0, np.deg2rad(self.last_rate)], q1)
        )
        return np.asarray(
            [q0, v0, 3 * (q1 - q0) - 2 * v0 - v1, 2 * (q0 - q1) + v0 + v1]
        )

    @cached_property
    def _polynomials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self._coefficients
        # Ascending-power coefficients have fixed sizes: p is cubic, |p|²
        # degree six, and the rate numerator degree five. Direct convolution
        # retains trailing zeros, avoiding polynomial trimming and repadding.
        dp = p[1:] * np.arange(1, 4)[:, None]
        norm = np.zeros(7)
        for component in p.T:
            norm += np.convolve(component, component)
        # Vector part of p' * conjugate(p); all polynomials use normalized time.
        rate = np.empty((6, 3))
        for i, j, k in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
            rate[:, i - 1] = (
                np.convolve(dp[:, i], p[:, 0])
                - np.convolve(dp[:, 0], p[:, i])
                - np.convolve(dp[:, j], p[:, k])
                + np.convolve(dp[:, k], p[:, j])
            )
        rate *= -2 * np.rad2deg(1.0) / self.duration
        dnorm = norm[1:] * np.arange(1, 7)
        drate = rate[1:] * np.arange(1, 6)[:, None]
        accel = np.empty((11, 3))
        for axis in range(3):
            accel[:, axis] = np.convolve(drate[:, axis], norm) - np.convolve(
                rate[:, axis], dnorm
            )
        accel /= self.duration
        return norm, rate, accel

    def evaluate(self, fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = polyval(fraction, self._coefficients)
        norm, rate, acceleration = self._polynomials
        denominator = float(polyval(fraction, norm))
        return (
            q / np.sqrt(denominator),
            polyval(fraction, rate) / denominator,
            polyval(fraction, acceleration) / denominator**2,
        )

    @cached_property
    def _rotation_polynomial(self) -> np.ndarray:
        """Numerator of R_eci_to_body; the denominator is |p|²."""
        p = self._coefficients
        scalar = np.convolve(p[:, 0], p[:, 0]) - sum(
            np.convolve(component, component) for component in p[:, 1:].T
        )
        rotation = scalar[:, None, None] * np.eye(3)
        for i in range(3):
            for j in range(3):
                rotation[:, i, j] += 2 * np.convolve(p[:, i + 1], p[:, j + 1])
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            cross = 2 * np.convolve(p[:, 0], p[:, k + 1])
            rotation[:, i, j] -= cross
            rotation[:, j, i] += cross
        return rotation

    def within_momentum_limit(
        self,
        first: float,
        last: float,
        inertia: np.ndarray,
        momentum_eci: np.ndarray,
        capacity: float,
    ) -> bool:
        """Bound actuator demand over a subinterval, not just its endpoints.

        ``momentum_eci`` contains three ascending-power vector coefficients in
        *local* interval time [0, 1]. A quadratic represents the integral of
        linearly interpolated external torque, matching trapezoidal integration.
        The rotating inertial momentum and I omega are combined before bounding
        their norm, retaining off-axis inertia and vector cancellation.
        """
        if not (0 <= first < last <= 1 and isfinite(capacity) and capacity > 0):
            return False
        # Compose each degree-six polynomial with s = first + (last-first) u.
        transform = np.asarray(
            [
                [
                    comb(j, i) * first ** (j - i) * (last - first) ** i
                    if j >= i
                    else 0.0
                    for j in range(7)
                ]
                for i in range(7)
            ]
        )
        norm, rate, _ = self._polynomials
        denominator = transform @ norm
        rotation = np.einsum("ij,jkl->ikl", transform, self._rotation_polynomial)
        numerator = np.zeros((9, 3))
        for power, vector in enumerate(momentum_eci):
            numerator[power : power + 7] += rotation @ vector
        numerator[:7] -= np.deg2rad(transform[:, :6] @ rate) @ inertia.T
        if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
            return False
        return _certify(
            _bernstein_matrix(6) @ denominator,
            _bernstein_matrix(8) @ (numerator / capacity),
            1,
        )

    def within_limits(
        self,
        rate_axes: tuple[float, float, float],
        acceleration_axes: tuple[float, float, float],
    ) -> bool:
        if not all(
            isfinite(value) and value > 0 for value in (*rate_axes, *acceleration_axes)
        ):
            return False
        norm, rate, acceleration = self._polynomials
        if not all(np.isfinite(values).all() for values in (norm, rate, acceleration)):
            return False
        denominator = _bernstein_matrix(len(norm) - 1) @ norm
        for coefficients, axes, power in (
            (rate, rate_axes, 1),
            (acceleration, acceleration_axes, 2),
        ):
            bounds = _bernstein_matrix(len(coefficients) - 1) @ (coefficients / axes)
            if not _certify(denominator, bounds, power):
                return False
        return True
