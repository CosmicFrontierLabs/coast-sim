"""Rate-continuous quaternion Hermite curves with interval limit checks.

Normalize a cubic quaternion polynomial p(s). For the ECI-to-body convention,
body rate is -2 vec(p' conjugate(p)) / |p|². Differentiating this rational
expression gives body angular acceleration. Bernstein convex-hull bounds check
both envelopes over whole intervals, not just a grid of evaluation times.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from math import comb

import numpy as np
from numpy.polynomial.polynomial import polyder, polymul, polyval

from .vector import _quat_mul


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
    if lower > 1e-12 and upper <= lower**power * (1 + 1e-10):
        return True
    if depth == 14:
        return False
    dl, dr = _split(denominator)
    nl, nr = _split(numerator)
    # An actual violation at the midpoint permits immediate rejection.
    if dl[-1] <= 1e-12 or np.linalg.norm(nl[-1]) > dl[-1] ** power * (1 + 1e-10):
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
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Curve duration must be finite and positive")
        for quaternion in (self.first, self.last):
            if not np.isclose(np.linalg.norm(quaternion), 1.0, rtol=0, atol=1e-10):
                raise ValueError("Curve endpoints must be finite unit quaternions")
        if not np.all(np.isfinite((self.first_rate, self.last_rate))):
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
        dp = polyder(p)
        products_norm = [polymul(component, component) for component in p.T]
        norm = sum(
            (np.pad(term, (0, 7 - len(term))) for term in products_norm),
            start=np.zeros(7),
        )
        # Vector part of p' * conjugate(p); all polynomials use normalized time.
        products = []
        for i, j, k in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
            terms = [
                polymul(dp[:, i], p[:, 0]),
                -polymul(dp[:, 0], p[:, i]),
                -polymul(dp[:, j], p[:, k]),
                polymul(dp[:, k], p[:, j]),
            ]
            products.append(
                sum(
                    (np.pad(term, (0, 6 - len(term))) for term in terms),
                    start=np.zeros(6),
                )
            )
        rate = -2 * np.rad2deg(1.0) / self.duration * np.asarray(products).T
        acceleration = []
        for component in rate.T:
            first = polymul(polyder(component), norm)
            second = polymul(component, polyder(norm))
            size = max(len(first), len(second))
            acceleration.append(
                np.pad(first, (0, size - len(first)))
                - np.pad(second, (0, size - len(second)))
            )
        size = max(map(len, acceleration))
        accel = (
            np.asarray([np.pad(term, (0, size - len(term))) for term in acceleration]).T
            / self.duration
        )
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

    def within_limits(
        self,
        rate_axes: tuple[float, float, float],
        acceleration_axes: tuple[float, float, float],
    ) -> bool:
        norm, rate, acceleration = self._polynomials
        if not all(
            np.all(np.isfinite(values))
            for values in (norm, rate, acceleration, rate_axes, acceleration_axes)
        ):
            return False
        if min(*rate_axes, *acceleration_axes) <= 0:
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
