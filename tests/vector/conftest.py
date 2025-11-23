"""Test fixtures for vector subsystem tests."""

import numpy as np
import pytest


@pytest.fixture
def x_axis():
    return np.array([1.0, 0.0, 0.0])


@pytest.fixture
def y_axis():
    return np.array([0.0, 1.0, 0.0])


@pytest.fixture
def z_axis():
    return np.array([0.0, 0.0, 1.0])


@pytest.fixture
def test_vector():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def ecivec_x():
    return np.array([1, 0, 0])


@pytest.fixture
def ecivec_y():
    return np.array([0, 1, 0])


@pytest.fixture
def ecivec_xyz():
    return np.array([1, 1, 1])


@pytest.fixture
def origin():
    return [0, 0]


@pytest.fixture
def equator_90():
    return [np.pi / 2, 0]


@pytest.fixture
def opposite():
    return [np.pi, 0]


@pytest.fixture
def small_npts():
    return 10


@pytest.fixture
def large_npts():
    return 50


@pytest.fixture
def no_rollover_angles():
    return [10, 20, 30, 40]


@pytest.fixture
def positive_rollover_angles():
    return [350, 355, 5, 10]


@pytest.fixture
def negative_rollover_angles():
    return [10, 5, 355, 350]


@pytest.fixture
def multiple_rollover_angles():
    return [350, 355, 5, 10, 350, 355]


@pytest.fixture
def single_angle():
    return [180]
