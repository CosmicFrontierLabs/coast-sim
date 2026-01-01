"""Test fixtures for target_queue subsystem tests."""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

from conops import Pointing, Queue


@pytest.fixture
def mock_target():
    """Fixture for a mock target."""
    target = Mock(spec=Pointing)
    target.merit = 100
    target.done = False
    target.exptime = 1000
    target.ss_min = 60
    target.ss_max = 120
    target.slewtime = 10
    target.ra = 0
    target.dec = 0

    def reset_func():
        target.done = False

    target.reset = Mock(side_effect=reset_func)
    target.visible.return_value = True
    target.calc_slewtime = Mock(return_value=10)
    return target


@pytest.fixture
def mock_targets(mock_target):
    """Fixture for a list of mock targets."""
    targets = []
    for i in range(5):
        t = Mock(spec=Pointing)
        t.merit = 100 - i * 10
        t.done = False
        t.exptime = 1000
        t.ss_min = 60
        t.ss_max = 120
        t.slewtime = 10
        t.ra = i * 10
        t.dec = i * 10

        # Create a closure to capture the target instance
        def create_reset_func(target_instance):
            def reset_func():
                target_instance.done = False

            return reset_func

        t.reset = Mock(side_effect=create_reset_func(t))
        t.visible.return_value = True
        t.calc_slewtime = Mock(return_value=10)
        targets.append(t)
    return targets


@pytest.fixture
def mock_config():
    """Fixture for a mock config."""
    config = Mock()
    config.constraint = Mock()
    config.spacecraft_bus = Mock()
    config.attitude_control = Mock()
    config.slew_distance_weight = 0.0
    return config


@pytest.fixture
def queue_instance(mock_targets, mock_config):
    """Fixture for a Queue instance."""

    queue = Queue(config=mock_config)
    for target in mock_targets:
        queue.append(target)
    queue.ephem = Mock()
    # Mocking ephem timestamps for a day in the future
    queue.ephem.timestamp = np.array(
        [datetime.fromtimestamp(1762924800), datetime.fromtimestamp(1763011200)]
    )
    queue.ephem.datetimes = np.array(
        [datetime.fromtimestamp(1762924800), datetime.fromtimestamp(1763011200)]
    )

    queue.utime = 1762924800.0
    return queue
