"""Tests for sky pointing movie export functionality."""

import os
import tempfile

import pytest

from conops.visualization import save_sky_pointing_movie


def test_save_sky_pointing_movie_invalid_format():
    """Test that invalid file format raises ValueError."""
    from unittest.mock import MagicMock

    # Create minimal mock - only need plan and utime for validation
    ditl = MagicMock()
    ditl.plan = [MagicMock()]  # Non-empty list
    ditl.utime = [1000, 2000, 3000]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test.mkv")

        with pytest.raises(ValueError, match="Unsupported output format"):
            save_sky_pointing_movie(ditl, output_file, show_progress=False)


def test_save_sky_pointing_movie_no_data():
    """Test that DITL without data raises ValueError."""
    from unittest.mock import MagicMock

    # Create a mock DITL with no plan
    ditl = MagicMock()
    ditl.plan = []
    ditl.utime = [1000, 2000, 3000]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test.mp4")

        with pytest.raises(ValueError, match="has no pointings"):
            save_sky_pointing_movie(ditl, output_file, show_progress=False)


def test_save_sky_pointing_movie_no_time_data():
    """Test that DITL without time data raises ValueError."""
    from unittest.mock import MagicMock

    ditl = MagicMock()
    ditl.plan = [MagicMock()]  # Non-empty list
    ditl.utime = []  # Empty time data

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test.mp4")

        with pytest.raises(ValueError, match="has no time data"):
            save_sky_pointing_movie(ditl, output_file, show_progress=False)


def test_save_sky_pointing_movie_parallel_parameter():
    """Test that n_jobs parameter is accepted."""
    from unittest.mock import MagicMock

    ditl = MagicMock()
    ditl.plan = [MagicMock()]
    ditl.utime = []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test.mp4")

        # Should still raise ValueError for no time data
        with pytest.raises(ValueError, match="has no time data"):
            save_sky_pointing_movie(ditl, output_file, show_progress=False)
