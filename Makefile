.PHONY: test-acs test install dev-install lint format typecheck

# MANDATORY test for ACS changes - uses 12h/1000-target config
test-acs:
	pytest tests/acs/test_ditl_adcs_debug.py::TestDITLADCSMomentumWarnings::test_no_pointing_or_momentum_warnings -v

# Run all tests
test:
	pytest tests/ -v

# Install package
install:
	pip install -e .

# Install with dev dependencies
dev-install:
	pip install -e ".[dev]"

# Lint code
lint:
	ruff check conops/

# Format code
format:
	ruff format conops/

# Type check
typecheck:
	mypy --strict conops/
