# COASTSim: ConOps Astronomical Space Telescope Simulator

A Python module for simulating Concept of Operations (ConOps) for space telescopes and astronomical spacecraft missions.

## Overview

COASTSim is a comprehensive simulation framework designed to model and analyze the operational behavior of space-based astronomical observatories. It enables mission planners and engineers to simulate Day-In-The-Life (DITL) scenarios, evaluate spacecraft performance, optimize observation schedules, and validate operational constraints before launch.

## Key Features

- **Spacecraft Bus Simulation**: Model power systems, attitude control, and thermal management
- **Orbit Propagation**: TLE-based ephemeris computation and orbit tracking
- **Observation Planning**: Target queue management and scheduling algorithms
- **Instrument Modeling**: Multi-instrument configurations with power and pointing requirements
- **Constraint Management**: Sun, Moon, Earth limb, and custom geometric constraints
- **Power Budget Analysis**: Solar panel modeling, battery management, and emergency charging scenarios
- **Ground Station Passes**: Communication window calculations and data downlink planning
- **Attitude Control System**: Slew modeling, pointing accuracy, and settle time simulation
- **South Atlantic Anomaly (SAA) Avoidance**: Radiation belt constraint handling
- **DITL Generation**: Comprehensive day-in-the-life timeline simulation

## Installation

### From Source

```bash
git clone https://github.com/CosmicFrontierLabs/coast-sim.git
cd coast-sim
pip install -e .
```

### Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependency list

Key dependencies include:

- `astropy` - Astronomical calculations and coordinate systems
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `rust-ephem` - Efficient ephemeris calculations
- `pydantic` - Configuration validation
- `shapely` - Geometric operations

## Quick Start

### Basic DITL Simulation

```python
from datetime import datetime, timedelta
from conops.config import Config
from conops.queue_ditl import QueueDITL
from conops.ephemeris import compute_tle_ephemeris

# Load configuration
config = Config.from_json("example_config.json")

# Set simulation period
begin = datetime(2025, 11, 1)
end = begin + timedelta(days=1)

# Compute orbit ephemeris
ephemeris = compute_tle_ephemeris("example.tle", begin, end)

# Run DITL simulation
ditl = QueueDITL(config, ephemeris, begin, end)
ditl.run()

# Analyze results
ditl.plot_timeline()
ditl.print_statistics()
```

### Configuration-Based Approach

Create a JSON configuration file defining your spacecraft parameters:

```json
{
    "name": "My Space Telescope",
    "spacecraft_bus": {
        "power_draw": {
            "nominal_power": 50.0,
            "peak_power": 300.0
        },
        "attitude_control": {
            "slew_acceleration": 0.01,
            "max_slew_rate": 1.0
        }
    },
    "solar_panel": {
        "panels": [...]
    },
    "instruments": {
        "instruments": [...]
    }
}
```

## Examples

Comprehensive examples are provided in the `examples/` directory as Jupyter notebooks:

- **`Example_Spacecraft_DITL.ipynb`**: Complete spacecraft DITL simulation with custom spacecraft configuration, including power modeling, attitude control, and observation scheduling
- **`Example_DITL_from_JSON.ipynb`**: Simplified workflow using JSON configuration files for quick simulations

To run the examples:

```bash
cd examples
jupyter notebook
```

## Core Components

### Configuration (`config.py`)

Pydantic-based configuration management for spacecraft parameters, instruments, and operational constraints.

### Ephemeris (`ephemeris.py`)

Orbit propagation using TLE data, providing spacecraft position and velocity vectors.

### Queue Scheduler (`queue_scheduler.py`, `queue_ditl.py`)

Target observation queue management and intelligent scheduling algorithms.

### Pointing (`pointing.py`)

Spacecraft pointing representation with coordinate transformations.

### Attitude Control System (`acs.py`, `spacecraft_bus.py`)

Slew modeling with realistic acceleration profiles, settle times, and pointing accuracy.

### Battery & Power (`battery.py`, `solar_panel.py`)

Power generation modeling, battery state tracking, and emergency charging scenarios.

### Instruments (`instrument.py`)

Multi-instrument support with individual power profiles and operational modes.

### Constraints (`constraint.py`)

Sun angle, Moon avoidance, Earth limb exclusion, and custom geometric constraints.

### Ground Stations (`groundstation.py`, `passes.py`)

Communication pass prediction and ground station network management.

### SAA Handling (`saa.py`)

South Atlantic Anomaly avoidance for radiation-sensitive instruments.

## Module Structure

```text
conops/
├── __init__.py           # Package initialization
├── config.py             # Configuration management
├── ditl.py               # Basic DITL simulation
├── queue_ditl.py         # Queue-based DITL
├── scheduler.py          # Scheduling algorithms
├── queue_scheduler.py    # Queue scheduling
├── ephemeris.py          # Orbit propagation
├── pointing.py           # Pointing and coordinates
├── acs.py                # Attitude control
├── spacecraft_bus.py     # Spacecraft bus modeling
├── battery.py            # Battery management
├── solar_panel.py        # Solar power generation
├── instrument.py         # Instrument modeling
├── constraint.py         # Observational constraints
├── groundstation.py      # Ground station network
├── passes.py             # Pass calculations
├── saa.py                # SAA handling
├── slew.py               # Slew modeling
├── roll.py               # Roll angle optimization
├── vector.py             # Vector mathematics
├── common.py             # Common utilities
└── constants.py          # Physical constants
```

## Testing

The project includes a comprehensive test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=conops tests/
```

## Documentation

Full API documentation is available in the `docs/` directory and can be built using Sphinx.

### Building Documentation

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build the HTML documentation:

```bash
cd docs
make html
```

The built documentation will be available in `docs/_build/html/index.html`.

For more information, see `docs/README.md`.

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Code Quality

This project uses:

- **ruff**: Linting and code formatting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

## Use Cases

- **Mission Planning**: Evaluate operational scenarios before launch
- **Performance Analysis**: Assess power budgets, observation efficiency, and data return
- **Schedule Optimization**: Test different scheduling algorithms and target prioritization
- **Constraint Validation**: Verify observational constraints are satisfied
- **Trade Studies**: Compare different spacecraft configurations and operational strategies
- **Operations Training**: Simulate realistic mission scenarios for operations teams

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code quality checks succeed
5. Submit a pull request

## License

See LICENSE file for details.

## Author

**Jamie A. Kennea**
Email: <jak51@psu.edu>

## Acknowledgments

Developed for space telescope mission planning and operational analysis.

## Project Status

Development Status: Production/Stable

---

For questions, issues, or feature requests, please open an issue on GitHub.
