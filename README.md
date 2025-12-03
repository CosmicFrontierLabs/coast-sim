<p align="center">
  <img src="docs/_static/coast-sim-logo.png" alt="COASTSim Logo" width="400">
</p>

# COASTSim: ConOps Astronomical Space Telescope Simulator

A Python module for simulating Concept of Operations (ConOps) for space telescopes and astronomical spacecraft missions.

## Overview

COASTSim is a comprehensive simulation framework designed to model and analyze the operational behavior of space-based astronomical observatories. It enables mission planners and engineers to simulate Day-In-The-Life (DITL) scenarios, evaluate spacecraft performance, optimize observation schedules, and validate operational constraints before launch.

The package is organized into logical submodules for better maintainability and clearer separation of concerns: `common` (utilities), `config` (configuration), `ditl` (simulation), `schedulers` (planning), `targets` (observations), and `simulation` (core components).

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
from conops import Config, QueueDITL
from rust_ephem import TLEEphemeris

# Load configuration
config = Config.from_json_file("example_config.json")

# Set simulation period
begin = datetime(2025, 11, 1)
end = begin + timedelta(days=1)

# Compute orbit ephemeris
ephemeris = TLEEphemeris(tle="example.tle", begin=begin, end=end)

# Run DITL simulation
ditl = QueueDITL(config=config)
ditl.ephem = ephemeris
ditl.begin = begin
ditl.end = end
ditl.calc()

# Analyze results
ditl.plot()
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

## Visualization

COASTSim includes a set of plotting utilities to visualize DITL simulations and telemetry. Each visualization function accepts an optional `config` parameter. If omitted, the plots will use `ditl.config.visualization` if present, or reasonable defaults defined by `VisualizationConfig`.

Key plotting utilities are in the `conops.visualization` module:

- `plot_ditl_telemetry()` — show RA, Dec, ACS mode, battery, power, and ObsIDs in a multi-panel timeline
- `plot_data_management_telemetry()` — recorder volume/fill fractions, cumulative data generated/downlinked, and alert timelines
- `plot_acs_mode_distribution()` — pie chart of time spent in each ACS mode; supports custom mode colors
- `plot_ditl_timeline()` — full timeline view with orbit numbers, observations, slews, SAA, and eclipses
- `plot_sky_pointing()` — interactive mollweide sky projection showing current pointing, patterns, and constraints

### How to customize fonts and colors

Custom visualization settings are controlled via the `VisualizationConfig` Pydantic model (`conops.config.visualization`). Important fields include:

- `font_family` (str): font used for titles, labels, and legends (default: `Helvetica`)
- `title_font_size`, `label_font_size`, `legend_font_size`, `tick_font_size` (int)
- `mode_colors` (dict[str, str]): color mapping for ACS modes used in plots such as `plot_acs_mode_distribution`

### Example — customizing fonts and colors

```python
from conops import Config, QueueDITL
from conops.visualization import plot_ditl_telemetry, plot_acs_mode_distribution
from datetime import datetime, timedelta
from rust_ephem import TLEEphemeris

cfg = Config.from_json_file("examples/example_config.json")
cfg.visualization.font_family = "Helvetica"
cfg.visualization.title_font_size = 14
cfg.visualization.mode_colors["SAA"] = "#800080"  # purple

begin = datetime.utcnow()
end = begin + timedelta(days=1)
ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)
ditl = QueueDITL(config=cfg)
ditl.ephem = ephem
ditl.calc()

# Render using the resolved visualization config
fig, axes = plot_ditl_telemetry(ditl)
fig2, ax2 = plot_acs_mode_distribution(ditl, config=cfg.visualization)
```

Tip: Set `cfg.visualization` to keep consistent fonts and colors across every plot end-to-end. Note that the requested font must be installed on your system; Matplotlib may fallback to another font (e.g., DejaVu Sans or Arial) if the requested family isn't available. To guarantee a specific font, use a `FontProperties` object with the font file path.

## Core Components

### Configuration (`conops.config`)

Pydantic-based configuration management for spacecraft parameters, instruments, and operational constraints.

### DITL Simulation (`conops.ditl`)

Day-In-The-Life simulation classes including DITL, DITLMixin, and QueueDITL for comprehensive timeline simulation.

### Scheduling (`conops.schedulers`)

Target observation queue management and intelligent scheduling algorithms (DumbScheduler, DumbQueueScheduler).

### Targets (`conops.targets`)

Target management classes including Pointing, Queue, Plan, PlanEntry, and Plan for observation planning.

### Simulation (`conops.simulation`)

Core simulation components including ACS (Attitude Control System), slew modeling, SAA handling, emergency charging, and other simulation utilities.

### Common Utilities (`conops.common`)

Shared utilities, enums (ACSMode, ChargeState, ACSCommandType), vector mathematics, and common functions.

## Module Structure

```text
conops/
├── __init__.py              # Package initialization and exports
├── _version.py              # Version information
├── common/                  # Shared utilities and common functions
│   ├── __init__.py
│   ├── common.py
│   ├── enums.py
│   └── vector.py
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── constraint.py
│   ├── battery.py
│   ├── solar_panel.py
│   ├── instrument.py
│   └── spacecraft_bus.py
├── ditl/                    # Day-In-The-Life simulation
│   ├── __init__.py
│   ├── ditl.py
│   ├── ditl_mixin.py
│   └── queue_ditl.py
├── schedulers/              # Scheduling algorithms
│   ├── __init__.py
│   ├── scheduler.py
│   └── queue_scheduler.py
├── targets/                 # Target management and planning
│   ├── __init__.py
│   ├── pointing.py
│   ├── plan.py
│   ├── plan_entry.py
│   └── target_queue.py
└── simulation/              # Core simulation components
    ├── __init__.py
    ├── acs.py
    ├── acs_command.py
    ├── emergency_charging.py
    ├── passes.py
    ├── roll.py
    ├── saa.py
    ├── slew.py
    └── ephemeris.py
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
