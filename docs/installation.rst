Installation
============

Requirements
------------

* Python >= 3.10
* See ``pyproject.toml`` for full dependency list

Key dependencies include:

* ``rust-ephem`` - Efficient ephemeris calculations
* ``numpy`` - Numerical computations
* ``matplotlib`` - Visualization
* ``pydantic`` - Configuration validation
* ``shapely`` / ``pyproj`` - Geometric operations

From Source
-----------

To install COASTSim from source:

.. code-block:: bash

   git clone https://github.com/CosmicFrontierLabs/coast-sim.git
   cd coast-sim
   pip install -e .

Development Installation
------------------------

For development, install with the optional development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"
   pre-commit install

This will install additional tools for development:

* **ruff**: Linting and code formatting
* **mypy**: Static type checking
* **pytest**: Testing framework
* **pre-commit**: Git hooks for code quality

Verifying Installation
----------------------

To verify that COASTSim is installed correctly:

.. code-block:: python

   import conops
   print(conops.__version__)

You can also run the test suite:

.. code-block:: bash

   pytest tests/
