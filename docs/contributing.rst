Contributing
============

Contributions to COASTSim are welcome! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/coast-sim.git
      cd coast-sim

3. Install in development mode with dev dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Set up pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Quality Standards
----------------------

This project uses several tools to maintain code quality:

Ruff
~~~~

For linting and code formatting:

.. code-block:: bash

   ruff check conops/
   ruff format conops/

Mypy
~~~~

For static type checking:

.. code-block:: bash

   mypy conops/

Pytest
~~~~~~

For running tests:

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with coverage
   pytest --cov=conops tests/

   # Run specific test file
   pytest tests/test_battery.py

Pre-commit
~~~~~~~~~~

Pre-commit hooks automatically run before each commit:

* Code formatting with ruff
* Linting checks
* Type checking with mypy
* Trailing whitespace removal
* YAML/JSON validation

Making Changes
--------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass:

   .. code-block:: bash

      pytest tests/

5. Check code quality:

   .. code-block:: bash

      ruff check conops/
      mypy conops/

6. Commit your changes:

   .. code-block:: bash

      git add .
      git commit -m "Description of your changes"

7. Push to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

8. Open a Pull Request on GitHub

Testing Guidelines
------------------

* Write tests for all new functionality
* Maintain or improve code coverage
* Use descriptive test names
* Include docstrings explaining what each test validates
* Use fixtures from ``tests/conftest.py`` when appropriate

Example test structure:

.. code-block:: python

   def test_battery_discharge():
       """Test that battery discharges correctly over time."""
       battery = Battery(capacity=100.0, initial_charge=100.0)
       battery.discharge(10.0, duration=3600)  # 10W for 1 hour

       expected_charge = 100.0 - (10.0 * 3600 / 3600)
       assert abs(battery.current_charge - expected_charge) < 1e-6

Documentation
-------------

* Add docstrings to all public functions, classes, and modules
* Use Google or NumPy style docstrings
* Include examples in docstrings when helpful
* Update documentation when changing functionality
* Build docs locally to verify:

  .. code-block:: bash

     cd docs
     make html
     open _build/html/index.html

Docstring Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_slew_time(angle: float, max_rate: float) -> float:
       """Compute the time required to slew a given angle.

       Args:
           angle: Slew angle in degrees.
           max_rate: Maximum slew rate in degrees per second.

       Returns:
           Time required for the slew in seconds.

       Example:
           >>> compute_slew_time(90.0, 1.0)
           90.0
       """
       return angle / max_rate

Pull Request Guidelines
-----------------------

When submitting a pull request:

* Provide a clear description of the changes
* Reference any related issues
* Ensure all tests pass
* Maintain or improve code coverage
* Follow the existing code style
* Update documentation as needed
* Keep changes focused and atomic

Reporting Issues
----------------

When reporting issues:

* Use a clear, descriptive title
* Describe the expected behavior
* Describe the actual behavior
* Provide steps to reproduce
* Include version information
* Add relevant code snippets or error messages

Feature Requests
----------------

Feature requests are welcome! Please:

* Check if the feature already exists or is planned
* Clearly describe the use case
* Explain why it would be valuable
* Provide examples if possible

Code of Conduct
---------------

* Be respectful and inclusive
* Welcome newcomers and help them get started
* Focus on constructive feedback
* Assume good intentions

Getting Help
------------

If you need help:

* Check the documentation
* Look through existing issues
* Open a new issue with your question
* Be specific about what you're trying to accomplish

License
-------

By contributing to COASTSim, you agree that your contributions will be
licensed under the same license as the project.

Thank You!
----------

Thank you for contributing to COASTSim! Your contributions help make
this project better for everyone.
