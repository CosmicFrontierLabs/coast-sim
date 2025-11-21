# COASTSim Documentation

This directory contains the Sphinx documentation for COASTSim.

## Building the Documentation

### Install Dependencies

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install from the requirements file:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Other Build Formats

Sphinx supports multiple output formats:

```bash
make latexpdf  # Build PDF documentation
make epub      # Build EPUB documentation
make man       # Build man pages
make help      # See all available formats
```

### Clean Build

To clean the build directory:

```bash
make clean
```

### Auto-rebuild During Development

For live reloading during documentation development, you can use `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then open http://127.0.0.1:8000 in your browser. The documentation will automatically rebuild when you save changes.

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation index
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `examples.rst` - Examples and tutorials
- `contributing.rst` - Contributing guidelines
- `api/` - API reference documentation
  - `modules.rst` - API reference index
  - `conops.*.rst` - Individual module documentation

## Documentation Style

- Write in reStructuredText (.rst) format
- Use Google or NumPy style docstrings in Python code
- Include code examples where helpful
- Keep documentation up to date with code changes
- Use appropriate Sphinx directives for notes, warnings, etc.

## Viewing Online

Once published, the documentation will be available at the project's documentation site.
