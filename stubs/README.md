# Local dependency stubs

`mypy.ini` adds this directory to mypy's search path, so optional dependencies
can be type-checked without being installed in the lint environment.

The `ppigrf` 2.1.0 stubs were generated with mypy's `stubgen`:

```sh
.venv/bin/stubgen -p ppigrf -o stubs
```

The three functions exported by `ppigrf` and `ppigrf.ppigrf.igrf_V` have manually
refined annotations for NumPy inputs, dates, and results. The internal helpers
retain draft types, marked explicitly with `_typeshed.Incomplete`.
Regenerating overwrites these refinements; review the diff
and restore the annotations after updating to a new upstream version.

These stubs are for development in this repository and do not install `ppigrf`
or change its optional runtime dependency status.
