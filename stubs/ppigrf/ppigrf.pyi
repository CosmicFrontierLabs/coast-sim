# ruff: noqa: N802, N803
# Upstream function and parameter names are preserved for keyword arguments.
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
from _typeshed import Incomplete

_DateLike = (
    datetime
    | np.datetime64
    | Sequence[datetime | np.datetime64]
    | npt.NDArray[np.datetime64]
    | npt.NDArray[np.object_]
)
_FieldComponents = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]
_Angle = np.float64 | npt.NDArray[np.float64]

basepath: str
shc_fn: str
shc_fn_igrf13: str
shc_fn_igrf14: str
RE: float
WGS84_e2: float
WGS84_a: float

def is_leapyear(year: Incomplete) -> Incomplete: ...
def yearfrac_to_datetime(fracyear: Incomplete) -> Incomplete: ...
def get_legendre(theta: Incomplete, keys: Incomplete) -> Incomplete: ...
def read_shc(filename: str = ...) -> Incomplete: ...
def geod2geoc(
    gdlat: Incomplete, height: Incomplete, Bn: Incomplete, Bu: Incomplete
) -> Incomplete: ...
def geoc2geod(
    theta: Incomplete, r: Incomplete, B_th: Incomplete, B_r: Incomplete
) -> Incomplete: ...
def igrf_gc(
    r: npt.ArrayLike,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    date: _DateLike,
    coeff_fn: str = ...,
    min_degree: int = 1,
    max_degree: int = 13,
) -> _FieldComponents: ...
def igrf(
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    h: npt.ArrayLike,
    date: _DateLike,
    coeff_fn: str = ...,
    min_degree: int = 1,
    max_degree: int = 13,
) -> _FieldComponents: ...
def igrf_V(
    r: npt.ArrayLike,
    theta: npt.ArrayLike,
    phi: npt.ArrayLike,
    date: _DateLike,
    coeff_fn: str = ...,
    min_degree: int = 1,
    max_degree: int = 13,
) -> npt.NDArray[np.float64]: ...
def get_inclination_declination(
    Be: float | npt.NDArray[np.float64],
    Bn: float | npt.NDArray[np.float64],
    Bu: float | npt.NDArray[np.float64],
    degrees: bool = True,
) -> tuple[_Angle, _Angle]: ...
