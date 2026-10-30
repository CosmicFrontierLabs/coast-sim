import numpy as np
import numpy.typing as npt
from enum import IntEnum
from pymsis import msis00f as msis00f, msis20f as msis20f, msis21f as msis21f
from pymsis.utils import get_f107_ap as get_f107_ap

class Variable(IntEnum):
    MASS_DENSITY = 0
    N2 = 1
    O2 = 2
    O = 3
    HE = 4
    H = 5
    AR = 6
    N = 7
    ANOMALOUS_O = 8
    NO = 9
    TEMPERATURE = 10

def calculate(
    dates: npt.ArrayLike,
    lons: npt.ArrayLike,
    lats: npt.ArrayLike,
    alts: npt.ArrayLike,
    f107s: npt.ArrayLike | None = None,
    f107as: npt.ArrayLike | None = None,
    aps: npt.ArrayLike | None = None,
    *,
    options: list[float] | None = None,
    version: float | str = 2.1,
    interpolate_indices: bool = False,
    **kwargs: float,
) -> npt.NDArray[np.float32]: ...

run = calculate

def create_options(
    f107: float = 1,
    time_independent: float = 1,
    symmetrical_annual: float = 1,
    symmetrical_semiannual: float = 1,
    asymmetrical_annual: float = 1,
    asymmetrical_semiannual: float = 1,
    diurnal: float = 1,
    semidiurnal: float = 1,
    geomagnetic_activity: float = 1,
    all_ut_effects: float = 1,
    longitudinal: float = 1,
    mixed_ut_long: float = 1,
    mixed_ap_ut_long: float = 1,
    terdiurnal: float = 1,
) -> list[float]: ...
def create_input(
    dates: npt.ArrayLike,
    lons: npt.ArrayLike,
    lats: npt.ArrayLike,
    alts: npt.ArrayLike,
    f107s: npt.ArrayLike | None = None,
    f107as: npt.ArrayLike | None = None,
    aps: npt.ArrayLike | None = None,
    interpolate_indices: bool = False,
) -> tuple[tuple[int, ...], npt.NDArray[np.float32]]: ...
