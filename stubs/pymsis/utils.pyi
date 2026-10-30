import numpy as np
import numpy.typing as npt
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path

@dataclass
class _SpaceWeatherState:
    path: Path
    data: dict[str, npt.NDArray[Incomplete]] | None = ...

def use_space_weather_file(file: str | Path | None = None) -> None: ...
def download_f107_ap() -> None: ...
def get_f107_ap(
    dates: npt.ArrayLike, interpolate: bool = False
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]: ...
