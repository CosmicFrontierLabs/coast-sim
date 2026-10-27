"""Named, independent stores and explicit data/link routing.

Null definitions are useful while collecting hardware inputs. They are not
unlimited recorders: execution requires a complete definition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator

from ..common.enums import ACSMode
from ._base import ConfigModel
from .data_generator import ScienceData
from .recorder import OnboardRecorder

if TYPE_CHECKING:
    from .config import MissionConfig


class DataStore(ConfigModel):
    owner: Literal["payload", "spacecraft_bus"]
    installed_capacity_gb: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    recorder: OnboardRecorder | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _capacity(self) -> DataStore:
        self.check_capacity()
        return self

    def check_capacity(self) -> None:
        if self.recorder is not None:
            missing = {
                "capacity_gb",
                "current_volume_gb",
            } - self.recorder.model_fields_set
            if missing:
                raise ValueError(
                    "named recorders require explicit "
                    + ", ".join(sorted(missing))
                    + "; use recorder: null for unresolved hardware"
                )
        if self.recorder is not None and not all(
            math.isfinite(v)
            for v in (self.recorder.capacity_gb, self.recorder.current_volume_gb)
        ):
            raise ValueError("recorder capacity and initial volume must be finite")
        if (
            self.recorder is not None
            and self.installed_capacity_gb is not None
            and self.recorder.capacity_gb > self.installed_capacity_gb
        ):
            raise ValueError("usable recorder capacity exceeds installed capacity")


class DownlinkRoute(ConfigModel):
    """Reserved shares of one physical downlink, not a full link per store.

    Unused shares are not reassigned. This conservative policy is identical
    in planning and execution. A station has one active link in this model.
    """

    station: str
    band: Literal["UHF", "S", "X", "Ka", "Ku", "L", "C", "K"]
    shares: dict[str, float]

    @field_validator("station")
    @classmethod
    def _normalize_station(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _shares(self) -> DownlinkRoute:
        self.check_shares()
        return self

    def check_shares(self) -> None:
        if not self.shares or any(
            not math.isfinite(v) or not 0 < v <= 1 for v in self.shares.values()
        ):
            raise ValueError("downlink shares must be finite and in (0, 1]")
        if sum(self.shares.values()) > 1.0 + 1e-12:
            raise ValueError("stores cannot reserve more than one physical downlink")


class DataStorage(ConfigModel):
    stores: dict[str, DataStore]
    bus_housekeeping: str | None = None
    instrument_science: dict[str, str | None] = Field(default_factory=dict)
    instrument_housekeeping: dict[str, str | None] = Field(default_factory=dict)
    downlinks: list[DownlinkRoute] | None = None
    """None is unresolved; [] explicitly models no downlink."""

    @model_validator(mode="after")
    def _references(self) -> DataStorage:
        self.check_references()
        return self

    def check_references(self) -> None:
        if not self.stores or any(not name or "." in name for name in self.stores):
            raise ValueError("stores need nonempty identifiers without dots")
        recorder_owners: dict[int, str] = {}
        for name, store in self.stores.items():
            if store.recorder is None:
                continue
            identity = id(store.recorder)
            if identity in recorder_owners:
                raise ValueError(
                    f"stores {recorder_owners[identity]!r} and {name!r} use the "
                    "same recorder instance; create a separate OnboardRecorder "
                    "for each store"
                )
            recorder_owners[identity] = name
        destinations = [
            self.bus_housekeeping,
            *self.instrument_science.values(),
            *self.instrument_housekeeping.values(),
        ]
        stations = set()
        for link in self.downlinks or []:
            link.check_shares()
            if link.station in stations:
                raise ValueError("a station must have one shared downlink route")
            stations.add(link.station)
            destinations.extend(link.shares)
        if any(name is not None and name not in self.stores for name in destinations):
            raise ValueError("data route references an unknown store")

    def unresolved(self, config: MissionConfig) -> list[str]:
        # Pydantic validates assignment, but edits inside dictionaries also need checks.
        self.check_references()
        for store in self.stores.values():
            store.check_capacity()
        missing = [
            f"stores.{name}.recorder"
            for name, store in self.stores.items()
            if store.recorder is None
        ]
        if self.bus_housekeeping is None:
            missing.append("bus_housekeeping")
        names = {instrument.name for instrument in config.payload.instruments}
        for label in ("instrument_science", "instrument_housekeeping"):
            routes = getattr(self, label)
            missing.extend(
                f"{label}.{name}" for name in sorted(names) if routes.get(name) is None
            )
            unknown = set(routes) - names
            if unknown:
                raise ValueError(
                    f"{label} references unknown instruments: {sorted(unknown)}"
                )
        if self.downlinks is None:
            missing.append("downlinks")
        for link in self.downlinks or []:
            try:
                station = config.ground_stations.get(link.station)
            except KeyError as error:
                raise ValueError(f"unknown downlink station: {link.station}") from error
            comms = config.spacecraft_bus.communications
            if (
                comms is None
                or comms.get_downlink_rate(link.band) <= 0
                or (station.get_downlink_rate(link.band) or 0) <= 0
            ):
                raise ValueError(
                    f"{link.station} has no compatible {link.band} downlink"
                )
        return missing

    def require_ready(self, config: MissionConfig) -> None:
        missing = self.unresolved(config)
        if missing:
            raise ValueError(
                "Unresolved data storage configuration: " + ", ".join(missing)
            )

    def downlink_rate(
        self,
        config: MissionConfig,
        store: str,
        station: str,
        range_km: float | None = None,
    ) -> float:
        for link in self.downlinks or []:
            if link.station == station:
                ground = config.ground_stations.get(station)
                comms = config.spacecraft_bus.communications
                if comms is None:
                    return 0.0
                rate = min(
                    ground.get_downlink_rate(link.band, range_km) or 0.0,
                    comms.get_downlink_rate(link.band, range_km),
                )
                return rate * link.shares.get(store, 0.0)
        return 0.0


class RecorderView:
    """The existing recorder algorithms' inputs, restricted to one store."""

    def __init__(self, config: MissionConfig, name: str):
        self.mission = config
        self.name = name
        storage = config.data_storage
        assert storage is not None
        recorder = storage.stores[name].recorder
        assert recorder is not None
        self.recorder = recorder
        self.spacecraft_bus = config.spacecraft_bus
        self.payload = config.payload
        self.ground_stations = config.ground_stations

    def housekeeping_generated(
        self, seconds: float, *, mode: ACSMode | None = None
    ) -> float:
        storage = self.mission.data_storage
        assert storage is not None
        result = (
            self.spacecraft_bus.data_generated(seconds, mode=mode)
            if storage.bus_housekeeping == self.name
            else 0.0
        )
        for instrument in self.payload.instruments:
            if storage.instrument_housekeeping.get(instrument.name) == self.name:
                result += instrument.data_generation.housekeeping.data_generated(
                    seconds, mode=mode
                )
        return result

    def housekeeping_while_waiting(self, seconds: float) -> float:
        from .config import housekeeping_while_waiting

        return housekeeping_while_waiting(self, seconds)

    def science_generated(
        self,
        seconds: float,
        science: Mapping[str, ScienceData] | None = None,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> float:
        storage = self.mission.data_storage
        assert storage is not None
        # Reject an override that would otherwise silently drop science data.
        unknown = set(science or {}) - {
            instrument.name for instrument in self.payload.instruments
        }
        if unknown:
            raise ValueError(
                f"science overrides reference unknown instruments: {sorted(unknown)}"
            )
        return sum(
            data.data_generated(seconds)
            for instrument in self.payload.instruments
            if storage.instrument_science.get(instrument.name) == self.name
            if (
                data := self.payload._science_of(
                    instrument, science or {}, instrument_modes
                )
            )
            is not None
        )

    def downlink_rate(self, station: str, range_km: float | None = None) -> float:
        storage = self.mission.data_storage
        assert storage is not None
        return storage.downlink_rate(self.mission, self.name, station, range_km)


class RecorderSummary:
    """Read-only summary for legacy displays; never used for admission."""

    def __init__(self, views: Mapping[str, MissionConfig | RecorderView]):
        self.recorders = {name: view.recorder for name, view in views.items()}

    @property
    def current_volume_gb(self) -> float:
        return sum(recorder.current_volume_gb for recorder in self.recorders.values())

    @property
    def capacity_gb(self) -> float:
        return sum(recorder.capacity_gb for recorder in self.recorders.values())

    def get_fill_fraction(self) -> float:
        return max(recorder.get_fill_fraction() for recorder in self.recorders.values())

    def get_alert_level(self) -> int:
        return max(recorder.get_alert_level() for recorder in self.recorders.values())
