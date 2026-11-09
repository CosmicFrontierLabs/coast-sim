"""Attitude requests shared by scheduled holds and thruster dumps."""

from typing import Literal

from pydantic import Field, model_validator

from ._base import ConfigModel


class ActivityAttitude(ConfigModel):
    """Inherit the preceding attitude, or request a fixed physical body attitude.

    Fixed coordinates use the same GCRS RA/Dec/roll convention as plan entries.
    They do not infer a spacecraft's thruster mounting or a Sun-relative rule.
    """

    kind: Literal["inherited", "fixed"] = "inherited"
    ra_deg: float | None = Field(default=None, ge=0, lt=360, allow_inf_nan=False)
    dec_deg: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    roll_deg: float | None = Field(default=None, ge=0, lt=360, allow_inf_nan=False)

    @model_validator(mode="after")
    def _coordinates_match_kind(self) -> "ActivityAttitude":
        values = (self.ra_deg, self.dec_deg, self.roll_deg)
        if self.kind == "fixed" and any(v is None for v in values):
            raise ValueError(
                "a fixed activity attitude needs ra_deg, dec_deg and roll_deg"
            )
        if self.kind == "inherited" and any(v is not None for v in values):
            raise ValueError(
                "an inherited activity attitude cannot specify coordinates"
            )
        return self

    @property
    def fixed_pointing(self) -> tuple[float, float, float] | None:
        if self.kind == "inherited":
            return None
        assert (
            self.ra_deg is not None
            and self.dec_deg is not None
            and self.roll_deg is not None
        )
        return self.ra_deg, self.dec_deg, self.roll_deg
