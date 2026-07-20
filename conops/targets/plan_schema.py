"""Compatibility shim for the old conops.targets.plan_schema module.

PlanSchema/PlanEntrySchema used to be a separate pydantic mirror of
Plan/PlanEntry, built solely to make plans serializable. Plan and PlanEntry
are now directly serializable, so that mirror layer was removed. This module
re-exports the replacement names under their old names, plus a PlanSchema
subclass carrying the old from_plan() conversion entry point, so that
``from conops.targets.plan_schema import ...`` and
``from conops.targets import ...`` keep working for existing callers.
"""

from .plan import (
    AttitudeSampleSchema,
    AttitudeTimeseriesSchema,
    OrbitStateSampleSchema,
    OrbitStateTimeseriesSchema,
    Plan,
)
from .plan_entry import PlanEntry as PlanEntrySchema


class PlanSchema(Plan):
    """Compatibility subclass for the old conops.targets.plan_schema.PlanSchema.

    Plan is directly serializable now, so PlanSchema no longer needs to be a
    separate mirror type. This subclass exists only to keep the legacy import
    path and the from_plan() conversion entry point working for existing
    callers.
    """

    @classmethod
    def from_plan(cls, plan: Plan) -> "PlanSchema":
        """Build a PlanSchema from an existing Plan instance.

        Validates from the source object's attributes, so all state carries
        over — including the ``exclude=True`` attitude/orbit-state timeseries,
        which save() needs in order to write the sidecar files and link them
        from the plan JSON.
        """
        return cls.model_validate(plan, from_attributes=True)


__all__ = [
    "AttitudeSampleSchema",
    "AttitudeTimeseriesSchema",
    "OrbitStateSampleSchema",
    "OrbitStateTimeseriesSchema",
    "PlanEntrySchema",
    "PlanSchema",
]
