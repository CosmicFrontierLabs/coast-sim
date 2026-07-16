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
    separate mirror type. This subclass exists only to keep the legacy
    from_plan() conversion entry point working for existing callers.
    """

    @classmethod
    def from_plan(cls, plan: Plan) -> "PlanSchema":
        """Build a PlanSchema from an existing Plan instance.

        Round-trips through JSON (matching Plan.save()/Plan.load()) so the
        result is fully independent of ``plan`` and drops the same
        runtime-only fields (config, ephem, constraint, acs_config, ...)
        that the old PlanEntrySchema never carried either.
        """
        return cls.model_validate(plan.model_dump(mode="json"))


__all__ = [
    "AttitudeSampleSchema",
    "AttitudeTimeseriesSchema",
    "OrbitStateSampleSchema",
    "OrbitStateTimeseriesSchema",
    "PlanEntrySchema",
    "PlanSchema",
]
