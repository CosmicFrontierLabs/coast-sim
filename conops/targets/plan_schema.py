"""Compatibility shim for the old conops.targets.plan_schema module.

PlanSchema/PlanEntrySchema used to be a separate pydantic mirror of
Plan/PlanEntry, built solely to make plans serializable. Plan and PlanEntry
are now directly serializable, so that mirror layer was removed. This
module re-exports Plan as PlanSchema so that
``from conops.targets.plan_schema import PlanSchema`` and
``from conops.targets import PlanSchema`` keep working for existing callers.
"""

from .plan import Plan as PlanSchema

__all__ = ["PlanSchema"]
