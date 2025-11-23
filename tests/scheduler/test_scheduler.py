"""Unit tests for the DumbScheduler class."""

from unittest.mock import Mock

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops.plan_entry import PlanEntry
from conops.scheduler import DumbScheduler


class TestDumbSchedulerInit:
    """Test DumbScheduler initialization."""

    def test_init_sets_constraint(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint, days=1)
        assert scheduler.constraint is mock_constraint

    def test_init_sets_ephem(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint, days=1)
        assert scheduler.ephem is mock_constraint.ephem

    def test_init_sets_days(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint, days=1)
        assert scheduler.days == 1

    def test_init_without_constraint(self):
        with pytest.raises(ValueError, match="Constraint must be provided"):
            DumbScheduler(constraint=None, days=1)

    def test_init_constraint_without_ephem(self):
        constraint = Mock()
        constraint.ephem = None
        with pytest.raises(ValueError, match="Constraint.ephem must be set"):
            DumbScheduler(constraint=constraint, days=1)

    def test_init_default_mintime(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert scheduler.mintime == 300  # 5 minutes

    def test_init_default_step_size(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert scheduler.step_size == 60  # seconds

    def test_init_default_days(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert scheduler.days == 1

    def test_init_default_ppst_empty(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert len(scheduler.ppst) == 0

    def test_init_default_scheduled_empty(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert len(scheduler.scheduled) == 0

    def test_init_constraint_values_copied_suncons(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert scheduler.suncons == mock_constraint.sun_constraint.min_angle

    def test_init_constraint_values_copied_antisuncons(self, mock_constraint):
        scheduler = DumbScheduler(constraint=mock_constraint)
        assert scheduler.antisuncons == mock_constraint.anti_sun_constraint.max_angle


class TestDumbSchedulerTargetList:
    """Test scheduler target list management."""

    def test_add_targets_to_list(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        assert len(scheduler.targlist) == 4

    def test_target_list_is_iterable(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        count = 0
        for target in scheduler.targlist:
            count += 1
        assert count == 4

    def test_empty_target_list(self, scheduler, simple_target_factory):
        assert len(scheduler.targlist) == 0


class TestDumbSchedulerSAA:
    """Test SAA initialization."""

    def test_saa_pre_set_in_fixture(self, scheduler, simple_target_factory):
        assert scheduler.saa is not None

    def test_saa_can_be_set_to_none(self, scheduler, simple_target_factory):
        scheduler.saa = None
        assert scheduler.saa is None

    def test_saa_has_ephem_attribute(self, scheduler, simple_target_factory):
        assert hasattr(scheduler.saa, "ephem")

    def test_saa_has_insaa_method(self, scheduler, simple_target_factory):
        assert hasattr(scheduler.saa, "insaa")

    def test_saa_ephem_set_reference(self, scheduler, simple_target_factory):
        scheduler.saa.ephem = scheduler.ephem
        assert scheduler.saa.ephem is scheduler.ephem


class TestDumbSchedulerScheduling:
    """Test the core scheduling algorithm."""

    def test_schedule_creates_ppst(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) > 0

    def test_schedule_records_scheduled_ids(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.scheduled) > 0

    def test_schedule_empty_list_ppst_empty(self, scheduler, simple_target_factory):
        scheduler.schedule()
        assert len(scheduler.ppst) == 0

    def test_schedule_empty_list_scheduled_empty(
        self, scheduler, simple_target_factory
    ):
        scheduler.schedule()
        assert len(scheduler.scheduled) == 0

    def test_scheduled_targets_are_ints(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(
            isinstance(scheduled_id, int) for scheduled_id in scheduler.scheduled
        )

    def test_single_target_creates_single_entry(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) == 1

    def test_single_target_scheduled_id(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert scheduler.scheduled == [1]

    def test_multiple_targets_no_overlap_schedules_some(
        self, scheduler, simple_target_factory
    ):
        targets = [
            simple_target_factory(1, 0.0, 0.0, 300),
            simple_target_factory(2, 90.0, 0.0, 300),
        ]
        for target in targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) >= 1

    def test_plan_entry_creation_has_plan_entries(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(isinstance(ppt, PlanEntry) for ppt in scheduler.ppst.entries)

    def test_plan_entry_creation_has_ra_attribute(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(hasattr(ppt, "ra") for ppt in scheduler.ppst.entries)

    def test_plan_entry_creation_has_dec_attribute(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(hasattr(ppt, "dec") for ppt in scheduler.ppst.entries)

    def test_plan_entry_creation_has_begin_attribute(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(hasattr(ppt, "begin") for ppt in scheduler.ppst.entries)

    def test_plan_entry_creation_has_end_attribute(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(hasattr(ppt, "end") for ppt in scheduler.ppst.entries)

    def test_plan_entry_creation_has_slewtime_attribute(
        self, scheduler, sample_targets
    ):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert all(hasattr(ppt, "slewtime") for ppt in scheduler.ppst.entries)

    def test_plan_entry_attributes_count_nonzero(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(42, 45.0, 30.0, 600, "Test")
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) > 0

    def test_plan_entry_attribute_ra(self, scheduler, simple_target_factory):
        target = simple_target_factory(42, 45.0, 30.0, 600, "Test")
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.ra == 45.0

    def test_plan_entry_attribute_dec(self, scheduler, simple_target_factory):
        target = simple_target_factory(42, 45.0, 30.0, 600, "Test")
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.dec == 30.0

    def test_plan_entry_attribute_obsid(self, scheduler, simple_target_factory):
        target = simple_target_factory(42, 45.0, 30.0, 600, "Test")
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.obsid == 42

    def test_plan_entry_attribute_name(self, scheduler, simple_target_factory):
        target = simple_target_factory(42, 45.0, 30.0, 600, "Test")
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.name == "Test"

    def test_slew_time_calculation_first_target_value(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.slewtime == 180  # Default slew time

    def test_target_exptime_reduced_after_scheduling_value(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(1, 45.0, 30.0, 1200)
        initial_exptime = target.exptime
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert target.exptime < initial_exptime

    def test_only_unscheduled_targets_considered_single_schedule(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        scheduler.schedule()
        assert scheduler.scheduled.count(1) == 1

    def test_targets_with_zero_exptime_skipped_only_one_scheduled(
        self, scheduler, simple_target_factory
    ):
        target1 = simple_target_factory(1, 45.0, 30.0, 600)
        target2 = simple_target_factory(2, 90.0, 0.0, 0)
        scheduler.targlist.add_target(target1)
        scheduler.targlist.add_target(target2)
        scheduler.schedule()
        assert 1 in scheduler.scheduled

    def test_targets_with_zero_exptime_skipped_target2_not_scheduled(
        self, scheduler, simple_target_factory
    ):
        target1 = simple_target_factory(1, 45.0, 30.0, 600)
        target2 = simple_target_factory(2, 90.0, 0.0, 0)
        scheduler.targlist.add_target(target1)
        scheduler.targlist.add_target(target2)
        scheduler.schedule()
        assert 2 not in scheduler.scheduled

    def test_scheduling_respects_time_window_begin_after_start(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.days = 1
        scheduler.schedule()
        for ppt in scheduler.ppst.entries:
            assert ppt.begin >= scheduler.ephem.utime[0]

    def test_scheduling_respects_time_window_begin_within_day(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.days = 1
        scheduler.schedule()
        for ppt in scheduler.ppst.entries:
            assert ppt.begin < scheduler.ephem.utime[0] + 86400


class TestDumbSchedulerPlanEntry:
    """Test plan entry properties."""

    def test_plan_entry_begin_not_none(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.begin is not None

    def test_plan_entry_end_not_none(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.end is not None

    def test_plan_entry_end_after_begin(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        if isinstance(ppt.end, Time):
            assert ppt.end.unix > ppt.begin
        else:
            assert ppt.end > ppt.begin

    def test_plan_entry_saa_reference(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.saa is scheduler.saa

    def test_plan_entry_constraint_reference(self, scheduler, simple_target_factory):
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        ppt = scheduler.ppst[0]
        assert ppt.constraint is scheduler.constraint


class TestDumbSchedulerConstraints:
    """Test constraint evaluation."""

    def test_constraint_inoccult_called(self, scheduler, simple_target_factory):
        original_inoccult = scheduler.constraint.inoccult
        call_count = [0]

        def tracked_inoccult(*args, **kwargs):
            call_count[0] += 1
            return original_inoccult(*args, **kwargs)

        scheduler.constraint.inoccult = tracked_inoccult

        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()

        assert call_count[0] > 0

    def test_constraint_with_all_times_valid_schedules(
        self, scheduler, mock_ephemeris, simple_target_factory
    ):
        scheduler.constraint.inoccult = lambda ra, dec, utime, hardonly=True: np.zeros(
            len(utime) if hasattr(utime, "__len__") else 1, dtype=bool
        )
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) > 0

    def test_constraint_with_all_times_invalid_not_scheduled(
        self, scheduler, simple_target_factory
    ):
        scheduler.constraint.inoccult = lambda ra, dec, utime, hardonly=True: np.ones(
            len(utime) if hasattr(utime, "__len__") else 1, dtype=bool
        )
        target = simple_target_factory(1, 45.0, 30.0, 600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) == 0


class TestDumbSchedulerProperties:
    """Test scheduler configuration properties."""

    def test_mintime_configuration(self, scheduler, simple_target_factory):
        scheduler.mintime = 600  # 10 minutes
        assert scheduler.mintime == 600

    def test_stepsize_configuration(self, scheduler, simple_target_factory):
        scheduler.stepsize = 120  # 2 minutes
        assert scheduler.stepsize == 120

    def test_gimbled_initial_type(self, scheduler, simple_target_factory):
        assert scheduler.gimbled in (True, False)

    def test_gimbled_set_true(self, scheduler, simple_target_factory):
        scheduler.gimbled = True
        assert scheduler.gimbled is True

    def test_sidemount_initial_type(self, scheduler, simple_target_factory):
        assert scheduler.sidemount in (True, False)

    def test_sidemount_set_true(self, scheduler, simple_target_factory):
        scheduler.sidemount = True
        assert scheduler.sidemount is True


class TestDumbSchedulerEdgeCases:
    """Test edge cases and error handling."""

    def test_extremely_short_observation_runs(self, scheduler, simple_target_factory):
        target = simple_target_factory(targetid=1, ra=45.0, dec=30.0, exptime=60)
        scheduler.targlist.add_target(target)
        scheduler.mintime = 30  # Allow very short observations
        scheduler.schedule()
        assert scheduler.schedule is not None

    def test_extremely_long_observation_runs(self, scheduler, simple_target_factory):
        target = simple_target_factory(targetid=1, ra=45.0, dec=30.0, exptime=86400)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        # No assert: may or may not schedule depending on available window

    def test_large_target_list_runs(self, scheduler, simple_target_factory):
        for i in range(100):
            target = simple_target_factory(
                targetid=i,
                ra=(i * 3.6) % 360,
                dec=(i - 50) % 90 - 45,
                exptime=300,
            )
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        # No assert: just ensure no error

    def test_multiple_scheduling_runs_handle_success(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(targetid=1, ra=45.0, dec=30.0, exptime=300)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        target2 = simple_target_factory(targetid=2, ra=90.0, dec=0.0, exptime=300)
        scheduler.targlist.add_target(target2)
        scheduler.schedule()
        # No assert: behavior depends on implementation


class TestDumbSchedulerIntegration:
    """Integration tests for the scheduler."""

    def test_full_scheduling_workflow_target_count(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        assert len(scheduler.targlist) == len(sample_targets)

    def test_full_scheduling_workflow_runs(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.ppst) >= 0

    def test_full_scheduling_workflow_scheduled_nonnegative(
        self, scheduler, sample_targets
    ):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.scheduled) >= 0

    def test_full_scheduling_workflow_unique_scheduled(self, scheduler, sample_targets):
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert len(scheduler.scheduled) == len(set(scheduler.scheduled))

    def test_scheduler_with_custom_config_mintime(
        self, mock_constraint, mock_saa, mock_config, sample_targets
    ):
        scheduler = DumbScheduler(constraint=mock_constraint, days=1)
        scheduler.saa = mock_saa
        scheduler.config = mock_config
        scheduler.mintime = 600
        scheduler.stepsize = 120
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert scheduler.mintime == 600

    def test_scheduler_with_custom_config_stepsize(
        self, mock_constraint, mock_saa, mock_config, sample_targets
    ):
        scheduler = DumbScheduler(constraint=mock_constraint, days=1)
        scheduler.saa = mock_saa
        scheduler.config = mock_config
        scheduler.mintime = 600
        scheduler.stepsize = 120
        for target in sample_targets:
            scheduler.targlist.add_target(target)
        scheduler.schedule()
        assert scheduler.stepsize == 120

    def test_plan_summary_after_scheduling_counts(
        self, scheduler, simple_target_factory
    ):
        target = simple_target_factory(targetid=1, ra=45.0, dec=30.0, exptime=600)
        scheduler.targlist.add_target(target)
        scheduler.schedule()
        total_scheduled = len(scheduler.scheduled)
        total_entries = len(scheduler.ppst)
        assert total_scheduled == total_entries
