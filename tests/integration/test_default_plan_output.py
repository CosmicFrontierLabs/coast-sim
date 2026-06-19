from scripts.check_default_plan_output import (
    DEFAULT_BASELINE,
    DEFAULT_NUMERIC_ABS_TOL,
    build_default_plan_payload,
    compare_to_baseline,
)


def test_default_plan_output_matches_baseline() -> None:
    actual = build_default_plan_payload()
    diffs = compare_to_baseline(
        actual, DEFAULT_BASELINE, abs_tol=DEFAULT_NUMERIC_ABS_TOL
    )
    assert not diffs, "\n".join(diffs[:25])
