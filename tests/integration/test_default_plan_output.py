from scripts.check_default_plan_output import (
    DEFAULT_BASELINE,
    build_default_plan_payload,
    compare_to_baseline,
)


def test_default_plan_output_matches_baseline() -> None:
    actual = build_default_plan_payload()
    diffs = compare_to_baseline(actual, DEFAULT_BASELINE, abs_tol=1e-9)
    assert not diffs, "\n".join(diffs[:25])
