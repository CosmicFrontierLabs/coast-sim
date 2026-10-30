COAST #318 review reproductions
==============================

These checks reproduce the three review findings against commit
``7b8c9af63ffae835e9a1ada6d47690c9dd964f36``. They intentionally live outside
``tests/`` so the regular test suite does not collect known failing review
cases. The underlying production code is unchanged on this review branch.

Review: https://github.com/CosmicFrontierLabs/coast-sim/pull/318#pullrequestreview-5707481618

Run from the repository root in an environment with this checkout installed
and its test dependencies::

    python -m pip install -e '.[test]'
    PYTHONPATH=. python review/pr318/reproduce.py
    PYTHONPATH=. python -m pytest -n 0 review/pr318/test_review_regressions.py -q

Expected results on the reviewed commit: three assertion failures.

* Multiple contacts within one step: the simulation reports 0.6 Gb downlinked;
  chronological processing gives 0.5 Gb and 0.1 Gb lost to recorder overflow.
* Variable-rate placement: the scheduler returns no observations despite four
  verified feasible start times. Supplying an explicit earliest start places
  the same observation successfully.
* Restored booking boundaries: 0.15 Gb is credited after the station reservation
  ends when its endpoints lie between ephemeris samples.

Validation on the reviewed commit: the existing suite passed with 3,909 tests
and one skip. These three additional regressions failed with the results above.
The review environment used Python 3.12 and rust-ephem 0.15.1.

When testing a later fix, carry these files onto the updated branch and rerun
the explicit command. Passing assertions are the desired fixed behavior.
