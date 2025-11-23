from conops import AttitudeControlSystem, Slew


class DummyConstraint:
    def __init__(self):
        # Slew asserts constraint.ephem is not None, but does not use it during calc
        self.ephem = object()


def test_slew_uses_acs_config():
    acs = AttitudeControlSystem(
        slew_acceleration=1.0, max_slew_rate=0.5, settle_time=90.0
    )
    s = Slew(constraint=DummyConstraint(), acs_config=acs)
    s.startra = 0
    s.startdec = 0
    s.endra = 90
    s.enddec = 0
    expected = acs.slew_time(90.0)
    calc = s.calc_slewtime()
    assert calc == round(expected)


def test_slew_path_and_secs_lengths():
    acs = AttitudeControlSystem()
    s = Slew(constraint=DummyConstraint(), acs_config=acs)
    s.startra = 10
    s.startdec = 5
    s.endra = 20
    s.enddec = 15
    st = s.calc_slewtime()
    assert isinstance(s.slewpath, tuple)
    assert len(s.slewpath) == 2
    # With ACS-driven kinematics, slew_ra_dec no longer uses self.slewsecs
    # so we don't require a time grid to be generated here.
    assert st > 0
