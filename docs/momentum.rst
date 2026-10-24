Stored Momentum
===============

A spacecraft that holds an attitude against the environment's torques stores
angular momentum in its reaction wheels, and the wheels have a capacity. COAST
tracks that momentum from the torques themselves, unloads it as magnetorquers and
thruster dumps would, and plans and dispatches science so the wheels stay within
their capacity. This page describes the model, how it is configured, what the
simulations and the planners do with it, and what it leaves out.

The model
---------

**What is tracked.** The wheels store the angular impulse of every environmental
torque the spacecraft rejects while holding or moving its attitude. COAST
integrates the sum of the configured torques in inertial coordinates by the
trapezoidal rule, so changing attitude alone creates no momentum, and reports the
stored vector in the current body frame together with its norm. The integrator is
:class:`~conops.simulation.StoredMomentumTracker`; the configuration is
:class:`~conops.config.momentum.StoredMomentumConfig` under
``spacecraft_bus.attitude_control.stored_momentum``, and the inertia tensor it needs
is ``spacecraft_bus.inertia_tensor_body_kg_m2``.

**Sources.** Each torque source is evaluated in the body frame at every sample from
the ephemeris and the attitude, and reported by name in the telemetry
(``disturbance_torques_body_n_m``), so a run shows which term fills the wheels.

* *Gravity gradient* (``gravity_gradient_enabled``): the central body's torque on
  the inertia tensor, :math:`3\mu/r^3\,(\hat r \times I\hat r)`. For an inertial
  hold the radial direction sweeps the body once per orbit and the torque is
  secular; it is the dominant term for a spacecraft with a few hundred kg m² of
  inertia difference, filling a few N m s of wheel capacity in hours to a day.
* *Solar radiation pressure* (``solar_radiation_pressure_enabled``): sunlight on the
  spacecraft's surfaces. Each plate the Sun shines on takes the force of what it
  absorbs and reflects, specularly and diffusely,
  :math:`F = -P A \cos\theta\,[(1-\rho_s)\hat s + 2(\rho_s\cos\theta + \rho_d/3)\hat n]`,
  with the pressure scaled from one astronomical unit by the Sun's distance; plates
  facing away take none and nothing does in eclipse. Secular for an inertial or
  Sun-referenced hold.
* *Aerodynamic drag* (``aerodynamic_drag_enabled``): the atmosphere on the same
  surfaces. The flow is the spacecraft's velocity relative to the atmosphere, which
  rotates with the Earth, :math:`v_{rel} = v - \omega_\oplus \times r`; each plate
  facing it takes :math:`F = -\tfrac12 \rho |v_{rel}|^2 C_d A (\hat n \cdot \hat v)^+ \hat v`
  with its own drag coefficient. The flow direction turns once per orbit, so for an
  inertial hold the torque largely averages out; the density is the uncertain
  input, a factor of twenty between quiet Sun and solar maximum.
* *Residual dipole* (``residual_dipole_body_a_m2``): the spacecraft's own magnetic
  dipole, torqued by the geomagnetic field, :math:`m \times B`. Of the order of
  solar radiation pressure for a few A m² in low Earth orbit.
* *Magnetorquer unloading* (``magnetorquer_max_dipole_a_m2``,
  ``magnetorquer_time_constant_s``): a negative source. The torquers command the
  dipole that would remove the stored momentum over the time constant, capped at
  their greatest dipole, so at the cap the unloading torque is the cap times the
  field. They turn the spacecraft only about axes across the field, so they take out
  the momentum's component across it and leave the component along it, which turns
  as the field turns round the orbit.

**Surfaces.** Sunlight and drag act on ``spacecraft_bus.surfaces``
(:class:`~conops.config.SurfaceModel`): flat plates with a centre, an outward
normal, an area, specular and diffuse reflectivities and a drag coefficient, about
a centre of mass. :func:`~conops.config.box_surfaces`,
:func:`~conops.config.panel_surfaces` and :func:`~conops.config.telescope_surfaces`
build plates from a box, the solar panels with a geometry and the telescopes'
apertures. With no surfaces configured, :func:`~conops.config.guess_surfaces`
builds a placeholder, a 1.5 × 1.5 × 2 m box, the panels and the apertures about a
centre of mass 5 cm behind the boresight, whose torques are of the right order and
nothing more; a mission's mass properties replace it. Plates do not shadow one
another.

**Atmosphere.** The density drag uses comes from ``stored_momentum.atmosphere``
(:class:`~conops.config.AtmosphereConfig`):

* ``model: "table"`` (the default): a piecewise exponential profile by geodetic
  height, Vallado's table for moderate solar activity, needing no data. It gives the
  density to within a factor of a few at any height a spacecraft flies.
* ``model: "msis"``: NRLMSIS 2.1 through the optional ``pymsis`` package, at the
  solar flux (``f107``, ``f107a``) and geomagnetic activity (``ap``) the run assumes,
  so a run states one level of activity rather than fetching the indices of its
  dates. Use it where the density matters. Install it with the ``momentum`` extra
  (see :doc:`installation`).
* ``density_scale``: a factor on either, for sensitivity studies.

**Geomagnetic field.** The field the torquers unload against and the residual
dipole is torqued by is ``stored_momentum.geomagnetic_field``:

* ``"dipole"`` (the default): a tilted dipole, 31.2 µT at the magnetic equator on
  the surface, falling with the cube of the radius, its axis through the
  geomagnetic poles. Strength and direction to ten or twenty per cent of the real
  field, enough for the torques' order of magnitude.
* ``"igrf"``: the International Geomagnetic Reference Field through the optional
  ``ppigrf`` package, at the sample's epoch, in the ``momentum`` extra.

The field at each sample is reported in body coordinates
(``geomagnetic_field_body_t``), and the atmosphere's density where drag is tracked
(``atmosphere_density_kg_m3``).

**Sampling.** The torque is taken as linear between samples. A held attitude is
sampled at the simulation's step, the ephemeris resolution, which is enough since
the torques then vary only with the orbit. A slew is sampled more finely wherever
it falls: the simulations ask the ACS for the attitude flown at intermediate times
within the step, every ``max_attitude_step_deg`` (5° by default) of motion at the
fastest configured slew rate, with the orbit interpolated between the step's ends.
The planners' forecast samples the same way. Tracking therefore works at whatever
resolution the ephemeris has; the attitude step is a convergence control, not an
integration-error guarantee.

Configuration
-------------

Under ``spacecraft_bus.attitude_control.stored_momentum``:

* ``gravity_gradient_enabled``, ``solar_radiation_pressure_enabled``,
  ``aerodynamic_drag_enabled`` (bool, default ``False``): the sources tracked.
* ``atmosphere`` (:class:`~conops.config.AtmosphereConfig`): ``model`` (``table`` or
  ``msis``), ``f107``, ``f107a``, ``ap`` (150, 150 and 4 by default), ``density_scale``.
* ``geomagnetic_field`` (``dipole`` or ``igrf``, default ``dipole``).
* ``magnetorquer_max_dipole_a_m2`` (float or None) and
  ``magnetorquer_time_constant_s`` (default 1800): the torquers; None for none.
* ``residual_dipole_body_a_m2`` (three floats or None): the spacecraft's dipole.
* ``initial_momentum_body_n_m_s`` (three floats): the wheels' momentum at the start.
* ``max_attitude_step_deg`` (default 5): the attitude step along slews.
* ``capacity_n_m_s`` (float or None): the wheels' capacity, taken as the same about
  every axis; None sets none.
* ``planning_level`` (default 0.8): the fraction of the capacity at which a thruster
  dump is called for, the rest being margin.
* ``thruster_dump`` (:class:`~conops.config.ThrusterDumpConfig` or None):
  ``duration_seconds`` (default 120) the dump holds the spacecraft, and
  ``residual_fraction`` (default 0) it leaves in the wheels. A dump needs a
  capacity. None: no thrusters, so the wheels fill until a run reports it.

The inertia tensor's principal moments must be positive and satisfy the triangle
inequalities; a planar body's equality is allowed within rounding. The tracker
reloads the configuration at the start of every run, so changes between runs take
effect without rebuilding the simulation. Enabling any source with no inertia
tensor is an error.

A minimal configuration that tracks gravity gradient and drag with thruster dumps:

.. code-block:: yaml

   spacecraft_bus:
     inertia_tensor_body_kg_m2: [[1000, 0, 0], [0, 1200, 0], [0, 0, 800]]
     attitude_control:
       stored_momentum:
         gravity_gradient_enabled: true
         aerodynamic_drag_enabled: true
         atmosphere: {model: msis, f107: 120, f107a: 120, ap: 8}
         capacity_n_m_s: 4.0
         planning_level: 0.8
         thruster_dump: {duration_seconds: 180, residual_fraction: 0.05}

Dumps
-----

Two mechanisms unload the wheels, different in kind. Magnetorquers unload
continuously and need no block; they are a torque source in the integrator. A
thruster dump is a hold during which no science is observed: a desaturation
activity (``ScheduledActivity.desaturation``) that holds the attitude the spacecraft
has for the dump's length, in the ``DESAT`` ACS mode, and unloads the wheels when it
ends. While it holds, the telemetry reports ``DESAT``, an instrument that defines a
mode named ``DESAT`` is in it, and the attitude constraint scopes (hardware safety
unless set) and the bus power table (the nominal power unless set) take an entry
for it. Dumps are never timer-based and never change the attitude: the firing
attitude is not modelled.

**Dispatch.** Dispatch calls for a dump when the stored momentum reaches the
planning level of the capacity. The dump goes between observations: the observation
under way runs on, and the dump begins once it has ended and before another starts;
the margin between the level and the capacity is what the wait draws on. Before
dispatch begins an observation it forecasts the momentum over it from the wheels as
they are, over a visit of a sequence whole, every step at its own pointing with the
slews between them and the cleanup after the last. Where the forecast reaches the
level during the observation but not the capacity, the observation runs and the
dump follows it. Where it would reach the capacity, a plain observation is bounded
to end where the level is reached, or where its minimum snapshot is collected if
that comes later and still fits, and the dump follows; a visit, which is not cut,
has the wheels emptied first where it fits from empty and is rejected and marked
done where it does not, with the reason logged. Where the level is reached before
the observation would collect, the wheels are emptied first.

**The capacity is a red line.** Where the stored momentum reaches it, fault
management records a ``momentum_capacity`` event against the ``stored_momentum``
state, naming the observation under way if any, and the dump is an emergency: it
begins once the observation's cleanup and handoff budget is paid, with its
collection stopped a budget before, as before a target-of-opportunity preemption.
No safe mode is requested; the unload clears the state with a second event. A
planned simulation records the event the same way against the plan entry under way
and relies on the plan's dumps, unloading the wheels at the end of each entry marked
``desaturation``.

**Planners.** Every plan built with momentum tracking on is forecast for it.
:class:`~conops.schedulers.momentum.MomentumForecast` integrates the configured
sources along the plan's timeline with the same tracker: each slew along its path,
each observation, contact, charge and activity at its attitude, and the idle time
between them at the attitude held, with each thruster dump unloading the wheels as
it ends. Where the forecast reaches the planning level and the spacecraft has
thrusters, the planner reserves a dump, a desaturation activity holding the attitude
of whatever precedes it, and plans again from the start with the dumps so far. The
dump goes between observations: after the one the level is reached inside, where
the capacity is not reached through the dump's own length; before it, where the
observation fits from empty wheels; a visit that does not fit from empty is left
out, a plain observation that does not is cut at the level. A dump found earlier
than one already reserved replaces it and those after it. The passes of one plan
share the spans integrated so far, keyed on each span's times, attitude or slew and
the momentum it starts from, so a pass integrates only from its new dump onwards.
A rolling-horizon run gives each replan the momentum it will have when its horizon
starts. Without thrusters the plan stands and the log says where the forecast
passes the capacity. See :doc:`planning`, Planning within the wheels.

Telemetry and reporting
-----------------------

Housekeeping telemetry carries, at every step, each source's torque by name
(``disturbance_torques_body_n_m``) and their total (``disturbance_torque_body_n_m``),
the stored momentum in body coordinates and its norm (``stored_momentum_body_n_m_s``,
``stored_momentum_norm_n_m_s``), the geomagnetic field in body coordinates where a
source uses one (``geomagnetic_field_body_t``) and the atmosphere's density where
drag is tracked (``atmosphere_density_kg_m3``). The event log carries ``MOMENTUM``
events for each dump called for, each unload and each selection decision the
forecast makes; fault management carries the capacity events. The timelines
(:doc:`visualization`) draw dumps on the activities row, and the momentum example
notebook plots the stored momentum with each source's share, emptied at every dump.

What is left out
----------------

* The rigid-body momentum of the slews themselves (:math:`I\omega`), a transient the
  wheels give back when the slew ends.
* Self-shadowing between plates, for sunlight and drag alike.
* A thruster firing attitude: a dump holds the attitude the spacecraft has.
* Propellant accounting for dumps.
* The accuracy of the environment: the built-in atmosphere table and the dipole
  field are order-of-magnitude models; the optional ``msis`` and ``igrf`` models are
  the ones to use where the numbers matter, and the density's solar-activity inputs
  are the run's assumption, not a forecast.
