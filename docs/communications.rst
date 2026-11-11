Communications System Configuration
====================================

Overview
--------

The Communications System configuration defines the onboard communications capabilities for spacecraft downlink and uplink operations. It supports multiple frequency bands, different antenna types, and configurable data rates for realistic mission planning and link budget analysis.

Key Features
------------

* **Multiple frequency bands**: UHF, S, X, Ka, Ku, L, C, K bands
* **Flexible antenna types**: Omnidirectional, fixed, and gimbaled antennas
* **Per-band data rates**: Independent uplink and downlink rates for each frequency band
* **Standard defaults**: Automatic rate assignment based on common band characteristics
* **Antenna pointing**: Configurable mounting and pointing requirements
* **Polarization support**: Linear and circular polarization options
* **Ground station integration**: Per-band ground station downlink capabilities

Components
----------

BandCapability
^^^^^^^^^^^^^^

Defines the capabilities for a specific frequency band.

**Fields:**

* ``band``: Frequency band identifier (``"UHF"``, ``"S"``, ``"X"``, ``"Ka"``, ``"Ku"``, ``"L"``, ``"C"``, ``"K"``)
* ``uplink_rate_mbps``: Uplink data rate in Mbps (≥ 0)
* ``downlink_rate_mbps``: Downlink data rate in Mbps (≥ 0)

**Standard Defaults:**

When only the ``band`` is specified, standard defaults are automatically applied:

* **S-band**: uplink 2.0 Mbps, downlink 10.0 Mbps
* **X-band**: uplink 10.0 Mbps, downlink 150.0 Mbps
* **Ka-band**: uplink 20.0 Mbps, downlink 300.0 Mbps
* **Ku-band**: uplink 5.0 Mbps, downlink 50.0 Mbps
* **L-band**: uplink 0.5 Mbps, downlink 1.0 Mbps
* **C-band**: uplink 2.0 Mbps, downlink 20.0 Mbps
* **K-band**: uplink 15.0 Mbps, downlink 200.0 Mbps

UHF has no generic default rate; specify mission-specific uplink and downlink
rates explicitly. Explicitly specified values override all defaults.

A band's rate can vary with the spacecraft-to-station range:
``downlink_rate_vs_range_km`` is a table of ``(range_km, rate_mbps)`` rows in
increasing range, interpolated linearly between rows and held at the end rows
beyond them. Either side of a link may carry one, and the effective rate at a
range is the lower of the two sides' rates there, on the best band. The table is
used where the range is known: the contacts of an omni or gimballed antenna take
the rate at each step's range, so a contact at low elevation downlinks less than
one overhead, and each contact span reports its time-weighted mean rate. A
body-fixed antenna's passes use ``downlink_rate_mbps`` throughout.

AntennaPointing
^^^^^^^^^^^^^^^

Defines how the antenna is mounted and pointed.

**Antenna Types:**

* ``OMNI``: Omnidirectional antenna (no pointing requirement)
* ``FIXED``: Body-fixed antenna with defined pointing direction
* ``GIMBALED``: Steerable antenna with angular range

**Fields:**

* ``antenna_type``: Type of antenna mounting
* ``fixed_boresight_body``: Fixed antenna boresight as a unit vector in spacecraft body frame.
  The default ``(-1, 0, 0)`` preserves the legacy ground-station-pass tracking convention.
* ``fixed_azimuth_deg`` / ``fixed_elevation_deg``: Legacy metadata fields. Use
  ``fixed_boresight_body`` for GSP attitude generation. Nonzero values are accepted
  with a deprecation warning and do not affect generated GSP attitudes.
* ``gimbal_range_deg``: Angular range for gimbaled antennas (from boresight)
* ``repoint_threshold_deg``: For a gimballed antenna, the pointing allowance: the
  radius of the cap about the antenna's pointing that the tracked station is kept in,
  the antenna being re-pointed ahead of the station where it would leave (default
  None: no tracking repoints)
* ``repoint_interval_vs_range_km``: Instead of the threshold, the interval between
  repoints in seconds against the spacecraft's distance from the Earth's centre, as
  ``(range_km, seconds)`` rows interpolated at the range at each repoint
* ``repoint_seconds``: Seconds a repoint takes, during which nothing is downlinked and
  an observation of a category that forbids repoints cannot collect (default 0)

The three repoint fields apply to a ``GIMBALED`` antenna only: an omni antenna has no
direction to re-point and a body-fixed one moves with the spacecraft.

For omni and gimballed antennas, ``contact_policy="continuous"`` is the
communications default. :class:`~conops.simulation.StationCoverage` retains
all available visibility windows, including their overlapping tails and
windows shorter than eight minutes. It selects the station with the highest
instantaneous effective downlink rate, then highest elevation and station
code for a stable tie. It tracks that station until it sets or its rate
becomes zero, then hands over to the best usable station. A faster station
rising does not interrupt the current link. A chain of overlapping windows
therefore supplies continuous network coverage instead of being reduced to
whole, mutually exclusive passes.

Station access is still distinct from visibility: ``schedule_probability``
is drawn once per station visibility window and ``random_seed`` reproduces
those draws. Zero rejects the window; one accepts it. Availability is decided
once over the full ephemeris, shared by dispatch and every replan.

Set ``contact_policy="scheduled"`` to use whole-pass booking instead.
:class:`~conops.simulation.PassTimes` then applies its eight-minute minimum
and selects whole non-overlapping passes by downlink volume, duration and
earlier start. Range-dependent effective rates contribute to that volume.
Body-fixed antennas always use this pass-booking path, with spacecraft
tracking profiles and ``PASS`` activities that pause science; the contact
policy does not change their behavior. Only fixed antennas require
``schedule_for_tracking`` eligibility.

Both flexible-antenna policies select one station at a time and use the same
acquisition, handover and repointing accounting. Every contact starts
``acquisition_seconds`` after selection; a direct station switch adds
``handover_seconds``. Nothing is downlinked during those outages. A fresh
acquisition after a genuine availability gap pays no handover. Rates use the
bands shared by station and spacecraft at their range at each step.
Execution also checks a gimballed antenna's reach from the actual body pose;
planning takes its selected station as reachable.

The recorder ledgers, dispatch and planners consume the same selected
contact timeline and its outages. Plans save the timeline's exact raw
``(begin, end, station)`` segments in ``ground_contacts``, before acquisition
and repoint costs. Replay restores those segments without selecting again
or repeating availability draws, including when the seed or contact policy
has changed. An empty list restores no contacts. A historical saved plan
therefore retains its original bookings; rebuild from the program to use a
new policy. Plans without ``ground_contacts`` predict from the configuration.

**Antenna repoints.** A gimballed antenna is re-pointed at each acquisition and
handover, and within a contact as the spacecraft moves along its orbit; an omni antenna
is never re-pointed, so it has no repoint events. While an
attitude is held the station's direction moves in the body frame exactly as it moves
across the sky. Tracking repoint times are therefore calculated from the station
geometry from the contact's start. With ``repoint_threshold_deg`` set, the antenna
leads the station: at each repoint it is pointed at the centre of the longest arc of
the station's coming track that fits within a cap of that radius, so the station
enters the cap on one side and leaves on the other, and the next repoint comes where
the arc no longer fits. Taking each arc as long as it can be gives the fewest repoints
for the allowance, about half as many as re-pointing at the station each time it
moves the allowance away. With ``repoint_interval_vs_range_km`` set instead, repoints
follow the table's interval at the spacecraft's range from the Earth's centre, each
pointing at the station. Each repoint records the antenna's pointing and how far it
turned, and ``print_statistics`` reports the mean and largest turn. A repoint takes ``repoint_seconds``, during which nothing is
downlinked, so a contact is split around it; an acquisition or handover repoint lasts
its latency. The simulation logs each repoint as an ``ANTENNA`` event with the station
and the observation in progress, and ``print_statistics`` reports them by kind. An
observation category's ``antenna_repoints`` setting says what a repoint during one of
its observations means: ``allowed`` (the default), ``reported`` (counted and logged) or
``forbidden``, in which case the schedulers cut the category's visibility windows at
each repoint so snapshots are placed between them, and dispatch neither starts such an
observation during a repoint nor lets it run into the next one (see :doc:`planning`).

A configuration without a communications system keeps the body-fixed behaviour.
With coverage, a DITL's ``coverage`` holds the model, its data telemetry shows
the downlink, and ``print_statistics`` reports the number of contacts and
handovers and the fraction of the run in contact. The timeline plots show the contacts
in their ``Ground Contact`` row, coloured by station: one bar per span on a run
up to two days long, and on a longer run one bar per hour with a contact, as
dark as the hour's fraction in contact against the run's fullest hour, since a
span of minutes is narrower than a pixel at that scale.

Polarization
^^^^^^^^^^^^

Defines antenna polarization type.

**Options:**

* ``LINEAR_HORIZONTAL``: Linear horizontal polarization
* ``LINEAR_VERTICAL``: Linear vertical polarization
* ``CIRCULAR_RIGHT``: Right-hand circular polarization (RHCP)
* ``CIRCULAR_LEFT``: Left-hand circular polarization (LHCP)
* ``DUAL``: Supports multiple polarizations

CommunicationsSystem
^^^^^^^^^^^^^^^^^^^^

Main configuration class for the communications system.

**Fields:**

* ``name``: Descriptive name for the communications system
* ``band_capabilities``: List of supported frequency bands with data rates
* ``antenna_pointing``: Antenna pointing configuration
* ``pointing_accuracy_deg``: Maximum pointing error to maintain good signal (degrees)
* ``polarization``: Antenna polarization type
* ``contact_policy``: ``continuous`` (default) for station handovers without
  dropping overlaps, or ``scheduled`` for whole-pass booking; fixed antennas
  always use tracking passes
* ``acquisition_seconds``: Seconds to acquire the link during passes without
  spacecraft tracking; no data is downlinked during acquisition (default 0)
* ``handover_seconds``: Seconds to switch the link straight from one ground station to
  another during passes without spacecraft tracking, on top of acquisition, during which
  nothing is downlinked (default 0)

**Methods:**

* ``get_band(band: str)``: Get capability for a specific band
* ``get_downlink_rate(band: str)``: Get downlink rate in Mbps
* ``get_uplink_rate(band: str)``: Get uplink rate in Mbps
* ``can_communicate(pointing_error_deg: float)``: Check if communication is possible

Configuration Examples
----------------------

Python Mission Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.config import (
       CommunicationsSystem,
       BandCapability,
       AntennaPointing,
       AntennaType,
       Polarization,
   )

   # LEO satellite with body-fixed -X S-band (using defaults)
   leo_comms = CommunicationsSystem(
       name="LEO S-band",
       band_capabilities=[
           BandCapability(band="S")  # Uses defaults: 2.0 Mbps up, 10.0 Mbps down
       ],
       antenna_pointing=AntennaPointing(
           antenna_type=AntennaType.FIXED,
           fixed_boresight_body=(-1.0, 0.0, 0.0),
       ),
       polarization=Polarization.CIRCULAR_RIGHT,
       pointing_accuracy_deg=10.0,
   )

   # High-rate X-band with gimbaled antenna (custom rates)
   xband_comms = CommunicationsSystem(
       name="High-Rate X-band",
       band_capabilities=[
           BandCapability(
               band="X",
               uplink_rate_mbps=10.0,
               downlink_rate_mbps=300.0,
           )
       ],
       antenna_pointing=AntennaPointing(
           antenna_type=AntennaType.GIMBALED,
           gimbal_range_deg=60.0,
       ),
       polarization=Polarization.CIRCULAR_RIGHT,
       pointing_accuracy_deg=3.0,
   )

   # Check communication capability
   if xband_comms.can_communicate(pointing_error=2.5):
       rate = xband_comms.get_downlink_rate("X")
       print(f"Can downlink at {rate} Mbps")

JSON Configuration
^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "name": "S/X Dual Band",
     "band_capabilities": [
       {
         "band": "S",
         "uplink_rate_mbps": 2.0,
         "downlink_rate_mbps": 10.0
       },
       {
         "band": "X",
         "uplink_rate_mbps": 0.0,
         "downlink_rate_mbps": 150.0
       }
     ],
     "antenna_pointing": {
       "antenna_type": "gimbaled",
       "gimbal_range_deg": 45.0
     },
     "polarization": "dual",
     "pointing_accuracy_deg": 5.0
   }

Ground Station Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ground stations now store band capabilities directly on the ``GroundStation`` object:

.. code-block:: python

   from conops.config import GroundStation, BandCapability

   # Ground station with X-band capability
   station = GroundStation(
       code="SGS",
       name="Singapore Ground Station",
       latitude_deg=1.3521,
       longitude_deg=103.8198,
       bands=[
           BandCapability(band="X", downlink_rate_mbps=100.0)
       ],
       gain_db=45.0
   )

   # Access methods
   supported = station.supported_bands()  # ["X"]
   rate = station.get_downlink_rate("X")  # 100.0
   max_rate = station.get_overall_max_downlink()  # 100.0

Integration with Spacecraft Bus
--------------------------------

The communications system integrates with the spacecraft bus configuration:

.. code-block:: python

   from conops.config import SpacecraftBus, CommunicationsSystem

   bus = SpacecraftBus(
       name="My Spacecraft",
       communications=CommunicationsSystem(
           name="X-band Comms",
           band_capabilities=[
               BandCapability(band="X", downlink_rate_mbps=150.0)
           ]
       ),
   )

Common Configurations
---------------------

See ``examples/example_communications_configs.json`` for complete examples:

* LEO S-band (body-fixed -X)
* High-rate X-band (gimbaled)
* Dual S/X-band
* Deep space Ka-band
* CubeSat omnidirectional
* Zenith-pointing inter-satellite link

Use Cases
---------

Data Management
^^^^^^^^^^^^^^^

The communications configuration is used to:

* Calculate data transfer rates during ground station passes
* Determine data volume that can be downlinked per pass
* Manage onboard data storage requirements

Pass Definition
^^^^^^^^^^^^^^^

The configuration affects pass planning:

* ``pointing_accuracy_deg`` determines if spacecraft pointing is adequate
* ``antenna_type`` affects whether spacecraft must point at ground station
* Gimbaled antennas may allow communication without spacecraft slewing

Link Budget Analysis
^^^^^^^^^^^^^^^^^^^^

Data rates and pointing requirements support:

* Link budget calculations
* Communication window planning
* Data transfer optimization

Data Rate Matching
^^^^^^^^^^^^^^^^^^

During ground station passes, the effective downlink rate is calculated as:

1. For each common band between ground station and spacecraft:

   * Compute ``min(GS downlink rate, SC downlink rate)``

2. Select the maximum effective rate across all common bands
3. This becomes the link-limiting factor for data downlink

Example:

.. code-block:: python

   # Ground station: X-band @ 100 Mbps
   # Spacecraft: X-band @ 150 Mbps
   # Effective rate: 100 Mbps (GS limited)

   # Ground station: S @ 50 Mbps, X @ 200 Mbps
   # Spacecraft: S @ 10 Mbps, X @ 150 Mbps
   # Effective rate: 150 Mbps (X-band, SC limited)

Ground Station Pass (GSP) Plan Entries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :class:`~conops.ditl.queue_ditl.QueueDITL` executes a Day-In-The-Life simulation,
spacecraft tracking activities for ground-station passes are exported as ``GSP``
entries in the observation plan. Pass bookings without tracking are stored in
``ground_contacts``. GSP entries capture:

* **Station identification**: Which ground station is used (``station`` field)
* **Contact window**: Actual pass start/end times (``contact_begin``, ``contact_end``)
* **Preparation time**: Slew time to point at the station before contact begins
* **Deconfliction**: When multiple stations are simultaneously visible, the system selects
  the pass with the highest expected data volume (downlink rate × duration)

**Key behavior:**

* GSP entries are created only for passes the spacecraft actually commands and executes
* Overlapping pass opportunities are automatically deconflicted; the highest-value pass is selected
* Passes are not commanded (and no GSP entry is created) when the spacecraft is in SAFE mode
* GSP entries are excluded from science observation timelines in visualization plots

**Example workflow:**

.. code-block:: python

   from conops import QueueDITL, MissionConfig
   from conops.ephemeris import TLEEphemeris

   # Run DITL with ground station passes
   ditl = QueueDITL(config=config, ephem=ephem, queue=queue)
   ditl.calc()

   # Export plan with GSP entries
   ditl.plan.save("observation_plan.json")

   # GSP entries appear in the saved plan with obstype="GSP"
   for entry in ditl.plan:
       if entry.obstype == ObsType.GSP:
           print(f"Ground pass: {entry.station} from "
                 f"{entry.contact_begin} to {entry.contact_end}")

See :doc:`plan_serialization` for details on the GSP entry format and serialization.

Implementation Notes
--------------------

**Fixed-antenna pointing**: Ground-station-pass profiles use the unit body-frame vector in
``fixed_boresight_body``. The default ``(-1, 0, 0)`` preserves the historical tracking
convention. ``fixed_azimuth_deg`` and ``fixed_elevation_deg`` are retained only as deprecated
metadata and do not steer the generated profile.

**Omni antennas**: Always return ``True`` for ``can_communicate()`` regardless of pointing

**Multiple bands**: System can support multiple bands simultaneously (e.g., S-band for commands, X-band for science data)

**Data rates**: Set to 0.0 if a band doesn't support uplink or downlink in that direction

**Standard defaults**: Automatically applied when only band name is specified; override by providing explicit rates

Validation
----------

All configuration parameters are validated using Pydantic:

* Data rates must be non-negative
* Angles must be within valid ranges
* Band types must be from supported list
* See tests in ``tests/config/test_communications.py`` for examples

API Reference
-------------

For detailed API documentation, see:

* :py:class:`conops.config.BandCapability`
* :py:class:`conops.config.CommunicationsSystem`
* :py:class:`conops.config.AntennaPointing`
* :py:class:`conops.config.AntennaType`
* :py:class:`conops.config.Polarization`
* :py:class:`conops.config.GroundStation`
