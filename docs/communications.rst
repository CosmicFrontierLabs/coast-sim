Communications System Configuration
====================================

Overview
--------

The Communications System configuration defines the onboard communications capabilities for spacecraft downlink and uplink operations. It supports multiple frequency bands, different antenna types, and configurable data rates for realistic mission planning and link budget analysis.

Key Features
------------

* **Multiple frequency bands**: S, X, Ka, Ku, L, C, K bands
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

* ``band``: Frequency band identifier (``"S"``, ``"X"``, ``"Ka"``, ``"Ku"``, ``"L"``, ``"C"``, ``"K"``)
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

Explicitly specified values override these defaults.

AntennaPointing
^^^^^^^^^^^^^^^

Defines how the antenna is mounted and pointed.

**Antenna Types:**

* ``OMNI``: Omnidirectional antenna (no pointing requirement)
* ``FIXED``: Body-fixed antenna with defined pointing direction
* ``GIMBALED``: Steerable antenna with angular range

**Fields:**

* ``antenna_type``: Type of antenna mounting
* ``fixed_azimuth_deg``: Azimuth in spacecraft body frame (0° = nadir, 180° = zenith)
* ``fixed_elevation_deg``: Elevation angle in body frame
* ``gimbal_range_deg``: Angular range for gimbaled antennas (from boresight)

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

   # LEO satellite with nadir-pointing S-band (using defaults)
   leo_comms = CommunicationsSystem(
       name="LEO S-band",
       band_capabilities=[
           BandCapability(band="S")  # Uses defaults: 2.0 Mbps up, 10.0 Mbps down
       ],
       antenna_pointing=AntennaPointing(
           antenna_type=AntennaType.FIXED,
           fixed_azimuth_deg=0.0,  # Nadir pointing
           fixed_elevation_deg=0.0,
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

* LEO S-band (nadir-pointing)
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

Implementation Notes
--------------------

**Nadir pointing**: Default for fixed antennas (azimuth=0°, elevation=0°), points opposite to telescope/payload

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
