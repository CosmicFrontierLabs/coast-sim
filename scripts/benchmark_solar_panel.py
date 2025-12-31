#!/usr/bin/env python3
"""Benchmark script for solar panel illumination vectorization.

Runs A/B comparison between:
- OLD: Loop over panels, calling panel_illumination_fraction() for each
- NEW: Vectorized implementation with single sun/eclipse lookup

Usage:
    python scripts/benchmark_solar_panel.py
"""

import time
from datetime import datetime, timedelta

from rust_ephem import TLEEphemeris

from conops.config.solar_panel import SolarPanel, SolarPanelSet


def create_test_panel_set() -> SolarPanelSet:
    """Create a realistic multi-panel configuration."""
    panels = [
        SolarPanel(name="Panel1", sidemount=True, azimuth_deg=0, max_power=800),
        SolarPanel(name="Panel2", sidemount=True, azimuth_deg=180, max_power=800),
        SolarPanel(
            name="Panel3", sidemount=True, azimuth_deg=90, cant_x=15, max_power=400
        ),
        SolarPanel(
            name="Panel4", sidemount=True, azimuth_deg=270, cant_x=15, max_power=400
        ),
    ]
    return SolarPanelSet(panels=panels)


def run_benchmark(num_timesteps: int = 4320) -> dict:
    """Run A/B benchmark comparing old loop vs new vectorized implementation.

    Args:
        num_timesteps: Number of simulation steps (default 4320 = 12 hours at 10s)

    Returns:
        Dictionary with benchmark results
    """
    # Setup ephemeris (12 hours)
    begin = datetime(2025, 1, 1)
    end = begin + timedelta(hours=12)
    ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end, step_size=10)

    # Create two panel sets (separate instances to avoid cache contamination)
    panel_set_old = create_test_panel_set()
    panel_set_new = create_test_panel_set()

    # Generate timesteps (unix timestamps)
    timesteps = [begin.timestamp() + i * 10 for i in range(num_timesteps)]

    # Fixed pointing for benchmark
    ra, dec = 180.0, 45.0

    print("Benchmark configuration:")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Panels: {len(panel_set_old.panels)}")
    print(f"  Total illumination_and_power() calls: {num_timesteps:,}")
    print()

    # Warm up both implementations
    _ = panel_set_old._illumination_and_power_loop([begin], ra, dec, ephem)
    _ = panel_set_new.illumination_and_power(
        time=timesteps[0], ra=ra, dec=dec, ephem=ephem
    )

    # =========================================================================
    # OLD: Loop-based implementation
    # =========================================================================
    print("Running OLD (loop-based)...")
    old_illum_total = 0.0
    old_power_total = 0.0

    start = time.perf_counter()
    for t in timesteps:
        # Use the old loop implementation by wrapping as list
        illum, power = panel_set_old._illumination_and_power_loop(
            [begin + timedelta(seconds=t - begin.timestamp())], ra, dec, ephem
        )
        old_illum_total += float(illum[0])
        old_power_total += float(power[0])
    elapsed_old = time.perf_counter() - start

    # =========================================================================
    # NEW: Vectorized implementation
    # =========================================================================
    print("Running NEW (vectorized)...")
    new_illum_total = 0.0
    new_power_total = 0.0

    start = time.perf_counter()
    for t in timesteps:
        illum, power = panel_set_new.illumination_and_power(
            time=t, ra=ra, dec=dec, ephem=ephem
        )
        new_illum_total += illum
        new_power_total += power
    elapsed_new = time.perf_counter() - start

    # =========================================================================
    # Results
    # =========================================================================
    speedup = elapsed_old / elapsed_new if elapsed_new > 0 else 0
    time_saved = elapsed_old - elapsed_new
    pct_saved = (time_saved / elapsed_old * 100) if elapsed_old > 0 else 0

    # Verify results match
    illum_match = abs(old_illum_total - new_illum_total) < 0.01 * num_timesteps
    power_match = abs(old_power_total - new_power_total) < 0.01 * num_timesteps

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("Timing:")
    print(f"  OLD (loop):       {elapsed_old:.3f}s")
    print(f"  NEW (vectorized): {elapsed_new:.3f}s")
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Time saved:       {time_saved:.3f}s ({pct_saved:.0f}%)")
    print()
    print("Correctness check:")
    print(f"  Illumination match: {'PASS' if illum_match else 'FAIL'}")
    print(f"  Power match:        {'PASS' if power_match else 'FAIL'}")
    if not illum_match:
        print(f"    OLD illum: {old_illum_total:.2f}, NEW illum: {new_illum_total:.2f}")
    if not power_match:
        print(f"    OLD power: {old_power_total:.2f}, NEW power: {new_power_total:.2f}")

    return {
        "elapsed_old": elapsed_old,
        "elapsed_new": elapsed_new,
        "speedup": speedup,
        "time_saved": time_saved,
        "pct_saved": pct_saved,
        "illum_match": illum_match,
        "power_match": power_match,
    }


if __name__ == "__main__":
    run_benchmark()
