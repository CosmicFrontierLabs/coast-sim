#!/usr/bin/env python3
"""Benchmark script for constraint caching performance.

This script measures the effectiveness of constraint result caching by
simulating a typical DITL workload where the same constraints are checked
multiple times per timestep.

Runs actual A/B comparison:
- WITH cache: normal operation
- WITHOUT cache: clear cache after each check to force recomputation

Usage:
    python scripts/benchmark_constraint_cache.py

"""

import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from rust_ephem import TLEEphemeris

from conops.config.constraint import Constraint


def run_benchmark(
    num_timesteps: int = 100,
    num_targets: int = 20,
    checks_per_target: int = 8,
) -> dict[str, Any]:
    """Run constraint caching benchmark with actual A/B comparison.

    Args:
        num_timesteps: Number of simulation timesteps (default 100 = ~17 min at 10s)
        num_targets: Number of distinct pointing targets
        checks_per_target: How many times each target is checked per timestep
            (simulates redundant checks from scheduler, emergency charging, etc.)

    Returns:
        Dictionary with benchmark results
    """
    # Setup ephemeris
    begin = datetime(2025, 1, 1)
    end = begin + timedelta(hours=1)
    ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end, step_size=10)

    # Generate random targets
    np.random.seed(42)
    target_ras = np.random.uniform(0, 360, num_targets)
    target_decs = np.random.uniform(-90, 90, num_targets)

    # Generate timesteps
    timesteps = [begin.timestamp() + i * 10 for i in range(num_timesteps)]

    total_checks = num_timesteps * num_targets * checks_per_target
    print("Benchmark configuration:")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Targets: {num_targets}")
    print(f"  Checks per target per step: {checks_per_target}")
    print(f"  Total constraint checks: {total_checks:,}")
    print()

    # =========================================================================
    # Run WITH cache (normal operation)
    # =========================================================================
    print("Running WITH cache...")
    constraint_cached = Constraint()
    constraint_cached.ephem = ephem
    constraint_cached.clear_cache()
    constraint_cached._cache_hits = 0
    constraint_cached._cache_misses = 0

    start = time.perf_counter()
    violations_cached = 0

    for t in timesteps:
        for ra, dec in zip(target_ras, target_decs):
            for _ in range(checks_per_target):
                if constraint_cached.in_constraint(ra, dec, t):
                    violations_cached += 1

    elapsed_cached = time.perf_counter() - start
    hits, misses = constraint_cached.cache_stats()

    # =========================================================================
    # Run WITHOUT cache (clear after each check to force recomputation)
    # =========================================================================
    print("Running WITHOUT cache...")
    constraint_uncached = Constraint()
    constraint_uncached.ephem = ephem

    start = time.perf_counter()
    violations_uncached = 0

    for t in timesteps:
        for ra, dec in zip(target_ras, target_decs):
            for _ in range(checks_per_target):
                constraint_uncached.clear_cache()  # Force cache miss
                if constraint_uncached.in_constraint(ra, dec, t):
                    violations_uncached += 1

    elapsed_uncached = time.perf_counter() - start

    # =========================================================================
    # Results
    # =========================================================================
    hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    speedup = elapsed_uncached / elapsed_cached if elapsed_cached > 0 else 0
    time_saved = elapsed_uncached - elapsed_cached

    results = {
        "total_checks": total_checks,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": hit_rate,
        "elapsed_cached_sec": elapsed_cached,
        "elapsed_uncached_sec": elapsed_uncached,
        "speedup": speedup,
        "time_saved_sec": time_saved,
        "violations": violations_cached,
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nCache statistics:")
    print(f"  Cache hits: {hits:,}")
    print(f"  Cache misses: {misses:,}")
    print(f"  Hit rate: {hit_rate:.1f}%")
    print(f"  Cache entries: {len(constraint_cached._cache):,}")
    print("\nTiming:")
    print(f"  WITH cache:    {elapsed_cached:.3f}s")
    print(f"  WITHOUT cache: {elapsed_uncached:.3f}s")
    print(f"  Speedup:       {speedup:.2f}x")
    print(
        f"  Time saved:    {time_saved:.3f}s ({time_saved / elapsed_uncached * 100:.0f}%)"
    )
    print(f"\nViolations found: {violations_cached:,}")

    # Sanity check - both runs should find same violations
    if violations_cached != violations_uncached:
        print(
            f"\nWARNING: Violation count mismatch! cached={violations_cached}, uncached={violations_uncached}"
        )

    return results


if __name__ == "__main__":
    run_benchmark()
