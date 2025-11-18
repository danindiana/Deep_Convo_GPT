#!/usr/bin/env python3
"""Throughput benchmark example.

This example demonstrates how to benchmark network throughput
between interfaces using both socket-based and iPerf methods.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.diagnostics.nic_diagnostics import NICDiagnostics
from src.diagnostics.iperf_test import IperfTester
from src.utils.logging_config import setup_logging


def print_results_table(results: Dict[str, float]) -> None:
    """Print results in a formatted table.

    Args:
        results: Dictionary of test name to throughput (Mbps)
    """
    print("\n" + "=" * 60)
    print("Throughput Benchmark Results")
    print("=" * 60)
    print(f"{'Test':<40} {'Throughput':>15}")
    print("-" * 60)

    for test_name, throughput in results.items():
        if throughput:
            print(f"{test_name:<40} {throughput:>12.2f} MB/s")
        else:
            print(f"{test_name:<40} {'FAILED':>15}")

    print("=" * 60)


def main() -> int:
    """Run throughput benchmarks.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup logging
    logger = setup_logging(
        log_file="throughput_benchmark.log",
        log_level="INFO"
    )

    logger.info("=" * 60)
    logger.info("Throughput Benchmark Example")
    logger.info("=" * 60)

    # Load configuration
    config = Config()

    # Create diagnostic instances
    nic_diag = NICDiagnostics(config=config)
    iperf_tester = IperfTester(config=config)

    # Discover interfaces
    logger.info("Discovering network interfaces...")
    if not nic_diag.discover_interfaces():
        logger.error("Failed to discover interfaces")
        return 1

    interfaces = nic_diag.interfaces
    logger.info(f"Discovered {len(interfaces)} interface(s)")

    if len(interfaces) < 2:
        logger.error("Need at least 2 interfaces for throughput testing")
        return 1

    # Results storage
    results: Dict[str, float] = {}

    # Test 1: Socket-based throughput
    logger.info("\n" + "-" * 60)
    logger.info("Test 1: Socket-based data transfer")
    logger.info("-" * 60)

    port = config.network.default_port
    data_size = 100  # Smaller size for quick testing

    logger.info(f"Testing {interfaces[0].name} -> {interfaces[1].name}")

    # This is simplified - in production, use threading for listener
    # send_result = nic_diag.send_data(
    #     interfaces[1].ip_address,
    #     port,
    #     data_size
    # )

    # For this example, we'll mark as N/A
    results["Socket Transfer"] = None  # Would contain actual result

    # Test 2: iPerf throughput
    logger.info("\n" + "-" * 60)
    logger.info("Test 2: iPerf throughput test")
    logger.info("-" * 60)

    if iperf_tester.check_iperf_installed():
        # Run bidirectional test
        result1, result2 = iperf_tester.run_bidirectional_test(
            interfaces[0],
            interfaces[1]
        )

        if result1 and result1.success:
            results[f"iPerf: {interfaces[0].name} -> {interfaces[1].name}"] = \
                result1.bandwidth_mbps
        else:
            results[f"iPerf: {interfaces[0].name} -> {interfaces[1].name}"] = None

        if result2 and result2.success:
            results[f"iPerf: {interfaces[1].name} -> {interfaces[0].name}"] = \
                result2.bandwidth_mbps
        else:
            results[f"iPerf: {interfaces[1].name} -> {interfaces[0].name}"] = None
    else:
        logger.warning("iPerf not installed, skipping iPerf tests")
        results["iPerf Test"] = None

    # Print summary table
    print_results_table(results)

    # Determine success
    successful_tests = sum(1 for v in results.values() if v is not None and v > 0)
    total_tests = len(results)

    logger.info(f"\nCompleted {successful_tests}/{total_tests} tests successfully")

    return 0 if successful_tests > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
