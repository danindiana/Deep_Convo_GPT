#!/usr/bin/env python3
"""Basic connectivity test example.

This example demonstrates how to use the NIC diagnostics library
to test basic connectivity between network interfaces.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.nic_diagnostics import NICDiagnostics, TestResult
from src.utils.logging_config import setup_logging


def main() -> int:
    """Run basic connectivity tests.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup logging
    logger = setup_logging(
        log_file="basic_connectivity_test.log",
        log_level="INFO"
    )

    logger.info("=" * 60)
    logger.info("Basic Connectivity Test Example")
    logger.info("=" * 60)

    # Create diagnostics instance
    diag = NICDiagnostics()

    # Discover interfaces
    logger.info("Discovering network interfaces...")
    if not diag.discover_interfaces():
        logger.error("Failed to discover interfaces")
        return 1

    logger.info(f"Discovered {len(diag.interfaces)} interface(s)")

    # Test each interface with ping
    all_passed = True
    for interface in diag.interfaces:
        logger.info(f"\nTesting interface: {interface.name} ({interface.ip_address})")

        # Run ping test
        result = diag.ping_test(interface.ip_address, count=4)

        if result.success:
            logger.info(f"✓ {interface.name} is reachable")
        else:
            logger.error(f"✗ {interface.name} failed: {result.error_message}")
            all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All connectivity tests PASSED")
        return 0
    else:
        logger.error("Some connectivity tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
