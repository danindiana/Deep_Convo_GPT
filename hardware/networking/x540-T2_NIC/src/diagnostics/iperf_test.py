"""iPerf-based network testing."""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from src.config import Config
from src.utils.interface_discovery import InterfaceDiscovery, NetworkInterface
from src.utils.logging_config import LoggerMixin


logger = logging.getLogger(__name__)


@dataclass
class IperfResult:
    """Results from an iPerf test."""

    success: bool
    bandwidth_mbps: Optional[float] = None
    jitter_ms: Optional[float] = None
    packet_loss: Optional[float] = None
    error_message: Optional[str] = None
    raw_output: Optional[str] = None

    def __str__(self) -> str:
        """String representation of iPerf results."""
        if self.success and self.bandwidth_mbps:
            return f"Bandwidth: {self.bandwidth_mbps:.2f} Mbps"
        return f"Failed - {self.error_message}"


class IperfTester(LoggerMixin):
    """iPerf-based network performance testing."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize iPerf tester.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.server_process: Optional[subprocess.Popen] = None

    def check_iperf_installed(self) -> bool:
        """Check if iPerf is installed on the system.

        Returns:
            True if iPerf is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['which', 'iperf'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Error checking for iPerf: {e}")
            return False

    def start_server(self, server_ip: str, port: int = 5201) -> bool:
        """Start iPerf server.

        Args:
            server_ip: IP address to bind server to
            port: Port number for server

        Returns:
            True if server started successfully
        """
        if not self.check_iperf_installed():
            self.logger.error("iPerf is not installed. Please install it first.")
            return False

        try:
            self.logger.info(f"Starting iPerf server on {server_ip}:{port}")

            cmd = ['iperf', '-s', '-B', server_ip, '-p', str(port), '-i', '1']

            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(2)  # Give server time to start

            if self.server_process.poll() is not None:
                self.logger.error("iPerf server failed to start")
                return False

            self.logger.info("iPerf server started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting iPerf server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the iPerf server if running."""
        if self.server_process:
            self.logger.info("Stopping iPerf server...")
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.server_process = None

    def run_client(
        self,
        server_ip: str,
        client_ip: str,
        duration: Optional[int] = None,
        port: int = 5201
    ) -> IperfResult:
        """Run iPerf client test.

        Args:
            server_ip: Server IP address
            client_ip: Client IP address (bind address)
            duration: Test duration in seconds
            port: Server port number

        Returns:
            IperfResult with test results
        """
        if duration is None:
            duration = self.config.network.iperf_duration

        try:
            self.logger.info(
                f"Starting iPerf client: {client_ip} -> {server_ip} "
                f"for {duration} seconds"
            )

            cmd = [
                'iperf',
                '-c', server_ip,
                '-B', client_ip,
                '-p', str(port),
                '-t', str(duration),
                '-f', 'm'  # Format output in Mbits/sec
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=duration + 10,
                check=False
            )

            output = result.stdout.decode()
            self.logger.info(f"iPerf client output:\n{output}")

            if result.returncode == 0:
                # Parse bandwidth from output
                bandwidth = self._parse_bandwidth(output)
                return IperfResult(
                    success=True,
                    bandwidth_mbps=bandwidth,
                    raw_output=output
                )
            else:
                error_msg = result.stderr.decode()
                self.logger.error(f"iPerf client failed: {error_msg}")
                return IperfResult(
                    success=False,
                    error_message=error_msg,
                    raw_output=output
                )

        except subprocess.TimeoutExpired:
            error_msg = "iPerf client timeout"
            self.logger.error(error_msg)
            return IperfResult(success=False, error_message=error_msg)

        except Exception as e:
            error_msg = f"iPerf client error: {str(e)}"
            self.logger.error(error_msg)
            return IperfResult(success=False, error_message=error_msg)

    def _parse_bandwidth(self, output: str) -> Optional[float]:
        """Parse bandwidth from iPerf output.

        Args:
            output: iPerf output text

        Returns:
            Bandwidth in Mbps or None if parsing failed
        """
        try:
            # Look for lines with bandwidth information
            for line in output.split('\n'):
                if 'Mbits/sec' in line or 'Gbits/sec' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'bits/sec' in part and i > 0:
                            bandwidth = float(parts[i - 1])
                            if 'Gbits/sec' in part:
                                bandwidth *= 1000  # Convert to Mbps
                            return bandwidth
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Failed to parse bandwidth: {e}")

        return None

    def run_bidirectional_test(
        self,
        iface1: NetworkInterface,
        iface2: NetworkInterface
    ) -> tuple[IperfResult, IperfResult]:
        """Run bidirectional iPerf test.

        Args:
            iface1: First network interface
            iface2: Second network interface

        Returns:
            Tuple of (test1_result, test2_result)
        """
        self.logger.info(
            f"Running bidirectional test between {iface1.name} and {iface2.name}"
        )

        results = [None, None]

        def test1():
            if self.start_server(iface1.ip_address):
                results[0] = self.run_client(iface1.ip_address, iface2.ip_address)
                self.stop_server()

        def test2():
            time.sleep(1)  # Offset start time
            if self.start_server(iface2.ip_address):
                results[1] = self.run_client(iface2.ip_address, iface1.ip_address)
                self.stop_server()

        thread1 = threading.Thread(target=test1)
        thread2 = threading.Thread(target=test2)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        return results[0], results[1]

    def run_interface_tests(
        self,
        interfaces: Optional[List[NetworkInterface]] = None
    ) -> List[IperfResult]:
        """Run iPerf tests on discovered interfaces.

        Args:
            interfaces: List of interfaces to test (discovers if None)

        Returns:
            List of test results
        """
        if interfaces is None:
            try:
                interfaces = InterfaceDiscovery.discover_intel_x540_ports()
            except RuntimeError as e:
                self.logger.error(f"Failed to discover interfaces: {e}")
                return []

        if len(interfaces) < 2:
            self.logger.error("Need at least 2 interfaces for testing")
            return []

        results = []

        # Test all interface pairs
        for i in range(len(interfaces) - 1):
            result1, result2 = self.run_bidirectional_test(
                interfaces[i],
                interfaces[i + 1]
            )
            results.extend([result1, result2])

        return results
