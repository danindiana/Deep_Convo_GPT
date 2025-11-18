"""NIC diagnostic tools for throughput testing."""

import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from tqdm import tqdm

from src.config import Config
from src.utils.interface_discovery import InterfaceDiscovery, NetworkInterface
from src.utils.logging_config import LoggerMixin


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a network test."""

    success: bool
    throughput_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None

    def __str__(self) -> str:
        """String representation of test results."""
        if self.success:
            return f"Success - Throughput: {self.throughput_mbps:.2f} MB/s"
        return f"Failed - {self.error_message}"


class NICDiagnostics(LoggerMixin):
    """Comprehensive NIC diagnostic and throughput testing."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize NIC diagnostics.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.interfaces = []

    def discover_interfaces(self) -> bool:
        """Discover Intel X540-T2 NIC interfaces.

        Returns:
            True if discovery successful, False otherwise
        """
        try:
            self.interfaces = InterfaceDiscovery.discover_intel_x540_ports()
            self.logger.info(
                f"Discovered {len(self.interfaces)} interfaces: "
                f"{', '.join(str(i) for i in self.interfaces)}"
            )
            return True
        except RuntimeError as e:
            self.logger.error(f"Interface discovery failed: {e}")
            return False

    def ping_test(self, ip_address: str, count: int = 4) -> TestResult:
        """Test connectivity using ping.

        Args:
            ip_address: IP address to ping
            count: Number of ping packets

        Returns:
            TestResult with ping statistics
        """
        self.logger.info(f"Pinging {ip_address} to check connectivity...")

        try:
            result = subprocess.run(
                ['ping', '-c', str(count), ip_address],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False
            )

            if result.returncode == 0:
                self.logger.info(f"Ping successful to {ip_address}")
                # Parse latency from output if needed
                return TestResult(success=True)
            else:
                error_msg = f"Ping failed to {ip_address}"
                self.logger.error(error_msg)
                return TestResult(success=False, error_message=error_msg)

        except subprocess.TimeoutExpired:
            error_msg = f"Ping timeout for {ip_address}"
            self.logger.error(error_msg)
            return TestResult(success=False, error_message=error_msg)
        except Exception as e:
            error_msg = f"Ping error: {str(e)}"
            self.logger.error(error_msg)
            return TestResult(success=False, error_message=error_msg)

    def start_listener(
        self,
        ip_address: str,
        port: int
    ) -> Optional[socket.socket]:
        """Start a listener socket with retry logic.

        Args:
            ip_address: IP address to bind to
            port: Port number to listen on

        Returns:
            Connected socket or None if failed
        """
        max_retries = self.config.network.max_retries
        timeout = self.config.network.listener_timeout

        for attempt in range(max_retries):
            self.logger.info(
                f"Setting up listener on {ip_address}:{port} "
                f"(Attempt {attempt + 1}/{max_retries})..."
            )

            listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                listener_socket.bind((ip_address, port))
                listener_socket.listen(1)
                listener_socket.settimeout(timeout)

                self.logger.info(f"Listening on {ip_address}:{port} for incoming data...")
                conn, addr = listener_socket.accept()
                self.logger.info(f"Connection accepted from {addr}")
                return conn

            except socket.timeout:
                self.logger.warning(
                    f"Listener timed out after {timeout} seconds on {ip_address}:{port}"
                )
            except Exception as e:
                self.logger.error(f"Listener error: {e}")
            finally:
                listener_socket.close()

        self.logger.error(
            f"Failed to establish listener connection after {max_retries} attempts"
        )
        return None

    def send_data(
        self,
        ip_address: str,
        port: int,
        data_size_mb: Optional[int] = None
    ) -> TestResult:
        """Send data and measure throughput.

        Args:
            ip_address: Destination IP address
            port: Destination port
            data_size_mb: Amount of data to send in MB

        Returns:
            TestResult with throughput information
        """
        if data_size_mb is None:
            data_size_mb = self.config.network.data_size_mb

        block_size = self.config.network.block_size

        self.logger.info(
            f"Attempting to connect to {ip_address}:{port} "
            f"for data transfer of {data_size_mb}MB..."
        )

        time.sleep(2)  # Ensure listener is ready

        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sender_socket.connect((ip_address, port))
            self.logger.info(f"Connected to {ip_address}:{port}, starting data transfer")

            total_sent = 0
            total_bytes = data_size_mb * block_size
            start_time = time.time()

            with tqdm(total=data_size_mb, unit='MB', desc="Sending") as pbar:
                while total_sent < total_bytes:
                    chunk = b'\0' * block_size
                    sender_socket.sendall(chunk)
                    total_sent += len(chunk)
                    pbar.update(1)

            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = data_size_mb / elapsed_time

            self.logger.info(f"Data transfer complete. Throughput: {throughput:.2f} MB/s")

            return TestResult(success=True, throughput_mbps=throughput)

        except Exception as e:
            error_msg = f"Error during data transfer: {str(e)}"
            self.logger.error(error_msg)
            return TestResult(success=False, error_message=error_msg)

        finally:
            sender_socket.close()

    def receive_data(
        self,
        conn: socket.socket,
        data_size_mb: Optional[int] = None
    ) -> TestResult:
        """Receive data from a socket.

        Args:
            conn: Connected socket
            data_size_mb: Expected amount of data in MB

        Returns:
            TestResult with reception information
        """
        if data_size_mb is None:
            data_size_mb = self.config.network.data_size_mb

        if not conn:
            error_msg = "No connection established for receiving data"
            self.logger.error(error_msg)
            return TestResult(success=False, error_message=error_msg)

        block_size = self.config.network.block_size
        self.logger.info(f"Receiving data of {data_size_mb}MB...")

        try:
            total_received = 0
            total_bytes = data_size_mb * block_size
            start_time = time.time()

            with tqdm(total=data_size_mb, unit='MB', desc="Receiving") as pbar:
                while total_received < total_bytes:
                    data = conn.recv(block_size)
                    if not data:
                        break
                    total_received += len(data)
                    pbar.update(len(data) / block_size)

            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = (total_received / block_size) / elapsed_time

            self.logger.info("Data reception complete")
            return TestResult(success=True, throughput_mbps=throughput)

        except Exception as e:
            error_msg = f"Error during data reception: {str(e)}"
            self.logger.error(error_msg)
            return TestResult(success=False, error_message=error_msg)

        finally:
            conn.close()

    def run_full_diagnostic(self) -> bool:
        """Run complete diagnostic test suite.

        Returns:
            True if all tests passed, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Full NIC Diagnostic Test")
        self.logger.info("=" * 60)

        # Discover interfaces
        if not self.discover_interfaces():
            return False

        if len(self.interfaces) < 2:
            self.logger.error("Need at least 2 interfaces for testing")
            return False

        port_a, port_b = self.interfaces[0], self.interfaces[1]
        test_port = self.config.network.default_port

        # Test connectivity
        ping_result = self.ping_test(port_b.ip_address)
        if not ping_result.success:
            self.logger.error("Ping test failed, aborting further tests")
            return False

        # Throughput test
        self.logger.info("Starting throughput test between ports...")

        # Start listener in background (in real implementation, use threading)
        listener_conn = self.start_listener(port_b.ip_address, test_port)

        if not listener_conn:
            self.logger.error("Failed to establish listener connection")
            return False

        # Send data
        send_result = self.send_data(port_b.ip_address, test_port)
        recv_result = self.receive_data(listener_conn)

        # Report results
        self.logger.info("=" * 60)
        self.logger.info("Test Results")
        self.logger.info("=" * 60)
        self.logger.info(f"Send test: {send_result}")
        self.logger.info(f"Receive test: {recv_result}")

        return send_result.success and recv_result.success
