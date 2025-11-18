"""Tests for NIC diagnostics module."""

import pytest
from unittest.mock import patch, MagicMock, call
import socket

from src.diagnostics.nic_diagnostics import NICDiagnostics, TestResult
from src.config import Config


class TestTestResult:
    """Tests for TestResult class."""

    def test_successful_result(self):
        """Test successful test result."""
        result = TestResult(
            success=True,
            throughput_mbps=950.5
        )

        assert result.success is True
        assert result.throughput_mbps == 950.5
        assert "950.50 MB/s" in str(result)

    def test_failed_result(self):
        """Test failed test result."""
        result = TestResult(
            success=False,
            error_message="Connection timeout"
        )

        assert result.success is False
        assert result.error_message == "Connection timeout"
        assert "Connection timeout" in str(result)


class TestNICDiagnostics:
    """Tests for NICDiagnostics class."""

    def test_initialization(self, test_config: Config):
        """Test NICDiagnostics initialization."""
        diag = NICDiagnostics(config=test_config)

        assert diag.config == test_config
        assert diag.interfaces == []

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        diag = NICDiagnostics()

        assert isinstance(diag.config, Config)
        assert diag.interfaces == []

    @patch('src.utils.interface_discovery.InterfaceDiscovery.discover_intel_x540_ports')
    def test_discover_interfaces_success(self, mock_discover, mock_interface_list):
        """Test successful interface discovery."""
        mock_discover.return_value = mock_interface_list

        diag = NICDiagnostics()
        result = diag.discover_interfaces()

        assert result is True
        assert len(diag.interfaces) == 2
        mock_discover.assert_called_once()

    @patch('src.utils.interface_discovery.InterfaceDiscovery.discover_intel_x540_ports')
    def test_discover_interfaces_failure(self, mock_discover):
        """Test interface discovery failure."""
        mock_discover.side_effect = RuntimeError("No interfaces found")

        diag = NICDiagnostics()
        result = diag.discover_interfaces()

        assert result is False
        assert len(diag.interfaces) == 0

    @patch('subprocess.run')
    def test_ping_test_success(self, mock_run):
        """Test successful ping test."""
        mock_run.return_value = MagicMock(returncode=0)

        diag = NICDiagnostics()
        result = diag.ping_test("192.168.1.1")

        assert result.success is True
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_ping_test_failure(self, mock_run):
        """Test failed ping test."""
        mock_run.return_value = MagicMock(returncode=1)

        diag = NICDiagnostics()
        result = diag.ping_test("192.168.1.1")

        assert result.success is False
        assert result.error_message is not None

    @patch('socket.socket')
    def test_start_listener_success(self, mock_socket):
        """Test successful listener setup."""
        mock_sock = MagicMock()
        mock_conn = MagicMock()
        mock_sock.accept.return_value = (mock_conn, ("192.168.1.100", 12345))
        mock_socket.return_value = mock_sock

        diag = NICDiagnostics()
        conn = diag.start_listener("192.168.1.1", 12345)

        assert conn is not None
        mock_sock.bind.assert_called_once_with(("192.168.1.1", 12345))
        mock_sock.listen.assert_called_once_with(1)

    @patch('socket.socket')
    def test_start_listener_timeout(self, mock_socket):
        """Test listener timeout."""
        mock_sock = MagicMock()
        mock_sock.accept.side_effect = socket.timeout()
        mock_socket.return_value = mock_sock

        diag = NICDiagnostics()
        diag.config.network.max_retries = 1  # Reduce retries for testing
        conn = diag.start_listener("192.168.1.1", 12345)

        assert conn is None

    def test_receive_data_no_connection(self):
        """Test receive_data with no connection."""
        diag = NICDiagnostics()
        result = diag.receive_data(None)

        assert result.success is False
        assert "No connection" in result.error_message


class TestNICDiagnosticsIntegration:
    """Integration tests for NICDiagnostics."""

    @patch('src.utils.interface_discovery.InterfaceDiscovery.discover_intel_x540_ports')
    @patch('subprocess.run')
    def test_basic_workflow(self, mock_run, mock_discover, mock_interface_list):
        """Test basic diagnostic workflow."""
        # Setup mocks
        mock_discover.return_value = mock_interface_list
        mock_run.return_value = MagicMock(returncode=0)

        # Run workflow
        diag = NICDiagnostics()

        # Discover interfaces
        assert diag.discover_interfaces() is True
        assert len(diag.interfaces) == 2

        # Ping test
        result = diag.ping_test(diag.interfaces[0].ip_address)
        assert result.success is True
