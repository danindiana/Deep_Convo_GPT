"""Tests for interface discovery module."""

import pytest
from unittest.mock import patch, MagicMock
import socket

from src.utils.interface_discovery import (
    NetworkInterface,
    InterfaceDiscovery
)


class TestNetworkInterface:
    """Tests for NetworkInterface class."""

    def test_initialization(self):
        """Test interface initialization."""
        interface = NetworkInterface(
            name="eth0",
            ip_address="192.168.1.100",
            mac_address="00:11:22:33:44:55",
            is_up=True,
            speed=10000
        )

        assert interface.name == "eth0"
        assert interface.ip_address == "192.168.1.100"
        assert interface.mac_address == "00:11:22:33:44:55"
        assert interface.is_up is True
        assert interface.speed == 10000

    def test_str_representation(self):
        """Test string representation."""
        interface = NetworkInterface(
            name="eth0",
            ip_address="192.168.1.100"
        )

        assert str(interface) == "eth0 (192.168.1.100)"

    def test_optional_fields(self):
        """Test that optional fields work."""
        interface = NetworkInterface(
            name="eth0",
            ip_address="192.168.1.100"
        )

        assert interface.mac_address is None
        assert interface.is_up is True  # Default value
        assert interface.speed is None


class TestInterfaceDiscovery:
    """Tests for InterfaceDiscovery class."""

    @patch('psutil.net_if_addrs')
    @patch('psutil.net_if_stats')
    def test_get_all_interfaces(self, mock_stats, mock_addrs):
        """Test getting all interfaces."""
        # Mock network interface data
        mock_addr = MagicMock()
        mock_addr.family = socket.AF_INET
        mock_addr.address = "192.168.1.100"

        mock_addrs.return_value = {
            'eth0': [mock_addr],
            'lo': [mock_addr],  # Should be skipped
        }

        mock_stat = MagicMock()
        mock_stat.isup = True
        mock_stat.speed = 1000
        mock_stats.return_value = {
            'eth0': mock_stat,
        }

        interfaces = InterfaceDiscovery.get_all_interfaces()

        assert len(interfaces) == 1
        assert interfaces[0].name == "eth0"
        assert interfaces[0].ip_address == "192.168.1.100"

    @patch('psutil.net_if_addrs')
    @patch('psutil.net_if_stats')
    def test_discover_intel_x540_ports_success(self, mock_stats, mock_addrs):
        """Test successful discovery of Intel NIC ports."""
        # Mock two ethernet interfaces
        mock_addr1 = MagicMock()
        mock_addr1.family = socket.AF_INET
        mock_addr1.address = "192.168.10.1"

        mock_addr2 = MagicMock()
        mock_addr2.family = socket.AF_INET
        mock_addr2.address = "192.168.10.2"

        mock_addrs.return_value = {
            'enp3s0f0': [mock_addr1],
            'enp3s0f1': [mock_addr2],
        }

        mock_stat = MagicMock()
        mock_stat.isup = True
        mock_stat.speed = 10000
        mock_stats.return_value = {
            'enp3s0f0': mock_stat,
            'enp3s0f1': mock_stat,
        }

        interfaces = InterfaceDiscovery.discover_intel_x540_ports()

        assert len(interfaces) == 2
        assert interfaces[0].name == "enp3s0f0"
        assert interfaces[1].name == "enp3s0f1"

    @patch('psutil.net_if_addrs')
    def test_discover_intel_x540_ports_insufficient(self, mock_addrs):
        """Test error when insufficient ports found."""
        # Mock only one interface
        mock_addr = MagicMock()
        mock_addr.family = socket.AF_INET
        mock_addr.address = "192.168.10.1"

        mock_addrs.return_value = {
            'enp3s0f0': [mock_addr],
        }

        with pytest.raises(RuntimeError, match="Failed to discover two Intel NIC ports"):
            InterfaceDiscovery.discover_intel_x540_ports()

    @patch('psutil.net_if_addrs')
    def test_get_ip_address_success(self, mock_addrs):
        """Test getting IP address for an interface."""
        mock_addr = MagicMock()
        mock_addr.family = socket.AF_INET
        mock_addr.address = "192.168.1.100"

        mock_addrs.return_value = {
            'eth0': [mock_addr],
        }

        ip = InterfaceDiscovery.get_ip_address('eth0')

        assert ip == "192.168.1.100"

    @patch('psutil.net_if_addrs')
    def test_get_ip_address_not_found(self, mock_addrs):
        """Test getting IP for nonexistent interface."""
        mock_addrs.return_value = {}

        ip = InterfaceDiscovery.get_ip_address('nonexistent')

        assert ip is None
