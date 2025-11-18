"""Network interface discovery utilities."""

import logging
import socket
from dataclasses import dataclass
from typing import List, Optional, Tuple

import psutil


logger = logging.getLogger(__name__)


@dataclass
class NetworkInterface:
    """Represents a network interface."""

    name: str
    ip_address: str
    mac_address: Optional[str] = None
    is_up: bool = True
    speed: Optional[int] = None  # Mbps

    def __str__(self) -> str:
        """String representation of the interface."""
        return f"{self.name} ({self.ip_address})"


class InterfaceDiscovery:
    """Discover and manage network interfaces."""

    @staticmethod
    def get_all_interfaces() -> List[NetworkInterface]:
        """Get all available network interfaces.

        Returns:
            List of NetworkInterface objects
        """
        interfaces = []

        for iface_name, iface_addrs in psutil.net_if_addrs().items():
            # Skip loopback
            if iface_name == 'lo':
                continue

            ip_addr = None
            mac_addr = None

            for addr in iface_addrs:
                if addr.family == socket.AF_INET:
                    ip_addr = addr.address
                elif addr.family == psutil.AF_LINK:
                    mac_addr = addr.address

            if ip_addr:
                # Get interface stats
                stats = psutil.net_if_stats().get(iface_name)
                is_up = stats.isup if stats else False
                speed = stats.speed if stats else None

                interfaces.append(NetworkInterface(
                    name=iface_name,
                    ip_address=ip_addr,
                    mac_address=mac_addr,
                    is_up=is_up,
                    speed=speed
                ))

        return interfaces

    @staticmethod
    def discover_intel_x540_ports() -> List[NetworkInterface]:
        """Discover Intel X540-T2 NIC ports specifically.

        Returns:
            List of NetworkInterface objects for Intel NICs

        Raises:
            RuntimeError: If fewer than 2 Intel NIC ports are found
        """
        logger.info("Starting interface discovery for Intel X540-T2 NIC...")
        intel_ports = []

        for iface_name, iface_addrs in psutil.net_if_addrs().items():
            logger.debug(f"Checking interface {iface_name}")

            # Look for ethernet interfaces (common naming patterns)
            if any(prefix in iface_name for prefix in ['eth', 'enp', 'eno']):
                ip_addr = None
                mac_addr = None

                for addr in iface_addrs:
                    if addr.family == socket.AF_INET:
                        ip_addr = addr.address
                    elif addr.family == psutil.AF_LINK:
                        mac_addr = addr.address

                if ip_addr:
                    stats = psutil.net_if_stats().get(iface_name)
                    is_up = stats.isup if stats else False
                    speed = stats.speed if stats else None

                    interface = NetworkInterface(
                        name=iface_name,
                        ip_address=ip_addr,
                        mac_address=mac_addr,
                        is_up=is_up,
                        speed=speed
                    )

                    intel_ports.append(interface)
                    logger.info(f"Discovered Intel port: {interface}")

        if len(intel_ports) < 2:
            error_msg = (
                f"Failed to discover two Intel NIC ports. "
                f"Found {len(intel_ports)} port(s). "
                f"Check NIC configuration and ensure interfaces are up."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Successfully discovered {len(intel_ports)} Intel X540-T2 NIC ports")
        return intel_ports[:2]  # Return first two

    @staticmethod
    def get_interface_by_name(name: str) -> Optional[NetworkInterface]:
        """Get specific interface by name.

        Args:
            name: Interface name (e.g., 'eth0')

        Returns:
            NetworkInterface object or None if not found
        """
        interfaces = InterfaceDiscovery.get_all_interfaces()
        for iface in interfaces:
            if iface.name == name:
                return iface
        return None

    @staticmethod
    def get_ip_address(iface_name: str) -> Optional[str]:
        """Get IP address for a specific interface.

        Args:
            iface_name: Name of the network interface

        Returns:
            IP address string or None if not found
        """
        addrs = psutil.net_if_addrs()
        if iface_name in addrs:
            for addr in addrs[iface_name]:
                if addr.family == socket.AF_INET:
                    return addr.address

        logger.warning(f"Could not retrieve IP address for {iface_name}")
        return None
