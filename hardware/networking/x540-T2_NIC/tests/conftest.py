"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.config import Config, NetworkConfig, LoggingConfig
from src.utils.interface_discovery import NetworkInterface


@pytest.fixture
def test_config() -> Config:
    """Create a test configuration."""
    network_config = NetworkConfig(
        data_size_mb=10,  # Small size for testing
        block_size=1024,
        listener_timeout=5,
        max_retries=2,
        iperf_duration=5,
        default_port=15000
    )

    logging_config = LoggingConfig(
        log_file="test_network.log",
        log_level="DEBUG",
        console_output=False
    )

    return Config(network=network_config, logging=logging_config)


@pytest.fixture
def mock_interface() -> NetworkInterface:
    """Create a mock network interface."""
    return NetworkInterface(
        name="eth0",
        ip_address="192.168.1.100",
        mac_address="00:11:22:33:44:55",
        is_up=True,
        speed=10000
    )


@pytest.fixture
def mock_interface_list() -> list[NetworkInterface]:
    """Create a list of mock network interfaces."""
    return [
        NetworkInterface(
            name="enp3s0f0",
            ip_address="192.168.10.1",
            mac_address="00:11:22:33:44:55",
            is_up=True,
            speed=10000
        ),
        NetworkInterface(
            name="enp3s0f1",
            ip_address="192.168.10.2",
            mac_address="00:11:22:33:44:66",
            is_up=True,
            speed=10000
        ),
    ]


@pytest.fixture
def temp_config_file(tmp_path: Path, test_config: Config) -> Path:
    """Create a temporary configuration file."""
    config_file = tmp_path / "test_config.yaml"
    test_config.to_yaml(config_file)
    return config_file


@pytest.fixture(autouse=True)
def cleanup_logs(tmp_path: Path):
    """Cleanup test log files after each test."""
    yield
    # Cleanup code runs after test
    for log_file in Path(".").glob("test_*.log"):
        log_file.unlink(missing_ok=True)
