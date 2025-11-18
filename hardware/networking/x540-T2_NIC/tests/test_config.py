"""Tests for configuration module."""

import pytest
from pathlib import Path

from src.config import Config, NetworkConfig, LoggingConfig


class TestNetworkConfig:
    """Tests for NetworkConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NetworkConfig()

        assert config.data_size_mb == 500
        assert config.block_size == 1024 * 1024
        assert config.listener_timeout == 15
        assert config.max_retries == 3
        assert config.iperf_duration == 10
        assert config.default_port == 12345

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NetworkConfig(
            data_size_mb=1000,
            block_size=2048,
            listener_timeout=30,
            max_retries=5,
            iperf_duration=20,
            default_port=15000
        )

        assert config.data_size_mb == 1000
        assert config.block_size == 2048
        assert config.listener_timeout == 30
        assert config.max_retries == 5
        assert config.iperf_duration == 20
        assert config.default_port == 15000


class TestLoggingConfig:
    """Tests for LoggingConfig class."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.log_file == "x540t2_network_test.log"
        assert config.log_level == "INFO"
        assert config.console_output is True

    def test_custom_values(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            log_file="custom.log",
            log_level="DEBUG",
            console_output=False
        )

        assert config.log_file == "custom.log"
        assert config.log_level == "DEBUG"
        assert config.console_output is False


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()

        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_from_yaml_nonexistent_file(self, tmp_path: Path):
        """Test loading from nonexistent file returns default config."""
        config_file = tmp_path / "nonexistent.yaml"
        config = Config.from_yaml(config_file)

        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_to_yaml_and_from_yaml(self, tmp_path: Path):
        """Test saving and loading configuration."""
        # Create custom config
        original_config = Config(
            network=NetworkConfig(data_size_mb=1000),
            logging=LoggingConfig(log_level="DEBUG")
        )

        # Save to file
        config_file = tmp_path / "config.yaml"
        original_config.to_yaml(config_file)

        # Load from file
        loaded_config = Config.from_yaml(config_file)

        # Verify values match
        assert loaded_config.network.data_size_mb == 1000
        assert loaded_config.logging.log_level == "DEBUG"

    def test_yaml_file_creation(self, tmp_path: Path):
        """Test that YAML file is created properly."""
        config = Config()
        config_file = tmp_path / "test.yaml"

        config.to_yaml(config_file)

        assert config_file.exists()
        assert config_file.is_file()
