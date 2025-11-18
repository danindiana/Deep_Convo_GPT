"""Configuration management for NIC diagnostics."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class NetworkConfig:
    """Network testing configuration."""

    data_size_mb: int = 500
    block_size: int = 1024 * 1024  # 1 MB
    listener_timeout: int = 15
    max_retries: int = 3
    iperf_duration: int = 10
    default_port: int = 12345


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_file: str = "x540t2_network_test.log"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True


@dataclass
class Config:
    """Main configuration class."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance with loaded settings
        """
        if not config_path.exists():
            return cls()

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        network_config = NetworkConfig(**data.get('network', {}))
        logging_config = LoggingConfig(**data.get('logging', {}))

        return cls(network=network_config, logging=logging_config)

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Path where to save configuration
        """
        data = {
            'network': {
                'data_size_mb': self.network.data_size_mb,
                'block_size': self.network.block_size,
                'listener_timeout': self.network.listener_timeout,
                'max_retries': self.network.max_retries,
                'iperf_duration': self.network.iperf_duration,
                'default_port': self.network.default_port,
            },
            'logging': {
                'log_file': self.logging.log_file,
                'log_level': self.logging.log_level,
                'log_format': self.logging.log_format,
                'console_output': self.logging.console_output,
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
