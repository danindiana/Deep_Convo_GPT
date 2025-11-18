# Quick Start Guide

Get up and running with the Intel X540-T2 NIC Tools in minutes.

## Prerequisites

- Ubuntu 22.04+ (or compatible Linux distribution)
- Python 3.10 or higher
- Intel X540-T2 NIC installed
- Root/sudo access

## 5-Minute Quick Start

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/danindiana/Deep_Convo_GPT.git
cd Deep_Convo_GPT/hardware/networking/x540-T2_NIC

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
make install
```

### 2. Verify NIC Detection

```bash
# Check if NIC is installed
lspci | grep -i ethernet

# Expected output:
# 03:00.0 Ethernet controller: Intel Corporation Ethernet Controller 10-Gigabit X540-AT2
# 03:00.1 Ethernet controller: Intel Corporation Ethernet Controller 10-Gigabit X540-AT2
```

### 3. Configure Interfaces

```bash
# Bring interfaces up
sudo ip link set enp3s0f0 up
sudo ip link set enp3s0f1 up

# Assign IP addresses (for direct connection testing)
sudo ip addr add 192.168.10.1/24 dev enp3s0f0
sudo ip addr add 192.168.10.2/24 dev enp3s0f1

# Verify configuration
ip addr show enp3s0f0
ip addr show enp3s0f1
```

### 4. Run Basic Diagnostics

```bash
# Run the diagnostic suite
make run-diag

# Or manually
python -m src.diagnostics.nic_diagnostics
```

### 5. Run iPerf Tests

```bash
# Run iPerf benchmarks
make run-iperf

# Or manually
python -m src.diagnostics.iperf_test
```

## Example Output

Successful diagnostic output:

```
[INFO] Starting interface discovery for Intel X540-T2 NIC...
[INFO] Discovered Intel port: enp3s0f0 with IP 192.168.10.1
[INFO] Discovered Intel port: enp3s0f1 with IP 192.168.10.2
[INFO] Pinging 192.168.10.2 to check connectivity...
[INFO] Ping successful to 192.168.10.2
[INFO] Starting data transfer test between ports...
[INFO] Connected to 192.168.10.2:12345, starting data transfer
Sending: 100%|██████████| 500/500 [00:05<00:00, 95.23MB/s]
[INFO] Data transfer complete. Throughput: 952.30 MB/s
[INFO] Test completed. Achieved throughput: 952.30 MB/s
```

## Configuration

### Using Custom Configuration

```python
from pathlib import Path
from src.config import Config
from src.diagnostics.nic_diagnostics import NICDiagnostics

# Load custom configuration
config = Config.from_yaml(Path("config/example_config.yaml"))

# Run diagnostics
diag = NICDiagnostics(config=config)
diag.run_full_diagnostic()
```

### Configuration Options

Edit `config/default_config.yaml`:

```yaml
network:
  data_size_mb: 500          # Test data size
  listener_timeout: 15       # Connection timeout
  iperf_duration: 10         # iPerf test duration

logging:
  log_level: "INFO"          # Logging level
  log_file: "network_test.log"
```

## Common Operations

### Check Interface Status

```bash
# View interface status
ip link show enp3s0f0

# Check link speed
ethtool enp3s0f0 | grep Speed

# View statistics
ip -s link show enp3s0f0
```

### Run Individual Tests

```python
from src.diagnostics.nic_diagnostics import NICDiagnostics

diag = NICDiagnostics()

# Discover interfaces
diag.discover_interfaces()

# Ping test
result = diag.ping_test("192.168.10.2")
print(f"Ping test: {result}")

# Throughput test
throughput = diag.send_data("192.168.10.2", 12345, data_size_mb=100)
print(f"Throughput: {throughput}")
```

## Troubleshooting

### Issue: Interfaces not found

```bash
# Check driver
lsmod | grep ixgbe

# Load driver if missing
sudo modprobe ixgbe

# Verify NIC is recognized
lspci -v | grep -A 10 Ethernet
```

### Issue: No IP address

```bash
# Manually assign IP
sudo ip addr add 192.168.10.1/24 dev enp3s0f0

# Or use DHCP (if available)
sudo dhclient enp3s0f0
```

### Issue: Low throughput

```bash
# Enable offloading features
sudo ethtool -K enp3s0f0 tso on gso on gro on

# Increase ring buffer
sudo ethtool -G enp3s0f0 rx 4096 tx 4096

# Check for errors
ethtool -S enp3s0f0 | grep -i error
```

## Next Steps

1. **Advanced Testing**: See [examples/](../../examples/) for advanced usage
2. **DPDK Setup**: Follow [DPDK Introduction](../dpdk/introduction.md) for high-performance
3. **Direct Connection**: Check [Direct Connection Guide](direct_connection.md) for peer-to-peer setup
4. **Production Deployment**: Review [INSTALL.md](../../INSTALL.md) for complete installation

## Docker Quick Start

For a containerized environment:

```bash
# Build Docker image
make docker-build

# Run diagnostics in container
make docker-run
```

## Getting Help

- Check the [main README](../../README.md)
- Review [troubleshooting guide](troubleshooting.md)
- Open an [issue on GitHub](https://github.com/danindiana/Deep_Convo_GPT/issues)

---

You're now ready to use the X540-T2 NIC Tools! For more advanced features and detailed documentation, explore the `docs/` directory.
