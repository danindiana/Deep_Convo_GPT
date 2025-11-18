# Installation Guide

Complete installation instructions for the Intel X540-T2 NIC Diagnostic & Testing Suite.

## Table of Contents

- [System Requirements](#system-requirements)
- [Hardware Requirements](#hardware-requirements)
- [Software Installation](#software-installation)
- [Driver Installation](#driver-installation)
- [DPDK Installation](#dpdk-installation-optional)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Operating System

- **Recommended**: Ubuntu 22.04 LTS or later
- **Supported**: Any modern Linux distribution with kernel 5.15+
- **Architecture**: x86_64 (AMD64)

### Software Requirements

- Python 3.10 or higher
- pip (Python package manager)
- Git
- GCC compiler
- Make

### Kernel Requirements

```bash
# Check kernel version (minimum 5.15)
uname -r
```

## Hardware Requirements

### Intel X540-T2 NIC

- **Model**: Intel Ethernet Server Adapter X540-T2
- **Interface**: PCIe 2.1 x8 (compatible with PCIe 3.0/4.0)
- **Ports**: Dual 10GBASE-T RJ45
- **Chipset**: Intel X540

### System Specifications

- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: 8 GB minimum (16 GB recommended for DPDK)
- **PCIe**: One available PCIe x8 slot
- **Storage**: 2 GB free space

### Network Requirements

- Cat6a or Cat7 Ethernet cables for 10GbE
- 10GbE switch (for multi-system testing) or direct connection

## Software Installation

### Step 1: Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2: Install System Dependencies

```bash
sudo apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    python3 \
    python3-pip \
    python3-venv \
    git \
    pciutils \
    net-tools \
    ethtool \
    iproute2 \
    iperf \
    iperf3 \
    netcat \
    pv \
    make
```

### Step 3: Clone Repository

```bash
# Clone the repository
git clone https://github.com/danindiana/Deep_Convo_GPT.git
cd Deep_Convo_GPT/hardware/networking/x540-T2_NIC
```

### Step 4: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 5: Install Python Package

```bash
# Install with make
make install

# Or install manually
pip install -r requirements.txt
pip install -e .
```

### Development Installation

For development work:

```bash
make install-dev

# Or manually
pip install -r requirements.txt
pip install -e ".[dev,docs]"
```

## Driver Installation

### Verify NIC Detection

```bash
# Check if NIC is detected
lspci | grep -i ethernet

# Expected output (may vary):
# 03:00.0 Ethernet controller: Intel Corporation Ethernet Controller 10-Gigabit X540-AT2
# 03:00.1 Ethernet controller: Intel Corporation Ethernet Controller 10-Gigabit X540-AT2
```

### Install ixgbe Driver

The ixgbe driver is usually included in the Linux kernel, but you may need to load it:

```bash
# Check if driver is loaded
lsmod | grep ixgbe

# Load driver if not present
sudo modprobe ixgbe

# Make it load on boot
echo "ixgbe" | sudo tee -a /etc/modules
```

### Install Latest Driver (Optional)

For the latest features:

```bash
# Download latest driver from Intel
wget https://downloadmirror.intel.com/13663/eng/ixgbe-5.19.9.tar.gz

# Extract and build
tar xzf ixgbe-5.19.9.tar.gz
cd ixgbe-5.19.9/src
make
sudo make install

# Load new driver
sudo modprobe -r ixgbe
sudo modprobe ixgbe

# Verify version
modinfo ixgbe | grep version
```

## Network Configuration

### Configure Interfaces

```bash
# List interfaces
ip link show

# Bring interfaces up
sudo ip link set enp3s0f0 up
sudo ip link set enp3s0f1 up

# Assign IP addresses (for direct connection)
sudo ip addr add 192.168.10.1/24 dev enp3s0f0
sudo ip addr add 192.168.10.2/24 dev enp3s0f1

# Verify configuration
ip addr show enp3s0f0
ip addr show enp3s0f1
```

### Persistent Configuration (Ubuntu/Netplan)

Create `/etc/netplan/60-x540.yaml`:

```yaml
network:
  version: 2
  ethernets:
    enp3s0f0:
      addresses:
        - 192.168.10.1/24
      dhcp4: false
      optional: true

    enp3s0f1:
      addresses:
        - 192.168.10.2/24
      dhcp4: false
      optional: true
```

Apply configuration:

```bash
sudo netplan apply
```

## DPDK Installation (Optional)

For high-performance packet processing:

### Install DPDK Packages

```bash
sudo apt-get install -y \
    dpdk \
    dpdk-dev \
    libdpdk-dev \
    meson \
    ninja-build \
    pkg-config \
    libnuma-dev
```

### Configure Huge Pages

```bash
# Allocate huge pages
echo 1024 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Make persistent
echo "vm.nr_hugepages=1024" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Mount huge pages
sudo mkdir -p /mnt/huge
sudo mount -t hugetlbfs nodev /mnt/huge

# Make mount persistent
echo "nodev /mnt/huge hugetlbfs defaults 0 0" | sudo tee -a /etc/fstab
```

### Bind Interfaces to DPDK

```bash
# Install DPDK Python bindings
pip install dpdk-devbind

# Show current bindings
sudo dpdk-devbind.py --status

# Unbind from kernel driver
sudo dpdk-devbind.py --unbind 03:00.0 03:00.1

# Bind to DPDK driver (UIO or VFIO)
sudo modprobe vfio-pci
sudo dpdk-devbind.py --bind=vfio-pci 03:00.0 03:00.1

# Verify binding
sudo dpdk-devbind.py --status
```

## Verification

### Test Basic Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Check Python version
python --version  # Should be 3.10+

# Verify package installation
python -c "import src; print(src.__version__)"
```

### Test NIC Detection

```bash
# Run interface discovery
python -c "from src.utils.interface_discovery import InterfaceDiscovery; print(InterfaceDiscovery.get_all_interfaces())"
```

### Run Diagnostic Tests

```bash
# Run basic diagnostics
make run-diag

# Run iPerf tests
make run-iperf
```

### Verify Driver

```bash
# Check driver information
ethtool -i enp3s0f0

# Expected output:
# driver: ixgbe
# version: 5.1.0-k
# firmware-version: ...
```

### Test Link Speed

```bash
# Check link speed
ethtool enp3s0f0 | grep Speed

# Expected output:
# Speed: 10000Mb/s
```

## Troubleshooting

### NIC Not Detected

```bash
# Rescan PCI bus
echo 1 | sudo tee /sys/bus/pci/rescan

# Check dmesg for errors
sudo dmesg | grep -i ixgbe

# Verify BIOS settings (ensure PCIe slot is enabled)
```

### Driver Issues

```bash
# Remove and reload driver
sudo modprobe -r ixgbe
sudo modprobe ixgbe

# Check for errors
dmesg | tail -50
```

### No Link Detected

```bash
# Check cable connection
ethtool enp3s0f0 | grep "Link detected"

# Force link up
sudo ip link set enp3s0f0 up

# Check physical connection and cable
```

### Permission Errors

```bash
# Add user to appropriate groups
sudo usermod -a -G netdev $USER
sudo usermod -a -G plugdev $USER

# Log out and back in for changes to take effect
```

### Low Performance

```bash
# Disable power management
sudo ethtool -s enp3s0f0 wol d

# Enable hardware offloading
sudo ethtool -K enp3s0f0 tso on gso on gro on

# Increase ring buffer size
sudo ethtool -G enp3s0f0 rx 4096 tx 4096

# Adjust interrupt coalescing
sudo ethtool -C enp3s0f0 rx-usecs 50
```

### Python Package Issues

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Clear pip cache
pip cache purge

# Use verbose installation
pip install -v -e .
```

## Docker Installation

For a containerized environment:

```bash
# Build Docker image
make docker-build

# Run container
make docker-run

# Or manually
docker build -t x540-t2-nic-tools .
docker run -it --rm --network=host --privileged x540-t2-nic-tools
```

## Uninstallation

### Remove Python Package

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Or if installed globally
pip uninstall x540-t2-nic-tools
```

### Remove System Packages

```bash
sudo apt-get remove --purge dpdk dpdk-dev libdpdk-dev
sudo apt-get autoremove
```

### Restore Network Configuration

```bash
# Remove netplan configuration
sudo rm /etc/netplan/60-x540.yaml
sudo netplan apply

# Remove huge pages configuration
sudo sed -i '/vm.nr_hugepages/d' /etc/sysctl.conf
sudo sysctl -p
```

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [system logs](#checking-logs)
3. Consult the [main README](README.md)
4. Open an [issue on GitHub](https://github.com/danindiana/Deep_Convo_GPT/issues)

### Checking Logs

```bash
# System logs
sudo journalctl -xe

# Kernel messages
sudo dmesg | grep -i ixgbe

# Network logs
sudo journalctl -u NetworkManager
```

---

**Installation complete!** Proceed to the [Quick Start Guide](docs/guides/quick_start.md) to begin testing your NIC.
