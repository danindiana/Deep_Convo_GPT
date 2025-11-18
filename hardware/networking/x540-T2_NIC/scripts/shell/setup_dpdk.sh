#!/bin/bash

# DPDK Environment Setup Script for Intel X540-T2
# Requires sudo privileges

set -euo pipefail

readonly COLOR_RESET='\033[0m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_RED='\033[0;31m'

log_info() { echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $*"; }
log_success() { echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $*"; }
log_warning() { echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $*"; }
log_error() { echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*"; }

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run with sudo"
    exit 1
fi

log_info "Setting up DPDK environment for Intel X540-T2..."

# Install DPDK packages
log_info "Installing DPDK packages..."
apt-get update
apt-get install -y dpdk dpdk-dev libdpdk-dev meson ninja-build pkg-config libnuma-dev

# Configure huge pages
log_info "Configuring huge pages..."
HUGEPAGES=1024

# Set huge pages for current session
echo $HUGEPAGES > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Make persistent
if ! grep -q "vm.nr_hugepages" /etc/sysctl.conf; then
    echo "vm.nr_hugepages=$HUGEPAGES" >> /etc/sysctl.conf
    log_success "Huge pages configuration added to /etc/sysctl.conf"
fi

sysctl -p

# Mount huge pages
log_info "Mounting huge pages filesystem..."
if ! mountpoint -q /mnt/huge; then
    mkdir -p /mnt/huge
    mount -t hugetlbfs nodev /mnt/huge

    # Make mount persistent
    if ! grep -q "/mnt/huge" /etc/fstab; then
        echo "nodev /mnt/huge hugetlbfs defaults 0 0" >> /etc/fstab
        log_success "Huge pages mount added to /etc/fstab"
    fi
else
    log_info "Huge pages already mounted"
fi

# Load required kernel modules
log_info "Loading kernel modules..."
modprobe uio
modprobe vfio-pci
modprobe uio_pci_generic 2>/dev/null || log_warning "uio_pci_generic not available"

# Make modules load on boot
for module in uio vfio-pci; do
    if ! grep -q "^$module" /etc/modules; then
        echo "$module" >> /etc/modules
    fi
done

# Set IOMMU if available
if [ -d /sys/kernel/iommu_groups ]; then
    log_info "IOMMU is available"
    # Enable IOMMU in GRUB (manual step required)
    log_warning "For best DPDK performance, enable IOMMU in GRUB:"
    log_warning "Add 'intel_iommu=on iommu=pt' to GRUB_CMDLINE_LINUX in /etc/default/grub"
    log_warning "Then run: sudo update-grub && sudo reboot"
fi

# Display huge pages status
log_info "Huge pages status:"
grep -i huge /proc/meminfo

# Display available DPDK-compatible NICs
log_info "Available network devices:"
lspci | grep -i ethernet

log_success "DPDK environment setup complete!"
log_info "To bind interfaces to DPDK, use:"
log_info "  sudo dpdk-devbind.py --status"
log_info "  sudo dpdk-devbind.py --bind=vfio-pci <PCI_ADDRESS>"
