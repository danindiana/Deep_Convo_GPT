# DPDK Examples

This directory contains examples for using DPDK (Data Plane Development Kit) with Intel X540-T2 NIC.

## Prerequisites

1. DPDK installed on your system
2. Huge pages configured
3. NIC bound to DPDK-compatible driver (vfio-pci or uio_pci_generic)

## Setup

Run the DPDK setup script:

```bash
sudo ../../scripts/shell/setup_dpdk.sh
```

## Binding NICs to DPDK

```bash
# Check current bindings
sudo dpdk-devbind.py --status

# Unbind from kernel driver
sudo dpdk-devbind.py --unbind 03:00.0 03:00.1

# Bind to DPDK driver
sudo modprobe vfio-pci
sudo dpdk-devbind.py --bind=vfio-pci 03:00.0 03:00.1

# Verify binding
sudo dpdk-devbind.py --status
```

## Examples

### 1. Basic Port Initialization

See `docs/dpdk/introduction.md` for basic DPDK concepts.

### 2. Memory Pool Management

Refer to `docs/dpdk/memory_pools.md` for memory pool examples.

### 3. Virtual Function (VF) Port Representors

Check `docs/dpdk/vf_port_representors.md` for VF configuration examples.

## Testing DPDK

### Test PMD Application

```bash
# Run testpmd with 2 cores and 2048 MB memory
sudo dpdk-testpmd -l 0-1 -n 4 --socket-mem=2048 -- -i

# Inside testpmd:
testpmd> show port info all
testpmd> start
testpmd> stop
testpmd> quit
```

### L2 Forwarding

```bash
# Run l2fwd example with 2 cores
sudo dpdk-l2fwd -l 0-1 -n 4 -- -p 0x3 -q 1
```

## Performance Tuning

### CPU Isolation

Isolate CPUs for DPDK in GRUB configuration:

```bash
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX="isolcpus=2,3,4,5 nohz_full=2,3,4,5"

# Update GRUB
sudo update-grub
sudo reboot
```

### Interrupt Affinity

```bash
# Set IRQ affinity for NIC
echo 2 > /proc/irq/<IRQ_NUMBER>/smp_affinity_list
```

### Huge Page Configuration

```bash
# Check huge pages
cat /proc/meminfo | grep Huge

# Increase huge pages (1024 x 2MB = 2GB)
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Or use 1GB huge pages (if supported)
echo 4 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
```

## Troubleshooting

### Issue: Cannot bind to vfio-pci

**Solution**: Enable IOMMU in BIOS and kernel

```bash
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"

sudo update-grub
sudo reboot
```

### Issue: Low packet rate

**Solutions**:
1. Use isolated CPUs
2. Increase huge pages
3. Adjust rx/tx descriptors
4. Disable CPU frequency scaling

```bash
# Set performance governor
sudo cpupower frequency-set -g performance
```

### Issue: Huge pages allocation fails

**Solution**: Allocate huge pages at boot

```bash
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX="default_hugepagesz=1G hugepagesz=1G hugepages=4"

sudo update-grub
sudo reboot
```

## Resources

- [DPDK Documentation](https://doc.dpdk.org/)
- [DPDK Getting Started Guide](https://doc.dpdk.org/guides/linux_gsg/)
- [Intel X540 DPDK Guide](https://doc.dpdk.org/guides/nics/ixgbe.html)

## Advanced Topics

For advanced DPDK usage, see:
- `../../docs/dpdk/` - Detailed DPDK documentation
- `../../docs/advanced/asic_repurposing.md` - Advanced hardware concepts
