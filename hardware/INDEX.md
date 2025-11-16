# Hardware Domain Index

Hardware acceleration, AI accelerators, GPUs, and networking infrastructure.

## 📂 Directory Structure

### AI Accelerators
- **[accelerators/tenstorrent/](accelerators/tenstorrent/)** - Tenstorrent AI accelerator development
  - PyBuda custom kernels
  - Convolution, LSTM, activation implementations
  - Analog multiply and matrix-vector engines
  - Dynamic cortex architectures

- **[accelerators/colossus/](accelerators/colossus/)** - Colossus hardware specifications

### GPU Systems
- **[gpu/GPU_new_uses/](gpu/GPU_new_uses/)** - Novel GPU applications and optimization
  - GPU RAID configurations
  - New use cases for GPU acceleration
  - Performance tuning

### Networking
- **[networking/x540-T2_NIC/](networking/x540-T2_NIC/)** - Intel x540-T2 NIC optimization
  - DPDK integration and memory pools
  - Network performance testing (iperf)
  - NIC diagnostics and tuning
  - ASIC research

- **[networking/networking/](networking/networking/)** - General network infrastructure
  - NIC card comparisons and selection

- **[voomrisc/](voomrisc/)** - VoomRISC architecture research

## Key Technologies

| Technology | Application | Directory |
|------------|-------------|-----------|
| **Tenstorrent** | AI Acceleration | accelerators/tenstorrent/ |
| **GPU RAID** | Parallel Computing | gpu/GPU_new_uses/ |
| **DPDK** | High-performance networking | networking/x540-T2_NIC/ |
| **Custom Kernels** | ML Optimization | accelerators/tenstorrent/ |

## Related Sections

- [Implementations](../implementations/) - Software utilizing hardware acceleration
- [Theory](../theory/) - Theoretical foundations of hardware design
- [Research](../research/) - Emerging hardware technologies

## Quick Start

Browse accelerator implementations for AI workloads, or explore networking optimizations for high-throughput applications.
