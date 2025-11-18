# Documentation Index

Welcome to the Intel X540-T2 NIC Tools documentation.

## Table of Contents

### Getting Started

- **[Quick Start Guide](guides/quick_start.md)** - Get running in 5 minutes
- **[Installation Guide](../INSTALL.md)** - Complete installation instructions
- **[Direct Connection Setup](guides/direct_connection.md)** - Peer-to-peer configuration

### DPDK Documentation

- **[DPDK Introduction](dpdk/introduction.md)** - What is DPDK and why use it
- **[Memory Pool Management](dpdk/memory_pools.md)** - DPDK memory pool concepts
- **[Memory Pool Diagrams](dpdk/memory_pools_diagram.md)** - Visual workflow
- **[VF Port Representors](dpdk/vf_port_representors.md)** - Virtual function management

### Guides

- **[Troubleshooting](guides/troubleshooting.md)** - Common issues and solutions
- **[Performance Tuning](#)** - Optimize your NIC performance
- **[Best Practices](#)** - Recommended configurations

### Advanced Topics

- **[ASIC Repurposing](advanced/asic_repurposing.md)** - Advanced hardware concepts
- **[Neuromorphic Computing](#)** - Research applications
- **[Hybrid GPU+NIC Architectures](#)** - Multi-accelerator systems

## Documentation Structure

```
docs/
├── README.md                      # This file
├── dpdk/                         # DPDK-specific documentation
│   ├── introduction.md
│   ├── memory_pools.md
│   ├── memory_pools_diagram.md
│   └── vf_port_representors.md
├── guides/                       # User guides
│   ├── quick_start.md
│   ├── direct_connection.md
│   └── troubleshooting.md
└── advanced/                     # Advanced topics
    └── asic_repurposing.md
```

## Contributing to Documentation

We welcome documentation improvements! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Documentation Standards

- Use Markdown format
- Include code examples
- Add diagrams where helpful (Mermaid format preferred)
- Keep language clear and concise
- Test all code examples

## API Reference

For API documentation, see the inline docstrings in the source code:

- `src/diagnostics/` - Diagnostic tools
- `src/utils/` - Utility functions
- `src/config.py` - Configuration management

## Examples

Practical examples are available in:

- `examples/` - Python usage examples
- `scripts/` - Shell script examples

## Additional Resources

### External Documentation

- [Intel X540 Datasheet](https://www.intel.com/content/www/us/en/products/details/ethernet/gigabit-controllers/x540-controllers.html)
- [DPDK Official Documentation](https://doc.dpdk.org/)
- [Linux Network Driver Documentation](https://www.kernel.org/doc/Documentation/networking/)

### Community

- [GitHub Repository](https://github.com/danindiana/Deep_Convo_GPT)
- [Issue Tracker](https://github.com/danindiana/Deep_Convo_GPT/issues)
- [Discussions](https://github.com/danindiana/Deep_Convo_GPT/discussions)

## Getting Help

If you can't find what you're looking for:

1. Check the [Quick Start Guide](guides/quick_start.md)
2. Review the [Troubleshooting Guide](guides/troubleshooting.md)
3. Search [existing issues](https://github.com/danindiana/Deep_Convo_GPT/issues)
4. Ask in [Discussions](https://github.com/danindiana/Deep_Convo_GPT/discussions)

---

**Last Updated**: 2025-01-18
