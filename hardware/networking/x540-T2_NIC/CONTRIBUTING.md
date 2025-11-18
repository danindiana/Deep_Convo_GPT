# Contributing to X540-T2 NIC Tools

Thank you for your interest in contributing to the Intel X540-T2 NIC Diagnostic & Testing Suite! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive criticism
- Accept gracefully when your contributions are not accepted

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Deep_Convo_GPT.git
   cd Deep_Convo_GPT/hardware/networking/x540-T2_NIC
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/danindiana/Deep_Convo_GPT.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Make (optional but recommended)

### Setup Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
make install-dev

# Or manually
pip install -r requirements.txt
pip install -e ".[dev,docs]"
```

### Verify Setup

```bash
# Run tests
make test

# Run linting
make lint

# Check formatting
make format-check
```

## Making Changes

### Branch Strategy

1. **Create a feature branch** from the main branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Development Workflow

1. **Make your changes** in small, logical commits
2. **Write or update tests** for your changes
3. **Update documentation** as needed
4. **Run tests and linting** before committing:
   ```bash
   make test
   make lint
   ```

### Commit Messages

Follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when applicable

Good examples:
```
Add support for multi-NIC testing

Implement bidirectional throughput testing between multiple NICs.
Includes configuration options and comprehensive logging.

Fixes #123
```

## Testing

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_diagnostics.py -v

# Run tests matching a pattern
pytest -k "test_interface_discovery" -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Aim for high code coverage (>80%)

Example test:

```python
import pytest
from src.utils.interface_discovery import InterfaceDiscovery

def test_get_all_interfaces():
    """Test interface discovery returns a list."""
    interfaces = InterfaceDiscovery.get_all_interfaces()
    assert isinstance(interfaces, list)
    assert len(interfaces) >= 0
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for all functions
- Docstrings: Google style
- Code formatting: Black
- Import sorting: isort

### Formatting Code

```bash
# Auto-format code
make format

# Check formatting without changes
make format-check
```

### Type Checking

All code should include type hints:

```python
from typing import List, Optional

def get_interfaces(filter_name: Optional[str] = None) -> List[str]:
    """Get network interfaces.

    Args:
        filter_name: Optional filter for interface names

    Returns:
        List of interface names
    """
    ...
```

Run type checking:
```bash
mypy src/
```

### Documentation Style

- Use Google-style docstrings
- Include type information
- Provide examples for complex functions
- Keep documentation up to date with code changes

Example:

```python
def send_data(self, ip_address: str, port: int, data_size_mb: int) -> TestResult:
    """Send data and measure throughput.

    Args:
        ip_address: Destination IP address
        port: Destination port number
        data_size_mb: Amount of data to send in megabytes

    Returns:
        TestResult object containing throughput metrics

    Raises:
        ConnectionError: If connection to destination fails
        ValueError: If data_size_mb is negative

    Example:
        >>> diag = NICDiagnostics()
        >>> result = diag.send_data("192.168.1.100", 12345, 500)
        >>> print(f"Throughput: {result.throughput_mbps} MB/s")
    """
    ...
```

## Submitting Changes

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite**:
   ```bash
   make all
   ```

3. **Update CHANGELOG** if applicable

### Creating a Pull Request

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

4. **Wait for review**:
   - Address reviewer comments
   - Make requested changes
   - Keep the PR updated with main branch

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts with main
- [ ] CHANGELOG is updated (if applicable)

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior**
- **Actual behavior**
- **Environment details**:
  - OS version
  - Python version
  - NIC model and driver version
  - Relevant configuration

Example:

```markdown
**Bug**: Interface discovery fails on Ubuntu 24.04

**Steps to Reproduce**:
1. Install package on Ubuntu 24.04
2. Run `python -m src.diagnostics.nic_diagnostics`
3. Observe error

**Expected**: Interfaces should be discovered

**Actual**: RuntimeError raised

**Environment**:
- OS: Ubuntu 24.04 LTS
- Python: 3.12.1
- NIC: Intel X540-T2
- Driver: ixgbe 5.1.0-k
```

### Feature Requests

When requesting features, include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: What alternatives have you considered?
- **Additional context**: Any other relevant information

## Questions?

If you have questions:

- Check the [documentation](docs/)
- Search [existing issues](https://github.com/danindiana/Deep_Convo_GPT/issues)
- Ask in [GitHub Discussions](https://github.com/danindiana/Deep_Convo_GPT/discussions)

## Thank You!

Your contributions help make this project better for everyone. We appreciate your time and effort!
