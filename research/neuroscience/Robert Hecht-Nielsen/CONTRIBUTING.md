# Contributing to the Robert Hecht-Nielsen Archive

Thank you for your interest in contributing to this archive! This project aims to preserve and extend the legacy of Robert Hecht-Nielsen, a pioneer in neural networks and AI.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Getting Started](#getting-started)
4. [Contribution Workflow](#contribution-workflow)
5. [Style Guidelines](#style-guidelines)
6. [Commit Message Guidelines](#commit-message-guidelines)
7. [Pull Request Process](#pull-request-process)
8. [Community](#community)

---

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## How Can I Contribute?

### 🎓 Historical Information

- **Biographical details** - Personal history, career milestones
- **Photos and images** - Historical photos, diagrams from papers
- **Publications** - PDF links, citations, paper summaries
- **Anecdotes and memories** - Stories from colleagues, students

### 💻 Code Contributions

- **Implementations** - C++, Python, Rust, or other languages
- **Examples** - Demonstration code for algorithms
- **Tests** - Unit tests, integration tests
- **Optimizations** - Performance improvements
- **Bug fixes** - Corrections to existing code

### 📚 Documentation

- **Tutorials** - Step-by-step guides
- **Explanations** - Algorithm descriptions, theory
- **API documentation** - Code documentation
- **Translations** - Non-English documentation

### 🎨 Visualizations

- **Mermaid diagrams** - Flowcharts, sequence diagrams
- **Infographics** - Visual representations
- **Interactive demos** - Web-based visualizations
- **Animations** - GIFs, videos

### 🔬 Research

- **Literature reviews** - Summaries of related work
- **Citation analysis** - Impact metrics
- **Modern connections** - Linking to current AI research
- **Applications** - Real-world uses of his techniques

---

## Getting Started

### Prerequisites

- Git installed
- GitHub account
- Appropriate development tools (C++ compiler, Python 3.8+, etc.)

### Fork the Repository

1. Navigate to [Deep_Convo_GPT](https://github.com/danindiana/Deep_Convo_GPT)
2. Click "Fork" in the top-right corner
3. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/Deep_Convo_GPT.git
cd Deep_Convo_GPT
```

### Set Up Development Environment

#### For Python Development

```bash
cd "research/neuroscience/Robert Hecht-Nielsen/implementations/python"
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

#### For C++ Development

```bash
cd "research/neuroscience/Robert Hecht-Nielsen/implementations/cpp"
mkdir build && cd build
cmake ..
make
```

#### For Rust Development

```bash
cd "research/neuroscience/Robert Hecht-Nielsen/implementations/rust"
cargo build
cargo test
```

---

## Contribution Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Adding tests
- `chore/` - Maintenance tasks

Examples:
- `feature/hebbian-learning-rust-impl`
- `docs/update-falcon-architecture`
- `fix/associative-memory-bug`

### 2. Make Your Changes

- Write clear, readable code
- Follow style guidelines (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

#### Python

```bash
# Run tests
pytest tests/ -v

# Run linting
flake8 .
black . --check
mypy .

# Run type checking
pyright
```

#### C++

```bash
# Build and run tests
cd build
cmake ..
make
ctest --verbose
```

#### Documentation

```bash
# Check markdown formatting
markdownlint **/*.md

# Validate links
markdown-link-check README.md
```

### 4. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add .
git commit -m "type: brief description"
```

See [Commit Message Guidelines](#commit-message-guidelines) below.

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Fill out the PR template
4. Submit for review

---

## Style Guidelines

### Markdown Documentation

- Use ATX-style headers (`#` not `===`)
- One sentence per line (makes diffs cleaner)
- Use fenced code blocks with language specifiers
- Include alt text for images
- Keep lines under 100 characters when possible
- Use relative links for internal references

**Example:**

```markdown
## Section Title

This is a paragraph.
Each sentence is on its own line.
This makes diffs easier to read.

```python
def example():
    return "code with language specifier"
```

![Alt text for image](images/example.png)
```

### Python Code Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with some modifications:

- **Line length:** 88 characters (Black default)
- **Formatter:** Black
- **Linter:** Flake8
- **Type hints:** Required for public APIs
- **Docstrings:** Google style

**Example:**

```python
"""Module docstring explaining purpose."""

import numpy as np
from typing import List, Tuple


def associative_memory(
    patterns: List[np.ndarray],
    input_pattern: np.ndarray,
) -> np.ndarray:
    """Retrieve pattern from associative memory.

    Args:
        patterns: List of stored patterns
        input_pattern: Partial or noisy input

    Returns:
        Retrieved complete pattern

    Raises:
        ValueError: If patterns are incompatible sizes
    """
    # Implementation here
    pass
```

### C++ Code Style

- **Standard:** C++17 or later
- **Naming:**
  - Classes: `PascalCase`
  - Functions: `camelCase`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`
- **Indentation:** 4 spaces
- **Braces:** K&R style

**Example:**

```cpp
/**
 * @brief Associative memory network implementation
 */
class AssociativeMemory {
private:
    std::vector<std::vector<double>> weight_matrix_;
    int size_;

public:
    /**
     * @brief Construct a new Associative Memory object
     * @param size Number of neurons
     */
    explicit AssociativeMemory(int size) : size_(size) {
        weight_matrix_ = std::vector<std::vector<double>>(
            size, std::vector<double>(size, 0.0)
        );
    }

    /**
     * @brief Store a pattern in memory
     * @param pattern Binary pattern to store
     */
    void storePattern(const std::vector<int>& pattern);
};
```

### Rust Code Style

- **Formatter:** `rustfmt`
- **Linter:** `clippy`
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

**Example:**

```rust
/// Associative memory network
pub struct AssociativeMemory {
    weight_matrix: Vec<Vec<f64>>,
    size: usize,
}

impl AssociativeMemory {
    /// Create a new associative memory network
    ///
    /// # Arguments
    ///
    /// * `size` - Number of neurons in the network
    pub fn new(size: usize) -> Self {
        Self {
            weight_matrix: vec![vec![0.0; size]; size],
            size,
        }
    }

    /// Store a pattern in the network
    pub fn store_pattern(&mut self, pattern: &[i8]) -> Result<(), Error> {
        // Implementation
        Ok(())
    }
}
```

---

## Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) for clear history and automated versioning.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, no logic change)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

### Examples

**Simple:**
```
docs: add Hebbian learning tutorial
```

**With scope:**
```
feat(python): implement self-organizing map
```

**With body:**
```
fix(cpp): correct weight matrix initialization

The weight matrix was not properly initialized to zero,
causing incorrect pattern retrieval. This commit fixes
the constructor to properly initialize all weights.

Fixes #42
```

**Breaking change:**
```
feat(api)!: change AssociativeMemory constructor signature

BREAKING CHANGE: Constructor now requires explicit size parameter
instead of inferring from first pattern.
```

---

## Pull Request Process

### Before Submitting

- [ ] Code compiles/runs without errors
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### PR Title

Follow conventional commit format:

```
feat: add counterpropagation network implementation
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Changes Made
- List specific changes
- With bullet points
- Be detailed

## Testing
Describe testing performed:
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Screenshots (if applicable)
Add screenshots for visual changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #(issue number)
```

### Review Process

1. **Automated Checks** - CI/CD must pass
2. **Peer Review** - At least one approval required
3. **Maintainer Review** - Final approval from maintainer
4. **Merge** - Squash and merge or rebase

### After Merge

- Delete your feature branch
- Update your fork
- Close related issues

---

## Testing Guidelines

### Test Coverage

Aim for:
- **80%+** code coverage for new code
- **100%** coverage for critical algorithms

### Test Structure

```python
# tests/test_associative_memory.py

import pytest
from implementations.python.associative_memory import AssociativeMemory


class TestAssociativeMemory:
    """Tests for AssociativeMemory class."""

    def test_initialization(self):
        """Test network initializes correctly."""
        memory = AssociativeMemory(size=10)
        assert memory.size == 10

    def test_store_pattern(self):
        """Test pattern storage."""
        memory = AssociativeMemory(size=4)
        pattern = [1, -1, 1, -1]
        memory.store_pattern(pattern)
        # Assertions here

    def test_retrieve_pattern(self):
        """Test pattern retrieval."""
        # Test implementation
        pass

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_capacity(self, size):
        """Test storage capacity for different sizes."""
        # Test implementation
        pass
```

---

## Documentation Guidelines

### Code Comments

- Explain **why**, not **what**
- Document complex algorithms
- Include references to papers
- Use TODO/FIXME for future work

```python
# Good
# Use Hebbian learning to strengthen co-activated connections
# as described in Hecht-Nielsen (1989), page 47
weight += learning_rate * input * output

# Bad
# Add to weight
weight += learning_rate * input * output
```

### API Documentation

All public APIs must have docstrings:

```python
def counterpropagation(
    input_data: np.ndarray,
    num_kohonen: int = 10,
    learning_rate: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a counterpropagation network.

    Implements the hybrid architecture from Hecht-Nielsen (1987)
    combining Kohonen competitive layer with Grossberg outstar.

    Args:
        input_data: Training data matrix (n_samples, n_features)
        num_kohonen: Number of Kohonen neurons
        learning_rate: Initial learning rate (0 < lr < 1)

    Returns:
        Tuple of (kohonen_weights, grossberg_weights)

    Raises:
        ValueError: If learning_rate not in valid range

    Example:
        >>> data = np.random.rand(100, 5)
        >>> k_weights, g_weights = counterpropagation(data, num_kohonen=10)

    References:
        Hecht-Nielsen, R. (1987). Counterpropagation networks.
        Applied Optics, 26(23), 4979-4984.
    """
    pass
```

---

## Community

### Getting Help

- **GitHub Discussions:** [Project Discussions](https://github.com/danindiana/Deep_Convo_GPT/discussions)
- **Issues:** Search existing issues before creating new ones
- **Email:** Contact maintainers for sensitive matters

### Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Mentioned in release notes
- Credited in documentation they create

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have questions about contributing:

1. Check this guide
2. Search existing issues
3. Ask in GitHub Discussions
4. Contact maintainers

Thank you for helping preserve and extend Robert Hecht-Nielsen's legacy! 🙏

---

**Last Updated:** January 2025
