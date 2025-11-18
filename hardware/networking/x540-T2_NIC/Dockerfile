# Multi-stage Dockerfile for Intel X540-T2 NIC Tools with DPDK support

FROM ubuntu:22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    linux-headers-generic \
    python3.10 \
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
    make \
    pkg-config \
    libnuma-dev \
    meson \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml setup.py ./
COPY README.md LICENSE ./

# Install the package
RUN pip3 install -e .

# Create non-root user
RUN useradd -m -u 1000 nicuser && \
    chown -R nicuser:nicuser /app

# Switch to non-root user (can be overridden for privileged operations)
USER nicuser

# Set default command
CMD ["python3", "-m", "src.diagnostics.nic_diagnostics"]


# DPDK-enabled stage
FROM base AS dpdk

USER root

# Install DPDK
RUN apt-get update && apt-get install -y \
    dpdk \
    dpdk-dev \
    libdpdk-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure huge pages
RUN echo "vm.nr_hugepages=1024" >> /etc/sysctl.conf

# Create huge pages mount point
RUN mkdir -p /mnt/huge

# Install DPDK Python bindings
RUN pip3 install --no-cache-dir dpdk-py || true

USER nicuser

CMD ["python3", "-m", "src.diagnostics.nic_diagnostics"]


# Development stage
FROM dpdk AS development

USER root

# Install development tools
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    isort \
    ipython \
    jupyter

USER nicuser

CMD ["/bin/bash"]
