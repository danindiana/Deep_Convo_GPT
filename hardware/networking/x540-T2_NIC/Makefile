.PHONY: help install install-dev test lint format clean docs docker-build docker-run

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
VENV := .venv
ACTIVATE := $(VENV)/bin/activate

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Intel X540-T2 NIC Tools - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install package and dependencies
	@echo "$(BLUE)Installing package and dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing package with development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,docs]"
	@echo "$(GREEN)Development installation complete!$(NC)"

venv: ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created!$(NC)"
	@echo "$(YELLOW)Activate with: source $(ACTIVATE)$(NC)"

test: ## Run tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Tests complete! Coverage report: htmlcov/index.html$(NC)"

test-quick: ## Run tests without coverage
	@echo "$(BLUE)Running quick tests...$(NC)"
	$(PYTEST) -v
	@echo "$(GREEN)Tests complete!$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(FLAKE8) src tests
	$(MYPY) src
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) src tests scripts
	$(ISORT) src tests scripts
	@echo "$(GREEN)Code formatting complete!$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(BLACK) --check src tests scripts
	$(ISORT) --check-only src tests scripts
	@echo "$(GREEN)Format check complete!$(NC)"

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && $(PYTHON) -m sphinx -b html . _build/html
	@echo "$(GREEN)Documentation built! Open docs/_build/html/index.html$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t x540-t2-nic-tools:latest .
	@echo "$(GREEN)Docker image built!$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm --network=host --privileged x540-t2-nic-tools:latest
	@echo "$(GREEN)Docker container stopped$(NC)"

run-diag: ## Run NIC diagnostics
	@echo "$(BLUE)Running NIC diagnostics...$(NC)"
	$(PYTHON) -m src.diagnostics.nic_diagnostics

run-iperf: ## Run iPerf tests
	@echo "$(BLUE)Running iPerf tests...$(NC)"
	$(PYTHON) -m src.diagnostics.iperf_test

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(PIP) list --outdated
	@echo "$(GREEN)Dependency check complete!$(NC)"

update-deps: ## Update all dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)Dependencies updated!$(NC)"

setup-dpdk: ## Setup DPDK environment (requires sudo)
	@echo "$(BLUE)Setting up DPDK environment...$(NC)"
	@echo "$(YELLOW)This requires sudo access...$(NC)"
	./scripts/shell/setup_dpdk.sh
	@echo "$(GREEN)DPDK setup complete!$(NC)"

all: clean install test lint ## Run complete build pipeline
	@echo "$(GREEN)All tasks completed successfully!$(NC)"

.DEFAULT_GOAL := help
