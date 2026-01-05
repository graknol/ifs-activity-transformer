# Makefile for IFS Activity Transformer
# Provides convenient commands for development, testing, and deployment
#
# NOTE FOR WINDOWS USERS:
# This Makefile requires GNU Make which needs MinGW, WSL, or Cygwin on Windows.
# For a Windows-friendly alternative, use the cross-platform Python scripts:
#   - python run_tests.py          # Cross-platform test runner
#   - run_tests.bat                # Windows batch wrapper
# These scripts provide all the same functionality without requiring Make.

.PHONY: help install install-dev test test-cov test-fast lint format clean run docker-build docker-run

# Default target
help:
	@echo "IFS Activity Transformer - Available Commands:"
	@echo ""
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install dev dependencies including tests"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-fast        Run fast tests only (skip slow/integration)"
	@echo "  make lint             Run code linters"
	@echo "  make format           Auto-format code with black and isort"
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make security         Run security checks"
	@echo "  make clean            Clean up temporary files"
	@echo "  make run              Run the application"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run application in Docker"

# Install dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-test.txt

# Testing
test:
	pytest

test-cov:
	pytest --cov=app --cov=services --cov-report=html --cov-report=term-missing

test-fast:
	pytest -m "not slow and not integration"

test-watch:
	pytest-watch

# Code quality
lint:
	flake8 app services tests
	pylint app services

format:
	black app services tests
	isort app services tests

type-check:
	mypy app services --ignore-missing-imports

security:
	bandit -r app services
	safety check

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage.*" -delete
	rm -f coverage.xml

# Running
run:
	python run.py

run-dev:
	FLASK_ENV=development python run.py

# Docker
docker-build:
	docker build -t ifs-activity-transformer .

docker-run:
	docker run -p 5000:5000 --env-file .env ifs-activity-transformer

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "See README.md, QUICKSTART.md, WINDOWS_SETUP.md"
	@echo "See tests/README.md for testing documentation"

# All checks (for CI)
ci: lint type-check security test-cov
	@echo "All checks passed!"
