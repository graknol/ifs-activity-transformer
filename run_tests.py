#!/usr/bin/env python3
"""
Cross-platform test runner script for IFS Activity Transformer.
Works on Windows, Linux, and macOS without requiring Make or MinGW.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --cov        # Run with coverage
    python run_tests.py --fast       # Run fast tests only
    python run_tests.py --lint       # Run linters
    python run_tests.py --format     # Format code
    python run_tests.py --clean      # Clean up temporary files
    python run_tests.py --all        # Run all checks (CI mode)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=None):
    """Run a command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description or 'Command'} failed with exit code {e.returncode}")
        return False


def run_tests(args):
    """Run pytest with various options."""
    cmd = "pytest"
    
    if args.cov:
        cmd += " --cov=app --cov=services --cov-report=html --cov-report=term-missing"
    
    if args.fast:
        cmd += ' -m "not slow and not integration"'
    
    if args.verbose:
        cmd += " -v"
    
    return run_command(cmd, "Running tests")


def run_lint(args):
    """Run code linters."""
    success = True
    
    # Flake8
    success &= run_command(
        "flake8 app services tests",
        "Running flake8"
    )
    
    # Pylint
    success &= run_command(
        "pylint app services",
        "Running pylint"
    )
    
    # Black check
    success &= run_command(
        "black --check app services tests",
        "Checking code formatting (black)"
    )
    
    # Isort check
    success &= run_command(
        "isort --check-only app services tests",
        "Checking import order (isort)"
    )
    
    return success


def format_code(args):
    """Auto-format code."""
    success = True
    
    success &= run_command(
        "black app services tests",
        "Formatting code with black"
    )
    
    success &= run_command(
        "isort app services tests",
        "Sorting imports with isort"
    )
    
    return success


def type_check(args):
    """Run type checking."""
    return run_command(
        "mypy app services --ignore-missing-imports",
        "Running type checker (mypy)"
    )


def security_check(args):
    """Run security checks."""
    success = True
    
    success &= run_command(
        "bandit -r app services",
        "Running security scanner (bandit)"
    )
    
    success &= run_command(
        "safety check",
        "Checking dependencies (safety)"
    )
    
    return success


def clean(args):
    """Clean up temporary files and directories."""
    print("\n" + "="*60)
    print("  Cleaning up temporary files")
    print("="*60 + "\n")
    
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.egg-info",
        ".pytest_cache",
        ".coverage",
        ".coverage.*",
        "htmlcov",
        "coverage.xml",
        ".mypy_cache",
        "dist",
        "build"
    ]
    
    root = Path.cwd()
    removed_count = 0
    
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()
                removed_count += 1
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed_count += 1
    
    print(f"✅ Cleaned up {removed_count} items")
    return True


def install_deps(args):
    """Install dependencies."""
    success = True
    
    success &= run_command(
        "pip install -r requirements.txt",
        "Installing production dependencies"
    )
    
    if args.dev:
        success &= run_command(
            "pip install -r requirements-test.txt",
            "Installing test dependencies"
        )
    
    return success


def run_all_checks(args):
    """Run all checks (CI mode)."""
    print("\n" + "="*60)
    print("  Running all checks (CI mode)")
    print("="*60 + "\n")
    
    success = True
    success &= run_lint(args)
    success &= type_check(args)
    success &= security_check(args)
    
    # Run tests with coverage (create a new args object to avoid mutation)
    import copy
    test_args = copy.copy(args)
    test_args.cov = True
    success &= run_tests(test_args)
    
    if success:
        print("\n" + "="*60)
        print("  ✅ All checks passed!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("  ❌ Some checks failed")
        print("="*60 + "\n")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform test runner for IFS Activity Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--test", action="store_true",
        help="Run tests"
    )
    parser.add_argument(
        "--cov", action="store_true",
        help="Run tests with coverage"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run fast tests only (skip slow/integration)"
    )
    parser.add_argument(
        "--lint", action="store_true",
        help="Run code linters"
    )
    parser.add_argument(
        "--format", action="store_true",
        help="Auto-format code"
    )
    parser.add_argument(
        "--type-check", action="store_true",
        help="Run type checking"
    )
    parser.add_argument(
        "--security", action="store_true",
        help="Run security checks"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clean up temporary files"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Install dependencies"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Install dev dependencies (use with --install)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all checks (CI mode)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Handle default behavior: run tests if no arguments provided
    no_args = len(sys.argv) == 1
    
    if args.clean:
        success &= clean(args)
    
    if args.install:
        success &= install_deps(args)
    
    if args.format:
        success &= format_code(args)
    
    if args.lint:
        success &= run_lint(args)
    
    if args.type_check:
        success &= type_check(args)
    
    if args.security:
        success &= security_check(args)
    
    if args.all:
        success &= run_all_checks(args)
    elif args.test or args.cov or args.fast or no_args:
        # Run tests if explicitly requested or if no arguments provided
        success &= run_tests(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
