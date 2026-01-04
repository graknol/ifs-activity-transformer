@echo off
REM Cross-platform test runner for Windows
REM Wrapper around run_tests.py for convenient Windows usage

uv run python run_tests.py %*
