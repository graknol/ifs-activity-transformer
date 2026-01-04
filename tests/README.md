# IFS Activity Transformer - Test Suite

Comprehensive unit test suite for the IFS Activity Transformer application, following pytest best practices and enterprise Python testing patterns.

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_config.py
```

### Run specific test class
```bash
pytest tests/test_config.py::TestDatabaseConfig
```

### Run specific test
```bash
pytest tests/test_config.py::TestDatabaseConfig::test_from_env_with_all_values
```

### Run with coverage
```bash
pytest --cov=app --cov=services --cov-report=html
```

### Run only fast tests (skip slow integration tests)
```bash
pytest -m "not slow"
```

### Run only unit tests
```bash
pytest -m unit
```

### Run with verbose output
```bash
pytest -v
```

### Run with output capture disabled (see print statements)
```bash
pytest -s
```

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── test_config.py              # Configuration module tests
├── test_container.py           # Dependency injection tests
├── test_utils.py               # Utility functions tests
├── test_active_learning.py     # Active learning service tests
├── test_model_manager.py       # Model management tests
└── README.md                   # This file
```

## Test Coverage

Current test coverage targets:
- **Target:** 70% minimum coverage
- **app/ modules:** All core business logic
- **services/ modules:** Dependency injection container
- **Unit tests:** Fast, isolated tests
- **Integration tests:** Marked with `@pytest.mark.integration`

View coverage report:
```bash
pytest --cov=app --cov=services --cov-report=html
open htmlcov/index.html
```

## Test Markers

Tests can be marked with the following markers:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.slow` - Slow tests (e.g., model training)
- `@pytest.mark.integration` - Integration tests (may require network)
- `@pytest.mark.gpu` - Tests that require GPU

Example:
```python
@pytest.mark.slow
@pytest.mark.integration
def test_download_large_model():
    pass
```

## Fixtures

Reusable test fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `mock_env` - Mock environment variables
- `database_config` - Test database configuration
- `model_config` - Test model configuration
- `flask_config` - Test Flask configuration
- `azure_config` - Test Azure configuration
- `service_container` - Fresh service container
- `sample_dataframe` - Sample DataFrame for testing
- `sample_annotations` - Sample annotations DataFrame
- `mock_flask_app` - Mock Flask application
- `mock_db_connection` - Mock database connection
- `mock_model` - Mock ML model

## Writing Tests

### Test Structure
Follow the Arrange-Act-Assert pattern:

```python
def test_example(fixture):
    # Arrange: Set up test data
    config = Config()
    
    # Act: Perform the action
    result = config.get_value('key')
    
    # Assert: Verify the result
    assert result == expected_value
```

### Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Best Practices
1. **Isolation:** Each test should be independent
2. **Clear names:** Test names should describe what they test
3. **One assertion per test:** Focus on testing one thing
4. **Use fixtures:** Reuse setup code with fixtures
5. **Mock external dependencies:** Use mocks for databases, APIs, etc.

## CI/CD Integration

### GitHub Actions
Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest
```

## Continuous Testing

### Watch mode (requires pytest-watch)
```bash
pip install pytest-watch
ptw
```

### Auto-reload on changes
```bash
pytest --looponfail
```

## Debugging Tests

### Run with debugger
```bash
pytest --pdb
```

### Drop into debugger on failure
```bash
pytest --pdb --maxfail=1
```

### Print output
```bash
pytest -s --log-cli-level=DEBUG
```

## Test Data

Test data should be:
- **Minimal:** Only what's needed for the test
- **Realistic:** Representative of actual data
- **Isolated:** Created fresh for each test
- **Cleaned up:** Automatically removed after tests

## Performance

- **Fast tests:** Unit tests should run in < 1 second
- **Parallel execution:** Use `pytest-xdist` for parallel runs
- **Skip slow tests:** Use markers to skip in development

```bash
# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

## Troubleshooting

### Tests fail with import errors
```bash
pip install -e .
```

### Coverage not showing all files
Check `pytest.ini` and ensure source paths are correct

### Fixtures not found
Ensure `conftest.py` is in the tests directory

### Tests pass locally but fail in CI
Check environment variables and external dependencies

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://testdriven.io/)
