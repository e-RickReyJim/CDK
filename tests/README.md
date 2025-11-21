# Running Tests

## Quick Start

```powershell
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils.py

# Run specific test class
pytest tests/test_utils.py::TestCalculateEGFR

# Run specific test
pytest tests/test_utils.py::TestCalculateEGFR::test_valid_inputs
```

## Test Categories

Run tests by marker:

```powershell
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Model tests (require model files)
pytest -m model

# Visualization tests
pytest -m visualization

# Validation tests
pytest -m validation

# Performance tests
pytest -m slow
```

## Coverage Report

```powershell
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux

# Terminal coverage report
pytest --cov=src --cov-report=term-missing
```

## Parallel Execution

```powershell
# Run tests in parallel (faster)
pytest -n auto
```

## Verbose Output

```powershell
# Verbose mode
pytest -v

# Very verbose mode
pytest -vv

# Show print statements
pytest -s
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test data
├── test_utils.py            # Utility function tests (80+ tests)
├── test_models.py           # Model loading/prediction tests (50+ tests)
├── test_app.py              # Application tests (40+ tests)
├── test_integration.py      # End-to-end tests (30+ tests)
└── test_data_validation.py  # Input validation tests (60+ tests)
```

All test data is defined in `conftest.py` fixtures.

## Test Configuration

- **pytest.ini**: Test discovery and configuration
- **.coveragerc**: Coverage settings
- **conftest.py**: Shared fixtures and setup

## Continuous Integration

See `.github/workflows/tests.yml` for CI/CD configuration.
