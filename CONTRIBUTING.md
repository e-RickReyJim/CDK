# Contributing to CKD Stage Predictor

First off, thank you for considering contributing to CKD Stage Predictor! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and respectful environment. Please be kind and courteous.

---

## How Can I Contribute?

### 🐛 Reporting Bugs

**Before submitting a bug report:**
- Check the existing issues to avoid duplicates
- Collect relevant information (error messages, screenshots, environment details)

**Bug Report Should Include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, package versions)
- Error messages and stack traces
- Screenshots if applicable

### 💡 Suggesting Enhancements

**Enhancement Suggestions Should Include:**
- Clear description of the proposed feature
- Use cases and benefits
- Potential implementation approach
- Any alternatives considered

### 🔧 Code Contributions

We welcome code contributions! Here are areas where you can help:

#### High Priority
- Additional visualization types
- Model explainability features (SHAP, LIME)
- Unit tests and integration tests
- Performance optimization
- Documentation improvements

#### Medium Priority
- Support for additional eGFR equations (CKD-EPI, Cockcroft-Gault)
- API endpoints for integration
- Docker containerization
- CI/CD pipeline setup

#### Nice to Have
- Mobile-responsive UI improvements
- Internationalization (i18n)
- Dark mode theme
- Export reports as PDF

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/CDK.git
cd CDK

# Add upstream remote
git remote add upstream https://github.com/e-RickReyJim/CDK.git
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (future)
# pip install -r requirements-dev.txt
```

### 3. Verify Setup

```bash
# Test the application
python src/app.py

# Run tests (when available)
# pytest tests/
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to functions
- Include type hints where applicable
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run tests
# pytest tests/

# Run the application
python src/app.py

# Test with different inputs
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: brief description

Detailed explanation of changes:
- What was changed
- Why it was changed
- Any breaking changes or side effects"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then:
1. Go to GitHub and create a Pull Request
2. Fill out the PR template
3. Link related issues
4. Wait for review

### 6. Code Review

- Respond to feedback promptly
- Make requested changes in new commits
- Don't force-push after review starts (unless requested)

---

## Coding Standards

### Python Style Guide

Follow **PEP 8** guidelines:

```python
# Good: Clear function names, type hints, docstrings
def calculate_egfr(serum_creatinine: float, age: float) -> float:
    """
    Calculate eGFR using MDRD equation.
    
    Parameters:
    -----------
    serum_creatinine : float
        Serum creatinine level in mg/dL
    age : float
        Patient age in years
    
    Returns:
    --------
    float
        Estimated GFR value
    """
    return 175 * (serum_creatinine ** -1.154) * (age ** -0.203)
```

### Code Organization

```python
# Imports: stdlib → third-party → local
import os
import sys

import pandas as pd
import numpy as np

from src.utils import calculate_egfr

# Constants at module level
MAX_AGE = 120
MIN_CREATININE = 0.1

# Functions and classes below
```

### Documentation

- **Docstrings:** Use Google or NumPy style for all public functions
- **Comments:** Explain "why", not "what"
- **README:** Update if adding new features
- **Type Hints:** Use for function parameters and returns

---

## Testing Guidelines

### Writing Tests

```python
import pytest
from src.utils import calculate_egfr, get_stage_from_egfr

def test_calculate_egfr_normal():
    """Test eGFR calculation with normal values"""
    result = calculate_egfr(serum_creatinine=1.0, age=50)
    assert 80 <= result <= 100, "eGFR should be in normal range"

def test_get_stage_from_egfr_stage1():
    """Test stage classification for Stage 1"""
    stage = get_stage_from_egfr(95)
    assert stage == 1, "eGFR >= 90 should be Stage 1"

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    result = calculate_egfr(serum_creatinine=-1, age=50)
    assert result is None, "Should return None for invalid inputs"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_calculate_egfr_normal
```

---

## Project-Specific Guidelines

### Machine Learning

- Document model training parameters
- Save model artifacts in `models/` directory
- Include performance metrics in documentation
- Version models appropriately

### Gradio Interface

- Keep UI intuitive and accessible
- Test with different screen sizes
- Include helpful tooltips and labels
- Add appropriate error handling

### Data Processing

- Validate inputs thoroughly
- Handle missing data appropriately
- Document data transformations
- Maintain reproducibility

---

## Questions?

- **General Questions:** Open a GitHub Discussion
- **Bug Reports:** Open an Issue
- **Feature Requests:** Open an Issue with "enhancement" label
- **Security Issues:** Email maintainer directly (see README)

---

Thank you for contributing! 🙏

Your efforts help make healthcare AI more accessible and reliable.
