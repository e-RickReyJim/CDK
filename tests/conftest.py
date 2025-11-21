"""
Pytest Configuration and Shared Fixtures
=========================================
Shared test fixtures, sample data, and utilities for CKD prediction tests.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import pickle
import os
import sys
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# Sample Patient Data Fixtures
# ============================================================================

@pytest.fixture
def sample_patient_normal():
    """Sample patient with Stage 1 CKD (normal kidney function)"""
    return {
        'age': 48,
        'blood_pressure': 80,
        'specific_gravity': 1.020,
        'albumin': 0,
        'sugar': 0,
        'blood_glucose_random': 121,
        'blood_urea': 36,
        'serum_creatinine': 1.2,
        'sodium': 138,
        'potassium': 4.6,
        'haemoglobin': 15.4,
        'packed_cell_volume': 44,
        'white_blood_cell_count': 7800,
        'red_blood_cell_count': 5.2,
        'red_blood_cells': 'normal',
        'pus_cell': 'normal',
        'pus_cell_clumps': 'notpresent',
        'bacteria': 'notpresent',
        'hypertension': 'no',
        'diabetes_mellitus': 'no',
        'coronary_artery_disease': 'no',
        'appetite': 'good',
        'peda_edema': 'no',
        'anemia': 'no'
    }


@pytest.fixture
def sample_patient_stage3():
    """Sample patient with Stage 3 CKD (moderate)"""
    return {
        'age': 62,
        'blood_pressure': 90,
        'specific_gravity': 1.015,
        'albumin': 2,
        'sugar': 0,
        'blood_glucose_random': 110,
        'blood_urea': 60,
        'serum_creatinine': 2.1,
        'sodium': 135,
        'potassium': 5.0,
        'haemoglobin': 12.5,
        'packed_cell_volume': 38,
        'white_blood_cell_count': 8500,
        'red_blood_cell_count': 4.5,
        'red_blood_cells': 'abnormal',
        'pus_cell': 'normal',
        'pus_cell_clumps': 'notpresent',
        'bacteria': 'notpresent',
        'hypertension': 'yes',
        'diabetes_mellitus': 'no',
        'coronary_artery_disease': 'no',
        'appetite': 'good',
        'peda_edema': 'no',
        'anemia': 'no'
    }


@pytest.fixture
def sample_patient_stage5():
    """Sample patient with Stage 5 CKD (kidney failure)"""
    return {
        'age': 70,
        'blood_pressure': 100,
        'specific_gravity': 1.010,
        'albumin': 4,
        'sugar': 3,
        'blood_glucose_random': 157,
        'blood_urea': 135,
        'serum_creatinine': 7.8,
        'sodium': 130,
        'potassium': 6.2,
        'haemoglobin': 8.9,
        'packed_cell_volume': 28,
        'white_blood_cell_count': 12000,
        'red_blood_cell_count': 3.2,
        'red_blood_cells': 'abnormal',
        'pus_cell': 'abnormal',
        'pus_cell_clumps': 'present',
        'bacteria': 'present',
        'hypertension': 'yes',
        'diabetes_mellitus': 'yes',
        'coronary_artery_disease': 'yes',
        'appetite': 'poor',
        'peda_edema': 'yes',
        'anemia': 'yes'
    }


@pytest.fixture
def sample_patients_batch():
    """Batch of sample patients for multiple predictions"""
    return [
        {'age': 48, 'serum_creatinine': 1.2, 'expected_stage': 1},
        {'age': 55, 'serum_creatinine': 1.8, 'expected_stage': 2},
        {'age': 62, 'serum_creatinine': 2.5, 'expected_stage': 3},
        {'age': 68, 'serum_creatinine': 4.2, 'expected_stage': 4},
        {'age': 70, 'serum_creatinine': 7.8, 'expected_stage': 5}
    ]


@pytest.fixture
def edge_case_inputs():
    """Edge cases and boundary values for testing"""
    return {
        'zero_age': {'age': 0, 'serum_creatinine': 1.2},
        'negative_age': {'age': -5, 'serum_creatinine': 1.2},
        'zero_creatinine': {'age': 50, 'serum_creatinine': 0},
        'negative_creatinine': {'age': 50, 'serum_creatinine': -1.0},
        'none_age': {'age': None, 'serum_creatinine': 1.2},
        'none_creatinine': {'age': 50, 'serum_creatinine': None},
        'very_high_age': {'age': 150, 'serum_creatinine': 1.2},
        'very_high_creatinine': {'age': 50, 'serum_creatinine': 15.0}
    }


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def models_path():
    """Path to models directory"""
    return os.path.join(os.path.dirname(__file__), '..', 'models')


@pytest.fixture
def mock_pca_pipeline():
    """Mock PCA pipeline for fast testing"""
    mock_pipeline = Mock()
    mock_pipeline.transform = Mock(return_value=np.random.randn(1, 20))
    return mock_pipeline


@pytest.fixture
def mock_model():
    """Mock ML model for fast testing"""
    mock = Mock()
    mock.predict = Mock(return_value=np.array([3]))
    mock.predict_proba = Mock(return_value=np.array([[0.05, 0.10, 0.60, 0.20, 0.05]]))
    return mock


@pytest.fixture
def mock_all_models():
    """Mock all 6 models for ensemble testing"""
    models = {}
    model_names = [
        'Logistic Regression',
        'Random Forest',
        'Gradient Boosting',
        'SVM',
        'Naive Bayes',
        'K-Nearest Neighbors'
    ]
    
    for name in model_names:
        mock = Mock()
        mock.predict = Mock(return_value=np.array([3]))
        mock.predict_proba = Mock(return_value=np.array([[0.05, 0.10, 0.60, 0.20, 0.05]]))
        models[name] = mock
    
    return models


@pytest.fixture
def mock_feature_info():
    """Mock feature info dictionary"""
    return {
        'best_model_name': 'Random Forest',
        'feature_names': [f'feature_{i}' for i in range(24)],
        'pca_components': 20,
        'n_classes': 5
    }


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_probabilities():
    """Sample probability distribution for all 5 stages"""
    return {1: 0.05, 2: 0.10, 3: 0.60, 4: 0.20, 5: 0.05}


@pytest.fixture
def sample_egfr_values():
    """Sample eGFR values for each stage"""
    return {
        'stage_1': 95.0,   # >= 90
        'stage_2': 75.0,   # 60-89
        'stage_3': 45.0,   # 30-59
        'stage_4': 20.0,   # 15-29
        'stage_5': 10.0    # < 15
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def suppress_plots():
    """Suppress matplotlib plot displays during testing"""
    import matplotlib
    matplotlib.use('Agg')
    yield
    matplotlib.pyplot.close('all')


# ============================================================================
# Parametrized Test Data
# ============================================================================

@pytest.fixture(params=[
    (1.0, 50, 1),   # Stage 1
    (1.5, 60, 2),   # Stage 2
    (2.5, 65, 3),   # Stage 3
    (4.5, 70, 4),   # Stage 4
    (8.0, 75, 5)    # Stage 5
])
def egfr_test_cases(request):
    """Parametrized test cases for eGFR calculation"""
    creatinine, age, expected_stage = request.param
    return {
        'creatinine': creatinine,
        'age': age,
        'expected_stage': expected_stage
    }


# ============================================================================
# Setup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib state before each test"""
    import matplotlib.pyplot as plt
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup before test session starts"""
    print("\n" + "="*70)
    print("Starting CKD Predictor Test Suite")
    print("="*70)
    yield
    print("\n" + "="*70)
    print("Test Suite Complete")
    print("="*70)
