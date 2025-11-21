"""
Tests for Model Loading and Predictions
=======================================
Tests for ML model loading, predictions, and consistency.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import pickle
import os
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Model Loading Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
class TestModelLoading:
    """Tests for model loading functionality"""
    
    def test_models_directory_exists(self, models_path):
        """Test that models directory exists"""
        assert os.path.exists(models_path), f"Models directory not found: {models_path}"
    
    def test_all_model_files_exist(self, models_path):
        """Test that all required model files exist"""
        required_files = [
            'pca_pipeline.pkl',
            'best_model.pkl',
            'feature_info.pkl',
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'svm_model.pkl',
            'naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        for filename in required_files:
            filepath = os.path.join(models_path, filename)
            assert os.path.exists(filepath), f"Model file not found: {filename}"
    
    def test_pca_pipeline_loads(self, models_path):
        """Test that PCA pipeline loads successfully"""
        pca_path = os.path.join(models_path, 'pca_pipeline.pkl')
        pca_pipeline = joblib.load(pca_path)
        
        assert pca_pipeline is not None
        assert hasattr(pca_pipeline, 'transform')
    
    def test_best_model_loads(self, models_path):
        """Test that best model loads successfully"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_feature_info_loads(self, models_path):
        """Test that feature info loads successfully"""
        info_path = os.path.join(models_path, 'feature_info.pkl')
        with open(info_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        assert feature_info is not None
        assert isinstance(feature_info, dict)
        assert 'best_model_name' in feature_info
    
    def test_all_six_models_load(self, models_path):
        """Test that all 6 models load successfully"""
        model_files = [
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'svm_model.pkl',
            'naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            model = joblib.load(model_path)
            
            assert model is not None
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')


# ============================================================================
# Model Prediction Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
class TestModelPredictions:
    """Tests for model predictions"""
    
    def test_best_model_predict_shape(self, models_path):
        """Test that best model prediction has correct shape"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        # Create dummy input (20 PCA components)
        X_test = np.random.randn(1, 20)
        prediction = model.predict(X_test)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [1, 2, 3, 4, 5]
    
    def test_best_model_predict_proba_shape(self, models_path):
        """Test that prediction probabilities have correct shape"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        # Create dummy input
        X_test = np.random.randn(1, 20)
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (1, 5)  # 1 sample, 5 classes
        assert np.allclose(proba.sum(), 1.0, atol=1e-5)  # Probabilities sum to 1
    
    def test_probabilities_valid_range(self, models_path):
        """Test that probabilities are between 0 and 1"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        X_test = np.random.randn(10, 20)
        proba = model.predict_proba(X_test)
        
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
    
    def test_prediction_consistency(self, models_path):
        """Test that same input produces same prediction"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        X_test = np.random.randn(1, 20)
        
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        pred3 = model.predict(X_test)
        
        assert np.array_equal(pred1, pred2)
        assert np.array_equal(pred2, pred3)
    
    def test_all_models_predict_same_input(self, models_path):
        """Test that all 6 models can predict the same input"""
        model_files = [
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'svm_model.pkl',
            'naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        X_test = np.random.randn(1, 20)
        predictions = []
        
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            model = joblib.load(model_path)
            pred = model.predict(X_test)
            
            assert pred[0] in [1, 2, 3, 4, 5]
            predictions.append(pred[0])
        
        # All predictions should be valid stages
        assert len(predictions) == 6
        assert all(p in [1, 2, 3, 4, 5] for p in predictions)
    
    def test_batch_predictions(self, models_path):
        """Test predictions on batch of samples"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        # Batch of 10 samples
        X_test = np.random.randn(10, 20)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert all(p in [1, 2, 3, 4, 5] for p in predictions)


# ============================================================================
# PCA Pipeline Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
class TestPCAPipeline:
    """Tests for PCA preprocessing pipeline"""
    
    def test_pca_transform_shape(self, models_path):
        """Test that PCA transform produces correct output shape"""
        pca_path = os.path.join(models_path, 'pca_pipeline.pkl')
        pca_pipeline = joblib.load(pca_path)
        
        # Create dummy input (24 original features)
        X_original = np.random.randn(1, 24)
        X_transformed = pca_pipeline.transform(X_original)
        
        # Should reduce to 20 PCA components
        assert X_transformed.shape == (1, 20)
    
    def test_pca_preserves_batch_size(self, models_path):
        """Test that PCA preserves batch size"""
        pca_path = os.path.join(models_path, 'pca_pipeline.pkl')
        pca_pipeline = joblib.load(pca_path)
        
        # Batch of 5 samples
        X_original = np.random.randn(5, 24)
        X_transformed = pca_pipeline.transform(X_original)
        
        assert X_transformed.shape == (5, 20)
    
    def test_pca_consistency(self, models_path):
        """Test that PCA produces consistent results"""
        pca_path = os.path.join(models_path, 'pca_pipeline.pkl')
        pca_pipeline = joblib.load(pca_path)
        
        X_original = np.random.randn(1, 24)
        
        X_transformed1 = pca_pipeline.transform(X_original)
        X_transformed2 = pca_pipeline.transform(X_original)
        
        assert np.allclose(X_transformed1, X_transformed2)


# ============================================================================
# Feature Info Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
class TestFeatureInfo:
    """Tests for feature info dictionary"""
    
    def test_feature_info_structure(self, models_path):
        """Test that feature info has required keys"""
        info_path = os.path.join(models_path, 'feature_info.pkl')
        with open(info_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        required_keys = ['best_model_name']
        for key in required_keys:
            assert key in feature_info, f"Missing required key: {key}"
    
    def test_best_model_name_valid(self, models_path):
        """Test that best model name is one of the 6 models"""
        info_path = os.path.join(models_path, 'feature_info.pkl')
        with open(info_path, 'rb') as f:
            feature_info = pickle.load(f)
        
        valid_names = [
            'Logistic Regression',
            'Random Forest',
            'Gradient Boosting',
            'SVM',
            'Naive Bayes',
            'K-Nearest Neighbors'
        ]
        
        assert feature_info['best_model_name'] in valid_names


# ============================================================================
# Mock Model Tests
# ============================================================================

@pytest.mark.unit
class TestMockModels:
    """Tests using mock models for fast testing"""
    
    def test_mock_model_predict(self, mock_model):
        """Test mock model prediction"""
        X_test = np.random.randn(1, 20)
        prediction = mock_model.predict(X_test)
        
        assert prediction is not None
        assert len(prediction) == 1
    
    def test_mock_model_predict_proba(self, mock_model):
        """Test mock model probability prediction"""
        X_test = np.random.randn(1, 20)
        proba = mock_model.predict_proba(X_test)
        
        assert proba is not None
        assert proba.shape == (1, 5)
    
    def test_mock_all_models(self, mock_all_models):
        """Test all mock models"""
        assert len(mock_all_models) == 6
        
        X_test = np.random.randn(1, 20)
        
        for name, model in mock_all_models.items():
            prediction = model.predict(X_test)
            assert prediction is not None
    
    def test_mock_pca_pipeline(self, mock_pca_pipeline):
        """Test mock PCA pipeline"""
        X_original = np.random.randn(1, 24)
        X_transformed = mock_pca_pipeline.transform(X_original)
        
        assert X_transformed is not None
        assert X_transformed.shape == (1, 20)


# ============================================================================
# Model Ensemble Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
class TestModelEnsemble:
    """Tests for model ensemble functionality"""
    
    def test_ensemble_predictions_valid(self, models_path):
        """Test that ensemble predictions are all valid stages"""
        model_files = [
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'svm_model.pkl',
            'naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        X_test = np.random.randn(1, 20)
        predictions = []
        
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            model = joblib.load(model_path)
            pred = model.predict(X_test)[0]
            predictions.append(pred)
        
        # Check all predictions are valid
        assert all(p in [1, 2, 3, 4, 5] for p in predictions)
    
    def test_ensemble_majority_vote(self, models_path):
        """Test ensemble majority voting"""
        model_files = [
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'svm_model.pkl',
            'naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        X_test = np.random.randn(1, 20)
        predictions = []
        
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            model = joblib.load(model_path)
            pred = model.predict(X_test)[0]
            predictions.append(pred)
        
        # Calculate majority vote
        from collections import Counter
        vote_counts = Counter(predictions)
        majority_prediction = vote_counts.most_common(1)[0][0]
        
        assert majority_prediction in [1, 2, 3, 4, 5]
    
    def test_ensemble_agreement_rate(self, models_path):
        """Test ensemble agreement rate"""
        model_files = [
            'logistic_regression_model.pkl',
            'random_forest_model.pkl',
            'gradient_boosting_model.pkl',
            'svm_model.pkl',
            'naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        X_test = np.random.randn(1, 20)
        predictions = []
        
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            model = joblib.load(model_path)
            pred = model.predict(X_test)[0]
            predictions.append(pred)
        
        # Calculate agreement rate
        from collections import Counter
        vote_counts = Counter(predictions)
        max_agreement = vote_counts.most_common(1)[0][1]
        agreement_rate = max_agreement / len(predictions)
        
        # Agreement rate should be between 0 and 1
        assert 0 < agreement_rate <= 1.0


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.model
class TestModelErrorHandling:
    """Tests for model error handling"""
    
    def test_missing_model_file(self):
        """Test handling of missing model file"""
        fake_path = os.path.join('nonexistent', 'fake_model.pkl')
        
        with pytest.raises(FileNotFoundError):
            joblib.load(fake_path)
    
    def test_invalid_input_shape(self, models_path):
        """Test model with invalid input shape"""
        model_path = os.path.join(models_path, 'best_model.pkl')
        model = joblib.load(model_path)
        
        # Wrong number of features
        X_wrong = np.random.randn(1, 10)  # Should be 20
        
        with pytest.raises(ValueError):
            model.predict(X_wrong)
    
    def test_pca_invalid_input_shape(self, models_path):
        """Test PCA with invalid input shape"""
        pca_path = os.path.join(models_path, 'pca_pipeline.pkl')
        pca_pipeline = joblib.load(pca_path)
        
        # Wrong number of features
        X_wrong = np.random.randn(1, 10)  # Should be 24
        
        with pytest.raises(ValueError):
            pca_pipeline.transform(X_wrong)
