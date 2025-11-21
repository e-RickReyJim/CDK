"""
Integration Tests
=================
End-to-end integration tests for complete prediction workflows.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================

@pytest.mark.integration
class TestFullPredictionPipeline:
    """End-to-end tests for complete prediction pipeline"""
    
    def test_complete_pipeline_stage1(self, sample_patient_normal):
        """Test complete pipeline from input to output for Stage 1 patient"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(os.path.join(models_path, 'best_model.pkl')):
            pytest.skip("Model files not available")
        
        from src.app import predict_ckd
        from src.utils import calculate_egfr, get_stage_from_egfr
        
        # Step 1: Calculate eGFR
        egfr = calculate_egfr(
            sample_patient_normal['serum_creatinine'],
            sample_patient_normal['age']
        )
        assert egfr is not None
        
        # Step 2: Get stage from eGFR
        egfr_stage = get_stage_from_egfr(egfr)
        assert egfr_stage in [1, 2, 3, 4, 5]
        
        # Step 3: Full prediction
        result = predict_ckd(**sample_patient_normal)
        assert len(result) == 6
        assert result[0] is not None  # Result HTML
    
    def test_complete_pipeline_stage5(self, sample_patient_stage5):
        """Test complete pipeline for Stage 5 patient"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(os.path.join(models_path, 'best_model.pkl')):
            pytest.skip("Model files not available")
        
        from src.app import predict_ckd
        from src.utils import calculate_egfr, get_stage_from_egfr
        
        # Calculate eGFR - should be very low for stage 5
        egfr = calculate_egfr(
            sample_patient_stage5['serum_creatinine'],
            sample_patient_stage5['age']
        )
        assert egfr is not None
        assert egfr < 15  # Stage 5 threshold
        
        # Get stage
        egfr_stage = get_stage_from_egfr(egfr)
        assert egfr_stage == 5
        
        # Full prediction
        result = predict_ckd(**sample_patient_stage5)
        assert len(result) == 6
    
    def test_batch_prediction_consistency(self, sample_patients_batch):
        """Test batch predictions are consistent and reasonable"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(os.path.join(models_path, 'best_model.pkl')):
            pytest.skip("Model files not available")
        
        from src.utils import calculate_egfr, get_stage_from_egfr
        
        for patient in sample_patients_batch:
            egfr = calculate_egfr(
                patient['serum_creatinine'],
                patient['age']
            )
            stage = get_stage_from_egfr(egfr)
            
            # Check stage is within reasonable range of expected
            assert abs(stage - patient['expected_stage']) <= 1


# ============================================================================
# Model Ensemble Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.model
class TestModelEnsembleIntegration:
    """Integration tests for model ensemble"""
    
    def test_all_models_agree_on_obvious_case(self):
        """Test that all models agree on obvious Stage 5 case"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(models_path):
            pytest.skip("Model files not available")
        
        import joblib
        
        # Load all models
        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'Gradient Boosting': 'gradient_boosting_model.pkl',
            'SVM': 'svm_model.pkl',
            'Naive Bayes': 'naive_bayes_model.pkl',
            'K-Nearest Neighbors': 'k-nearest_neighbors_model.pkl'
        }
        
        # Load PCA pipeline
        pca_pipeline = joblib.load(os.path.join(models_path, 'pca_pipeline.pkl'))
        
        # Create extreme Stage 5 case (24 features with poor values)
        X_extreme = np.array([[
            70, 110, 1.010, 4, 3, 0, 0, 0, 0,  # Age, BP, SG, albumin, sugar, categoricals
            157, 135, 7.8, 130, 6.2,  # BGlucose, Urea, Creatinine, Na, K
            8.9, 28, 12000, 3.2,  # Hb, PCV, WBC, RBC
            1, 1, 1, 1, 1, 0  # More categoricals
        ]])
        
        # Transform with PCA
        X_pca = pca_pipeline.transform(X_extreme)
        
        # Get predictions from all models
        predictions = []
        for name, filename in model_files.items():
            model = joblib.load(os.path.join(models_path, filename))
            pred = model.predict(X_pca)[0]
            predictions.append(pred)
        
        # Most models should predict severe stage (4 or 5)
        severe_predictions = sum(1 for p in predictions if p >= 4)
        assert severe_predictions >= 4  # At least 4 out of 6 should agree
    
    def test_ensemble_predictions_distribution(self, sample_patient_normal):
        """Test ensemble prediction distribution"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(os.path.join(models_path, 'best_model.pkl')):
            pytest.skip("Model files not available")
        
        from src.app import predict_ckd
        
        result = predict_ckd(**sample_patient_normal)
        
        # Result should contain information about all models
        comparison_html = result[3]
        
        # Should mention multiple models
        assert 'Logistic Regression' in comparison_html or 'Model' in comparison_html


# ============================================================================
# Visualization Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.visualization
class TestVisualizationIntegration:
    """Integration tests for visualizations in pipeline"""
    
    def test_all_visualizations_generated(self, sample_patient_normal):
        """Test that all visualizations are generated"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(os.path.join(models_path, 'best_model.pkl')):
            pytest.skip("Model files not available")
        
        from src.app import predict_ckd
        import matplotlib.pyplot as plt
        
        result = predict_ckd(**sample_patient_normal)
        
        # Should have probability plot, eGFR gauge, and agreement plot
        # (indices 1, 2, 4 in result tuple)
        assert result[1] is not None  # Probability plot
        assert result[2] is not None  # eGFR gauge
        assert result[4] is not None  # Agreement plot
        
        plt.close('all')
    
    def test_visualizations_valid_for_all_stages(self, sample_patients_batch):
        """Test visualizations work for all stages"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(models_path):
            pytest.skip("Model files not available")
        
        from src.utils import (
            calculate_egfr,
            get_stage_from_egfr,
            create_stage_probability_plot,
            create_egfr_gauge,
            get_stage_info
        )
        import matplotlib.pyplot as plt
        
        for patient in sample_patients_batch:
            egfr = calculate_egfr(
                patient['serum_creatinine'],
                patient['age']
            )
            stage = get_stage_from_egfr(egfr)
            
            # Create visualizations
            probs = {i: 0.2 for i in range(1, 6)}
            fig1 = create_stage_probability_plot(probs)
            fig2 = create_egfr_gauge(egfr)
            
            assert fig1 is not None
            assert fig2 is not None
            
            plt.close('all')


# ============================================================================
# Data Flow Integration Tests
# ============================================================================

@pytest.mark.integration
class TestDataFlow:
    """Integration tests for data flow through pipeline"""
    
    def test_data_transformation_pipeline(self):
        """Test data transformation from input to model"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(models_path):
            pytest.skip("Model files not available")
        
        import joblib
        
        # Load pipeline
        pca_pipeline = joblib.load(os.path.join(models_path, 'pca_pipeline.pkl'))
        best_model = joblib.load(os.path.join(models_path, 'best_model.pkl'))
        
        # Create sample input (24 features)
        X_original = np.random.randn(1, 24)
        
        # Transform with PCA (24 -> 20)
        X_pca = pca_pipeline.transform(X_original)
        assert X_pca.shape == (1, 20)
        
        # Predict with model
        prediction = best_model.predict(X_pca)
        assert prediction[0] in [1, 2, 3, 4, 5]
        
        # Get probabilities
        proba = best_model.predict_proba(X_pca)
        assert proba.shape == (1, 5)
        assert np.allclose(proba.sum(), 1.0)
    
    def test_feature_mapping_consistency(self, sample_patient_normal):
        """Test that feature mapping is consistent"""
        # Test categorical mapping
        categorical_mapping = {
            'Normal': 1, 'Abnormal': 0,
            'Not Present': 0, 'Present': 1,
            'No': 0, 'Yes': 1,
            'Good': 1, 'Poor': 0
        }
        
        # Map sample patient categoricals
        mapped_features = {
            'red_blood_cells': categorical_mapping[sample_patient_normal['red_blood_cells']],
            'pus_cell': categorical_mapping[sample_patient_normal['pus_cell']],
            'pus_cell_clumps': categorical_mapping[sample_patient_normal['pus_cell_clumps']],
            'bacteria': categorical_mapping[sample_patient_normal['bacteria']],
            'hypertension': categorical_mapping[sample_patient_normal['hypertension']],
            'diabetes_mellitus': categorical_mapping[sample_patient_normal['diabetes_mellitus']],
            'coronary_artery_disease': categorical_mapping[sample_patient_normal['coronary_artery_disease']],
            'appetite': categorical_mapping[sample_patient_normal['appetite']],
            'peda_edema': categorical_mapping[sample_patient_normal['peda_edema']],
            'anemia': categorical_mapping[sample_patient_normal['anemia']]
        }
        
        # All should be 0 or 1
        assert all(val in [0, 1] for val in mapped_features.values())


# ============================================================================
# eGFR Consistency Tests
# ============================================================================

@pytest.mark.integration
class TestEGFRConsistency:
    """Integration tests for eGFR calculation consistency"""
    
    def test_egfr_stage_alignment(self, sample_patients_batch):
        """Test that eGFR calculation aligns with stage classification"""
        from src.utils import calculate_egfr, get_stage_from_egfr
        
        for patient in sample_patients_batch:
            egfr = calculate_egfr(
                patient['serum_creatinine'],
                patient['age']
            )
            stage = get_stage_from_egfr(egfr)
            
            # Verify stage boundaries
            if stage == 1:
                assert egfr >= 90
            elif stage == 2:
                assert 60 <= egfr < 90
            elif stage == 3:
                assert 30 <= egfr < 60
            elif stage == 4:
                assert 15 <= egfr < 30
            elif stage == 5:
                assert egfr < 15
    
    def test_egfr_monotonicity(self):
        """Test that eGFR decreases as creatinine increases"""
        from src.utils import calculate_egfr
        
        age = 60
        creatinines = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        egfrs = [calculate_egfr(cr, age) for cr in creatinines]
        
        # eGFR should decrease as creatinine increases
        for i in range(len(egfrs) - 1):
            assert egfrs[i] > egfrs[i + 1]
    
    def test_egfr_age_effect(self):
        """Test that eGFR decreases with age (for same creatinine)"""
        from src.utils import calculate_egfr
        
        creatinine = 1.5
        ages = [30, 40, 50, 60, 70]
        
        egfrs = [calculate_egfr(creatinine, age) for age in ages]
        
        # eGFR should decrease with age
        for i in range(len(egfrs) - 1):
            assert egfrs[i] > egfrs[i + 1]


# ============================================================================
# Error Recovery Integration Tests
# ============================================================================

@pytest.mark.integration
class TestErrorRecovery:
    """Integration tests for error handling and recovery"""
    
    def test_graceful_handling_of_edge_cases(self, edge_case_inputs):
        """Test graceful handling of edge case inputs"""
        from src.utils import calculate_egfr, get_stage_from_egfr
        
        for case_name, inputs in edge_case_inputs.items():
            egfr = calculate_egfr(
                inputs['serum_creatinine'],
                inputs['age']
            )
            
            # Should return None for invalid inputs
            if inputs['age'] <= 0 or inputs['serum_creatinine'] <= 0 or \
               inputs['age'] is None or inputs['serum_creatinine'] is None:
                assert egfr is None
            
            # Stage classification should handle None
            stage = get_stage_from_egfr(egfr)
            if egfr is None:
                assert stage is None
    
    def test_model_prediction_robustness(self):
        """Test model predictions are robust to input variations"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(models_path):
            pytest.skip("Model files not available")
        
        import joblib
        
        pca_pipeline = joblib.load(os.path.join(models_path, 'pca_pipeline.pkl'))
        best_model = joblib.load(os.path.join(models_path, 'best_model.pkl'))
        
        # Test with various input scales
        for _ in range(10):
            X = np.random.randn(1, 24) * np.random.uniform(0.5, 2.0)
            X_pca = pca_pipeline.transform(X)
            prediction = best_model.predict(X_pca)
            
            # Should always predict valid stage
            assert prediction[0] in [1, 2, 3, 4, 5]


# ============================================================================
# Performance Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for performance"""
    
    def test_prediction_speed(self, sample_patient_normal):
        """Test that prediction completes in reasonable time"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(os.path.join(models_path, 'best_model.pkl')):
            pytest.skip("Model files not available")
        
        from src.app import predict_ckd
        import time
        
        start = time.time()
        result = predict_ckd(**sample_patient_normal)
        elapsed = time.time() - start
        
        # Prediction should complete in under 2 seconds
        assert elapsed < 2.0
        assert result is not None
    
    def test_batch_prediction_efficiency(self, sample_patients_batch):
        """Test efficiency of multiple predictions"""
        # Skip if models not available
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        if not os.path.exists(models_path):
            pytest.skip("Model files not available")
        
        from src.utils import calculate_egfr, get_stage_from_egfr
        import time
        
        start = time.time()
        
        for patient in sample_patients_batch:
            egfr = calculate_egfr(
                patient['serum_creatinine'],
                patient['age']
            )
            stage = get_stage_from_egfr(egfr)
        
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 0.1
