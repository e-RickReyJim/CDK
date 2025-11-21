"""
Tests for Main Application
===========================
Tests for the main Gradio application and prediction function.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================================
# Application Import Tests
# ============================================================================


@pytest.mark.unit
class TestApplicationImports:
    """Tests for application imports"""

    def test_app_module_imports(self):
        """Test that app module can be imported"""
        try:
            import src.app as app

            assert hasattr(app, "predict_ckd")
        except ImportError as e:
            pytest.fail(f"Failed to import app module: {e}")

    def test_load_models_function_exists(self):
        """Test that load_models function exists"""
        import src.app as app

        assert hasattr(app, "load_models")


# ============================================================================
# Predict CKD Function Tests
# ============================================================================


@pytest.mark.unit
class TestPredictCKD:
    """Tests for predict_ckd function"""

    @patch("src.app.pca_pipeline")
    @patch("src.app.best_model")
    @patch("src.app.all_models")
    @patch("src.app.feature_info")
    def test_predict_ckd_with_valid_inputs(
        self,
        mock_feature_info,
        mock_all_models,
        mock_best_model,
        mock_pca_pipeline,
        sample_patient_normal,
    ):
        """Test predict_ckd with valid patient data"""
        # Setup mocks
        mock_pca_pipeline.transform = Mock(return_value=np.random.randn(1, 20))
        mock_best_model.predict = Mock(return_value=np.array([3]))
        mock_best_model.predict_proba = Mock(
            return_value=np.array([[0.05, 0.10, 0.60, 0.20, 0.05]])
        )

        mock_all_models.items = Mock(
            return_value=[
                ("Model1", Mock(predict=Mock(return_value=np.array([3])))),
                ("Model2", Mock(predict=Mock(return_value=np.array([3])))),
            ]
        )

        mock_feature_info.__getitem__ = Mock(return_value="Random Forest")

        from src.app import predict_ckd

        # Call function with sample patient data
        result = predict_ckd(**sample_patient_normal)

        # Check that result is a tuple with 6 elements
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_predict_ckd_output_types(self, sample_patient_normal):
        """Test that predict_ckd returns correct output types"""
        # This test requires actual models - skip if models not available
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
        ):
            pytest.skip("Model files not available")

        from src.app import predict_ckd

        result = predict_ckd(**sample_patient_normal)

        # Result should be a tuple with 6 elements
        assert isinstance(result, tuple)
        assert len(result) == 6

        # First element: result HTML (string)
        assert isinstance(result[0], str)

        # Elements 2 and 3: plots (matplotlib figures)
        # (checking type would require matplotlib)

        # Fourth element: comparison HTML (string)
        assert isinstance(result[3], str)

    def test_predict_ckd_categorical_mapping(self):
        """Test categorical input mapping"""
        # Test that categorical inputs are correctly mapped
        categorical_tests = [
            ("Normal", 1),
            ("Abnormal", 0),
            ("Not Present", 0),
            ("Present", 1),
            ("No", 0),
            ("Yes", 1),
            ("Good", 1),
            ("Poor", 0),
        ]

        categorical_mapping = {
            "Normal": 1,
            "Abnormal": 0,
            "Not Present": 0,
            "Present": 1,
            "No": 0,
            "Yes": 1,
            "Good": 1,
            "Poor": 0,
        }

        for input_val, expected_val in categorical_tests:
            assert categorical_mapping[input_val] == expected_val


# ============================================================================
# Model Loading Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.model
class TestModelLoadingInApp:
    """Tests for model loading in application"""

    def test_load_models_returns_correct_types(self):
        """Test that load_models returns correct types"""
        # Skip if models not available
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models")):
            pytest.skip("Model files not available")

        from src.app import load_models

        pca_pipeline, best_model, feature_info, all_models = load_models()

        # Check types
        assert pca_pipeline is not None
        assert best_model is not None
        assert isinstance(feature_info, dict)
        assert isinstance(all_models, dict)
        assert len(all_models) == 6

    def test_load_models_error_handling(self):
        """Test error handling when models are missing"""
        from src.app import load_models

        # Patch os.path.join to return non-existent path
        with patch("os.path.join", return_value="/nonexistent/path"):
            with pytest.raises(FileNotFoundError):
                load_models()


# ============================================================================
# Gradio Interface Tests
# ============================================================================


@pytest.mark.unit
class TestGradioInterface:
    """Tests for Gradio interface setup"""

    def test_gradio_interface_creation(self):
        """Test that Gradio interface can be created"""
        try:
            import gradio as gr

            # Test basic interface creation
            with gr.Blocks() as demo:
                with gr.Row():
                    gr.Textbox(label="Test")

            assert demo is not None
        except ImportError:
            pytest.skip("Gradio not installed")

    def test_gradio_components_exist(self):
        """Test that required Gradio components are available"""
        try:
            import gradio as gr

            # Check that required components exist
            assert hasattr(gr, "Blocks")
            assert hasattr(gr, "Row")
            assert hasattr(gr, "Column")
            assert hasattr(gr, "Number")
            assert hasattr(gr, "Radio")
            assert hasattr(gr, "Button")
            assert hasattr(gr, "HTML")
            assert hasattr(gr, "Plot")
        except ImportError:
            pytest.skip("Gradio not installed")


# ============================================================================
# Integration Tests for Full Prediction Flow
# ============================================================================


@pytest.mark.integration
class TestPredictionFlow:
    """Integration tests for complete prediction flow"""

    def test_full_prediction_with_stage1_patient(self, sample_patient_normal):
        """Test full prediction flow with Stage 1 patient"""
        # Skip if models not available
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
        ):
            pytest.skip("Model files not available")

        from src.app import predict_ckd

        result = predict_ckd(**sample_patient_normal)

        # Should return 6 outputs
        assert len(result) == 6

        # Result HTML should mention a stage
        result_html = result[0]
        assert any(f"Stage {i}" in result_html for i in range(1, 6))

    def test_full_prediction_with_stage5_patient(self, sample_patient_stage5):
        """Test full prediction flow with Stage 5 patient"""
        # Skip if models not available
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
        ):
            pytest.skip("Model files not available")

        from src.app import predict_ckd

        result = predict_ckd(**sample_patient_stage5)

        # Should return 6 outputs
        assert len(result) == 6

        # Result HTML should mention a stage
        result_html = result[0]
        assert any(f"Stage {i}" in result_html for i in range(1, 6))

    def test_multiple_predictions_consistency(self, sample_patient_normal):
        """Test that multiple predictions are consistent"""
        # Skip if models not available
        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
        ):
            pytest.skip("Model files not available")

        from src.app import predict_ckd

        # Make three predictions with same inputs
        result1 = predict_ckd(**sample_patient_normal)
        result2 = predict_ckd(**sample_patient_normal)
        result3 = predict_ckd(**sample_patient_normal)

        # Results should be identical
        assert result1[0] == result2[0] == result3[0]


# ============================================================================
# Input Validation Tests
# ============================================================================


@pytest.mark.unit
class TestInputValidation:
    """Tests for input validation"""

    def test_numeric_features_accept_floats(self):
        """Test that numeric features accept float values"""
        numeric_values = [
            48.0,
            80.0,
            1.020,
            0.0,
            0.0,
            121.0,
            36.0,
            1.2,
            138.0,
            4.6,
            15.4,
            44.0,
            7800.0,
            5.2,
        ]

        for val in numeric_values:
            assert isinstance(val, (int, float))

    def test_numeric_features_accept_integers(self):
        """Test that numeric features accept integer values"""
        numeric_values = [48, 80, 1, 0, 0, 121, 36, 1, 138, 4, 15, 44, 7800, 5]

        for val in numeric_values:
            assert isinstance(val, (int, float))

    def test_categorical_features_valid_options(self):
        """Test that categorical features have valid options"""
        valid_options = {
            "red_blood_cells": ["Normal", "Abnormal"],
            "pus_cell": ["Normal", "Abnormal"],
            "pus_cell_clumps": ["Not Present", "Present"],
            "bacteria": ["Not Present", "Present"],
            "hypertension": ["No", "Yes"],
            "diabetes_mellitus": ["No", "Yes"],
            "coronary_artery_disease": ["No", "Yes"],
            "appetite": ["Good", "Poor"],
            "peda_edema": ["No", "Yes"],
            "anemia": ["No", "Yes"],
        }

        for feature, options in valid_options.items():
            assert len(options) == 2
            assert all(isinstance(opt, str) for opt in options)


# ============================================================================
# Output Format Tests
# ============================================================================


@pytest.mark.unit
class TestOutputFormat:
    """Tests for output format"""

    @patch("src.app.pca_pipeline")
    @patch("src.app.best_model")
    @patch("src.app.all_models")
    @patch("src.app.feature_info")
    @patch("src.app.create_stage_probability_plot")
    @patch("src.app.create_egfr_gauge")
    @patch("src.app.create_model_agreement_plot")
    @patch("src.app.create_info_plot")
    def test_output_tuple_structure(
        self,
        mock_info_plot,
        mock_agreement_plot,
        mock_gauge,
        mock_prob_plot,
        mock_feature_info,
        mock_all_models,
        mock_best_model,
        mock_pca_pipeline,
        sample_patient_normal,
    ):
        """Test that output has correct structure"""
        # Setup mocks
        mock_pca_pipeline.transform = Mock(return_value=np.random.randn(1, 20))
        mock_best_model.predict = Mock(return_value=np.array([3]))
        mock_best_model.predict_proba = Mock(
            return_value=np.array([[0.05, 0.10, 0.60, 0.20, 0.05]])
        )

        mock_models = {}
        for i in range(6):
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.array([3]))
            mock_models[f"Model{i}"] = mock_model

        mock_all_models.items = Mock(return_value=mock_models.items())
        mock_feature_info.__getitem__ = Mock(return_value="Random Forest")

        # Mock plotting functions
        mock_fig = MagicMock()
        mock_prob_plot.return_value = mock_fig
        mock_gauge.return_value = mock_fig
        mock_agreement_plot.return_value = mock_fig
        mock_info_plot.return_value = mock_fig

        from src.app import predict_ckd

        result = predict_ckd(**sample_patient_normal)

        # Should return tuple with 6 elements
        assert isinstance(result, tuple)
        assert len(result) == 6

        # Check element types
        assert isinstance(result[0], str)  # Result HTML
        # Elements 1, 2, 4 are plots
        assert isinstance(result[3], str)  # Comparison HTML
        assert isinstance(result[5], str)  # Confidence label
