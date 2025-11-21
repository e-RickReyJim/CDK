"""
Unit Tests for Utility Functions
=================================
Tests for eGFR calculation, stage classification, and visualization functions.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import (
    calculate_egfr,
    get_stage_from_egfr,
    get_stage_info,
    create_stage_probability_plot,
    create_egfr_gauge,
    create_model_agreement_plot,
    create_info_plot
)


# ============================================================================
# eGFR Calculation Tests
# ============================================================================

@pytest.mark.unit
class TestCalculateEGFR:
    """Tests for calculate_egfr function"""
    
    def test_valid_inputs(self):
        """Test eGFR calculation with valid inputs"""
        egfr = calculate_egfr(serum_creatinine=1.2, age=50)
        assert egfr is not None
        assert isinstance(egfr, (int, float))
        assert 0 < egfr < 200  # Reasonable range
    
    def test_typical_values(self):
        """Test with typical clinical values"""
        # Stage 1: Low creatinine, young age
        egfr1 = calculate_egfr(1.0, 30)
        assert egfr1 > 90
        
        # Stage 3: Moderate creatinine, older age
        egfr3 = calculate_egfr(2.0, 60)
        assert 30 <= egfr3 < 60
        
        # Stage 5: High creatinine, elderly
        egfr5 = calculate_egfr(6.0, 70)
        assert egfr5 < 15
    
    def test_zero_creatinine(self):
        """Test with zero creatinine (invalid)"""
        egfr = calculate_egfr(0, 50)
        assert egfr is None
    
    def test_negative_creatinine(self):
        """Test with negative creatinine (invalid)"""
        egfr = calculate_egfr(-1.2, 50)
        assert egfr is None
    
    def test_zero_age(self):
        """Test with zero age (invalid)"""
        egfr = calculate_egfr(1.2, 0)
        assert egfr is None
    
    def test_negative_age(self):
        """Test with negative age (invalid)"""
        egfr = calculate_egfr(1.2, -50)
        assert egfr is None
    
    def test_none_creatinine(self):
        """Test with None creatinine"""
        egfr = calculate_egfr(None, 50)
        assert egfr is None
    
    def test_none_age(self):
        """Test with None age"""
        egfr = calculate_egfr(1.2, None)
        assert egfr is None
    
    def test_both_none(self):
        """Test with both parameters None"""
        egfr = calculate_egfr(None, None)
        assert egfr is None
    
    @pytest.mark.parametrize("creatinine,age,expected_min,expected_max", [
        (0.8, 25, 100, 150),  # Young, low creatinine
        (1.2, 50, 60, 100),   # Middle age, normal creatinine
        (3.0, 70, 15, 40),    # Elderly, high creatinine
        (8.0, 80, 5, 15)      # Very high creatinine
    ])
    def test_egfr_ranges(self, creatinine, age, expected_min, expected_max):
        """Test eGFR falls within expected ranges"""
        egfr = calculate_egfr(creatinine, age)
        assert expected_min <= egfr <= expected_max


# ============================================================================
# Stage Classification Tests
# ============================================================================

@pytest.mark.unit
class TestGetStageFromEGFR:
    """Tests for get_stage_from_egfr function"""
    
    def test_stage_1_high(self):
        """Test Stage 1 classification (eGFR >= 90)"""
        assert get_stage_from_egfr(90) == 1
        assert get_stage_from_egfr(100) == 1
        assert get_stage_from_egfr(120) == 1
    
    def test_stage_2(self):
        """Test Stage 2 classification (eGFR 60-89)"""
        assert get_stage_from_egfr(60) == 2
        assert get_stage_from_egfr(75) == 2
        assert get_stage_from_egfr(89) == 2
    
    def test_stage_3(self):
        """Test Stage 3 classification (eGFR 30-59)"""
        assert get_stage_from_egfr(30) == 3
        assert get_stage_from_egfr(45) == 3
        assert get_stage_from_egfr(59) == 3
    
    def test_stage_4(self):
        """Test Stage 4 classification (eGFR 15-29)"""
        assert get_stage_from_egfr(15) == 4
        assert get_stage_from_egfr(22) == 4
        assert get_stage_from_egfr(29) == 4
    
    def test_stage_5(self):
        """Test Stage 5 classification (eGFR < 15)"""
        assert get_stage_from_egfr(14) == 5
        assert get_stage_from_egfr(10) == 5
        assert get_stage_from_egfr(5) == 5
    
    def test_boundary_values(self):
        """Test exact boundary values between stages"""
        assert get_stage_from_egfr(90.0) == 1  # Stage 1/2 boundary
        assert get_stage_from_egfr(89.9) == 2
        assert get_stage_from_egfr(60.0) == 2  # Stage 2/3 boundary
        assert get_stage_from_egfr(59.9) == 3
        assert get_stage_from_egfr(30.0) == 3  # Stage 3/4 boundary
        assert get_stage_from_egfr(29.9) == 4
        assert get_stage_from_egfr(15.0) == 4  # Stage 4/5 boundary
        assert get_stage_from_egfr(14.9) == 5
    
    def test_none_input(self):
        """Test with None input"""
        assert get_stage_from_egfr(None) is None
    
    def test_nan_input(self):
        """Test with NaN input"""
        assert get_stage_from_egfr(np.nan) is None
    
    def test_negative_egfr(self):
        """Test with negative eGFR (should still classify as stage 5)"""
        assert get_stage_from_egfr(-5) == 5


# ============================================================================
# Stage Info Tests
# ============================================================================

@pytest.mark.unit
class TestGetStageInfo:
    """Tests for get_stage_info function"""
    
    def test_stage_1_info(self):
        """Test Stage 1 info"""
        info = get_stage_info(1)
        assert info['name'] == 'Stage 1'
        assert 'Normal' in info['severity']
        assert info['color'] == '#28a745'
        assert 'normal function' in info['description'].lower()
    
    def test_stage_2_info(self):
        """Test Stage 2 info"""
        info = get_stage_info(2)
        assert info['name'] == 'Stage 2'
        assert 'Mild' in info['severity']
        assert info['color'] == '#90ee90'
    
    def test_stage_3_info(self):
        """Test Stage 3 info"""
        info = get_stage_info(3)
        assert info['name'] == 'Stage 3'
        assert 'Moderate' in info['severity']
        assert info['color'] == '#ffc107'
    
    def test_stage_4_info(self):
        """Test Stage 4 info"""
        info = get_stage_info(4)
        assert info['name'] == 'Stage 4'
        assert 'Severe' in info['severity']
        assert info['color'] == '#ff8c00'
    
    def test_stage_5_info(self):
        """Test Stage 5 info"""
        info = get_stage_info(5)
        assert info['name'] == 'Stage 5'
        assert 'Failure' in info['severity']
        assert info['color'] == '#dc3545'
        assert 'dialysis' in info['description'].lower()
    
    def test_all_stages_have_required_keys(self):
        """Test all stages have required keys"""
        required_keys = {'name', 'severity', 'color', 'description'}
        for stage in range(1, 6):
            info = get_stage_info(stage)
            assert required_keys.issubset(info.keys())
    
    def test_invalid_stage(self):
        """Test with invalid stage number"""
        info = get_stage_info(99)
        assert info['name'] == 'Unknown'
        assert info['severity'] == 'Unknown'
    
    def test_colors_are_hex(self):
        """Test all colors are valid hex codes"""
        for stage in range(1, 6):
            info = get_stage_info(stage)
            assert info['color'].startswith('#')
            assert len(info['color']) == 7


# ============================================================================
# Visualization Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.visualization
class TestVisualizationFunctions:
    """Tests for visualization functions"""
    
    def test_create_stage_probability_plot(self, sample_probabilities, suppress_plots):
        """Test stage probability plot creation"""
        fig = create_stage_probability_plot(sample_probabilities)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1  # Should have one axis
        
        # Check bars are created
        ax = fig.axes[0]
        assert len(ax.patches) == 5  # 5 stages = 5 bars
        
        plt.close(fig)
    
    def test_probability_plot_with_zeros(self, suppress_plots):
        """Test probability plot with zero probabilities"""
        probs = {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}
        fig = create_stage_probability_plot(probs)
        
        assert fig is not None
        plt.close(fig)
    
    def test_probability_plot_missing_stages(self, suppress_plots):
        """Test probability plot with missing stages"""
        probs = {1: 0.2, 3: 0.8}  # Missing stages 2, 4, 5
        fig = create_stage_probability_plot(probs)
        
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.patches) == 5  # Should still create 5 bars
        
        plt.close(fig)
    
    def test_create_egfr_gauge(self, sample_egfr_values, suppress_plots):
        """Test eGFR gauge creation"""
        for stage_name, egfr_value in sample_egfr_values.items():
            fig = create_egfr_gauge(egfr_value)
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            
            plt.close(fig)
    
    def test_egfr_gauge_extreme_values(self, suppress_plots):
        """Test eGFR gauge with extreme values"""
        # Very low eGFR
        fig1 = create_egfr_gauge(5.0)
        assert fig1 is not None
        plt.close(fig1)
        
        # Very high eGFR
        fig2 = create_egfr_gauge(150.0)
        assert fig2 is not None
        plt.close(fig2)
    
    def test_create_model_agreement_plot(self, suppress_plots):
        """Test model agreement plot creation"""
        model_predictions = {
            'Logistic Regression': 3,
            'Random Forest': 3,
            'Gradient Boosting': 3,
            'SVM': 4,
            'Naive Bayes': 3,
            'K-Nearest Neighbors': 3
        }
        
        fig = create_model_agreement_plot(model_predictions)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_model_agreement_all_agree(self, suppress_plots):
        """Test model agreement when all models agree"""
        model_predictions = {f'Model {i}': 3 for i in range(6)}
        
        fig = create_model_agreement_plot(model_predictions)
        assert fig is not None
        
        plt.close(fig)
    
    def test_model_agreement_all_differ(self, suppress_plots):
        """Test model agreement when all models differ"""
        model_predictions = {f'Model {i}': i+1 for i in range(5)}
        model_predictions['Model 6'] = 1
        
        fig = create_model_agreement_plot(model_predictions)
        assert fig is not None
        
        plt.close(fig)
    
    def test_create_info_plot(self, suppress_plots):
        """Test info plot creation"""
        fig = create_info_plot()
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_all_plots_have_titles(self, sample_probabilities, suppress_plots):
        """Test all plots have titles"""
        # Probability plot
        fig1 = create_stage_probability_plot(sample_probabilities)
        assert fig1.axes[0].get_title() != ''
        plt.close(fig1)
        
        # eGFR gauge
        fig2 = create_egfr_gauge(75.0)
        assert fig2.axes[0].get_title() != ''
        plt.close(fig2)
        
        # Info plot
        fig3 = create_info_plot()
        assert fig3.axes[0].get_title() != ''
        plt.close(fig3)


# ============================================================================
# Integration Tests for Utility Functions
# ============================================================================

@pytest.mark.unit
class TestUtilityIntegration:
    """Integration tests for utility functions working together"""
    
    def test_egfr_to_stage_pipeline(self):
        """Test full pipeline from creatinine to stage"""
        # Calculate eGFR
        egfr = calculate_egfr(serum_creatinine=2.5, age=65)
        assert egfr is not None
        
        # Get stage from eGFR
        stage = get_stage_from_egfr(egfr)
        assert stage in [1, 2, 3, 4, 5]
        
        # Get stage info
        info = get_stage_info(stage)
        assert info['name'] != 'Unknown'
    
    def test_multiple_patients_classification(self, sample_patients_batch):
        """Test classifying multiple patients"""
        for patient in sample_patients_batch:
            egfr = calculate_egfr(
                patient['serum_creatinine'],
                patient['age']
            )
            stage = get_stage_from_egfr(egfr)
            
            # Check stage is reasonable (within +/- 1 of expected)
            assert abs(stage - patient['expected_stage']) <= 1
    
    def test_consistency_across_calls(self):
        """Test that same inputs produce same outputs"""
        creatinine, age = 1.8, 60
        
        # Call multiple times
        egfr1 = calculate_egfr(creatinine, age)
        egfr2 = calculate_egfr(creatinine, age)
        egfr3 = calculate_egfr(creatinine, age)
        
        assert egfr1 == egfr2 == egfr3
        
        stage1 = get_stage_from_egfr(egfr1)
        stage2 = get_stage_from_egfr(egfr2)
        
        assert stage1 == stage2
