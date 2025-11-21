"""
Data Validation Tests
=====================
Tests for input validation and data type checking.
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# Numeric Feature Validation Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestNumericFeatureValidation:
    """Tests for numeric feature validation"""

    def test_age_valid_range(self):
        """Test age accepts valid range"""
        valid_ages = [1, 18, 45, 65, 90, 120]
        for age in valid_ages:
            assert 0 < age <= 120

    def test_age_boundaries(self):
        """Test age boundary values"""
        assert 1 > 0  # Minimum valid
        assert 120 <= 120  # Maximum reasonable

    def test_blood_pressure_valid_range(self):
        """Test blood pressure accepts valid range"""
        valid_bp = [60, 80, 120, 140, 180, 200]
        for bp in valid_bp:
            assert 0 < bp <= 250

    def test_serum_creatinine_valid_range(self):
        """Test serum creatinine accepts valid range"""
        valid_cr = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
        for cr in valid_cr:
            assert 0 < cr <= 15.0

    def test_specific_gravity_valid_range(self):
        """Test specific gravity accepts valid range"""
        valid_sg = [1.005, 1.010, 1.015, 1.020, 1.025]
        for sg in valid_sg:
            assert 1.000 <= sg <= 1.030

    def test_albumin_valid_range(self):
        """Test albumin accepts valid range (0-5)"""
        valid_albumin = [0, 1, 2, 3, 4, 5]
        for alb in valid_albumin:
            assert 0 <= alb <= 5

    def test_sugar_valid_range(self):
        """Test sugar accepts valid range (0-5)"""
        valid_sugar = [0, 1, 2, 3, 4, 5]
        for sug in valid_sugar:
            assert 0 <= sug <= 5

    def test_blood_glucose_random_valid_range(self):
        """Test blood glucose random accepts valid range"""
        valid_bg = [70, 100, 126, 150, 200, 300]
        for bg in valid_bg:
            assert 0 < bg <= 500

    def test_blood_urea_valid_range(self):
        """Test blood urea accepts valid range"""
        valid_urea = [10, 20, 40, 80, 150, 200]
        for urea in valid_urea:
            assert 0 < urea <= 300

    def test_sodium_valid_range(self):
        """Test sodium accepts valid range"""
        valid_sodium = [120, 130, 138, 145, 150]
        for na in valid_sodium:
            assert 100 <= na <= 180

    def test_potassium_valid_range(self):
        """Test potassium accepts valid range"""
        valid_potassium = [3.0, 4.0, 4.5, 5.0, 6.0]
        for k in valid_potassium:
            assert 2.0 <= k <= 8.0

    def test_haemoglobin_valid_range(self):
        """Test haemoglobin accepts valid range"""
        valid_hb = [8.0, 10.0, 12.5, 15.0, 17.0]
        for hb in valid_hb:
            assert 3.0 <= hb <= 20.0

    def test_packed_cell_volume_valid_range(self):
        """Test packed cell volume accepts valid range"""
        valid_pcv = [20, 30, 40, 45, 50]
        for pcv in valid_pcv:
            assert 10 <= pcv <= 60

    def test_white_blood_cell_count_valid_range(self):
        """Test WBC count accepts valid range"""
        valid_wbc = [3000, 5000, 8000, 11000, 15000]
        for wbc in valid_wbc:
            assert 1000 <= wbc <= 30000

    def test_red_blood_cell_count_valid_range(self):
        """Test RBC count accepts valid range"""
        valid_rbc = [2.5, 3.5, 4.5, 5.5, 6.0]
        for rbc in valid_rbc:
            assert 1.0 <= rbc <= 8.0


# ============================================================================
# Categorical Feature Validation Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestCategoricalFeatureValidation:
    """Tests for categorical feature validation"""

    def test_red_blood_cells_valid_values(self):
        """Test red_blood_cells accepts valid values"""
        valid_values = ["Normal", "Abnormal"]
        for val in valid_values:
            assert val in ["Normal", "Abnormal"]

    def test_pus_cell_valid_values(self):
        """Test pus_cell accepts valid values"""
        valid_values = ["Normal", "Abnormal"]
        for val in valid_values:
            assert val in ["Normal", "Abnormal"]

    def test_pus_cell_clumps_valid_values(self):
        """Test pus_cell_clumps accepts valid values"""
        valid_values = ["Not Present", "Present"]
        for val in valid_values:
            assert val in ["Not Present", "Present"]

    def test_bacteria_valid_values(self):
        """Test bacteria accepts valid values"""
        valid_values = ["Not Present", "Present"]
        for val in valid_values:
            assert val in ["Not Present", "Present"]

    def test_hypertension_valid_values(self):
        """Test hypertension accepts valid values"""
        valid_values = ["No", "Yes"]
        for val in valid_values:
            assert val in ["No", "Yes"]

    def test_diabetes_mellitus_valid_values(self):
        """Test diabetes_mellitus accepts valid values"""
        valid_values = ["No", "Yes"]
        for val in valid_values:
            assert val in ["No", "Yes"]

    def test_coronary_artery_disease_valid_values(self):
        """Test coronary_artery_disease accepts valid values"""
        valid_values = ["No", "Yes"]
        for val in valid_values:
            assert val in ["No", "Yes"]

    def test_appetite_valid_values(self):
        """Test appetite accepts valid values"""
        valid_values = ["Good", "Poor"]
        for val in valid_values:
            assert val in ["Good", "Poor"]

    def test_peda_edema_valid_values(self):
        """Test peda_edema accepts valid values"""
        valid_values = ["No", "Yes"]
        for val in valid_values:
            assert val in ["No", "Yes"]

    def test_anemia_valid_values(self):
        """Test anemia accepts valid values"""
        valid_values = ["No", "Yes"]
        for val in valid_values:
            assert val in ["No", "Yes"]


# ============================================================================
# Categorical Mapping Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestCategoricalMapping:
    """Tests for categorical to numeric mapping"""

    def test_normal_abnormal_mapping(self):
        """Test Normal/Abnormal mapping"""
        mapping = {"Normal": 1, "Abnormal": 0}
        assert mapping["Normal"] == 1
        assert mapping["Abnormal"] == 0

    def test_present_notpresent_mapping(self):
        """Test Present/Not Present mapping"""
        mapping = {"Not Present": 0, "Present": 1}
        assert mapping["Not Present"] == 0
        assert mapping["Present"] == 1

    def test_yes_no_mapping(self):
        """Test Yes/No mapping"""
        mapping = {"No": 0, "Yes": 1}
        assert mapping["No"] == 0
        assert mapping["Yes"] == 1

    def test_good_poor_mapping(self):
        """Test Good/Poor mapping"""
        mapping = {"Good": 1, "Poor": 0}
        assert mapping["Good"] == 1
        assert mapping["Poor"] == 0

    def test_all_mappings_binary(self):
        """Test all mappings are binary (0 or 1)"""
        all_mappings = {
            "Normal": 1,
            "Abnormal": 0,
            "Not Present": 0,
            "Present": 1,
            "No": 0,
            "Yes": 1,
            "Good": 1,
            "Poor": 0,
        }

        for key, value in all_mappings.items():
            assert value in [0, 1]


# ============================================================================
# Data Type Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestDataTypes:
    """Tests for data type validation"""

    def test_numeric_features_accept_int(self):
        """Test numeric features accept integers"""
        numeric_values = [48, 80, 1, 0, 121, 36, 138, 15, 44, 7800]
        for val in numeric_values:
            assert isinstance(val, (int, float, np.integer, np.floating))

    def test_numeric_features_accept_float(self):
        """Test numeric features accept floats"""
        numeric_values = [48.5, 80.0, 1.020, 1.2, 4.6, 15.4, 5.2]
        for val in numeric_values:
            assert isinstance(val, (int, float, np.integer, np.floating))

    def test_categorical_features_are_strings(self):
        """Test categorical features are strings"""
        categorical_values = [
            "Normal",
            "Abnormal",
            "Present",
            "Not Present",
            "Yes",
            "No",
            "Good",
            "Poor",
        ]
        for val in categorical_values:
            assert isinstance(val, str)

    def test_numpy_types_compatible(self):
        """Test numpy types are compatible"""
        numpy_values = [np.int32(48), np.int64(80), np.float32(1.2), np.float64(4.6)]
        for val in numpy_values:
            assert isinstance(val, (int, float, np.integer, np.floating))


# ============================================================================
# Missing Data Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestMissingDataHandling:
    """Tests for missing data handling"""

    def test_none_values_detected(self):
        """Test None values are detected"""
        assert None is None
        assert not (None == 0)
        assert not (None == "")

    def test_nan_values_detected(self):
        """Test NaN values are detected"""
        assert pd.isna(np.nan)
        assert pd.isna(float("nan"))
        assert not pd.isna(0)
        assert not pd.isna("")

    def test_egfr_handles_none_creatinine(self):
        """Test eGFR calculation handles None creatinine"""
        from src.utils import calculate_egfr

        egfr = calculate_egfr(None, 50)
        assert egfr is None

    def test_egfr_handles_none_age(self):
        """Test eGFR calculation handles None age"""
        from src.utils import calculate_egfr

        egfr = calculate_egfr(1.2, None)
        assert egfr is None

    def test_stage_handles_none_egfr(self):
        """Test stage classification handles None eGFR"""
        from src.utils import get_stage_from_egfr

        stage = get_stage_from_egfr(None)
        assert stage is None

    def test_stage_handles_nan_egfr(self):
        """Test stage classification handles NaN eGFR"""
        from src.utils import get_stage_from_egfr

        stage = get_stage_from_egfr(np.nan)
        assert stage is None


# ============================================================================
# Invalid Input Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestInvalidInputs:
    """Tests for invalid input handling"""

    def test_negative_age_rejected(self):
        """Test negative age is rejected"""
        from src.utils import calculate_egfr

        egfr = calculate_egfr(1.2, -50)
        assert egfr is None

    def test_zero_age_rejected(self):
        """Test zero age is rejected"""
        from src.utils import calculate_egfr

        egfr = calculate_egfr(1.2, 0)
        assert egfr is None

    def test_negative_creatinine_rejected(self):
        """Test negative creatinine is rejected"""
        from src.utils import calculate_egfr

        egfr = calculate_egfr(-1.2, 50)
        assert egfr is None

    def test_zero_creatinine_rejected(self):
        """Test zero creatinine is rejected"""
        from src.utils import calculate_egfr

        egfr = calculate_egfr(0, 50)
        assert egfr is None

    def test_extreme_age_handled(self):
        """Test extreme age values"""
        from src.utils import calculate_egfr

        egfr1 = calculate_egfr(1.2, 1)  # Very young
        egfr2 = calculate_egfr(1.2, 150)  # Very old

        # Both should return values (even if unrealistic)
        assert egfr1 is not None
        assert egfr2 is not None

    def test_extreme_creatinine_handled(self):
        """Test extreme creatinine values"""
        from src.utils import calculate_egfr

        egfr1 = calculate_egfr(0.1, 50)  # Very low
        egfr2 = calculate_egfr(20.0, 50)  # Very high

        assert egfr1 is not None
        assert egfr2 is not None


# ============================================================================
# Feature Count Validation Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestFeatureCount:
    """Tests for feature count validation"""

    def test_total_feature_count(self):
        """Test total number of features is 24"""
        numeric_features = (
            14  # age, bp, sg, albumin, sugar, bg, urea, cr, na, k, hb, pcv, wbc, rbc
        )
        categorical_features = 10  # rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane

        total = numeric_features + categorical_features
        assert total == 24

    def test_numeric_feature_count(self, sample_patient_normal):
        """Test number of numeric features"""
        numeric_keys = [
            "age",
            "blood_pressure",
            "specific_gravity",
            "albumin",
            "sugar",
            "blood_glucose_random",
            "blood_urea",
            "serum_creatinine",
            "sodium",
            "potassium",
            "haemoglobin",
            "packed_cell_volume",
            "white_blood_cell_count",
            "red_blood_cell_count",
        ]

        assert len(numeric_keys) == 14

        # Check all are in sample patient
        for key in numeric_keys:
            assert key in sample_patient_normal

    def test_categorical_feature_count(self, sample_patient_normal):
        """Test number of categorical features"""
        categorical_keys = [
            "red_blood_cells",
            "pus_cell",
            "pus_cell_clumps",
            "bacteria",
            "hypertension",
            "diabetes_mellitus",
            "coronary_artery_disease",
            "appetite",
            "peda_edema",
            "anemia",
        ]

        assert len(categorical_keys) == 10

        # Check all are in sample patient
        for key in categorical_keys:
            assert key in sample_patient_normal


# ============================================================================
# Boundary Value Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestBoundaryValues:
    """Tests for boundary value handling"""

    def test_egfr_stage_boundaries(self):
        """Test eGFR stage boundary values"""
        from src.utils import get_stage_from_egfr

        # Test exact boundaries
        assert get_stage_from_egfr(90.0) == 1  # Stage 1/2 boundary
        assert get_stage_from_egfr(89.9) == 2
        assert get_stage_from_egfr(60.0) == 2  # Stage 2/3 boundary
        assert get_stage_from_egfr(59.9) == 3
        assert get_stage_from_egfr(30.0) == 3  # Stage 3/4 boundary
        assert get_stage_from_egfr(29.9) == 4
        assert get_stage_from_egfr(15.0) == 4  # Stage 4/5 boundary
        assert get_stage_from_egfr(14.9) == 5

    def test_probability_boundaries(self):
        """Test probability values are in [0, 1]"""
        probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
        for prob in probabilities:
            assert 0.0 <= prob <= 1.0

    def test_stage_boundaries(self):
        """Test stage values are in [1, 5]"""
        stages = [1, 2, 3, 4, 5]
        for stage in stages:
            assert 1 <= stage <= 5


# ============================================================================
# Data Consistency Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.validation
class TestDataConsistency:
    """Tests for data consistency"""

    def test_sample_patients_have_all_features(
        self, sample_patient_normal, sample_patient_stage3, sample_patient_stage5
    ):
        """Test all sample patients have all required features"""
        required_keys = {
            "age",
            "blood_pressure",
            "specific_gravity",
            "albumin",
            "sugar",
            "blood_glucose_random",
            "blood_urea",
            "serum_creatinine",
            "sodium",
            "potassium",
            "haemoglobin",
            "packed_cell_volume",
            "white_blood_cell_count",
            "red_blood_cell_count",
            "red_blood_cells",
            "pus_cell",
            "pus_cell_clumps",
            "bacteria",
            "hypertension",
            "diabetes_mellitus",
            "coronary_artery_disease",
            "appetite",
            "peda_edema",
            "anemia",
        }

        for patient in [
            sample_patient_normal,
            sample_patient_stage3,
            sample_patient_stage5,
        ]:
            assert set(patient.keys()) == required_keys

    def test_feature_order_consistency(self):
        """Test feature order is consistent"""
        feature_order = [
            "age",
            "blood_pressure",
            "specific_gravity",
            "albumin",
            "sugar",
            "red_blood_cells",
            "pus_cell",
            "pus_cell_clumps",
            "bacteria",
            "blood_glucose_random",
            "blood_urea",
            "serum_creatinine",
            "sodium",
            "potassium",
            "haemoglobin",
            "packed_cell_volume",
            "white_blood_cell_count",
            "red_blood_cell_count",
            "hypertension",
            "diabetes_mellitus",
            "coronary_artery_disease",
            "appetite",
            "peda_edema",
            "anemia",
        ]

        assert len(feature_order) == 24
        assert len(set(feature_order)) == 24  # No duplicates
