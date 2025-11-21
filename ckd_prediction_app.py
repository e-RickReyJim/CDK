"""
CKD Prediction App - Gradio Interface
======================================
Interactive web application for Chronic Kidney Disease risk prediction.
Uses trained ML models with PCA preprocessing.

Usage:
    pip install gradio
    python ckd_prediction_app.py
"""

import gradio as gr
import joblib
import pickle
import pandas as pd
import numpy as np
import os

# ============================================================================
# Load Models and Configuration
# ============================================================================

def load_models():
    """Load all trained models and preprocessing pipeline"""
    try:
        pca_pipeline = joblib.load('models/pca_pipeline.pkl')
        best_model = joblib.load('models/best_model.pkl')
        
        with open('models/feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        # Load all 6 models
        all_models = {
            'Logistic Regression': joblib.load('models/logistic_regression_model.pkl'),
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'Gradient Boosting': joblib.load('models/gradient_boosting_model.pkl'),
            'SVM': joblib.load('models/svm_model.pkl'),
            'Naive Bayes': joblib.load('models/naive_bayes_model.pkl'),
            'K-Nearest Neighbors': joblib.load('models/k-nearest_neighbors_model.pkl')
        }
        
        print(f"✅ Models loaded successfully!")
        print(f"✅ Best model: {feature_info['best_model_name']}")
        print(f"✅ All 6 models loaded for comparison")
        return pca_pipeline, best_model, feature_info, all_models
    
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find model files in 'models/' directory")
        print(f"   Please run the training notebook first to generate the models")
        raise e

# Load models at startup
pca_pipeline, best_model, feature_info, all_models = load_models()

# ============================================================================
# Prediction Function
# ============================================================================

def predict_ckd(
    # Numeric features
    age, blood_pressure, specific_gravity, albumin, sugar,
    blood_glucose_random, blood_urea, serum_creatinine,
    sodium, potassium, haemoglobin, packed_cell_volume,
    white_blood_cell_count, red_blood_cell_count,
    # Binary features (categorical)
    red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
    hypertension, diabetes_mellitus, coronary_artery_disease,
    appetite, peda_edema, anemia
):
    """
    Predict CKD risk based on patient data using ALL models
    
    Returns:
        - Best model prediction
        - All models comparison table
        - Probability distribution chart
    """
    
    # Map categorical inputs to numeric values (0/1)
    categorical_mapping = {
        # For normal/abnormal
        'Normal': 1, 'Abnormal': 0,
        # For present/not present
        'Not Present': 0, 'Present': 1,
        # For yes/no
        'No': 0, 'Yes': 1,
        # For appetite
        'Good': 1, 'Poor': 0
    }
    
    # Create patient data dictionary
    patient_data = {
        'age': age,
        'blood_pressure': blood_pressure,
        'specific_gravity': specific_gravity,
        'albumin': albumin,
        'sugar': sugar,
        'red_blood_cells': categorical_mapping[red_blood_cells],
        'pus_cell': categorical_mapping[pus_cell],
        'pus_cell_clumps': categorical_mapping[pus_cell_clumps],
        'bacteria': categorical_mapping[bacteria],
        'blood_glucose_random': blood_glucose_random,
        'blood_urea': blood_urea,
        'serum_creatinine': serum_creatinine,
        'sodium': sodium,
        'potassium': potassium,
        'haemoglobin': haemoglobin,
        'packed_cell_volume': packed_cell_volume,
        'white_blood_cell_count': white_blood_cell_count,
        'red_blood_cell_count': red_blood_cell_count,
        'hypertension': categorical_mapping[hypertension],
        'diabetes_mellitus': categorical_mapping[diabetes_mellitus],
        'coronary_artery_disease': categorical_mapping[coronary_artery_disease],
        'appetite': categorical_mapping[appetite],
        'peda_edema': categorical_mapping[peda_edema],
        'anemia': categorical_mapping[anemia]
    }
    
    # Create DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure all required features are present (in correct order)
    for feature in feature_info['feature_names']:
        if feature not in patient_df.columns:
            patient_df[feature] = np.nan
    
    patient_df = patient_df[feature_info['feature_names']]
    
    # Apply PCA pipeline (includes preprocessing and imputation)
    patient_processed = pca_pipeline.transform(patient_df)
    
    # Get predictions from ALL models
    model_results = []
    for model_name, model in all_models.items():
        prediction = model.predict(patient_processed)[0]
        probabilities = model.predict_proba(patient_processed)[0]
        
        prob_no_ckd = probabilities[0]
        prob_ckd = probabilities[1]
        
        # Determine risk level
        if prob_ckd > 0.7:
            risk_level = "HIGH"
            risk_emoji = "🔴"
        elif prob_ckd > 0.4:
            risk_level = "MODERATE"
            risk_emoji = "🟡"
        else:
            risk_level = "LOW"
            risk_emoji = "🟢"
        
        is_best = "⭐" if model_name == feature_info['best_model_name'] else ""
        
        model_results.append({
            'model': model_name,
            'prediction': 'CKD' if prediction == 1 else 'No CKD',
            'prob_ckd': prob_ckd,
            'prob_no_ckd': prob_no_ckd,
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'is_best': is_best
        })
    
    # Best model prediction (main display)
    best_result = next(r for r in model_results if r['is_best'] == "⭐")
    
    result_html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
        <h2 style="text-align: center;">{'**CKD DETECTED**' if best_result['prediction'] == 'CKD' else '**NO CKD DETECTED**'}</h2>
        <p style="text-align: center; font-size: 14px; color: #666;">Best Model: {feature_info['best_model_name']} ⭐</p>
        <hr>
        <h3>{best_result['risk_emoji']} Risk Level: {best_result['risk_level']} RISK</h3>
        <p style="font-size: 18px;">
            <b>CKD Probability:</b> {best_result['prob_ckd']*100:.1f}%<br>
            <b>No CKD Probability:</b> {best_result['prob_no_ckd']*100:.1f}%
        </p>
        <hr>
        <p style="font-size: 14px; color: #666;">
            <b>Note:</b> This is a predictive model for screening purposes only. 
            Always consult with healthcare professionals for proper diagnosis.
        </p>
    </div>
    """
    
    # Create comparison table
    comparison_html = """
    <div style="padding: 20px;">
        <h3>📊 All Models Comparison</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <thead>
                <tr style="background-color: #e0e0e0; text-align: left;">
                    <th style="padding: 10px; border: 1px solid #ccc;">Model</th>
                    <th style="padding: 10px; border: 1px solid #ccc;">Prediction</th>
                    <th style="padding: 10px; border: 1px solid #ccc;">CKD Prob.</th>
                    <th style="padding: 10px; border: 1px solid #ccc;">Risk Level</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for result in model_results:
        row_style = "background-color: #fffacd;" if result['is_best'] else ""
        comparison_html += f"""
            <tr style="{row_style}">
                <td style="padding: 10px; border: 1px solid #ccc;"><b>{result['model']}</b> {result['is_best']}</td>
                <td style="padding: 10px; border: 1px solid #ccc;">{result['prediction']}</td>
                <td style="padding: 10px; border: 1px solid #ccc;"><b>{result['prob_ckd']*100:.1f}%</b></td>
                <td style="padding: 10px; border: 1px solid #ccc;">{result['risk_emoji']} {result['risk_level']}</td>
            </tr>
        """
    
    comparison_html += """
            </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 10px;">⭐ = Best performing model on test set</p>
    </div>
    """
    
    # Create probability chart data for best model
    probability_display = {
        "No CKD": best_result['prob_no_ckd'],
        "CKD": best_result['prob_ckd']
    }
    
    return result_html, comparison_html, probability_display

# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="CKD Risk Predictor", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🏥 Chronic Kidney Disease (CKD) Risk Predictor
            
            Enter patient data below to predict CKD risk using machine learning models trained with PCA.
            
            **Instructions:** Fill in all fields with patient measurements and clinical indicators.
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Numeric Measurements")
                
                age = gr.Number(label="Age (years)", info="Patient's age in years")
                blood_pressure = gr.Number(label="Blood Pressure (mm/Hg)", info="Diastolic blood pressure")
                specific_gravity = gr.Number(label="Specific Gravity", info="Urine specific gravity (1.005 - 1.025)")
                albumin = gr.Number(label="Albumin", info="Albumin level (0-5)")
                sugar = gr.Number(label="Sugar", info="Sugar level (0-5)")
                blood_glucose_random = gr.Number(label="Blood Glucose Random (mgs/dl)", info="Random blood glucose")
                blood_urea = gr.Number(label="Blood Urea (mgs/dl)", info="Blood urea level")
                serum_creatinine = gr.Number(label="Serum Creatinine (mgs/dl)", info="Serum creatinine level")
                sodium = gr.Number(label="Sodium (mEq/L)", info="Sodium level")
                potassium = gr.Number(label="Potassium (mEq/L)", info="Potassium level")
                haemoglobin = gr.Number(label="Haemoglobin (gms)", info="Hemoglobin level")
                packed_cell_volume = gr.Number(label="Packed Cell Volume", info="PCV percentage")
                white_blood_cell_count = gr.Number(label="White Blood Cell Count (cells/cumm)", info="WBC count")
                red_blood_cell_count = gr.Number(label="Red Blood Cell Count (millions/cmm)", info="RBC count")
            
            with gr.Column():
                gr.Markdown("### 🔬 Clinical Indicators")
                
                red_blood_cells = gr.Radio(
                    choices=["Normal", "Abnormal"],
                    label="Red Blood Cells",
                    info="Red blood cell condition"
                )
                
                pus_cell = gr.Radio(
                    choices=["Normal", "Abnormal"],
                    label="Pus Cell",
                    info="Pus cell condition"
                )
                
                pus_cell_clumps = gr.Radio(
                    choices=["Not Present", "Present"],
                    label="Pus Cell Clumps",
                    info="Presence of pus cell clumps"
                )
                
                bacteria = gr.Radio(
                    choices=["Not Present", "Present"],
                    label="Bacteria",
                    info="Presence of bacteria"
                )
                
                gr.Markdown("### 🏥 Medical History")
                
                hypertension = gr.Radio(
                    choices=["No", "Yes"],
                    label="Hypertension",
                    info="History of high blood pressure"
                )
                
                diabetes_mellitus = gr.Radio(
                    choices=["No", "Yes"],
                    label="Diabetes Mellitus",
                    info="History of diabetes"
                )
                
                coronary_artery_disease = gr.Radio(
                    choices=["No", "Yes"],
                    label="Coronary Artery Disease",
                    info="History of CAD"
                )
                
                appetite = gr.Radio(
                    choices=["Good", "Poor"],
                    label="Appetite",
                    info="Patient's appetite condition"
                )
                
                peda_edema = gr.Radio(
                    choices=["No", "Yes"],
                    label="Pedal Edema",
                    info="Swelling in feet/ankles"
                )
                
                anemia = gr.Radio(
                    choices=["No", "Yes"],
                    label="Anemia",
                    info="History of anemia"
                )
        
        # Prediction button
        predict_btn = gr.Button("🔍 Predict CKD Risk", variant="primary", size="lg")
        
        # Output section
        gr.Markdown("## 📋 Prediction Results")
        
        with gr.Row():
            with gr.Column(scale=2):
                result_output = gr.HTML(label="Best Model Diagnosis")
            with gr.Column(scale=1):
                probability_output = gr.Label(label="Best Model Probability", num_top_classes=2)
        
        # All models comparison
        comparison_output = gr.HTML(label="All Models Comparison")
        
        # Connect button to prediction function
        predict_btn.click(
            fn=predict_ckd,
            inputs=[
                age, blood_pressure, specific_gravity, albumin, sugar,
                blood_glucose_random, blood_urea, serum_creatinine,
                sodium, potassium, haemoglobin, packed_cell_volume,
                white_blood_cell_count, red_blood_cell_count,
                red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                hypertension, diabetes_mellitus, coronary_artery_disease,
                appetite, peda_edema, anemia
            ],
            outputs=[result_output, comparison_output, probability_output],
            api_name="predict"
        )
        
        # Examples section
        gr.Markdown("### 💡 Example Cases")
        gr.Examples(
            examples=[
                [
                    # Low risk patient
                    48, 80, 1.020, 0, 0, 121, 36, 1.2, 138, 4.5, 15.4, 44, 7800, 5.2,
                    "Normal", "Normal", "Not Present", "Not Present",
                    "No", "No", "No", "Good", "No", "No"
                ],
                [
                    # High risk patient
                    62, 90, 1.010, 4, 3, 157, 90, 4.1, 130, 5.8, 9.8, 28, 12000, 3.8,
                    "Abnormal", "Abnormal", "Present", "Present",
                    "Yes", "Yes", "No", "Poor", "Yes", "Yes"
                ],
            ],
            inputs=[
                age, blood_pressure, specific_gravity, albumin, sugar,
                blood_glucose_random, blood_urea, serum_creatinine,
                sodium, potassium, haemoglobin, packed_cell_volume,
                white_blood_cell_count, red_blood_cell_count,
                red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                hypertension, diabetes_mellitus, coronary_artery_disease,
                appetite, peda_edema, anemia
            ],
            label="Click to load example patient data"
        )
        
        gr.Markdown(
            """
            ---
            **Disclaimer:** This tool is for educational and screening purposes only. 
            It should not replace professional medical advice, diagnosis, or treatment.
            Always seek the advice of qualified health providers with any questions regarding medical conditions.
            """
        )
    
    return demo

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🏥 CKD Risk Predictor - Gradio Interface")
    print("=" * 70)
    print(f"Models directory: {os.path.abspath('models')}")
    print(f"Best model: {feature_info['best_model_name']}")
    print("=" * 70)
    
    # Create and launch the interface
    demo = create_interface()
    
    # Launch without queuing to avoid async issues
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7870,
        show_error=True,
        #inbrowser=True  # Automatically open browser
    )
