"""
CKD Stage Prediction App - Gradio Interface
============================================
Interactive web application for Chronic Kidney Disease stage prediction (1-5).
Uses trained ML models with PCA preprocessing.

Features:
- 5-stage CKD classification (Stage 1-5)
- eGFR estimation
- Interactive visualizations
- Multi-model comparison

Usage:
    python src/app.py
"""

import gradio as gr
import joblib
import pickle
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Load Models and Configuration
# ============================================================================

def load_models():
    """Load all trained models and preprocessing pipeline"""
    try:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'models')
        
        pca_pipeline = joblib.load(os.path.join(models_dir, 'pca_pipeline.pkl'))
        best_model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        
        with open(os.path.join(models_dir, 'feature_info.pkl'), 'rb') as f:
            feature_info = pickle.load(f)
        
        # Load all 6 models
        all_models = {
            'Logistic Regression': joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl')),
            'Random Forest': joblib.load(os.path.join(models_dir, 'random_forest_model.pkl')),
            'Gradient Boosting': joblib.load(os.path.join(models_dir, 'gradient_boosting_model.pkl')),
            'SVM': joblib.load(os.path.join(models_dir, 'svm_model.pkl')),
            'Naive Bayes': joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl')),
            'K-Nearest Neighbors': joblib.load(os.path.join(models_dir, 'k-nearest_neighbors_model.pkl'))
        }
        
        print(f"✅ Models loaded successfully from {models_dir}")
        print(f"✅ Best model: {feature_info['best_model_name']}")
        print(f"✅ All 6 models loaded for 5-stage prediction")
        return pca_pipeline, best_model, feature_info, all_models
    
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find model files in '{models_dir}' directory")
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
    Predict CKD stage (1-5) based on patient data using ALL models
    
    Returns:
        tuple: (result_html, prob_plot, egfr_plot, comparison_html, agreement_plot, confidence_label)
        - Best model prediction with eGFR
        - Stage probability plot
        - eGFR gauge visualization
        - All models comparison table
        - Model agreement heatmap
        - Confidence breakdown
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
    
    # Calculate eGFR
    egfr = calculate_egfr(serum_creatinine, age)
    egfr_stage = get_stage_from_egfr(egfr) if egfr else None
    
    # Create DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure all required features are present (in correct order)
    for feature in feature_info['feature_names']:
        if feature not in patient_df.columns:
            patient_df[feature] = np.nan
    
    patient_df = patient_df[feature_info['feature_names']]
    
    # Apply PCA pipeline (includes preprocessing and imputation)
    patient_processed = pca_pipeline.transform(patient_df)
    
    # Get predictions from best model
    predicted_stage = int(best_model.predict(patient_processed)[0])
    probabilities = best_model.predict_proba(patient_processed)[0]
    
    # Get stage info
    stage_info = get_stage_info(predicted_stage)
    desc = stage_info['description']
    color = stage_info['color']
    
    # Stage descriptions and severity
    stage_descriptions = {
        1: ("Normal/High kidney function", "eGFR ≥ 90", "🟢 LOW RISK", "#28a745"),
        2: ("Mild reduction in function", "eGFR 60-89", "🟡 LOW RISK", "#90ee90"),
        3: ("Moderate reduction", "eGFR 30-59", "🟠 MODERATE RISK", "#ffc107"),
        4: ("Severe reduction", "eGFR 15-29", "🔴 HIGH RISK", "#ff8c00"),
        5: ("Kidney failure", "eGFR < 15", "🚨 CRITICAL RISK", "#dc3545")
    }
    
    desc, egfr_range, risk_level, color = stage_descriptions[predicted_stage]
    confidence = probabilities[predicted_stage - 1] * 100
    
    # Create main result HTML
    result_html = f"""
    <div style="padding: 25px; border-radius: 15px; background: linear-gradient(135deg, {color}22, {color}44); border: 3px solid {color};">
        <h1 style="text-align: center; color: {color}; margin-bottom: 10px;">CKD STAGE {predicted_stage}</h1>
        <h3 style="text-align: center; color: #555; margin-top: 0;">{desc}</h3>
        <p style="text-align: center; font-size: 16px; color: #666;">Best Model: {feature_info['best_model_name']} ⭐</p>
        <hr style="border: 1px solid {color};">
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="text-align: center;">
                <h2 style="margin: 5px 0;">{risk_level}</h2>
                <p style="font-size: 14px; color: #666;">Risk Assessment</p>
            </div>
            <div style="text-align: center;">
                <h2 style="margin: 5px 0;">{confidence:.1f}%</h2>
                <p style="font-size: 14px; color: #666;">Prediction Confidence</p>
            </div>
        </div>
        
        <hr style="border: 1px solid {color};">
        
        <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 15px;">
            <h4 style="margin-top: 0;">📊 Clinical Information:</h4>
            <p><b>Expected eGFR Range:</b> {egfr_range} mL/min/1.73m²</p>
            {f'<p><b>Calculated eGFR:</b> {egfr:.1f} mL/min/1.73m² (Formula-based Stage: {egfr_stage})</p>' if egfr else '<p><b>eGFR:</b> Unable to calculate (missing age or creatinine)</p>'}
            <p><b>Stage Description:</b> {desc}</p>
        </div>
        
        <hr style="border: 1px solid {color};">
        
        <p style="font-size: 13px; color: #666; margin-bottom: 0; text-align: center;">
            ⚠️ <b>Medical Disclaimer:</b> This is a screening tool only. Consult healthcare professionals for diagnosis.
        </p>
    </div>
    """
    
    # Get predictions from ALL models for comparison
    model_predictions = {}
    model_results = []
    
    for model_name, model in all_models.items():
        prediction = int(model.predict(patient_processed)[0])
        model_predictions[model_name] = prediction
        probabilities_model = model.predict_proba(patient_processed)[0]
        
        is_best = "⭐" if model_name == feature_info['best_model_name'] else ""
        
        model_results.append({
            'model': model_name,
            'prediction': prediction,
            'confidence': probabilities_model[prediction - 1] * 100,
            'is_best': is_best
        })
    
    # Create comparison table HTML
    comparison_html = """
    <div style="padding: 20px;">
        <h3>📊 All Models Comparison (6 Models)</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <thead>
                <tr style="background-color: #e0e0e0; text-align: left;">
                    <th style="padding: 12px; border: 1px solid #ccc;">Model</th>
                    <th style="padding: 12px; border: 1px solid #ccc;">Predicted Stage</th>
                    <th style="padding: 12px; border: 1px solid #ccc;">Confidence</th>
                    <th style="padding: 12px; border: 1px solid #ccc;">Assessment</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for result in model_results:
        row_style = "background-color: #fffacd;" if result['is_best'] else ""
        stage = result['prediction']
        _, _, risk, color = stage_descriptions[stage]
        
        comparison_html += f"""
            <tr style="{row_style}">
                <td style="padding: 12px; border: 1px solid #ccc;"><b>{result['model']}</b> {result['is_best']}</td>
                <td style="padding: 12px; border: 1px solid #ccc; text-align: center;"><span style="background: {color}; padding: 5px 15px; border-radius: 5px; color: white; font-weight: bold;">Stage {stage}</span></td>
                <td style="padding: 12px; border: 1px solid #ccc; text-align: center;"><b>{result['confidence']:.1f}%</b></td>
                <td style="padding: 12px; border: 1px solid #ccc;">{risk}</td>
            </tr>
        """
    
    comparison_html += """
            </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 15px;">⭐ = Best performing model on test set</p>
    </div>
    """
    
    # Create visualizations
    prob_plot = create_stage_probability_plot({i: p for i, p in enumerate(probabilities, 1)})
    egfr_plot = create_egfr_gauge(egfr, predicted_stage) if egfr else create_info_plot()
    agreement_plot = create_model_agreement_plot(model_predictions)
    
    # Create confidence label data
    confidence_label = {
        f"Stage {i}": float(prob) 
        for i, prob in enumerate(probabilities, 1)
    }
    
    return result_html, prob_plot, egfr_plot, comparison_html, agreement_plot, confidence_label

# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="CKD Stage Predictor", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🏥 Chronic Kidney Disease (CKD) Stage Predictor
            
            ### Multi-Stage Classification (Stages 1-5) with eGFR Estimation
            
            This advanced ML system predicts CKD stages using **6 trained models** with PCA dimensionality reduction.
            
            **📋 Instructions:** Fill in all patient measurements and clinical indicators below.
            """
        )
        
        # Static informational plot
        with gr.Row():
            info_plot = gr.Plot(label="📚 CKD Stages Reference Guide", value=create_info_plot())
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Numeric Measurements")
                
                age = gr.Number(label="Age (years)", value=48, info="Patient's age in years")
                blood_pressure = gr.Number(label="Blood Pressure (mm/Hg)", value=80, info="Diastolic blood pressure")
                specific_gravity = gr.Number(label="Specific Gravity", value=1.020, info="Urine specific gravity (1.005 - 1.025)")
                albumin = gr.Number(label="Albumin", value=0, info="Albumin level (0-5)")
                sugar = gr.Number(label="Sugar", value=0, info="Sugar level (0-5)")
                blood_glucose_random = gr.Number(label="Blood Glucose Random (mgs/dl)", value=121, info="Random blood glucose")
                blood_urea = gr.Number(label="Blood Urea (mgs/dl)", value=36, info="Blood urea level")
                serum_creatinine = gr.Number(label="Serum Creatinine (mgs/dl)", value=1.2, info="⚠️ Required for eGFR calculation")
                sodium = gr.Number(label="Sodium (mEq/L)", value=138, info="Sodium level")
                potassium = gr.Number(label="Potassium (mEq/L)", value=4.5, info="Potassium level")
                haemoglobin = gr.Number(label="Haemoglobin (gms)", value=15.4, info="Hemoglobin level")
                packed_cell_volume = gr.Number(label="Packed Cell Volume", value=44, info="PCV percentage")
                white_blood_cell_count = gr.Number(label="White Blood Cell Count (cells/cumm)", value=7800, info="WBC count")
                red_blood_cell_count = gr.Number(label="Red Blood Cell Count (millions/cmm)", value=5.2, info="RBC count")
            
            with gr.Column():
                gr.Markdown("### 🔬 Clinical Indicators")
                
                red_blood_cells = gr.Radio(
                    choices=["Normal", "Abnormal"],
                    value="Normal",
                    label="Red Blood Cells",
                    info="Red blood cell condition"
                )
                
                pus_cell = gr.Radio(
                    choices=["Normal", "Abnormal"],
                    value="Normal",
                    label="Pus Cell",
                    info="Pus cell condition"
                )
                
                pus_cell_clumps = gr.Radio(
                    choices=["Not Present", "Present"],
                    value="Not Present",
                    label="Pus Cell Clumps",
                    info="Presence of pus cell clumps"
                )
                
                bacteria = gr.Radio(
                    choices=["Not Present", "Present"],
                    value="Not Present",
                    label="Bacteria",
                    info="Presence of bacteria"
                )
                
                gr.Markdown("### 🏥 Medical History")
                
                hypertension = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Hypertension",
                    info="History of high blood pressure"
                )
                
                diabetes_mellitus = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Diabetes Mellitus",
                    info="History of diabetes"
                )
                
                coronary_artery_disease = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Coronary Artery Disease",
                    info="History of CAD"
                )
                
                appetite = gr.Radio(
                    choices=["Good", "Poor"],
                    value="Good",
                    label="Appetite",
                    info="Patient's appetite condition"
                )
                
                peda_edema = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Pedal Edema",
                    info="Swelling in feet/ankles"
                )
                
                anemia = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Anemia",
                    info="History of anemia"
                )
        
        # Prediction button
        predict_btn = gr.Button("🔍 Predict CKD Stage", variant="primary", size="lg")
        
        # Output section
        gr.Markdown("## 📋 Prediction Results")
        
        # Main result
        result_output = gr.HTML(label="🎯 Primary Diagnosis (Best Model)")
        
        # Visualizations row
        with gr.Row():
            with gr.Column():
                prob_plot_output = gr.Plot(label="📊 Stage Probability Distribution")
            with gr.Column():
                confidence_output = gr.Label(label="📈 Confidence Breakdown", num_top_classes=5)
        
        # eGFR Gauge
        egfr_plot_output = gr.Plot(label="🩺 Kidney Function Assessment (eGFR)")
        
        # All models comparison
        comparison_output = gr.HTML(label="📊 Multi-Model Comparison (6 Models)")
        
        # Model agreement
        agreement_plot_output = gr.Plot(label="🤝 Model Predictions Agreement")
        
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
            outputs=[
                result_output, 
                prob_plot_output, 
                egfr_plot_output, 
                comparison_output, 
                agreement_plot_output, 
                confidence_output
            ],
            api_name="predict"
        )
        
        # Examples section
        gr.Markdown("### 💡 Example Cases")
        gr.Examples(
            examples=[
                [
                    # Stage 1-2: Low risk patient
                    48, 80, 1.020, 0, 0, 121, 36, 1.2, 138, 4.5, 15.4, 44, 7800, 5.2,
                    "Normal", "Normal", "Not Present", "Not Present",
                    "No", "No", "No", "Good", "No", "No"
                ],
                [
                    # Stage 3: Moderate risk
                    55, 85, 1.015, 2, 1, 140, 52, 2.1, 135, 4.8, 12.5, 38, 9200, 4.5,
                    "Normal", "Abnormal", "Not Present", "Not Present",
                    "Yes", "No", "No", "Good", "No", "No"
                ],
                [
                    # Stage 4-5: High risk patient
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
            ### 📖 About This System
            
            **Models Used:** Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes, K-Nearest Neighbors
            
            **Feature Engineering:** 24 features → 20 principal components (95% variance retained)
            
            **eGFR Formula:** MDRD equation: `eGFR = 175 × (SerumCreatinine)^-1.154 × (Age)^-0.203`
            
            ---
            **⚠️ Medical Disclaimer:** This tool is for educational and screening purposes only. 
            It should NOT replace professional medical advice, diagnosis, or treatment.
            Always seek the advice of qualified health providers with any questions regarding medical conditions.
            """
        )
    
    return demo

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🏥 CKD Stage Predictor - Gradio Interface")
    print("=" * 70)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project root: {project_root}")
    print(f"Models directory: {os.path.join(project_root, 'models')}")
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
        inbrowser=True  # Automatically open browser
    )
