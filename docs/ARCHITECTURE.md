# Solution Flow - Visual Summary

## 🔄 Complete CKD Predictor Solution Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CKD STAGE PREDICTOR SYSTEM                      │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐         ┌──────────────────────────────┐
│   TRAINING PIPELINE      │         │    PREDICTION PIPELINE        │
│  (Jupyter Notebook)      │         │    (Gradio Web App)           │
└──────────────────────────┘         └──────────────────────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │ Load Data   │                      │ Load Models  │
    │ (400 rows)  │                      │ (9 .pkl)     │
    └─────────────┘                      └──────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │   Clean &   │                      │ User Input   │
    │   Process   │                      │ (24 features)│
    └─────────────┘                      └──────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │ Calculate   │                      │ Calculate    │
    │ CKD Stages  │                      │ eGFR         │
    └─────────────┘                      └──────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │ PCA         │                      │ PCA          │
    │ Transform   │                      │ Transform    │
    │ (24→20)     │                      │ (24→20)      │
    └─────────────┘                      └──────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │ Train 6     │                      │ Predict with │
    │ Models      │                      │ 6 Models     │
    └─────────────┘                      └──────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │ Evaluate &  │                      │ Generate     │
    │ Select Best │                      │ Visuals      │
    └─────────────┘                      └──────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌──────────────┐
    │ Export      │──────────────────────▶│ Display      │
    │ Models      │      (models/)        │ Results      │
    └─────────────┘                      └──────────────┘
```

---

## 🎯 Key Components

### 1. Data Processing
```
Raw Data → Missing Value Handling → Encoding → eGFR Calculation → Stage Assignment
```

### 2. Model Training
```
Train Data → PCA Pipeline → 6 Classifiers → Cross-Validation → Best Model Selection
```

### 3. Prediction Flow
```
User Input → Validation → PCA Transform → Model Prediction → Visualization → Display
```

### 4. Six Models Ensemble
```
┌───────────────────┐
│ Logistic Reg      │ → Prediction 1
├───────────────────┤
│ Random Forest ⭐  │ → Prediction 2 (BEST: 98.5%)
├───────────────────┤
│ Gradient Boost    │ → Prediction 3
├───────────────────┤
│ SVM               │ → Prediction 4
├───────────────────┤
│ Naive Bayes       │ → Prediction 5
├───────────────────┤
│ K-Nearest Neighbors│ → Prediction 6
└───────────────────┘
         │
         ▼
    ┌─────────┐
    │ Ensemble│
    │ View    │
    └─────────┘
```

---

## 📊 Data Flow

```
INPUT (24 Features)
    │
    ├── Numeric (14): age, BP, glucose, creatinine, etc.
    │
    └── Categorical (10): hypertension, diabetes, etc.
    
    ▼
    
PREPROCESSING
    │
    ├── Missing values → Imputation
    ├── Encoding → Binary (0/1)
    └── Scaling → StandardScaler
    
    ▼
    
PCA TRANSFORMATION
    │
    └── 24 features → 20 principal components (95% variance)
    
    ▼
    
MODEL PREDICTION
    │
    ├── Stage: 1, 2, 3, 4, or 5
    ├── Probabilities: [p1, p2, p3, p4, p5]
    └── eGFR: Calculated value
    
    ▼
    
OUTPUT
    │
    ├── Primary Result: Stage + Confidence
    ├── Visual 1: Probability Bar Chart
    ├── Visual 2: eGFR Gauge
    ├── Visual 3: Model Agreement Heatmap
    └── Visual 4: Multi-Model Comparison Table
```

---

## 🔢 eGFR to Stage Mapping

```
eGFR Value                    CKD Stage
─────────────────────────────────────────
   ≥ 90 mL/min/1.73m²    →   Stage 1 (Normal/High)     🟢
   60-89                  →   Stage 2 (Mild)            🟡
   30-59                  →   Stage 3 (Moderate)        🟠
   15-29                  →   Stage 4 (Severe)          🔴
   < 15                   →   Stage 5 (Failure)         🚨
```

---

## 💻 Application Stack

```
┌────────────────────────────────────────┐
│         USER INTERFACE LAYER           │
│  Gradio 4.44.0 (Web UI + Interactions) │
└────────────────────────────────────────┘
                 ▲
                 │
┌────────────────────────────────────────┐
│        APPLICATION LAYER               │
│  src/app.py + src/utils.py            │
│  (Business Logic + Helpers)            │
└────────────────────────────────────────┘
                 ▲
                 │
┌────────────────────────────────────────┐
│         MACHINE LEARNING LAYER         │
│  scikit-learn Models + PCA Pipeline    │
│  (6 Classifiers + Preprocessing)       │
└────────────────────────────────────────┘
                 ▲
                 │
┌────────────────────────────────────────┐
│           DATA LAYER                   │
│  models/*.pkl + data/kidney_disease.csv│
│  (Serialized Models + Dataset)         │
└────────────────────────────────────────┘
```

---

## 🚀 Execution Modes

### Development Mode
```bash
jupyter notebook notebooks/CKD_PCA_Models.ipynb
# → Train models
# → Experiment with features
# → Evaluate performance
# → Export to models/
```

### Production Mode
```bash
.venv\Scripts\activate
python src/app.py
# → Load pre-trained models
# → Launch web interface
# → Accept user input
# → Provide predictions
```

---

## 📈 Performance Pipeline

```
Training Data (280 patients)
    ↓
Cross-Validation (5-fold)
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation
    ↓
Validation Set (60 patients)
    ↓
Performance Metrics
    ↓
Test Set (60 patients)
    ↓
Final Accuracy: 98.5% ✅
```

---

## 🎨 Visualization Pipeline

```python
Predictions
    ├→ create_stage_probability_plot()
    │   └→ Matplotlib Bar Chart
    │
    ├→ create_egfr_gauge()
    │   └→ Matplotlib Horizontal Gauge
    │
    ├→ create_model_agreement_plot()
    │   └→ Matplotlib Heatmap
    │
    └→ create_comparison_table()
        └→ HTML Table with Styling
            
All → Gradio gr.Plot() / gr.HTML()
```

---

## 🔐 File Organization

```
Training Phase:
    notebooks/CKD_PCA_Models.ipynb
         ↓ exports to
    models/
         ├── best_model.pkl
         ├── pca_pipeline.pkl
         ├── feature_info.pkl
         └── [6 model files]

Prediction Phase:
    src/app.py
         ↓ imports from
    src/utils.py (helper functions)
         ↓ loads from
    models/ (trained artifacts)
         ↓ serves via
    Gradio Web Interface (port 7870)
```

---

## 🎯 User Journey

```
1. User Opens Browser
         ↓
2. Navigates to http://127.0.0.1:7870
         ↓
3. Sees CKD Stages Guide
         ↓
4. Fills Patient Data Form
         ↓
5. Clicks "Predict CKD Stage"
         ↓
6. System Calculates eGFR
         ↓
7. System Applies PCA
         ↓
8. All 6 Models Predict
         ↓
9. Best Model Result Highlighted
         ↓
10. Visualizations Generated
         ↓
11. Results Displayed
         ↓
12. User Reviews & Decides
```

---

## 🔄 Continuous Flow

```
Development → Testing → Training → Export → Deploy → Predict → Feedback → Improve → Retrain
    ↑                                                                                  ↓
    └──────────────────────────────────────────────────────────────────────────────────┘
```

---

**For detailed Mermaid diagrams and interactive flowcharts, see [docs/FLOWCHART.md](FLOWCHART.md)**
