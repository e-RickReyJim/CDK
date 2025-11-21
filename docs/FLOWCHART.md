# CKD Stage Predictor - Solution Architecture & Flow

## 🔄 Complete System Flowchart

```mermaid
flowchart TB
    Start([👤 User/Developer]) --> Choice{Choose Path}
    
    %% Model Training Path
    Choice -->|Training Path| DataLoad[📊 Load Dataset<br/>kidney_disease.csv]
    DataLoad --> EDA[🔍 Exploratory Data Analysis<br/>Missing data, correlations]
    EDA --> Preprocess[🧹 Data Preprocessing<br/>Handle missing values<br/>Encode categoricals]
    Preprocess --> Stage[📐 Calculate CKD Stages<br/>eGFR = 175 × SC^-1.154 × Age^-0.203]
    Stage --> Split[✂️ Train/Val/Test Split<br/>70% / 15% / 15%<br/>Stratified by stage]
    Split --> PCA[🎯 PCA Transformation<br/>24 features → 20 PCs<br/>95% variance retained]
    PCA --> Train[🤖 Train 6 Models<br/>• Logistic Regression<br/>• Random Forest<br/>• Gradient Boosting<br/>• SVM<br/>• Naive Bayes<br/>• KNN]
    Train --> Evaluate[📈 Model Evaluation<br/>Cross-validation<br/>Confusion matrices<br/>Performance metrics]
    Evaluate --> Select[⭐ Select Best Model<br/>Random Forest: 98.5% acc]
    Select --> SaveModels[💾 Save Models<br/>models/*.pkl]
    SaveModels --> Ready[✅ Models Ready]
    
    %% Prediction Path
    Choice -->|Prediction Path| AppStart[🚀 Launch Gradio App<br/>python src/app.py]
    AppStart --> LoadModels[📦 Load Trained Models<br/>• PCA pipeline<br/>• Best model<br/>• All 6 models<br/>• Feature info]
    LoadModels --> Interface[🖥️ Web Interface Ready<br/>http://127.0.0.1:7870]
    Interface --> Input[📝 User Input<br/>24 clinical features]
    Input --> Validate[✓ Input Validation<br/>Check ranges<br/>Handle missing]
    Validate --> CalcEGFR[🧮 Calculate eGFR<br/>MDRD equation]
    CalcEGFR --> Transform[🔄 PCA Transform<br/>24 → 20 features]
    Transform --> Predict[🎯 Predictions<br/>• Best model<br/>• All 6 models<br/>• Probabilities]
    Predict --> Visualize[📊 Generate Visualizations<br/>• Probability chart<br/>• eGFR gauge<br/>• Model agreement<br/>• Comparison table]
    Visualize --> Display[🖼️ Display Results<br/>Stage, confidence,<br/>risk assessment]
    Display --> UserDecision{User Action?}
    UserDecision -->|New Patient| Input
    UserDecision -->|View Details| Visualize
    UserDecision -->|Exit| End([End])
    
    Ready -.->|Models Available| LoadModels
    
    %% Styling
    classDef training fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef prediction fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef ml fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef viz fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef io fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class DataLoad,EDA,Preprocess,Stage,Split training
    class AppStart,Interface,Input,Display io
    class PCA,Train,Evaluate,Select,Transform,Predict ml
    class Visualize,UserDecision viz
    class LoadModels,SaveModels prediction
```

## 📋 Detailed Component Flow

### 1️⃣ **Model Training Pipeline** (Jupyter Notebook)

```mermaid
graph LR
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Stage Calculation]
    D --> E[Data Split]
    E --> F[PCA Pipeline]
    F --> G[Model Training]
    G --> H[Model Evaluation]
    H --> I[Model Export]
    
    style A fill:#ffebee
    style I fill:#e8f5e9
```

**Steps:**
1. Load `kidney_disease.csv` (400 patients, 26 features)
2. Handle missing values (imputation, dropping)
3. Encode categorical variables (binary encoding)
4. Calculate eGFR using MDRD equation
5. Classify into 5 stages based on eGFR
6. Split data (70/15/15) with stratification
7. Apply PCA (reduce to 20 components)
8. Train 6 different classifiers
9. Evaluate with cross-validation
10. Export models to `models/` directory

---

### 2️⃣ **Prediction Pipeline** (Gradio Application)

```mermaid
sequenceDiagram
    participant User
    participant Gradio
    participant App
    participant Models
    participant Utils
    
    User->>Gradio: Enter patient data
    Gradio->>App: Form submission
    App->>App: Validate inputs
    App->>Utils: calculate_egfr(creatinine, age)
    Utils-->>App: eGFR value
    App->>App: Create DataFrame
    App->>Models: PCA transform
    Models-->>App: Transformed features
    App->>Models: Predict with best model
    Models-->>App: Stage & probabilities
    App->>Models: Predict with all 6 models
    Models-->>App: All predictions
    App->>Utils: Create visualizations
    Utils-->>App: Plots (probability, gauge, heatmap)
    App->>Gradio: Results + visualizations
    Gradio->>User: Display interactive results
```

**Steps:**
1. User fills web form with 24 features
2. App validates and preprocesses inputs
3. Calculate eGFR from creatinine and age
4. Transform inputs using PCA pipeline
5. Get prediction from best model (Random Forest)
6. Get predictions from all 6 models
7. Generate 4 visualizations:
   - Stage probability distribution
   - eGFR gauge with zones
   - Model agreement heatmap
   - Comparison table
8. Display results with HTML formatting

---

### 3️⃣ **Data Flow Architecture**

```mermaid
flowchart LR
    subgraph Input["📥 Input Layer"]
        Raw[24 Clinical Features]
    end
    
    subgraph Process["⚙️ Processing Layer"]
        Clean[Data Cleaning]
        Encode[Encoding]
        PCA[PCA Transform]
    end
    
    subgraph Models["🤖 Model Layer"]
        LR[Logistic Reg]
        RF[Random Forest]
        GB[Gradient Boost]
        SVM[SVM]
        NB[Naive Bayes]
        KNN[KNN]
    end
    
    subgraph Output["📤 Output Layer"]
        Stage[CKD Stage 1-5]
        Prob[Probabilities]
        Viz[Visualizations]
    end
    
    Raw --> Clean
    Clean --> Encode
    Encode --> PCA
    PCA --> LR & RF & GB & SVM & NB & KNN
    LR & RF & GB & SVM & NB & KNN --> Stage
    LR & RF & GB & SVM & NB & KNN --> Prob
    Stage & Prob --> Viz
    
    style Input fill:#e3f2fd
    style Process fill:#fff3e0
    style Models fill:#f3e5f5
    style Output fill:#e8f5e9
```

---

### 4️⃣ **System Architecture**

```mermaid
graph TB
    subgraph Frontend["🖥️ Frontend Layer"]
        UI[Gradio Web Interface<br/>Port 7870]
    end
    
    subgraph Application["⚙️ Application Layer"]
        App[src/app.py<br/>Main Application]
        Utils[src/utils.py<br/>Helper Functions]
    end
    
    subgraph ML["🤖 ML Layer"]
        Pipeline[PCA Pipeline]
        Best[Best Model]
        All[All 6 Models]
    end
    
    subgraph Data["💾 Data Layer"]
        Models[(models/<br/>*.pkl files)]
        Dataset[(data/<br/>kidney_disease.csv)]
    end
    
    subgraph Notebooks["📓 Development"]
        Jupyter[notebooks/<br/>CKD_PCA_Models.ipynb]
    end
    
    UI <--> App
    App --> Utils
    App --> Pipeline
    App --> Best
    App --> All
    Pipeline -.-> Models
    Best -.-> Models
    All -.-> Models
    Jupyter -.Create.-> Models
    Jupyter -.Load.-> Dataset
    
    style Frontend fill:#e3f2fd
    style Application fill:#fff3e0
    style ML fill:#f3e5f5
    style Data fill:#e8f5e9
    style Notebooks fill:#fce4ec
```

---

### 5️⃣ **eGFR Calculation Flow**

```mermaid
flowchart TD
    Start[Patient Data] --> Check{Has Age &<br/>Creatinine?}
    Check -->|Yes| Calc[Calculate eGFR<br/>MDRD Formula]
    Check -->|No| Error[Return None<br/>Show Error]
    Calc --> Formula["eGFR = 175 × SC^-1.154 × Age^-0.203"]
    Formula --> Stage{Classify Stage}
    Stage -->|eGFR ≥ 90| S1[Stage 1<br/>Normal/High]
    Stage -->|60-89| S2[Stage 2<br/>Mild]
    Stage -->|30-59| S3[Stage 3<br/>Moderate]
    Stage -->|15-29| S4[Stage 4<br/>Severe]
    Stage -->|< 15| S5[Stage 5<br/>Failure]
    S1 & S2 & S3 & S4 & S5 --> Display[Display Results]
    Error --> Display
    
    style S1 fill:#c8e6c9
    style S2 fill:#fff9c4
    style S3 fill:#ffe0b2
    style S4 fill:#ffccbc
    style S5 fill:#ffcdd2
```

---

### 6️⃣ **Visualization Generation**

```mermaid
flowchart LR
    Predictions[Model Predictions] --> V1[Probability Chart]
    Predictions --> V2[eGFR Gauge]
    Predictions --> V3[Model Agreement]
    Predictions --> V4[Comparison Table]
    
    V1 --> M1[Matplotlib Figure]
    V2 --> M2[Matplotlib Figure]
    V3 --> M3[Matplotlib Figure]
    V4 --> H[HTML Table]
    
    M1 & M2 & M3 & H --> Display[Gradio Display]
    
    style V1 fill:#e1bee7
    style V2 fill:#c5cae9
    style V3 fill:#b2dfdb
    style V4 fill:#f8bbd0
```

---

## 🔑 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | scikit-learn 1.7.2 | Model training & prediction |
| **Dimensionality** | PCA | 24 → 20 features (95% variance) |
| **Web Interface** | Gradio 4.44.0 | Interactive UI |
| **Visualization** | Matplotlib 3.10.7 | Charts & plots |
| **Data Processing** | Pandas 2.3.3 | DataFrame operations |
| **Numerical** | NumPy 2.3.5 | Array operations |
| **Backend** | FastAPI 0.121.3 | API support |

---

## 📊 Model Ensemble Strategy

```mermaid
graph TD
    Input[Patient Data] --> All[All 6 Models Predict]
    All --> LR[Logistic Regression]
    All --> RF[Random Forest ⭐]
    All --> GB[Gradient Boosting]
    All --> SVM[SVM]
    All --> NB[Naive Bayes]
    All --> KNN[KNN]
    
    RF --> Best[Best Model<br/>98.5% Accuracy]
    LR & GB & SVM & NB & KNN --> Compare[Comparison View]
    
    Best --> Primary[Primary Result]
    Compare --> Secondary[Secondary Results]
    
    Primary & Secondary --> User[User Decision]
    
    style RF fill:#ffd700,stroke:#ff8c00,stroke-width:3px
    style Best fill:#90ee90
```

---

## 🎯 Use Case Flow

### Scenario: New Patient Evaluation

```mermaid
stateDiagram-v2
    [*] --> PatientArrival: New patient
    PatientArrival --> DataCollection: Collect lab results
    DataCollection --> AppLaunch: Open application
    AppLaunch --> DataEntry: Enter measurements
    DataEntry --> Prediction: Click Predict
    Prediction --> ResultsReview: View stage & visualizations
    ResultsReview --> Decision: Clinical decision
    Decision --> [*]: Treatment plan
    
    note right of Prediction
        • eGFR calculated
        • 6 models predict
        • Visualizations generated
    end note
    
    note right of ResultsReview
        • Stage 1-5 classification
        • Probability distribution
        • Risk assessment
        • Model consensus
    end note
```

---

## 🔄 Development vs Production Flow

```mermaid
graph TB
    subgraph Development["👨‍💻 Development Mode"]
        D1[Jupyter Notebook]
        D2[Experiment & Train]
        D3[Evaluate Models]
        D4[Export Models]
    end
    
    subgraph Production["🚀 Production Mode"]
        P1[Load Pre-trained Models]
        P2[Gradio Web App]
        P3[Real-time Predictions]
        P4[User Interface]
    end
    
    D1 --> D2 --> D3 --> D4
    D4 -.Models.-> P1
    P1 --> P2 --> P3 --> P4
    
    style Development fill:#fff3e0
    style Production fill:#e8f5e9
```

---

## 📁 File Flow

```mermaid
flowchart LR
    subgraph Input
        CSV[kidney_disease.csv]
    end
    
    subgraph Processing
        NB[CKD_PCA_Models.ipynb]
    end
    
    subgraph Models_Dir[models/]
        PKL1[*.pkl files]
    end
    
    subgraph Application
        APP[src/app.py]
        UTIL[src/utils.py]
    end
    
    subgraph Output
        WEB[Web Interface]
    end
    
    CSV --> NB
    NB -->|Train & Export| PKL1
    PKL1 -->|Load| APP
    UTIL -.Used by.-> APP
    APP --> WEB
    
    style Input fill:#e3f2fd
    style Processing fill:#fff3e0
    style Models_Dir fill:#f3e5f5
    style Application fill:#e8f5e9
    style Output fill:#fce4ec
```

---

This flowchart document provides comprehensive visualization of:
- Complete system architecture
- Model training pipeline
- Prediction workflow
- Data flow
- Component interactions
- Use case scenarios

All diagrams use Mermaid syntax and will render beautifully in GitHub README!
