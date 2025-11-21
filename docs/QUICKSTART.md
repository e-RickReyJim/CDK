# Quick Start Guide

## 🚀 Running the Application

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
.venv\Scripts\activate
```

```bash
# Linux/Mac
source .venv/bin/activate
```

### 2. Launch the Web App

```bash
python src/app.py
```

The application will start at `http://127.0.0.1:7870`

### 3. Using the Interface

1. **Fill Patient Data**: Enter all measurements and clinical indicators
2. **Click Predict**: Get instant CKD stage prediction
3. **View Results**: 
   - Primary diagnosis with confidence
   - Probability distribution chart
   - eGFR gauge visualization
   - Multi-model comparison table
   - Model agreement heatmap

### 4. Example Workflows

#### Scenario 1: Low Risk Patient
```
Age: 48, Creatinine: 1.2, All indicators: Normal
Expected Result: Stage 1-2 (Low Risk)
```

#### Scenario 2: Moderate Risk Patient
```
Age: 55, Creatinine: 2.1, Some abnormal indicators
Expected Result: Stage 3 (Moderate Risk)
```

#### Scenario 3: High Risk Patient
```
Age: 62, Creatinine: 4.1, Multiple abnormal indicators
Expected Result: Stage 4-5 (Critical Risk)
```

---

## 📓 Working with Jupyter Notebook

### Launch Notebook

```bash
jupyter notebook notebooks/CKD_PCA_Models.ipynb
```

### Notebook Sections

1. **Data Loading**: Import and explore dataset
2. **EDA**: Visualizations and statistical analysis
3. **Feature Engineering**: PCA dimensionality reduction
4. **Model Training**: Train 6 different classifiers
5. **Evaluation**: Performance metrics and visualizations
6. **Model Export**: Save trained models to `models/`

---

## 🔧 Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Update Dependencies

```bash
# After installing new packages
pip freeze > requirements.txt
```

### Project Structure

```
src/
├── app.py          # Main Gradio application
└── utils.py        # Utility functions (eGFR, plots)

models/             # Trained ML models (*.pkl)
notebooks/          # Jupyter notebooks
data/               # Dataset files
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Activate virtual environment first
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Issue: Models not found

**Solution**: Run the training notebook to generate models
```bash
jupyter notebook notebooks/CKD_PCA_Models.ipynb
# Run all cells
```

### Issue: Port already in use

**Solution**: Change port in `src/app.py`
```python
demo.launch(server_port=7871)  # Use different port
```

---

## 📚 Additional Resources

- **Full Documentation**: See [README.md](README.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **License**: See [LICENSE](LICENSE)

---

**Need Help?** Open an issue on GitHub!
