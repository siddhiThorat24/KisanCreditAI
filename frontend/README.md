# 🌾 KisanCredit AI — Farmer Loan Risk Intelligence

> An end-to-end machine learning pipeline and REST API for predicting agricultural loan approval risk, built on the Kaggle Loan Prediction Problem dataset.

---

## 📁 Project Structure

```
ML_FarmerCreditRisk/
├── ml_pipeline.py       # Full ML pipeline (Steps 0–11)
├── app.py               # Flask REST API
├── index.html           # Frontend dashboard (single-page)
├── loan_train.csv       # Dataset (614 rows, 14 columns)
└── results_summary.json # Model performance summary
```

Generated at runtime:
```
data/
└── loan_train.csv       # Dataset symlink / working copy

models/
├── best_model.pkl       # Serialised best classifier
├── scaler.pkl           # StandardScaler
├── pca.pkl              # PCA transformer
├── label_encoders.pkl   # LabelEncoders per categorical column
└── feature_cols.pkl     # Ordered feature column list

outputs/
├── eda_overview.png          # EDA bar charts
├── income_distribution.png   # Income & loan amount histograms
├── pca_variance.png          # Cumulative explained variance plot
├── roc_auc.png               # ROC-AUC curves (all models)
├── pr_curve.png              # Precision-Recall curves
├── confusion_matrix.png      # Confusion matrix (best model)
├── feature_importance.png    # PCA component importances
├── correlation_heatmap.png   # Feature correlation heatmap
└── results_summary.json      # JSON performance summary
```

---

## 🔬 ML Pipeline — Step by Step

| Step | Function | Description |
|------|----------|-------------|
| 0 | `load()` | Reads `loan_train.csv` (614 rows × 14 cols) |
| 1 | `eda()` | Exploratory analysis — bar charts, histograms |
| 2 | `clean()` | Drops `Loan_ID`/`Applicant_Name`; imputes missing values |
| 3 | `engineer()` | Creates 8 new features (EMI, Debt-to-Income, etc.) |
| 4 | `encode()` | Label-encodes all categorical columns |
| 5 | `discretise()` | KBins (quantile, n=5) on income & loan amount |
| 6 | `normalise()` | StandardScaler on full feature matrix |
| 7 | `apply_pca()` | PCA retaining 95% variance |
| 8 | `split()` | Stratified 80/20 train-test split |
| 9 | `train_models()` | 3 classifiers with 5-fold stratified CV |
| 10 | `evaluate()` | Accuracy, ROC-AUC, PR-AUC, confusion matrix |
| 11 | `save_artefacts()` | Persists models and JSON summary |

### Engineered Features
- `TotalIncome` = ApplicantIncome + CoapplicantIncome
- `TotalIncome_log` = log1p(TotalIncome)
- `LoanAmount_log` = log1p(LoanAmount)
- `EMI` = LoanAmount / Loan_Amount_Term
- `BalanceIncome` = TotalIncome − (EMI × 1000)
- `Debt_to_Income` = LoanAmount / (TotalIncome + 1)
- `Income_per_Person` = TotalIncome / (Dependents + 2)
- `LoanToIncome_ratio` = LoanAmount / (ApplicantIncome + 1)

### Models Trained
| Model | CV AUC | Test AUC | Test Accuracy |
|-------|--------|----------|---------------|
| Random Forest | ~0.84 | 0.8424 | 76.42% |
| Gradient Boosting | ~0.80 | 0.8015 | 76.42% |
| **Logistic Regression** ✅ | ~0.88 | **0.8796** | **80.49%** |

> **Best Model**: Logistic Regression (ROC-AUC = 0.8796, PR-AUC = 0.9439)

---

## 🌐 Dataset

**Source**: [Kaggle — Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

**Download via Kaggle CLI**:
```bash
kaggle datasets download -d altruistdelhite04/loan-prediction-problem-dataset \
  --unzip -p data/
```

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | string | Unique identifier (dropped) |
| `Applicant_Name` | string | Farmer name (dropped) |
| `Gender` | categorical | Male / Female |
| `Married` | categorical | Yes / No |
| `Dependents` | categorical | 0 / 1 / 2 / 3+ |
| `Education` | categorical | Graduate / Not Graduate |
| `Self_Employed` | categorical | Yes / No |
| `ApplicantIncome` | numeric | Monthly income (₹) |
| `CoapplicantIncome` | numeric | Co-applicant monthly income (₹) |
| `LoanAmount` | numeric | Loan amount (thousands ₹) |
| `Loan_Amount_Term` | numeric | Term in months |
| `Credit_History` | binary | 1 = good history, 0 = poor |
| `Property_Area` | categorical | Rural / Semiurban / Urban |
| `Loan_Status` | target | Y = Approved, N = Rejected |

---

## 🚀 Setup & Run

### 1. Install Dependencies

```bash
pip install flask numpy pandas matplotlib seaborn scikit-learn joblib
```

### 2. Prepare Dataset

Place `loan_train.csv` inside a `data/` folder at the project root (see Download above).

### 3. Train the Pipeline

**Option A — Python directly:**
```bash
python ml_pipeline.py
```

**Option B — via API endpoint:**
```bash
curl http://localhost:5000/api/train
```

### 4. Start the Flask API

```bash
python app.py
```

API available at `http://localhost:5000`

### 5. Open the Dashboard

Open `index.html` in any modern browser.  
Make sure the Flask server is running on port 5000 (CORS is pre-configured).

---

## 🔌 REST API Reference

### `GET /api/health`
Returns server and model status.
```json
{ "status": "ok", "model_loaded": true, "model_type": "LogisticRegression" }
```

### `POST /api/predict`
Predict loan approval risk for a single applicant.

**Request body:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Semiurban",
  "LandAcres": 3
}
```

**Response:**
```json
{
  "approved": true,
  "probability": 0.8234,
  "risk_level": "LOW RISK",
  "risk_color": "#2d9e4e",
  "risk_score": 17.7,
  "recommendation": "Loan Approved — Low credit risk.",
  "factors": [
    { "factor": "Good credit history", "impact": "+", "weight": 0.38 },
    { "factor": "Strong income ₹6500", "impact": "+", "weight": 0.22 }
  ]
}
```

**Risk Levels:**
| Probability | Risk Level |
|-------------|-----------|
| ≥ 0.75 | LOW RISK 🟢 |
| 0.50 – 0.74 | MEDIUM RISK 🟡 |
| < 0.50 | HIGH RISK 🔴 |

### `GET /api/results`
Returns the full `results_summary.json` with model metrics.

### `GET /api/dataset/stats`
Returns dataset shape, column list, approval rate, and missing value counts.

### `GET /api/dataset/sample?n=10`
Returns the first `n` rows of the dataset (max 50).

### `GET /api/train` or `POST /api/train`
Triggers the full ML pipeline and reloads models in-memory.

---

## 🎨 Design System

The frontend and pipeline share a cohesive dark-green palette:

| Token | Value | Usage |
|-------|-------|-------|
| `--ink` | `#05100a` | Page background |
| `--g4` | `#34a853` | Primary green (Google-inspired) |
| `--a4` | `#fbbc04` | Amber accent / warnings |
| `--r5` | `#e03333` | Red / high risk |
| `--b5` | `#1a8cff` | Blue / info |
| `--fd` | Syne | Display / heading font |
| `--fb` | Instrument Sans | Body font |
| `--fm` | JetBrains Mono | Code / numeric font |

---

## 📊 Model Results Summary

```
Best Model : Logistic Regression
Dataset    : 614 rows (Kaggle Loan Prediction)
Train/Test : 80/20 stratified split

┌────────────────────┬──────────┬──────────┬──────────┐
│ Model              │ Accuracy │ ROC-AUC  │ PR-AUC   │
├────────────────────┼──────────┼──────────┼──────────┤
│ Random Forest      │ 76.42%   │ 0.8424   │ 0.9211   │
│ Gradient Boosting  │ 76.42%   │ 0.8015   │ 0.8847   │
│ Logistic Regression│ 80.49%   │ 0.8796   │ 0.9439   │
└────────────────────┴──────────┴──────────┴──────────┘

Confusion Matrix (Logistic Regression, test set):
              Predicted
              Rejected  Approved
Actual Rejected   25        13
       Approved   11        74
```

---

## 📝 License

MIT — free to use and modify for educational and agricultural credit research purposes.

---

*Built for Indian agricultural credit assessment. Trained on public Kaggle data — not intended for production lending decisions without further validation.*
