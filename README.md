# 💊 FESAR — FAERS Adverse Event Severity Prediction

A machine learning pipeline that predicts the **severity of adverse drug events** using real-world pharmacovigilance data from the FDA Adverse Event Reporting System (FAERS Q4 2025).

---

## 🚀 Links

| Resource | Link |
|----------|------|
| 🤗 Live Demo (HuggingFace Space) | [ADVERSE_EFFECT](https://huggingface.co/spaces/iamn0tken/ADVERSE_EFFECT) |
| 📁 Dataset & Model Files (Google Drive) | `[PASTE YOUR GOOGLE DRIVE LINK HERE]` |

---

## 🧠 What It Does

Given patient demographics, drug information, and report metadata, FESAR predicts the **severity class** of an adverse drug reaction across 5 levels:

| Code | Severity | Description |
|------|----------|-------------|
| 0 | 🟢 No Event | No adverse outcome recorded |
| 1 | 🟡 Other | Non-serious or unclassified adverse event |
| 3 | 🟠 Hospitalization | Required hospitalization, disability, or intervention |
| 4 | 🔴 Life-Threatening | Life-threatening adverse event |
| 5 | 💀 Death | Fatal outcome reported |

---

## 📂 Project Structure

```
Advers_Effect_Prediction/
│
├── Data_Cleaning.ipynb       # Loads & cleans raw FAERS Q4 2025 files
├── Model_Building.ipynb      # Feature engineering, XGBoost training & export
├── Model_Testing.ipynb       # Model validation & manual inference testing
├── app.py                    # Gradio web app deployed on HuggingFace Spaces
└── README.md
```

---

## ⚙️ Pipeline Overview

### 1. Data Cleaning (`Data_Cleaning.ipynb`)

Raw FAERS Q4 2025 files (`DEMO`, `DRUG`, `REAC`, `OUTC`, `INDI`, `THER`) are loaded and merged. Key cleaning steps:
- **Age normalization** — converts years, months, weeks, days, decades → years
- **Weight normalization** — converts lbs, grams → kg with country-aware heuristics
- **Severity labeling** — derived from FAERS outcome codes

### 2. Feature Engineering & Model Building (`Model_Building.ipynb`)

**Patient features**
- Age (raw + age group bucket: Infant → Elderly)
- Sex (binary encoded)
- Weight (kg, with missing flag)

**Drug features**
- Top 300 drug names (one-hot encoded)
- Number of drugs taken + polypharmacy risk bucket (None / Minor / Moderate / Major)
- Dose amount (first numeric value extracted via regex)
- Dechallenge flag (reaction stopped when drug stopped?)
- Rechallenge flag (reaction returned when drug restarted?)
- Dose form indicators (Tablet, Capsule, Injection, etc.)

**Indication features**
- Top 100 indications one-hot encoded
- Indication count (comorbidity burden)

**Report metadata**
- Report type: Expedited (2) / 30-Day (1) / Periodic/Direct (0)
- Reporter occupation encoded (MD > HP > PH > CN > OT)
- Country: US flag (largest reporter group, different reporting bias)

**Model**
- **Algorithm:** XGBoost Classifier (`multi:softprob`)
- **Hyperparameters:** 500 estimators, max depth 6, learning rate 0.05, L1 + L2 regularization
- **Training:** 5-Fold Stratified Cross-Validation with balanced class weights
- **Output:** Probability distribution across 5 severity classes

### 3. Model Testing (`Model_Testing.ipynb`)

- Manual inference with CLI inputs for rapid validation
- Drug and indication lists exported to `drug_list.txt` and `indi_list.txt`

---

## 🖥️ Running Locally

```bash
git clone https://github.com/KenezNonwar/Advers_Effect_Prediction.git
cd Advers_Effect_Prediction

pip install gradio pandas numpy joblib xgboost scikit-learn shap

# Download model files from Drive (link above) and place in root:
# - faers_xgb_severity_v1.json
# - feature_cols_v1.pkl

python app.py
```

Then open `http://localhost:7860` in your browser.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| XGBoost | Gradient boosting classifier |
| Pandas / NumPy | Data processing & feature engineering |
| Scikit-learn | Cross-validation, class balancing, metrics |
| SHAP | Feature importance analysis |
| Gradio | Web app interface |
| HuggingFace Spaces | Deployment |
| Google Drive | Dataset & model file storage |

---

## 📊 Results

> _(Paste your CV macro-F1 score, classification report, or confusion matrix here)_

---

## 📋 Data Source

**FDA FAERS Q4 2025** — A publicly available pharmacovigilance database containing adverse event reports submitted to the US FDA. Files used: `DEMO25Q4`, `DRUG25Q4`, `REAC25Q4`, `OUTC25Q4`, `INDI25Q4`, `THER25Q4`.

---

## 👤 Author

**KenezNonwar**
- GitHub: [@KenezNonwar](https://github.com/KenezNonwar)
- HuggingFace: [@iamn0tken](https://huggingface.co/iamn0tken)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
