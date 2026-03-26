# 💊 FESAR — FAERS Adverse Event Severity Prediction

A machine learning system that predicts the **severity of adverse drug events** using real-world pharmacovigilance data from the FDA Adverse Event Reporting System (FAERS).

---

## 🚀 Live Demo

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space (Live App) | [ADVERSE_EFFECT on HuggingFace](https://huggingface.co/spaces/iamn0tken/ADVERSE_EFFECT) |
| 📁 Dataset / Model Files (Drive) | `[PASTE YOUR GOOGLE DRIVE LINK HERE]` |

---

## 🧠 What It Does

Given patient demographics, drug information, and report metadata, FESAR predicts the **severity class** of an adverse drug reaction across 5 levels:

| Severity Level | Description |
|---|---|
| 🟢 No Event | No adverse outcome recorded |
| 🟡 Other | Non-serious or unclassified adverse event |
| 🟠 Hospitalization | Required hospitalization, disability, or intervention |
| 🔴 Life-Threatening | Life-threatening adverse event |
| 💀 Death | Fatal outcome reported |

---

## 📂 Project Structure

```
Advers_Effect_Prediction/
│
├── Data_Cleaning.ipynb       # Data preprocessing & feature engineering
├── Model_Building.ipynb      # Model training, evaluation & export
├── Model_Testing.ipynb       # Model validation & testing
├── app.py                    # Gradio web app (deployed on HuggingFace)
└── README.md
```

---

## ⚙️ How It Works

### Dataset
- Source: **FDA FAERS (Q4 2025)** — a real-world pharmacovigilance database containing millions of adverse event reports submitted to the FDA.

### Features Used

**Patient Info**
- Age, Age Group, Sex, Weight

**Drug Info**
- Drug name (one-hot encoded, top 300 drugs)
- Indication / condition being treated (top 100 indications)
- Dose amount
- Number of drugs currently taken (Polypharmacy risk)

**Report Metadata**
- Report type (Expedited / 30-Day / Periodic)
- Dechallenge (did reaction stop when drug was stopped?)
- Rechallenge (did reaction return when drug was restarted?)

### Model
- **Algorithm:** XGBoost Classifier (`faers_xgb_severity_v1.json`)
- **Output:** Probability distribution across 5 severity classes + predicted label

---

## 🖥️ Running Locally

```bash
git clone https://github.com/KenezNonwar/Advers_Effect_Prediction.git
cd Advers_Effect_Prediction

pip install gradio pandas numpy joblib xgboost

# Download model files from Drive (link above) and place in root folder:
# - faers_xgb_severity_v1.json
# - feature_cols_v1.pkl

python app.py
```

Then open `http://localhost:7860` in your browser.

---

## 🛠️ Tech Stack

- **Python** — core language
- **XGBoost** — gradient boosting classifier
- **Pandas / NumPy** — data processing
- **Gradio** — web app interface
- **Joblib** — model serialization
- **HuggingFace Spaces** — deployment

---

## 📊 Results

> _(Paste your model accuracy, F1-score, or confusion matrix here after testing)_

---

## 👤 Author

**KenezNonwar**
- GitHub: [@KenezNonwar](https://github.com/KenezNonwar)
- HuggingFace: [@iamn0tken](https://huggingface.co/iamn0tken)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
