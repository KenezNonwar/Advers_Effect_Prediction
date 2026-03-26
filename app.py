import gradio as gr
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ── Load model & features ──────────────────────────────────────────────────────
model = xgb.XGBClassifier()
model.load_model("faers_xgb_severity_v1.json")
feature_cols = joblib.load("feature_cols_v1.pkl")

# ── Extract drug & indication lists from feature_cols ─────────────────────────
drug_options = sorted([
    c.replace("drug_", "", 1).replace("_", " ")
    for c in feature_cols if c.startswith("drug_")
])
indi_options = sorted([
    c.replace("indi_", "", 1).replace("_", " ")
    for c in feature_cols if c.startswith("indi_")
])

# ── Severity label mapping ─────────────────────────────────────────────────────
SEVERITY_LABELS = {
    0: ("🟢 No Event",          "No adverse outcome recorded."),
    1: ("🟡 Other",             "Non-serious or unclassified adverse event."),
    3: ("🟠 Hospitalization",   "Required hospitalization, disability, or intervention."),
    4: ("🔴 Life-Threatening",  "Life-threatening adverse event."),
    5: ("💀 Death",             "Fatal outcome reported."),
}
INDEX_TO_SEVERITY = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5}

# ── Helper functions ───────────────────────────────────────────────────────────
def get_age_group(age):
    bins   = [0, 2, 12, 18, 40, 62, 85, 200]
    labels = [0, 1,  2,  3,  4,  5,  6]
    for i, upper in enumerate(bins[1:]):
        if age <= upper:
            return labels[i]
    return 6

def get_poly_risk(n_drugs):
    return int(pd.cut([n_drugs], bins=[-1, 1, 2, 4, 9999], labels=[0, 1, 2, 3])[0])

def safe_col(prefix, name):
    return prefix + name.upper().replace(" ", "_").replace("-", "_")[:40]

def build_row(age, sex_str, wt_kg, dose_amt, d_count, drug, indi, rept_str, dechal_str, rechal_str):
    sex    = 0 if sex_str == "Male" else 1
    rept   = {"Expedited": 2, "30-Day": 1, "Periodic / Direct": 0}[rept_str]
    dechal = 1 if dechal_str == "Yes" else 0
    rechal = 1 if rechal_str == "Yes" else 0

    row = pd.DataFrame([dict.fromkeys(feature_cols, 0)], dtype="float32")

    row["age_years"]    = age
    row["age_group"]    = get_age_group(age)
    row["missing_age"]  = 0
    row["sex_enc"]      = sex
    row["wt_kg"]        = wt_kg
    row["missing_wt"]   = 0
    row["dose_amt_val"] = dose_amt
    row["missing_dose"] = 0
    row["n_drugs"]      = d_count
    row["poly_risk"]    = get_poly_risk(d_count)
    row["dechal_pos"]   = dechal
    row["rechal_pos"]   = rechal
    row["rept_enc"]          = rept
    row["indication_count"]  = 1
    row["occp_enc"]          = 4
    row["is_us"]             = 1

    drug_col = safe_col("drug_", drug)
    indi_col = safe_col("indi_", indi)
    if drug_col in feature_cols:
        row[drug_col] = 1
    if indi_col in feature_cols:
        row[indi_col] = 1

    return row, drug_col, indi_col

# ── Age group & poly risk labels ───────────────────────────────────────────────
AGE_GROUP_LABELS = {
    0: "Infant (0–2)", 1: "Child (2–12)", 2: "Adolescent (12–18)",
    3: "Adult (18–40)", 4: "Middle-aged (40–62)", 5: "Senior (62–85)", 6: "Elderly (85+)"
}
POLY_LABELS = {
    0: "None (1 drug)", 1: "Minor (2 drugs)",
    2: "Moderate (3–4 drugs)", 3: "Major (5+ drugs)"
}

# ── Predict function ───────────────────────────────────────────────────────────
def predict(age, sex_str, wt_kg, dose_amt, d_count, drug, indi, rept_str, dechal_str, rechal_str):
    row, drug_col, indi_col = build_row(
        age, sex_str, wt_kg, dose_amt, d_count,
        drug, indi, rept_str, dechal_str, rechal_str
    )

    probs = model.predict_proba(row)[0]
    pred_index    = int(np.argmax(probs))
    pred_severity = INDEX_TO_SEVERITY[pred_index]
    label, description = SEVERITY_LABELS[pred_severity]

    # Auto-derived features
    age_grp  = AGE_GROUP_LABELS[get_age_group(age)]
    poly_risk = POLY_LABELS[get_poly_risk(d_count)]

    # Build result text
    result = f"## Prediction: {label}\n**{description}**\n\n"
    result += f"**Age Group:** {age_grp} &nbsp;|&nbsp; **Poly Risk:** {poly_risk}\n\n"

    # Warnings
    warnings = []
    if drug_col not in feature_cols:
        warnings.append(f"⚠️ Drug '{drug}' was not in the top-300 training drugs (column set to 0).")
    if indi_col not in feature_cols:
        warnings.append(f"⚠️ Indication '{indi}' was not in top-100 training indications (column set to 0).")
    if warnings:
        result += "\n".join(warnings) + "\n\n"

    # Probability breakdown
    result += "### Probability across severity levels\n"
    prob_rows = []
    for i, p in enumerate(probs):
        sev_key = INDEX_TO_SEVERITY[i]
        sev_label = SEVERITY_LABELS[sev_key][0]
        bar = "█" * int(p * 30)
        prob_rows.append(f"`{sev_label:<22}` {bar} {p:.1%}")
    result += "\n".join(prob_rows)

    return result

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="FAERS Severity Predictor") as demo:
    gr.Markdown(
        """
        # 💊 FAERS Adverse Event Severity Predictor
        Predicts the severity of an adverse drug event using the FAERS Q4 2025 dataset model.
        > Built with XGBoost · by [KenezNonwar](https://huggingface.co/KenezNonwar)
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧑 Patient Info")
            age    = gr.Slider(0, 120, value=35, step=1, label="Age (years)")
            sex    = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            wt_kg  = gr.Slider(2, 250, value=70, step=0.5, label="Weight (kg)")

        with gr.Column():
            gr.Markdown("### 💊 Drug Info")
            drug     = gr.Dropdown(drug_options, value=drug_options[0], label="Drug name")
            indi     = gr.Dropdown(indi_options, value=indi_options[0], label="Indication")
            dose_amt = gr.Number(value=500, label="Dose amount (mg)")
            d_count  = gr.Slider(1, 50, value=1, step=1, label="Number of drugs currently taking")

    gr.Markdown("### 📋 Report Info")
    with gr.Row():
        rept   = gr.Dropdown(["Expedited", "30-Day", "Periodic / Direct"], value="Expedited", label="Report type")
        dechal = gr.Radio(["No / Unknown", "Yes"], value="No / Unknown", label="Dechallenge (reaction stopped when drug stopped?)")
        rechal = gr.Radio(["No / Unknown", "Yes"], value="No / Unknown", label="Rechallenge (reaction returned when drug restarted?)")

    predict_btn = gr.Button("🔍 Predict Severity", variant="primary", size="lg")
    output      = gr.Markdown(label="Result")

    predict_btn.click(
        fn=predict,
        inputs=[age, sex, wt_kg, dose_amt, d_count, drug, indi, rept, dechal, rechal],
        outputs=output
    )

demo.launch(ssr_mode=False, server_name="0.0.0.0", server_port=7860)
