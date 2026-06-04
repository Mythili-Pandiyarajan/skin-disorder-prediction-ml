import gradio as gr
import numpy as np
import pickle

# ── Load model & scaler ───────────────────────────────────────────────────────
with open("skin_disorder_model.pkl",  "rb") as f: model  = pickle.load(f)
with open("skin_disorder_scaler.pkl", "rb") as f: scaler = pickle.load(f)

# ── Constants (exact from notebook) ──────────────────────────────────────────
CLASS_MAP = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris",
}

DISEASE_INFO = {
    "Psoriasis":               "Chronic autoimmune condition. Key markers: koebner phenomenon, acanthosis, parakeratosis, knee/elbow involvement.",
    "Seborrheic Dermatitis":   "Inflammatory skin condition affecting oily areas. Key markers: scaling, erythema, follicular papules.",
    "Lichen Planus":           "Inflammatory condition of skin/mucous membranes. Key markers: polygonal papules, melanin incontinence, band-like infiltrate, oral mucosal involvement.",
    "Pityriasis Rosea":        "Self-limiting skin rash. Key markers: scaling, definite borders, focal hypergranulosis.",
    "Chronic Dermatitis":      "Long-term skin inflammation. Key markers: acanthosis, spongiosis, PNL infiltrate, exocytosis.",
    "Pityriasis Rubra Pilaris": "Rare skin disorder causing reddish-orange scaly patches. Key markers: follicular horn plug, perifollicular parakeratosis.",
}

# Exact feature order matching X = df.drop('class', axis=1) from notebook
# 34 features: 11 clinical + 22 histopathological + Age
FEATURES = [
    # Clinical (11)
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_and_elbow_involvement",
    "scalp_involvement", "family_history",
    # Histopathological (22)
    "melanin_incontinence", "eosinophils_in_the_infiltrate", "PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
    "elongation_of_the_rete_ridges", "thinning_of_the_suprapapillary_epidermis",
    "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
    "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer",
    "spongiosis", "saw-tooth_appearance_of_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
    "band-like_infiltrate",
    # Age
    "Age",
]

DISPLAY_NAMES = {
    "erythema":                                "Erythema (Redness)",
    "scaling":                                 "Scaling",
    "definite_borders":                        "Definite Borders",
    "itching":                                 "Itching",
    "koebner_phenomenon":                      "Koebner Phenomenon",
    "polygonal_papules":                       "Polygonal Papules",
    "follicular_papules":                      "Follicular Papules",
    "oral_mucosal_involvement":                "Oral Mucosal Involvement",
    "knee_and_elbow_involvement":              "Knee & Elbow Involvement",
    "scalp_involvement":                       "Scalp Involvement",
    "family_history":                          "Family History",
    "melanin_incontinence":                    "Melanin Incontinence",
    "eosinophils_in_the_infiltrate":           "Eosinophils in Infiltrate",
    "PNL_infiltrate":                          "PNL Infiltrate",
    "fibrosis_of_the_papillary_dermis":        "Fibrosis of Papillary Dermis",
    "exocytosis":                              "Exocytosis",
    "acanthosis":                              "Acanthosis",
    "hyperkeratosis":                          "Hyperkeratosis",
    "parakeratosis":                           "Parakeratosis",
    "clubbing_of_the_rete_ridges":             "Clubbing of Rete Ridges",
    "elongation_of_the_rete_ridges":           "Elongation of Rete Ridges",
    "thinning_of_the_suprapapillary_epidermis":"Thinning of Suprapapillary Epidermis",
    "spongiform_pustule":                      "Spongiform Pustule",
    "munro_microabcess":                       "Munro Microabscess",
    "focal_hypergranulosis":                   "Focal Hypergranulosis",
    "disappearance_of_the_granular_layer":     "Disappearance of Granular Layer",
    "vacuolisation_and_damage_of_basal_layer": "Vacuolisation & Damage of Basal Layer",
    "spongiosis":                              "Spongiosis",
    "saw-tooth_appearance_of_retes":           "Saw-tooth Appearance of Retes",
    "follicular_horn_plug":                    "Follicular Horn Plug",
    "perifollicular_parakeratosis":            "Perifollicular Parakeratosis",
    "inflammatory_monoluclear_inflitrate":     "Inflammatory Mononuclear Infiltrate",
    "band-like_infiltrate":                    "Band-like Infiltrate",
    "Age":                                     "Patient Age (years)",
}

SCALE_01 = {"family_history"}   # binary feature

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(*args):
    values = np.array(args, dtype=float).reshape(1, -1)
    scaled  = scaler.transform(values)
    pred    = model.predict(scaled)[0]
    proba   = model.predict_proba(scaled)[0]

    disease    = CLASS_MAP[pred]
    confidence = max(proba) * 100

    # Probability table
    prob_rows = ""
    for i, p in enumerate(proba):
        disease_name = CLASS_MAP[i + 1]
        bar   = "█" * int(p * 20)
        check = "✅" if i + 1 == pred else "  "
        prob_rows += f"{check} {disease_name:<30} {bar:<20} {p*100:5.1f}%\n"

    result = f"""╔══════════════════════════════════════════════════╗
║           SKIN DISORDER PREDICTION RESULT        ║
╚══════════════════════════════════════════════════╝

🔬 Predicted Disease : {disease}
📊 Confidence        : {confidence:.1f}%

─────────────────────────────────────────────────
 Probability Breakdown
─────────────────────────────────────────────────
{prob_rows}
─────────────────────────────────────────────────
 Clinical Notes
─────────────────────────────────────────────────
{DISEASE_INFO[disease]}

⚠️  For educational/research use only.
    Consult a qualified dermatologist for diagnosis."""

    return result

# ── Build UI ──────────────────────────────────────────────────────────────────
ORDINAL_INFO = "0 = Absent  |  1 = Mild  |  2 = Moderate  |  3 = Severe"
BINARY_INFO  = "0 = No  |  1 = Yes"

def make_slider(feat):
    label = DISPLAY_NAMES[feat]
    if feat == "Age":
        return gr.Slider(0, 100, value=35, step=1, label=f"🎂 {label}")
    elif feat in SCALE_01:
        return gr.Slider(0, 1, value=0, step=1,
                         label=f"🔘 {label}  [{BINARY_INFO}]")
    else:
        return gr.Slider(0, 3, value=0, step=1,
                         label=f"📏 {label}  [{ORDINAL_INFO}]")

clinical_feats = FEATURES[:11]
histo_feats    = FEATURES[11:33]
age_feat       = FEATURES[33]

with gr.Blocks(title="Skin Disorder Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🩺 Skin Disorder Prediction — PRCP-1027
    **Model:** Random Forest (Tuned) &nbsp;|&nbsp;
    **Accuracy:** 97.30% &nbsp;|&nbsp;
    **F1 Macro:** 0.9694 &nbsp;|&nbsp;
    **Dataset:** UCI Dermatology (366 patients, 34 features)

    Enter the patient's clinical and histopathological scores.
    Ordinal features: **0 = Absent → 3 = Severe**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🏥 Clinical Features (11)")
            clinical_inputs = [make_slider(f) for f in clinical_feats]

        with gr.Column(scale=1):
            gr.Markdown("### 🔬 Histopathological Features (22)")
            histo_inputs = [make_slider(f) for f in histo_feats]

    with gr.Row():
        with gr.Column(scale=1):
            age_input = make_slider(age_feat)
        with gr.Column(scale=1):
            predict_btn = gr.Button("🔍 Predict Disease", variant="primary", size="lg")

    output = gr.Textbox(
        label="Prediction Result",
        lines=20,
        show_copy_button=True,
        font_family="monospace",
    )

    predict_btn.click(
        fn=predict,
        inputs=clinical_inputs + histo_inputs + [age_input],
        outputs=output,
    )

    gr.Markdown("""
    ---
    **6 Disease Classes:** Psoriasis · Seborrheic Dermatitis · Lichen Planus ·
    Pityriasis Rosea · Chronic Dermatitis · Pityriasis Rubra Pilaris

    *Built with ❤️ using Gradio · PRCP-1027 Skin Disorder Prediction Project*
    """)

if __name__ == "__main__":
    demo.launch()
