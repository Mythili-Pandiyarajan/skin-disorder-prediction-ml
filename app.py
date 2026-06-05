"""
Skin Disorder Prediction — Advanced Gradio App
Model: Random Forest Tuned (97.30% Accuracy | F1 Macro 0.9694)
Dataset: UCI Dermatology Dataset (366 patients, 34 features, 6 disease classes)
"""

import gradio as gr
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. CONSTANTS & CONFIG
# ──────────────────────────────────────────────

CLASS_MAP = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris",
}

CLASS_ICONS = {
    1: "🔴",
    2: "🟠",
    3: "🟣",
    4: "🟡",
    5: "🟤",
    6: "🔵",
}

CLASS_CLINICAL_NOTES = {
    1: (
        "Psoriasis is characterized by rapid skin cell turnover causing thick, scaly plaques. "
        "Key indicators: Koebner phenomenon (lesions at trauma sites), extensor surface involvement "
        "(knees, elbows), and acanthosis with parakeratosis on biopsy."
    ),
    2: (
        "Seborrheic Dermatitis affects sebaceous gland-rich areas. "
        "Look for greasy, yellowish scales on scalp and face. "
        "Follicular papules and scalp involvement are distinguishing clinical features."
    ),
    3: (
        "Lichen Planus presents with flat-topped, polygonal, violaceous papules. "
        "Oral mucosal involvement is near-exclusive to this condition. "
        "Histology shows melanin incontinence and band-like infiltrate."
    ),
    4: (
        "Pityriasis Rosea typically begins with a 'herald patch' followed by a "
        "Christmas tree-shaped rash on the trunk. Usually self-limiting within 6–8 weeks. "
        "Check for oval lesions with collarette scaling."
    ),
    5: (
        "Chronic Dermatitis often results from prolonged exposure to irritants/allergens. "
        "Spongiosis and exocytosis are key histopathological findings. "
        "Differentiate from Seborrheic Dermatitis via biopsy when in doubt."
    ),
    6: (
        "Pityriasis Rubra Pilaris is a rare, chronic papulosquamous disorder. "
        "Classic orange-red palmoplantar keratoderma and follicular keratotic papules "
        "are hallmarks. Often requires systemic therapy."
    ),
}

# ──────────────────────────────────────────────
# EXACT column order the model was trained on
# (from df.info() output — 34 features, no engineered cols)
# NOTE: hyphens preserved exactly as in the UCI CSV
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "erythema",
    "scaling",
    "definite_borders",
    "itching",
    "koebner_phenomenon",
    "polygonal_papules",
    "follicular_papules",
    "oral_mucosal_involvement",
    "knee_and_elbow_involvement",
    "scalp_involvement",
    "family_history",
    "melanin_incontinence",
    "eosinophils_in_the_infiltrate",
    "PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing_of_the_rete_ridges",
    "elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis",
    "spongiform_pustule",
    "munro_microabcess",
    "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer",
    "spongiosis",
    "saw-tooth_appearance_of_retes",        # hyphen — exact UCI name
    "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_monoluclear_inflitrate",  # original typo in UCI dataset
    "band-like_infiltrate",                 # hyphen — exact UCI name
    "Age",
]

# UI-friendly feature definitions (label, internal_key, tooltip, min, max, default)
# internal_key uses underscores for Gradio compatibility; mapped to FEATURE_COLUMNS on predict
FEATURE_GROUPS = {
    "Patient Info": [
        ("Age", "age_val", "Patient age in years", 0, 90, 35),
        ("Family History (0/1)", "family_history", "1 = family history of similar disease", 0, 1, 0),
    ],
    "🔴 Clinical Features (0–3 scale)": [
        ("Erythema", "erythema", "Redness of skin (0=none, 3=severe)", 0, 3, 0),
        ("Scaling", "scaling", "Scaling on skin surface", 0, 3, 0),
        ("Definite Borders", "definite_borders", "Sharpness of lesion boundary", 0, 3, 0),
        ("Itching", "itching", "Patient-reported itch severity", 0, 3, 0),
        ("Koebner Phenomenon", "koebner_phenomenon", "Lesions at trauma sites (Psoriasis marker)", 0, 3, 0),
        ("Polygonal Papules", "polygonal_papules", "Flat-topped polygonal bumps (Lichen Planus marker)", 0, 3, 0),
        ("Follicular Papules", "follicular_papules", "Hair follicle-centered bumps", 0, 3, 0),
        ("Oral Mucosal Involvement", "oral_mucosal_involvement", "Lesions inside mouth (near-exclusive to Lichen Planus)", 0, 3, 0),
        ("Knee & Elbow Involvement", "knee_and_elbow_involvement", "Extensor surface involvement", 0, 3, 0),
        ("Scalp Involvement", "scalp_involvement", "Lesions on scalp", 0, 3, 0),
    ],
    "🧫 Histopathological Features (0–3 scale)": [
        ("Inflammatory Mononuclear Infiltrate", "inflammatory_monoluclear_inflitrate", "Immune cell infiltration (original UCI spelling)", 0, 3, 0),
        ("Band-like Infiltrate", "band_like_infiltrate_ui", "Band-like immune infiltration (Lichen Planus)", 0, 3, 0),
        ("PNL Infiltrate", "PNL_infiltrate", "Polymorphonuclear leukocyte infiltrate", 0, 3, 0),
        ("Fibrosis of Papillary Dermis", "fibrosis_of_the_papillary_dermis", "Dermal fibrosis grade", 0, 3, 0),
        ("Exocytosis", "exocytosis", "Inflammatory cells in epidermis", 0, 3, 0),
        ("Acanthosis", "acanthosis", "Epidermal thickening (Psoriasis/Chronic Dermatitis)", 0, 3, 0),
        ("Hyperkeratosis", "hyperkeratosis", "Thickened stratum corneum", 0, 3, 0),
        ("Parakeratosis", "parakeratosis", "Nuclei retained in stratum corneum (Psoriasis)", 0, 3, 0),
        ("Clubbing of Rete Ridges", "clubbing_of_the_rete_ridges", "Bulbous rete ridges on biopsy", 0, 3, 0),
        ("Elongation of Rete Ridges", "elongation_of_the_rete_ridges", "Extended rete ridges", 0, 3, 0),
        ("Thinning of Suprapapillary Epidermis", "thinning_of_the_suprapapillary_epidermis", "Reduced epidermal thickness above papillae", 0, 3, 0),
        ("Spongiform Pustule", "spongiform_pustule", "Characteristic Psoriasis pustule", 0, 3, 0),
        ("Munro Microabscess", "munro_microabcess", "Psoriasis-specific microabscess", 0, 3, 0),
        ("Focal Hypergranulosis", "focal_hypergranulosis", "Lichen Planus granular layer feature", 0, 3, 0),
        ("Disappearance of Granular Layer", "disappearance_of_the_granular_layer", "Loss of granular layer", 0, 3, 0),
        ("Vacuolisation of Basal Layer", "vacuolisation_and_damage_of_basal_layer", "Basal cell damage (Lichen Planus)", 0, 3, 0),
        ("Spongiosis", "spongiosis", "Intercellular edema (Chronic Dermatitis)", 0, 3, 0),
        ("Saw-tooth Appearance of Retes", "saw_tooth_appearance_of_retes_ui", "Lichen Planus rete morphology", 0, 3, 0),
        ("Follicular Horn Plug", "follicular_horn_plug", "Follicular keratotic plugging", 0, 3, 0),
        ("Perifollicular Parakeratosis", "perifollicular_parakeratosis", "Parakeratosis around follicles", 0, 3, 0),
        ("Melanin Incontinence", "melanin_incontinence", "Melanin in dermis (Lichen Planus key marker)", 0, 3, 0),
        ("Eosinophils in Infiltrate", "eosinophils_in_the_infiltrate", "Eosinophil presence", 0, 3, 0),
    ],
}

# Mapping from UI internal_key → exact model column name (handles hyphenated names)
UI_KEY_TO_MODEL_COL = {
    "band_like_infiltrate_ui": "band-like_infiltrate",
    "saw_tooth_appearance_of_retes_ui": "saw-tooth_appearance_of_retes",
}

# ──────────────────────────────────────────────
# 2. MODEL LOADING
# ──────────────────────────────────────────────

def load_artifacts():
    model, scaler = None, None
    if os.path.exists("skin_disorder_model.pkl"):
        with open("skin_disorder_model.pkl", "rb") as f:
            model = pickle.load(f)
    if os.path.exists("skin_disorder_scaler.pkl"):
        with open("skin_disorder_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

MODEL, SCALER = load_artifacts()

# ──────────────────────────────────────────────
# 3. PREDICTION LOGIC
# ──────────────────────────────────────────────

def _get_ordinal_keys():
    """Return list of UI internal_keys for all non-patient-info features."""
    return [t[1] for group_label, items in FEATURE_GROUPS.items()
            for t in items if t[1] not in ("age_val", "family_history")]


def build_feature_vector(age, family_history, *ordinal_vals):
    """Assemble feature dict keyed by EXACT model column names."""
    ordinal_keys = _get_ordinal_keys()

    ui_feat = {}
    for k, v in zip(ordinal_keys, ordinal_vals):
        ui_feat[k] = int(v)

    # Build model-ready dict using exact column names
    feat = {}
    for ui_key, val in ui_feat.items():
        model_col = UI_KEY_TO_MODEL_COL.get(ui_key, ui_key)
        feat[model_col] = val

    feat["Age"] = float(age)
    feat["family_history"] = int(family_history)
    return feat


def predict(age, family_history, *ordinal_vals):
    feat = build_feature_vector(age, family_history, *ordinal_vals)

    # Build DataFrame in EXACT training column order (no engineered features)
    row = {col: feat.get(col, 0) for col in FEATURE_COLUMNS}
    df_input = pd.DataFrame([row])

    if MODEL and SCALER:
        X_scaled = SCALER.transform(df_input)
        pred_class = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]
        classes_order = MODEL.classes_
    else:
        pred_class, probabilities, classes_order = _demo_predict(feat)

    pred_name = CLASS_MAP[pred_class]
    pred_icon = CLASS_ICONS[pred_class]
    pred_conf = max(probabilities) * 100

    if pred_conf >= 85:
        conf_label, conf_color = "HIGH CONFIDENCE", "#22c55e"
    elif pred_conf >= 65:
        conf_label, conf_color = "MODERATE CONFIDENCE", "#f59e0b"
    else:
        conf_label, conf_color = "LOW CONFIDENCE — CONSULT SPECIALIST", "#ef4444"

    clinical_note = CLASS_CLINICAL_NOTES[pred_class]

    # Probability bars
    prob_rows = ""
    sorted_indices = np.argsort(probabilities)[::-1]
    for idx in sorted_indices:
        c = classes_order[idx]
        name = CLASS_MAP[c]
        icon = CLASS_ICONS[c]
        pct = probabilities[idx] * 100
        bar_color = conf_color if c == pred_class else "#64748b"
        bold = "font-weight:700;" if c == pred_class else ""
        prob_rows += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
            <span style="{bold}color:#e2e8f0;font-size:13px;">{icon} {name}</span>
            <span style="{bold}color:{bar_color};font-size:13px;">{pct:.1f}%</span>
          </div>
          <div style="background:#1e293b;border-radius:6px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{bar_color};height:100%;
                        border-radius:6px;transition:width 0.5s ease;"></div>
          </div>
        </div>"""

    # Clinical flags
    flags = []
    if feat.get("koebner_phenomenon", 0) >= 2:
        flags.append("⚠️ Strong Koebner phenomenon — high Psoriasis likelihood")
    if feat.get("oral_mucosal_involvement", 0) >= 1:
        flags.append("⚠️ Oral mucosal involvement — consider Lichen Planus")
    if feat.get("melanin_incontinence", 0) >= 2:
        flags.append("⚠️ Melanin incontinence on biopsy — Lichen Planus marker")
    if feat.get("band-like_infiltrate", 0) >= 2:
        flags.append("⚠️ Band-like infiltrate — Lichen Planus histological signature")
    if feat.get("spongiosis", 0) >= 2 and feat.get("exocytosis", 0) >= 2:
        flags.append("⚠️ Spongiosis + exocytosis — Chronic Dermatitis pattern")
    if feat.get("acanthosis", 0) >= 2 and feat.get("parakeratosis", 0) >= 2:
        flags.append("⚠️ Acanthosis + parakeratosis — Psoriasis/Chronic Dermatitis")

    flags_html = "".join(
        f'<div style="background:#1e293b;border-left:3px solid #f59e0b;'
        f'padding:8px 12px;border-radius:4px;margin-bottom:6px;'
        f'color:#fbbf24;font-size:12px;">{flag}</div>'
        for flag in flags
    ) if flags else (
        '<div style="color:#64748b;font-size:12px;font-style:italic;">No critical pattern flags detected.</div>'
    )

    model_note = "" if (MODEL and SCALER) else (
        '<div style="background:#7c2d12;border:1px solid #ef4444;padding:8px 12px;'
        'border-radius:6px;color:#fca5a5;font-size:11px;margin-bottom:12px;">'
        '⚠️ Model files not found — running in demo/heuristic mode. '
        'Upload skin_disorder_model.pkl and skin_disorder_scaler.pkl for real predictions.</div>'
    )

    # Compute simple scores for display
    clinical_score = sum(feat.get(k, 0) for k in [
        "erythema", "scaling", "itching", "koebner_phenomenon", "polygonal_papules"
    ])
    histo_score = sum(feat.get(k, 0) for k in [
        "acanthosis", "parakeratosis", "melanin_incontinence", "fibrosis_of_the_papillary_dermis"
    ])

    html = f"""
    <div style="font-family:'IBM Plex Mono',monospace;padding:4px;">
      {model_note}
      <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
                  border:1px solid #334155;border-radius:16px;padding:24px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
          <span style="font-size:48px;">{pred_icon}</span>
          <div>
            <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;text-transform:uppercase;
                        margin-bottom:4px;">PREDICTED DIAGNOSIS</div>
            <div style="color:#f1f5f9;font-size:26px;font-weight:700;line-height:1.1;">{pred_name}</div>
          </div>
          <div style="margin-left:auto;text-align:right;">
            <div style="font-size:32px;font-weight:700;color:{conf_color};">{pred_conf:.1f}%</div>
            <div style="font-size:10px;color:{conf_color};letter-spacing:1px;">{conf_label}</div>
          </div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
          <span style="background:#0f172a;border:1px solid #334155;border-radius:20px;
                       padding:4px 14px;color:#94a3b8;font-size:11px;">
            🔬 Clinical Score: {clinical_score}/15
          </span>
          <span style="background:#0f172a;border:1px solid #334155;border-radius:20px;
                       padding:4px 14px;color:#94a3b8;font-size:11px;">
            🧫 Histo Score: {histo_score}/12
          </span>
          <span style="background:#0f172a;border:1px solid #334155;border-radius:20px;
                       padding:4px 14px;color:#94a3b8;font-size:11px;">
            👤 Age: {int(age)} yrs
          </span>
          <span style="background:#0f172a;border:1px solid #334155;border-radius:20px;
                       padding:4px 14px;color:#94a3b8;font-size:11px;">
            🧬 Family History: {'Yes' if int(family_history) else 'No'}
          </span>
        </div>
        <div style="background:#0f172a;border-radius:10px;padding:14px;">
          <div style="color:#7dd3fc;font-size:11px;letter-spacing:1px;margin-bottom:6px;">
            📋 CLINICAL NOTE
          </div>
          <div style="color:#cbd5e1;font-size:13px;line-height:1.6;">{clinical_note}</div>
        </div>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                  padding:20px;margin-bottom:16px;">
        <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;margin-bottom:16px;">
          📊 CLASS PROBABILITIES
        </div>
        {prob_rows}
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;">
        <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;margin-bottom:12px;">
          🚦 CLINICAL PATTERN FLAGS
        </div>
        {flags_html}
      </div>
      <div style="margin-top:14px;padding:10px 14px;background:#1e293b;border-radius:8px;
                  color:#64748b;font-size:10px;line-height:1.5;">
        ⚕️ DISCLAIMER: This tool is for clinical decision support only.
        All predictions must be validated by a qualified dermatologist.
        Model accuracy: 97.30% on UCI Dermatology Dataset (366 patients).
      </div>
    </div>
    """
    return html


def _demo_predict(feat):
    """Heuristic fallback when no model is loaded."""
    scores = np.array([0.0] * 6)
    scores[0] += feat.get("koebner_phenomenon", 0) * 1.5
    scores[0] += feat.get("acanthosis", 0) * 1.2
    scores[0] += feat.get("parakeratosis", 0) * 1.2
    scores[2] += feat.get("polygonal_papules", 0) * 1.8
    scores[2] += feat.get("oral_mucosal_involvement", 0) * 2.0
    scores[2] += feat.get("melanin_incontinence", 0) * 1.5
    scores[4] += feat.get("spongiosis", 0) * 1.5
    scores[4] += feat.get("exocytosis", 0) * 1.5
    scores[1] += feat.get("scalp_involvement", 0) * 1.3
    scores[1] += feat.get("follicular_papules", 0) * 1.3
    scores[3] += feat.get("erythema", 0) * 0.5
    scores[5] += feat.get("follicular_horn_plug", 0) * 1.5
    scores += 0.5
    probs = scores / scores.sum()
    pred_class = int(np.argmax(probs)) + 1
    return pred_class, probs, list(range(1, 7))


def load_preset(preset_name):
    presets = {
        "Psoriasis Patient": {
            "age": 42, "family_history": 1,
            "erythema": 3, "scaling": 3, "koebner_phenomenon": 2,
            "acanthosis": 2, "parakeratosis": 2,
            "knee_and_elbow_involvement": 3, "itching": 2,
        },
        "Lichen Planus Patient": {
            "age": 38, "family_history": 0,
            "polygonal_papules": 3, "oral_mucosal_involvement": 2,
            "melanin_incontinence": 2, "band_like_infiltrate_ui": 3,
            "focal_hypergranulosis": 2, "itching": 3,
        },
        "Seborrheic Dermatitis": {
            "age": 30, "family_history": 0,
            "erythema": 2, "scaling": 2, "scalp_involvement": 3,
            "follicular_papules": 2,
        },
        "Chronic Dermatitis": {
            "age": 50, "family_history": 0,
            "erythema": 2, "itching": 3,
            "spongiosis": 2, "exocytosis": 2, "acanthosis": 1,
        },
        "Clear Patient": {
            "age": 25, "family_history": 0,
        },
    }

    if preset_name not in presets:
        return [gr.update()] * (2 + len(_get_ordinal_keys()))

    p = presets[preset_name]
    ordinal_keys = _get_ordinal_keys()

    updates = [
        gr.update(value=p.get("age", 35)),
        gr.update(value=p.get("family_history", 0)),
    ]
    for key in ordinal_keys:
        updates.append(gr.update(value=p.get(key, 0)))
    return updates


# ──────────────────────────────────────────────
# 4. UI
# ──────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
* { box-sizing: border-box; }
body, .gradio-container {
    background: #020617 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
.tab-nav button {
    background: #0f172a !important; color: #64748b !important;
    border: 1px solid #1e293b !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; letter-spacing: 1px !important; border-radius: 6px !important;
}
.tab-nav button.selected {
    background: #1e3a5f !important; color: #7dd3fc !important; border-color: #3b82f6 !important;
}
input[type=range] { accent-color: #3b82f6; }
input[type=number], .gr-number input {
    background: #0f172a !important; border: 1px solid #334155 !important;
    color: #e2e8f0 !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.gr-form > label, label.svelte-1hnfib2 {
    color: #94a3b8 !important; font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    border: none !important; font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
    border-radius: 8px !important; color: #fff !important;
    padding: 12px 28px !important; font-size: 13px !important; cursor: pointer !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
    transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.4) !important;
}
.gr-button-secondary {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #94a3b8 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 1px !important;
    border-radius: 6px !important; cursor: pointer !important;
}
.gr-button-secondary:hover { border-color: #3b82f6 !important; color: #7dd3fc !important; }
.gr-accordion { background: #0f172a !important; border: 1px solid #1e293b !important; border-radius: 10px !important; }
select {
    background: #0f172a !important; border: 1px solid #334155 !important;
    color: #e2e8f0 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; border-radius: 6px !important;
}
.gr-panel, .gr-box { background: #0f172a !important; border: 1px solid #1e293b !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
"""

HEADER_HTML = """
<div style="font-family:'IBM Plex Sans',sans-serif;padding:20px 0 10px;">
  <div style="display:flex;align-items:center;gap:20px;">
    <div style="font-size:52px;">🔬</div>
    <div>
      <div style="color:#3b82f6;font-size:11px;letter-spacing:3px;font-family:'IBM Plex Mono',monospace;
                  text-transform:uppercase;margin-bottom:4px;">Clinical Decision Support System</div>
      <div style="color:#f1f5f9;font-size:28px;font-weight:700;line-height:1.1;">
        Skin Disorder Prediction
      </div>
      <div style="color:#64748b;font-size:13px;margin-top:4px;">
        Erythemato-squamous Disease Classification · 6 Classes · Random Forest · 97.30% Accuracy
      </div>
    </div>
    <div style="margin-left:auto;display:flex;gap:24px;text-align:center;">
      <div>
        <div style="color:#22c55e;font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;">97.3%</div>
        <div style="color:#64748b;font-size:10px;letter-spacing:1px;">ACCURACY</div>
      </div>
      <div>
        <div style="color:#3b82f6;font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;">0.969</div>
        <div style="color:#64748b;font-size:10px;letter-spacing:1px;">F1 MACRO</div>
      </div>
      <div>
        <div style="color:#a78bfa;font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;">34</div>
        <div style="color:#64748b;font-size:10px;letter-spacing:1px;">FEATURES</div>
      </div>
      <div>
        <div style="color:#fb923c;font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;">6</div>
        <div style="color:#64748b;font-size:10px;letter-spacing:1px;">CLASSES</div>
      </div>
    </div>
  </div>
</div>
"""

INFO_HTML = """
<div style="font-family:'IBM Plex Sans',sans-serif;">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;">
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;">
      <div style="font-size:22px;margin-bottom:8px;">🔴</div>
      <div style="color:#f1f5f9;font-weight:600;margin-bottom:4px;">Psoriasis</div>
      <div style="color:#64748b;font-size:12px;">Koebner · Acanthosis · Parakeratosis · Extensor</div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;">
      <div style="font-size:22px;margin-bottom:8px;">🟠</div>
      <div style="color:#f1f5f9;font-weight:600;margin-bottom:4px;">Seborrheic Dermatitis</div>
      <div style="color:#64748b;font-size:12px;">Scalp · Follicular papules · Greasy scales</div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;">
      <div style="font-size:22px;margin-bottom:8px;">🟣</div>
      <div style="color:#f1f5f9;font-weight:600;margin-bottom:4px;">Lichen Planus</div>
      <div style="color:#64748b;font-size:12px;">Polygonal papules · Oral mucosa · Melanin incontinence</div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;">
      <div style="font-size:22px;margin-bottom:8px;">🟡</div>
      <div style="color:#f1f5f9;font-weight:600;margin-bottom:4px;">Pityriasis Rosea</div>
      <div style="color:#64748b;font-size:12px;">Herald patch · Christmas tree rash · Self-limiting</div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;">
      <div style="font-size:22px;margin-bottom:8px;">🟤</div>
      <div style="color:#f1f5f9;font-weight:600;margin-bottom:4px;">Chronic Dermatitis</div>
      <div style="color:#64748b;font-size:12px;">Spongiosis · Exocytosis · Prolonged exposure</div>
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;">
      <div style="font-size:22px;margin-bottom:8px;">🔵</div>
      <div style="color:#f1f5f9;font-weight:600;margin-bottom:4px;">Pityriasis Rubra Pilaris</div>
      <div style="color:#64748b;font-size:12px;">Orange keratoderma · Follicular plugging · Rare</div>
    </div>
  </div>
  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;">
    <div style="color:#7dd3fc;font-size:12px;letter-spacing:2px;margin-bottom:12px;">ℹ️ HOW TO USE</div>
    <ol style="color:#cbd5e1;font-size:13px;line-height:2;padding-left:20px;">
      <li>Select a <strong style="color:#f1f5f9;">demo preset</strong> or manually enter patient values</li>
      <li>Use the <strong style="color:#f1f5f9;">0–3 scale</strong> for all clinical/histopathological features (0=absent, 3=severe)</li>
      <li>Click <strong style="color:#3b82f6;">ANALYSE & PREDICT</strong></li>
      <li>Review the diagnosis card, confidence score, probability breakdown, and clinical flags</li>
      <li>Always validate AI output with physician assessment</li>
    </ol>
    <div style="margin-top:12px;padding:10px;background:#1e293b;border-radius:8px;color:#64748b;font-size:11px;">
      Upload <code>skin_disorder_model.pkl</code> and <code>skin_disorder_scaler.pkl</code>
      (from your notebook output) to the same folder as app.py to enable live model inference.
    </div>
  </div>
</div>
"""


def build_app():
    clinical_feature_keys = [
        "erythema", "scaling", "definite_borders", "itching",
        "koebner_phenomenon", "polygonal_papules", "follicular_papules",
        "oral_mucosal_involvement", "knee_and_elbow_involvement", "scalp_involvement",
    ]
    histo_feature_keys = [
        "inflammatory_monoluclear_inflitrate", "band_like_infiltrate_ui",
        "PNL_infiltrate", "fibrosis_of_the_papillary_dermis",
        "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis",
        "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges",
        "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule",
        "munro_microabcess", "focal_hypergranulosis",
        "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer",
        "spongiosis", "saw_tooth_appearance_of_retes_ui",
        "follicular_horn_plug", "perifollicular_parakeratosis",
        "melanin_incontinence", "eosinophils_in_the_infiltrate",
    ]

    # Flatten all ordinal features for lookup
    all_ordinal = [(t[0], t[1], t[2], t[3], t[4], t[5])
                   for grp, items in FEATURE_GROUPS.items()
                   for t in items if t[1] not in ("age_val", "family_history")]
    key_to_spec = {t[1]: t for t in all_ordinal}

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Skin Disorder Prediction | Clinical AI",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("IBM Plex Sans"),
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            # ── TAB 1: PREDICT ──
            with gr.Tab("🩺 PREDICT"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.HTML("""<div style="color:#94a3b8;font-size:11px;letter-spacing:2px;
                                        margin-bottom:8px;font-family:'IBM Plex Mono',monospace;">
                                        ⚡ QUICK LOAD DEMO PRESET</div>""")
                            preset_dd = gr.Dropdown(
                                choices=["Psoriasis Patient", "Lichen Planus Patient",
                                         "Seborrheic Dermatitis", "Chronic Dermatitis", "Clear Patient"],
                                label="Demo Preset", value=None, interactive=True,
                                info="Loads example patient values — then click Predict",
                            )

                        with gr.Accordion("👤 Patient Information", open=True):
                            age_sl = gr.Slider(0, 90, value=35, step=1, label="Age (years)")
                            fh_sl  = gr.Slider(0, 1,  value=0,  step=1, label="Family History (0=No, 1=Yes)")

                        # Clinical sliders
                        clinical_sliders = []
                        with gr.Accordion("🔴 Clinical Features (0–3)", open=True):
                            with gr.Row():
                                with gr.Column():
                                    for key in clinical_feature_keys[:5]:
                                        spec = key_to_spec[key]
                                        s = gr.Slider(spec[3], spec[4], value=spec[5], step=1,
                                                      label=spec[0], info=spec[2])
                                        clinical_sliders.append((key, s))
                                with gr.Column():
                                    for key in clinical_feature_keys[5:]:
                                        spec = key_to_spec[key]
                                        s = gr.Slider(spec[3], spec[4], value=spec[5], step=1,
                                                      label=spec[0], info=spec[2])
                                        clinical_sliders.append((key, s))

                        # Histo sliders
                        histo_sliders = []
                        with gr.Accordion("🧫 Histopathological Features (0–3)", open=False):
                            with gr.Row():
                                with gr.Column():
                                    for key in histo_feature_keys[:11]:
                                        spec = key_to_spec[key]
                                        s = gr.Slider(spec[3], spec[4], value=spec[5], step=1,
                                                      label=spec[0], info=spec[2])
                                        histo_sliders.append((key, s))
                                with gr.Column():
                                    for key in histo_feature_keys[11:]:
                                        spec = key_to_spec[key]
                                        s = gr.Slider(spec[3], spec[4], value=spec[5], step=1,
                                                      label=spec[0], info=spec[2])
                                        histo_sliders.append((key, s))

                        predict_btn = gr.Button("🔬  ANALYSE & PREDICT", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.HTML("""<div style="color:#94a3b8;font-size:11px;letter-spacing:2px;
                                    margin-bottom:12px;font-family:'IBM Plex Mono',monospace;">
                                    📋 PREDICTION RESULT</div>""")
                        result_html = gr.HTML(
                            value="""<div style="background:#0f172a;border:1px dashed #334155;
                                        border-radius:16px;padding:40px;text-align:center;">
                                      <div style="font-size:48px;margin-bottom:16px;">🩺</div>
                                      <div style="color:#64748b;font-size:14px;
                                                  font-family:'IBM Plex Mono',monospace;">
                                        Fill in patient data and click<br>ANALYSE & PREDICT
                                      </div></div>""",
                            label="",
                        )

                # Build ordered slider list matching _get_ordinal_keys() order
                key_to_slider = {k: s for k, s in clinical_sliders + histo_sliders}
                all_sliders_ordered = [key_to_slider[k] for k in _get_ordinal_keys() if k in key_to_slider]

                predict_btn.click(
                    fn=predict,
                    inputs=[age_sl, fh_sl] + all_sliders_ordered,
                    outputs=result_html,
                )
                preset_dd.change(
                    fn=load_preset,
                    inputs=[preset_dd],
                    outputs=[age_sl, fh_sl] + all_sliders_ordered,
                )

            # ── TAB 2: DISEASE GUIDE ──
            with gr.Tab("📖 DISEASE GUIDE"):
                gr.HTML(INFO_HTML)

            # ── TAB 3: MODEL PERFORMANCE ──
            with gr.Tab("📊 MODEL PERFORMANCE"):
                gr.HTML("""
                <div style="font-family:'IBM Plex Sans',sans-serif;">
                  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                              padding:24px;margin-bottom:16px;">
                    <div style="color:#7dd3fc;font-size:12px;letter-spacing:2px;margin-bottom:16px;">
                      🏆 MODEL COMPARISON TABLE
                    </div>
                    <table style="width:100%;border-collapse:collapse;font-size:13px;">
                      <thead>
                        <tr style="border-bottom:1px solid #334155;">
                          <th style="color:#94a3b8;text-align:left;padding:8px 12px;">Model</th>
                          <th style="color:#94a3b8;text-align:center;padding:8px 12px;">Type</th>
                          <th style="color:#94a3b8;text-align:center;padding:8px 12px;">Accuracy</th>
                          <th style="color:#94a3b8;text-align:center;padding:8px 12px;">F1 Macro</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style="background:#1e3a5f;border-bottom:1px solid #334155;">
                          <td style="color:#22c55e;padding:10px 12px;font-weight:700;">✅ Random Forest (Tuned) — Production</td>
                          <td style="color:#94a3b8;text-align:center;padding:10px 12px;">ML</td>
                          <td style="color:#22c55e;text-align:center;font-weight:700;padding:10px 12px;">0.9730</td>
                          <td style="color:#22c55e;text-align:center;font-weight:700;padding:10px 12px;">0.9694</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1e293b;">
                          <td style="color:#cbd5e1;padding:8px 12px;">Logistic Regression (Baseline)</td>
                          <td style="color:#64748b;text-align:center;">ML</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9595</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9574</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1e293b;">
                          <td style="color:#cbd5e1;padding:8px 12px;">Random Forest (Baseline)</td>
                          <td style="color:#64748b;text-align:center;">ML</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9595</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9545</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1e293b;">
                          <td style="color:#cbd5e1;padding:8px 12px;">Decision Tree (Tuned)</td>
                          <td style="color:#64748b;text-align:center;">ML</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9324</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9245</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1e293b;">
                          <td style="color:#cbd5e1;padding:8px 12px;">XGBoost (Tuned)</td>
                          <td style="color:#64748b;text-align:center;">ML</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9324</td>
                          <td style="color:#cbd5e1;text-align:center;">0.9257</td>
                        </tr>
                        <tr style="border-bottom:1px solid #1e293b;">
                          <td style="color:#a78bfa;padding:8px 12px;">1D CNN (Deep Learning)</td>
                          <td style="color:#a78bfa;text-align:center;">DL</td>
                          <td style="color:#a78bfa;text-align:center;">0.9324</td>
                          <td style="color:#a78bfa;text-align:center;">0.8837</td>
                        </tr>
                        <tr>
                          <td style="color:#a78bfa;padding:8px 12px;">2D CNN Heatmap</td>
                          <td style="color:#a78bfa;text-align:center;">DL</td>
                          <td style="color:#a78bfa;text-align:center;">0.3108</td>
                          <td style="color:#a78bfa;text-align:center;">0.0790</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;">
                      <div style="color:#7dd3fc;font-size:12px;letter-spacing:2px;margin-bottom:12px;">🎯 TOP PREDICTIVE FEATURES</div>
                      <ol style="color:#cbd5e1;font-size:13px;line-height:2.2;padding-left:16px;">
                        <li><strong style="color:#f1f5f9;">melanin_incontinence</strong> — Lichen Planus key marker</li>
                        <li><strong style="color:#f1f5f9;">band-like_infiltrate</strong> — Lichen Planus histology</li>
                        <li><strong style="color:#f1f5f9;">koebner_phenomenon</strong> — Psoriasis discriminator</li>
                        <li><strong style="color:#f1f5f9;">acanthosis</strong> — Psoriasis / Chronic Dermatitis</li>
                        <li><strong style="color:#f1f5f9;">parakeratosis</strong> — Psoriasis biopsy marker</li>
                        <li><strong style="color:#f1f5f9;">polygonal_papules</strong> — Lichen Planus clinical sign</li>
                        <li><strong style="color:#f1f5f9;">oral_mucosal_involvement</strong> — Lichen Planus specific</li>
                        <li><strong style="color:#f1f5f9;">Age</strong> — Contextual modifier</li>
                      </ol>
                    </div>
                    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;">
                      <div style="color:#7dd3fc;font-size:12px;letter-spacing:2px;margin-bottom:12px;">📋 CROSS-VALIDATION RESULTS (RF Tuned)</div>
                      <div style="color:#cbd5e1;font-size:13px;line-height:2;">
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #1e293b;padding:6px 0;">
                          <span>Test Accuracy</span><span style="color:#22c55e;font-weight:700;">97.30%</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #1e293b;padding:6px 0;">
                          <span>Test F1 Macro</span><span style="color:#22c55e;font-weight:700;">0.9694</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #1e293b;padding:6px 0;">
                          <span>Classes with perfect recall</span><span style="color:#22c55e;font-weight:700;">4 / 6</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #1e293b;padding:6px 0;">
                          <span>Dataset size</span><span style="color:#94a3b8;">366 patients</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;padding:6px 0;">
                          <span>Features</span><span style="color:#94a3b8;">34 clinical + histopathological</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                """)

    return demo


# ──────────────────────────────────────────────
# 5. LAUNCH
# ──────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )