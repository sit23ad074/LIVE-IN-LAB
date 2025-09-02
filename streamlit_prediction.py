# app.py
# ------------------------------------------------------------
# Cattle Disease Prediction System (Full-Feature Build with Unicode Fix)
# - Model choice (RF, DT, LR, SVM)
# - Evaluate all models
# - Predict with Top-3 probabilities
# - Show image for Top-1 disease + embed in PDF
# - PDF now uses NotoSans (full Unicode support)
# - Cleaned symptoms to avoid black squares
# ------------------------------------------------------------

# app.py
# ------------------------------------------------------------
# Cattle Disease Prediction System (Polished UI Version)
# - Sidebar inputs for model/symptoms/vitals
# - Tabs for results, report, and about
# - Removed model evaluation code
# - Styled UI for professional look
# ------------------------------------------------------------

import os
import io
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ReportLab (PDF)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ------------------------------------------------------------
# App Config
# ------------------------------------------------------------
st.set_page_config(page_title="Cattle Disease Prediction", layout="wide")
DATA_PATH = "Training.csv"
IMAGES_DIR = "disease_images"
UNICODE_FONT_TTF = "NotoSans-Regular.ttf"

# ------------------------------------------------------------
# Fonts: Register Unicode Font
# ------------------------------------------------------------
def register_unicode_font(ttf_path: str) -> str:
    try:
        if os.path.exists(ttf_path):
            pdfmetrics.registerFont(TTFont("NotoSans", ttf_path))
            return "NotoSans"
        else:
            return "Helvetica"
    except Exception:
        return "Helvetica"

PDF_FONT = register_unicode_font(UNICODE_FONT_TTF)

# ------------------------------------------------------------
# Data Loading & Preparation
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    return df

@st.cache_data(show_spinner=False)
def split_and_encode(df: pd.DataFrame):
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    return X_train, X_test, y_train, y_test, le, list(X.columns)

def get_model_dict():
    return {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def make_input_vector(symptom_columns: List[str], selected: List[str]) -> pd.DataFrame:
    bits = [1 if c in selected else 0 for c in symptom_columns]
    return pd.DataFrame([bits], columns=symptom_columns)

def scale_image_for_pdf(img_path: str, max_w: int = 350, max_h: int = 260) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        ratio = min(max_w / w, max_h / h)
        return int(w * ratio), int(h * ratio)
    except Exception:
        return None

def get_top3_from_probs(probs: np.ndarray, label_encoder: LabelEncoder) -> List[Tuple[str, float]]:
    top_idx = np.argsort(probs)[-3:][::-1]
    results: List[Tuple[str, float]] = []
    for idx in top_idx:
        disease_name = label_encoder.inverse_transform([idx])[0]
        results.append((disease_name, probs[idx] * 100.0))
    return results

def clean_text_list(items: List[str]) -> List[str]:
    cleaned = []
    for s in items:
        if not isinstance(s, str):
            s = str(s)
        s = s.encode("utf-8", "ignore").decode("utf-8")
        cleaned.append(s)
    return cleaned

# ------------------------------------------------------------
# PDF Builder
# ------------------------------------------------------------
def build_pdf(model_name: str, model_acc: float, vitals: Dict[str, str],
              symptoms: List[str], top3: List[Tuple[str, float]], top_image_path: Optional[str] = None) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)

    styles = getSampleStyleSheet()
    for style_name in styles.byName:
        styles[style_name].fontName = PDF_FONT

    styles.add(
        ParagraphStyle(
            name="Section",
            fontName=PDF_FONT,
            fontSize=13,
            leading=16,
            spaceAfter=8,
            textColor=colors.HexColor("#1F3B4D"),
        )
    )

    elements = []
    elements.append(Paragraph("Cattle Disease Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 14))

    if top_image_path and os.path.exists(top_image_path):
        dims = scale_image_for_pdf(top_image_path)
        if dims:
            elements.append(RLImage(top_image_path, width=dims[0], height=dims[1]))
            elements.append(Spacer(1, 12))

    elements.append(Paragraph("Model Information", styles["Section"]))
    elements.append(Paragraph(f"Model Used: {model_name}", styles["Normal"]))
    elements.append(Paragraph(f"Accuracy (test): {model_acc * 100:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Vital Stats", styles["Section"]))
    if vitals:
        for k, v in vitals.items():
            elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    else:
        elements.append(Paragraph("None provided.", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Predictions (Top 3)", styles["Section"]))
    data = [["Disease", "Probability (%)"]]
    for d, p in top3:
        data.append([d, f"{p:.2f}%"])
    table = Table(data, hAlign="LEFT", colWidths=[260, 140])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F618D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), PDF_FONT),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7F9F9")]),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Selected Symptoms", styles["Section"]))
    cleaned_symptoms = clean_text_list(symptoms)
    if cleaned_symptoms:
        for s in cleaned_symptoms:
            elements.append(Paragraph(f"- {s}", styles["Normal"]))
    else:
        elements.append(Paragraph("None selected.", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Italic"]))

    doc.build(elements)
    buf.seek(0)
    return buf

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #1F618D;'>🐄 Cattle Disease Prediction System</h1>", unsafe_allow_html=True)

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find dataset: {DATA_PATH}. Place 'Training.csv' next to this file.")
    st.stop()

df = load_dataset(DATA_PATH)
if "prognosis" not in df.columns:
    st.error("The dataset must contain a column named 'prognosis' as the label.")
    st.stop()

X_train, X_test, y_train, y_test, label_encoder, symptom_columns = split_and_encode(df)

# Sidebar Inputs
st.sidebar.header("⚙️ Model & Inputs")

selected_symptoms = st.sidebar.multiselect("Select Symptoms:", symptom_columns)

st.sidebar.subheader("Vital Stats (for report)")
temperature = st.sidebar.number_input("Temperature (°C)", 30.0, 45.0, value=38.5, step=0.1)
weight = st.sidebar.number_input("Weight (Kg)", 50.0, 800.0, value=250.0, step=1.0)
pulse_rate = st.sidebar.number_input("Pulse Rate (bpm)", 40.0, 180.0, value=70.0, step=1.0)

model_choice = st.sidebar.selectbox("Choose Model:", list(get_model_dict().keys()))
predict_btn = st.sidebar.button("🚀 Predict", use_container_width=True)

# Main Tabs
tabs = st.tabs(["🔍 Results", "📑 Report", "ℹ️ About"])

if predict_btn:
    if len(selected_symptoms) == 0:
        st.error("Please select at least one symptom from the sidebar.")
    else:
        models = get_model_dict()
        model = models[model_choice]
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        input_df = make_input_vector(symptom_columns, selected_symptoms)
        probs = model.predict_proba(input_df)[0]
        top3 = get_top3_from_probs(probs, label_encoder)
        top_disease = top3[0][0]

        # Results Tab
        with tabs[0]:
            st.success(f"Most likely disease: **{top_disease} ({top3[0][1]:.2f}%)**")
            st.metric("Model Accuracy", f"{test_acc*100:.2f}%")

            prob_df = pd.DataFrame({"Disease": [d for d, _ in top3], "Probability (%)": [p for _, p in top3]})
            st.subheader("Prediction Probabilities")
            st.bar_chart(prob_df.set_index("Disease"))

            img_path = os.path.join(IMAGES_DIR, f"{top_disease}.jpg")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=top_disease, use_container_width=True)
            else:
                st.warning(f"No image found for: {top_disease}")

        # Report Tab
        with tabs[1]:
            vitals_block = {
                "Temperature": f"{temperature} °C",
                "Weight": f"{weight} Kg",
                "Pulse Rate": f"{pulse_rate} bpm",
            }
            pdf_buffer = build_pdf(
                model_name=model_choice,
                model_acc=test_acc,
                vitals=vitals_block,
                symptoms=selected_symptoms,
                top3=top3,
                top_image_path=img_path if os.path.exists(img_path) else None,
            )
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_buffer,
                file_name=f"Cattle_Disease_Report_{top_disease}.pdf",
                mime="application/pdf",
                type="primary",
            )

# About Tab
with tabs[2]:
    st.markdown("""
    ### About this App  
    This system predicts **cattle diseases** using ML models trained on symptoms.  
    - Select symptoms from the sidebar  
    - Choose a model  
    - Enter vitals for report  
    - Get predictions + PDF report  

    **Tip**: Place cattle disease images inside `disease_images/` with filenames matching disease labels.  
    Place `NotoSans-Regular.ttf` in the project folder for Unicode PDF support.
    """)

# Footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
