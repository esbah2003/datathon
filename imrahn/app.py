import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Optional Gemini import (guarded)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load model bundle
here = Path(__file__).parent
bundle = joblib.load(here / "model.joblib")
pipe = bundle["pipeline"]
feature_names = bundle["feature_names"]

def classify(score: float) -> str:
    if score < 0.33: return "Low"
    if score < 0.66: return "Medium"
    return "High"

def generate_support_message(score: float, features: dict, context: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_AVAILABLE or not api_key:
        return "Gemini support message unavailable (missing library or API key)."
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are an academic wellness assistant.
Burnout score: {score:.2f}
Risk label: {classify(score)}
Features: {features}
User context: "{context}"
Tasks:
1. Interpret the burnout score.
2. Give 2 drivers from the inputs.
3. Provide 3 actionable, realistic recommendations.
4. Keep under 120 words. Warm, concise, non-judgmental.
"""
    resp = client.models.generate_text(model="gemini-pro", prompt=prompt)
    return (resp.text or "").strip()

def explain_importance(importances, features_dict):
    pairs = list(zip(feature_names, importances))
    top = sorted(pairs, key=lambda x: x[1], reverse=True)[:3]
    lines = [f"- {n}: contributed more (value={features_dict[n]})" for n, _ in top]
    return "Key contributors:\n" + "\n".join(lines)

st.title("Student Burnout Early-Intervention Assistant")
st.caption("Predict risk and receive supportive guidance.")

cols = st.columns(5)
study_hours = cols[0].number_input("Study hours (daily)", 0.0, 12.0, 4.0)
screen_time = cols[1].number_input("Screen time (daily)", 0.0, 12.0, 5.0)
stress_level = cols[2].slider("Stress level (1-10)", 1, 10, 5)
physical_activity = cols[3].number_input("Physical activity (hrs/week)", 0.0, 14.0, 3.0)
number_of_courses = cols[4].number_input("Course load", 1, 8, 4)

context = st.text_area("Optional: Add context about workload or feelings")

if st.button("Evaluate Burnout Risk"):
    row = {
        "study_hours": study_hours,
        "screen_time": screen_time,
        "stress_level": stress_level,
        "physical_activity": physical_activity,
        "number_of_courses": number_of_courses,
    }
    df = pd.DataFrame([row])
    score = float(pipe.predict(df)[0])
    label = classify(score)

    st.subheader(f"Burnout Risk: {label} ({score:.2f})")

    # Feature importance
    try:
        importances = pipe.named_steps["model"].feature_importances_
        st.markdown("### Interpretation")
        st.write(explain_importance(importances, row))
    except Exception:
        st.write("Importance unavailable.")

    # Gemini message
    st.markdown("### Support Message")
    support = generate_support_message(score, row, context)
    st.write(support)

    st.markdown("### Raw Inputs")
    st.json(row)

st.divider()
st.caption("Set GEMINI_API_KEY in environment to enable AI guidance.")