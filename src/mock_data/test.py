from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


here = Path(__file__).parent
model_path = here / "../model/model.joblib"
model_path = model_path.resolve()


if not model_path.exists():
    raise SystemExit("Model file not found. Run main.py to train and save the model first.")

bundle = joblib.load(model_path)
pipe = bundle["pipeline"]

# sample inputs
samples = [
    # Heavier workload, high stress, low activity: higher risk expected
    {
        "study_hours": 10.5,
        "screen_time": 9.0,
        "stress_level": 9,
        "physical_activity": 2.0,
        "number_of_courses": 7,
    },
    # Moderate workload, moderate stress, good activity: moderate risk
    {
        "study_hours": 6.0,
        "screen_time": 6.0,
        "stress_level": 6,
        "physical_activity": 7.0,
        "number_of_courses": 5,
    },
    # Light workload, low stress, high activity: low risk
    {
        "study_hours": 2.0,
        "screen_time": 2.0,
        "stress_level": 2,
        "physical_activity": 12.0,
        "number_of_courses": 2,
    },
]

df = pd.DataFrame(samples)
preds = pipe.predict(df)

#Example (Maslach Burnout Inventory commonly used thresholds):
#Low: bottom 25% of scores
#Moderate: middle 50%
#High: top 25%

low = np.percentile(preds, 25)
high = np.percentile(preds, 75)

def risk_category(score):
    if score < low:
        return "Low Risk"
    elif score < high:
        return "Medium Risk"
    else:
        return "High Risk"

for i, p in enumerate(preds):
    category = risk_category(p)
    print(f"Sample {i+1} prediction (burnout_risk): {category} (score: {p:.2f})")



