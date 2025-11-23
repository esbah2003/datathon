"""
Test the MBI burnout prediction model using fixed MBI thresholds.
Uses Maslach Burnout Inventory for Students (MBI-SS) standard thresholds:
- Low burnout: 0-16
- Moderate burnout: 17-26
- High burnout: 27-54
"""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

here = Path(__file__).parent
model_path = here / "mbi_model.joblib"

if not model_path.exists():
    raise SystemExit(
        "runtrain_mbi_model.py to train and save the model first."
    )

bundle = joblib.load(model_path)
pipe = bundle["pipeline"]
thresholds = bundle["mbi_thresholds"]

print("-" * 70)
print("MBI BURNOUT PREDICTION TEST")
print("-" * 70)
print("\nUsing Maslach Burnout Inventory for Students (MBI-SS) Thresholds:")
print(f"  Low burnout: 0 - {thresholds['low']}")
print(f"  Moderate burnout: {thresholds['low'] + 1} - {thresholds['moderate']}")
print(f"  High burnout:{thresholds['moderate'] + 1} - {thresholds['high']}")
print("\nModel Performance Metrics:")
print(f"  MAE: {bundle['metrics']['mae']:.2f} points")
print(f"  RMSE: {bundle['metrics']['rmse']:.2f} points")
print(f"  R² Score: {bundle['metrics']['r2']:.4f}")
print(f"  Category Accuracy: {bundle['metrics']['category_accuracy']:.1%}")
print("-" * 70)

# Define test samples
samples = [
    {
        "name": "High-Risk Student",
        "description": "Heavy workload, high stress, minimal activity",
        "study_hours": 10.5,
        "screen_time": 9.0,
        "stress_level": 9,
        "physical_activity": 2.0,
        "number_of_courses": 7,
    },
    {
        "name": "Moderate-Risk Student",
        "description": "Balanced workload, moderate stress, some activity",
        "study_hours": 6.0,
        "screen_time": 6.0,
        "stress_level": 6,
        "physical_activity": 7.0,
        "number_of_courses": 5,
    },
    {
        "name": "Low-Risk Student",
        "description": "Light workload, low stress, high activity",
        "study_hours": 2.0,
        "screen_time": 2.0,
        "stress_level": 2,
        "physical_activity": 12.0,
        "number_of_courses": 2,
    },
    {
        "name": "Medical Student",
        "description": "Extreme workload with high stress",
        "study_hours": 12.0,
        "screen_time": 10.0,
        "stress_level": 10,
        "physical_activity": 1.0,
        "number_of_courses": 8,
    },
    {
        "name": "Part-Time Student",
        "description": "Minimal course load, good balance",
        "study_hours": 3.0,
        "screen_time": 4.0,
        "stress_level": 3,
        "physical_activity": 10.0,
        "number_of_courses": 3,
    },
]

def get_mbi_category(score, thresholds):
    """Categorize MBI score using official thresholds."""
    if score <= thresholds['low']:
        return "Low Risk"
    elif score <= thresholds['moderate']:
        return "Moderate Risk"
    else:
        return "High Risk"

feature_data = []
for sample in samples:
    feature_data.append({
        "study_hours": sample["study_hours"],
        "screen_time": sample["screen_time"],
        "stress_level": sample["stress_level"],
        "physical_activity": sample["physical_activity"],
        "number_of_courses": sample["number_of_courses"],
    })

df = pd.DataFrame(feature_data)

predictions = pipe.predict(df)
predictions = np.clip(predictions, 0, 54)

# Display results
print("\nPREDICTIONS:")
print("-" * 70)

for i, (sample, score) in enumerate(zip(samples, predictions)):
    category = get_mbi_category(score, thresholds)
    
    print(f"\n{i+1}. {sample['name']}")
    print(f"   {sample['description']}")
    print(f"   ─────────────────────────────────────────────")
    print(f"   Study Hours:\t{sample['study_hours']:.1f} hrs/day")
    print(f"   Screen Time:\t{sample['screen_time']:.1f} hrs/day")
    print(f"   Stress Level:\t{sample['stress_level']}/10")
    print(f"   Physical Activity:\t{sample['physical_activity']:.1f} hrs/week")
    print(f"   Number of Courses:\t{sample['number_of_courses']}")
    print(f"   ─────────────────────────────────────────────")
    print(f"   MBI Score:\t{score:.1f} / 54")
    print(f"   Risk Category:\t{category}")
    
    # Add interpretation (replace with gemini)
    if category == "Low Risk":
        print(f"   Interpretation: Healthy balance, low burnout symptoms")
    elif category == "Moderate Risk":
        print(f"   Interpretation: Some burnout symptoms, monitor closely")
    else:
        print(f"   Interpretation: High burnout risk, intervention recommended")

print("\n" + "-" * 70)
print("NOTE: MBI scores are based on validated Maslach Burnout Inventory thresholds")
print("-" * 70)
