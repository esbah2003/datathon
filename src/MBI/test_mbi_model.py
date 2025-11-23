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

# Define test samples with all 12 features
samples = [
    {
        "name": "High-Risk Student (Severe Burnout)",
        "description": "Heavy workload, poor sleep, high anxiety/depression, minimal support",
        "study_hours": 10.5,
        "screen_time": 9.0,
        "stress_level": 9,
        "physical_activity": 2.0,
        "number_of_courses": 7,
        "sleep_hours": 5.0,
        "sleep_quality": 1,
        "anxiety_score": 9,
        "depression_score": 8,
        "coping_strategy_score": 3,
        "social_support_score": 2,
        "academic_self_efficacy": 3,
    },
    {
        "name": "Moderate-Risk Student",
        "description": "Balanced workload, moderate stress, average support",
        "study_hours": 6.0,
        "screen_time": 6.0,
        "stress_level": 6,
        "physical_activity": 7.0,
        "number_of_courses": 5,
        "sleep_hours": 7.0,
        "sleep_quality": 3,
        "anxiety_score": 5,
        "depression_score": 5,
        "coping_strategy_score": 6,
        "social_support_score": 6,
        "academic_self_efficacy": 6,
    },
    {
        "name": "Low-Risk Student (Thriving)",
        "description": "Light workload, good sleep, low stress, strong support",
        "study_hours": 4.0,
        "screen_time": 3.0,
        "stress_level": 2,
        "physical_activity": 12.0,
        "number_of_courses": 3,
        "sleep_hours": 8.5,
        "sleep_quality": 4,
        "anxiety_score": 2,
        "depression_score": 1,
        "coping_strategy_score": 9,
        "social_support_score": 9,
        "academic_self_efficacy": 8,
    },
    {
        "name": "Medical Student (Extreme Workload)",
        "description": "Extreme workload, poor sleep, high mental health symptoms",
        "study_hours": 12.0,
        "screen_time": 10.0,
        "stress_level": 10,
        "physical_activity": 1.0,
        "number_of_courses": 8,
        "sleep_hours": 4.5,
        "sleep_quality": 1,
        "anxiety_score": 10,
        "depression_score": 9,
        "coping_strategy_score": 4,
        "social_support_score": 5,
        "academic_self_efficacy": 5,
    },
    {
        "name": "Well-Supported Student",
        "description": "Moderate workload but excellent support systems",
        "study_hours": 7.0,
        "screen_time": 5.0,
        "stress_level": 5,
        "physical_activity": 10.0,
        "number_of_courses": 5,
        "sleep_hours": 8.0,
        "sleep_quality": 3,
        "anxiety_score": 4,
        "depression_score": 3,
        "coping_strategy_score": 8,
        "social_support_score": 9,
        "academic_self_efficacy": 8,
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
        "sleep_hours": sample["sleep_hours"],
        "sleep_quality": sample["sleep_quality"],
        "anxiety_score": sample["anxiety_score"],
        "depression_score": sample["depression_score"],
        "coping_strategy_score": sample["coping_strategy_score"],
        "social_support_score": sample["social_support_score"],
        "academic_self_efficacy": sample["academic_self_efficacy"],
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
    print(f"   📚 Academic & Lifestyle:")
    print(f"      Study Hours: {sample['study_hours']:.1f} hrs/day")
    print(f"      Screen Time: {sample['screen_time']:.1f} hrs/day")
    print(f"      Courses: {sample['number_of_courses']}")
    print(f"   💤 Sleep & Rest:")
    print(f"      Sleep Hours: {sample['sleep_hours']:.1f} hrs/night")
    print(f"      Sleep Quality: {sample['sleep_quality']}/4")
    print(f"   🧠 Mental Health:")
    print(f"      Stress: {sample['stress_level']}/10")
    print(f"      Anxiety: {sample['anxiety_score']}/10")
    print(f"      Depression: {sample['depression_score']}/10")
    print(f"   🤝 Support & Coping:")
    print(f"      Physical Activity: {sample['physical_activity']:.1f} hrs/week")
    print(f"      Coping Strategies: {sample['coping_strategy_score']}/10")
    print(f"      Social Support: {sample['social_support_score']}/10")
    print(f"      Academic Confidence: {sample['academic_self_efficacy']}/10")
    print(f"   ─────────────────────────────────────────────")
    print(f"   🎯 MBI Score:\t{score:.1f} / 54")
    print(f"   📊 Risk Category:\t{category}")
    
    # Add interpretation
    if category == "Low Risk":
        print(f"   ✅ Interpretation: Healthy balance, low burnout symptoms")
    elif category == "Moderate Risk":
        print(f"   ⚠️  Interpretation: Some burnout symptoms, monitor closely")
    else:
        print(f"   🚨 Interpretation: High burnout risk, intervention recommended")

print("\n" + "-" * 70)
print("NOTE: MBI scores are based on validated Maslach Burnout Inventory thresholds")
print("-" * 70)
