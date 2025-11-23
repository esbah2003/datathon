import joblib
import numpy as np
import pandas as pd

# Load the saved model and encoders
model = joblib.load("burnout_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

print("Model loaded successfully!")
print(f"Expected features: {feature_names}")
print(f"Burnout risk labels: {label_encoder.classes_}\n")

# Test cases
test_cases = [
    {
        "name": "High-risk student",
        "study_hours": 9.5,
        "screen_time": 8.0,
        "stress_level": 9,
        "physical_activity": 1.0,
        "number_of_courses": 7
    },
    {
        "name": "Low-risk student",
        "study_hours": 3.0,
        "screen_time": 4.0,
        "stress_level": 3,
        "physical_activity": 10.0,
        "number_of_courses": 3
    },
    {
        "name": "Medium-risk student",
        "study_hours": 6.0,
        "screen_time": 6.0,
        "stress_level": 6,
        "physical_activity": 5.0,
        "number_of_courses": 5
    }
]

# Make predictions
for test_case in test_cases:
    name = test_case.pop("name")
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([test_case])[feature_names]
    
    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Decode prediction
    risk_label = label_encoder.inverse_transform([prediction])[0]
    
    print(f"--- {name} ---")
    print(f"Input: {test_case}")
    print(f"Predicted burnout risk: {risk_label}")
    print(f"Confidence: {prediction_proba[prediction]:.2%}")
    print(f"Probabilities: {dict(zip(label_encoder.classes_, prediction_proba))}")
    print()

# Interactive prediction
print("\n=== Try your own input ===")
print("Enter student data (or press Ctrl+C to exit):\n")

try:
    study_hours = float(input("Study hours (0-10): "))
    screen_time = float(input("Screen time (0-10): "))
    stress_level = int(input("Stress level (1-10): "))
    physical_activity = float(input("Physical activity hours (0-14): "))
    number_of_courses = int(input("Number of courses (1-8): "))
    
    custom_input = pd.DataFrame([{
        "study_hours": study_hours,
        "screen_time": screen_time,
        "stress_level": stress_level,
        "physical_activity": physical_activity,
        "number_of_courses": number_of_courses
    }])[feature_names]
    
    prediction = model.predict(custom_input)[0]
    prediction_proba = model.predict_proba(custom_input)[0]
    risk_label = label_encoder.inverse_transform([prediction])[0]
    
    print(f"\n✓ Predicted burnout risk: {risk_label.upper()}")
    print(f"✓ Confidence: {prediction_proba[prediction]:.2%}")
    
except KeyboardInterrupt:
    print("\n\nExiting...")
except Exception as e:
    print(f"\nError: {e}")
