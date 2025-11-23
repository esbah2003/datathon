import joblib
import pandas as pd

# Load the saved model
model = joblib.load("burnout_model.pkl")

# mappings
FEATURE_NAMES = ['study_hours', 'screen_time', 'stress_level', 'physical_activity', 'number_of_courses']
RISK_LABELS = ['low', 'medium']  # Match training order

# Test inputs - modify these values as needed
test_input = {
    "study_hours": 8.0,
    "screen_time": 7.0,
    "stress_level": 8,
    "physical_activity": 2.5,
    "number_of_courses": 6
}

# Create DataFrame with correct feature order
input_df = pd.DataFrame([test_input])[FEATURE_NAMES]

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]
risk_label = RISK_LABELS[prediction]

# Display results
print("Input values:")
for feature, value in test_input.items():
    print(f"  {feature}: {value}")

print(f"\nPredicted burnout risk: {risk_label.upper()}")
print(f"Confidence: {prediction_proba[prediction]:.2%}")
print(f"\nProbability breakdown:")
for label, prob in zip(RISK_LABELS, prediction_proba):
    print(f"  {label}: {prob:.2%}") 