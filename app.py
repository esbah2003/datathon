from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model name
    print("Gemini AI configured successfully")
except Exception as e:
    print(f"Warning: Could not configure Gemini AI: {e}")
    gemini_model = None

# Load the trained model
model_path = Path(__file__).parent / "src" / "model" / "model.joblib"
pipeline = None
feature_names = ["study_hours", "screen_time", "stress_level", "physical_activity", "number_of_courses"]

try:
    if model_path.exists():
        model_data = joblib.load(model_path)
        pipeline = model_data.get("pipeline")
        feature_names = model_data.get("feature_names", feature_names)
        print("Model loaded successfully")
    else:
        print("Model file not found, using fallback calculations")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using fallback calculations")
    pipeline = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order as a dictionary for DataFrame
        features_dict = {
            'study_hours': float(data['study_hours']),
            'screen_time': float(data['screen_time']),
            'stress_level': float(data['stress_level']),
            'physical_activity': float(data['physical_activity']),
            'number_of_courses': int(data['number_of_courses'])
        }
        
        if pipeline:
            # Create DataFrame with feature names for model prediction
            features_df = pd.DataFrame([features_dict])
            print(f"DataFrame for model: \n{features_df}")
            print(f"DataFrame dtypes: \n{features_df.dtypes}")
            
            # Make prediction
            raw_prediction = pipeline.predict(features_df)[0]
            # Ensure prediction is between 0 and 1
            prediction = max(0.0, min(1.0, float(raw_prediction)))
            print(f"Model - Raw prediction: {raw_prediction}, Constrained: {prediction}")
        else:
            # Fallback calculation if model isn't loaded
            features_list = list(features_dict.values())
            prediction = calculate_burnout_risk(features_list)
            print(f"Fallback - Prediction: {prediction}")
            
        print(f"Final prediction sent to frontend: {prediction}")
        
        # Generate suggestions using Gemini AI
        suggestions = generate_suggestions(data, prediction)
        
        # Create risk interpretation
        risk_level = interpret_risk_level(prediction)
        
        return jsonify({
            'success': True,
            'burnout_risk': round(prediction, 3),
            'risk_level': risk_level,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def calculate_burnout_risk(features):
    """Fallback calculation if model isn't available"""
    study_hours, screen_time, stress_level, physical_activity, number_of_courses = features
    
    # Correct logic: Higher values increase risk (except physical activity which decreases risk)
    
    # Normalize inputs to 0-1 scale based on typical ranges
    study_risk = min(study_hours / 16.0, 1.0)  # More study hours = higher risk
    screen_risk = min(screen_time / 16.0, 1.0)  # More screen time = higher risk  
    stress_risk = stress_level / 10.0  # Higher stress = higher risk
    activity_protection = min(physical_activity / 8.0, 1.0)  # More activity = lower risk
    courses_risk = min(number_of_courses / 10.0, 1.0)  # More courses = higher risk
    
    # Calculate base risk (higher values = higher risk, except physical activity)
    base_risk = (
        study_risk * 0.20 +      # Study hours contribute to burnout
        screen_risk * 0.15 +     # Screen time contributes to burnout
        stress_risk * 0.35 +     # Stress is the biggest factor
        courses_risk * 0.10      # Course load contributes to burnout
    )
    
    # Physical activity reduces risk
    risk_after_activity = base_risk * (1.0 - (activity_protection * 0.3))
    
    # Add bonuses for extreme values
    if stress_level >= 8:
        risk_after_activity += 0.1  # Very high stress
    if study_hours >= 12:
        risk_after_activity += 0.05  # Excessive study hours
    if physical_activity <= 0.5:
        risk_after_activity += 0.1  # Very low activity
    if screen_time >= 10:
        risk_after_activity += 0.05  # Excessive screen time
    
    return min(max(risk_after_activity, 0.0), 1.0)

def interpret_risk_level(risk):
    """Convert numerical risk to descriptive level"""
    if risk < 0.3:
        return {
            'level': 'Low Risk',
            'color': '#10B981',
            'description': 'Your burnout risk is low. Keep up the good work!'
        }
    elif risk < 0.6:
        return {
            'level': 'Moderate Risk',
            'color': '#F59E0B',
            'description': 'You have moderate burnout risk. Consider some lifestyle adjustments.'
        }
    else:
        return {
            'level': 'High Risk',
            'color': '#EF4444',
            'description': 'Your burnout risk is high. Please prioritize your wellbeing.'
        }

def generate_suggestions(data, risk_score):
    """Generate personalized suggestions using Gemini AI"""
    if not gemini_model:
        return get_fallback_suggestions(data, risk_score)
    
    try:
        prompt = f"""
        A student has submitted their lifestyle data for burnout assessment. Please analyze this data and provide personalized recommendations:

        STUDENT PROFILE:
        - Study hours per day: {data['study_hours']} hours
        - Screen time per day: {data['screen_time']} hours  
        - Stress level: {data['stress_level']}/10
        - Physical activity per day: {data['physical_activity']} hours
        - Number of courses: {data['number_of_courses']} courses
        
        BURNOUT ASSESSMENT RESULT:
        - Calculated burnout risk: {risk_score:.1%} ({risk_score:.3f} on 0-1 scale)
        
        Based on this student's specific data and their {risk_score:.1%} burnout risk, please provide 3-4 specific, actionable suggestions to help them improve their wellbeing and reduce burnout risk. Focus on the areas that need the most attention based on their input values and risk level. Keep each suggestion practical and concise (1-2 sentences).
        """
        
        response = gemini_model.generate_content(prompt)
        suggestions_text = response.text
        
        # Parse the response into individual suggestions
        suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip() and not s.strip().startswith('#')]
        
        return suggestions[:4]  # Limit to 4 suggestions
        
    except Exception as e:
        print(f"Error generating AI suggestions: {e}")
        return get_fallback_suggestions(data, risk_score)

def get_fallback_suggestions(data, risk_score):
    """Fallback suggestions if AI is not available"""
    suggestions = []
    
    if float(data['study_hours']) > 8:
        suggestions.append("Consider reducing study hours and implementing more efficient study techniques like the Pomodoro method.")
    
    if float(data['screen_time']) > 8:
        suggestions.append("Try to reduce screen time by taking regular breaks and using apps to limit recreational screen use.")
    
    if float(data['stress_level']) > 6:
        suggestions.append("Practice stress management techniques like deep breathing, meditation, or yoga.")
    
    if float(data['physical_activity']) < 1:
        suggestions.append("Increase physical activity - even a 20-30 minute daily walk can significantly improve your wellbeing.")
    
    if len(suggestions) == 0:
        suggestions.append("Maintain your current healthy balance and consider incorporating mindfulness practices.")
    
    return suggestions

if __name__ == '__main__':
    app.run(debug=True)
