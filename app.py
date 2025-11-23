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

# Load the MBI trained model
model_path = Path(__file__).parent / "src" / "MBI" / "mbi_model.joblib"
pipeline = None
feature_cols = ["study_hours", "screen_time", "stress_level", "physical_activity", "number_of_courses"]
mbi_thresholds = {
    "low": 16,
    "moderate": 26,
    "high": 54
}

try:
    if model_path.exists():
        model_bundle = joblib.load(model_path)
        pipeline = model_bundle.get("pipeline")
        feature_cols = model_bundle.get("feature_cols", feature_cols)
        mbi_thresholds = model_bundle.get("mbi_thresholds", mbi_thresholds)
        print("MBI Model loaded successfully")
        print(f"MBI Thresholds: Low (0-{mbi_thresholds['low']}), Moderate ({mbi_thresholds['low']+1}-{mbi_thresholds['moderate']}), High ({mbi_thresholds['moderate']+1}-{mbi_thresholds['high']})")
    else:
        print("MBI Model file not found, using fallback calculations")
except Exception as e:
    print(f"Error loading MBI model: {e}")
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
            # Create DataFrame with feature names for MBI model prediction
            features_df = pd.DataFrame([features_dict])
            print(f"DataFrame for MBI model: \n{features_df}")
            print(f"DataFrame dtypes: \n{features_df.dtypes}")
            
            # Make prediction - MBI model returns score from 0-54
            raw_mbi_score = pipeline.predict(features_df)[0]
            # Clip to valid MBI range
            mbi_score = float(np.clip(raw_mbi_score, 0, 54))
            print(f"MBI Model - Raw score: {raw_mbi_score:.2f}, Clipped score: {mbi_score:.2f}")
        else:
            # Fallback calculation if model isn't loaded
            features_list = list(features_dict.values())
            mbi_score = calculate_mbi_fallback(features_list)
            print(f"Fallback - MBI Score: {mbi_score:.2f}")
            
        print(f"Final MBI score: {mbi_score:.2f} / 54")
        
        # Get MBI category and risk interpretation
        mbi_category = get_mbi_category(mbi_score)
        risk_level = interpret_mbi_risk(mbi_score, mbi_category)
        
        # Generate suggestions using Gemini AI
        suggestions = generate_suggestions(data, mbi_score, mbi_category)
        
        return jsonify({
            'success': True,
            'burnout_risk': round(mbi_score / 54, 3),  # Normalize to 0-1 for display
            'mbi_score': round(mbi_score, 1),
            'mbi_category': mbi_category,
            'risk_level': risk_level,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_mbi_category(mbi_score):
    """Categorize MBI score using official MBI-SS thresholds"""
    if mbi_score <= mbi_thresholds['low']:
        return "Low Risk"
    elif mbi_score <= mbi_thresholds['moderate']:
        return "Moderate Risk"
    else:
        return "High Risk"

def calculate_mbi_fallback(features):
    """Fallback MBI score calculation if model isn't available"""
    study_hours, screen_time, stress_level, physical_activity, number_of_courses = features
    
    # Calculate weighted contribution to burnout (0-54 scale)
    # Stress is the primary contributor in MBI
    stress_contribution = (stress_level / 10.0) * 22  # Max 22 points from stress
    
    # Study hours and screen time contribute to exhaustion
    study_contribution = min(study_hours / 16.0, 1.0) * 10  # Max 10 points
    screen_contribution = min(screen_time / 16.0, 1.0) * 8   # Max 8 points
    
    # Course load adds to overwhelm
    courses_contribution = min(number_of_courses / 10.0, 1.0) * 6  # Max 6 points
    
    # Physical activity reduces burnout
    activity_reduction = min(physical_activity / 8.0, 1.0) * 8  # Max 8 point reduction
    
    # Calculate base score
    base_score = (stress_contribution + study_contribution + 
                  screen_contribution + courses_contribution)
    
    # Apply activity reduction
    final_score = max(0, base_score - activity_reduction)
    
    # Add bonuses for extreme conditions
    if stress_level >= 9:
        final_score += 5
    if study_hours >= 12:
        final_score += 3
    if physical_activity <= 0.5:
        final_score += 4
    
    # Clip to valid MBI range
    return float(np.clip(final_score, 0, 54))

def interpret_mbi_risk(mbi_score, category):
    """Convert MBI score to descriptive risk level with color"""
    if category == "Low Risk":
        return {
            'level': 'Low Risk',
            'color': '#10B981',
            'description': f'MBI Score: {mbi_score:.1f}/54 - Healthy balance, minimal burnout symptoms. Keep up the good work!'
        }
    elif category == "Moderate Risk":
        return {
            'level': 'Moderate Risk',
            'color': '#F59E0B',
            'description': f'MBI Score: {mbi_score:.1f}/54 - Some burnout symptoms present. Consider lifestyle adjustments.'
        }
    else:
        return {
            'level': 'High Risk',
            'color': '#EF4444',
            'description': f'MBI Score: {mbi_score:.1f}/54 - Significant burnout risk. Intervention recommended - please prioritize your wellbeing.'
        }

def generate_suggestions(data, mbi_score, mbi_category):
    """Generate personalized suggestions using Gemini AI based on MBI assessment"""
    if not gemini_model:
        return get_fallback_suggestions(data, mbi_score, mbi_category)
    
    try:
        prompt = f"""
        A student has submitted their lifestyle data for burnout assessment using the Maslach Burnout Inventory for Students (MBI-SS). Please analyze this data and provide personalized recommendations:

        STUDENT PROFILE:
        - Study hours per day: {data['study_hours']} hours
        - Screen time per day: {data['screen_time']} hours  
        - Stress level: {data['stress_level']}/10
        - Physical activity per day: {data['physical_activity']} hours
        - Number of courses: {data['number_of_courses']} courses
        
        MBI BURNOUT ASSESSMENT RESULT:
        - MBI Score: {mbi_score:.1f} out of 54
        - Risk Category: {mbi_category}
        - MBI Thresholds: Low (0-16), Moderate (17-26), High (27-54)
        
        Based on this student's specific data and their MBI score of {mbi_score:.1f}/54 ({mbi_category}), please provide 3-4 specific, actionable suggestions to help them improve their wellbeing and reduce burnout risk. Focus on the areas that need the most attention based on their input values and MBI risk level. Keep each suggestion practical and concise (1-2 sentences).
        """
        
        response = gemini_model.generate_content(prompt)
        suggestions_text = response.text
        
        # Parse the response into individual suggestions
        suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip() and not s.strip().startswith('#')]
        
        return suggestions[:4]  # Limit to 4 suggestions
        
    except Exception as e:
        print(f"Error generating AI suggestions: {e}")
        return get_fallback_suggestions(data, mbi_score, mbi_category)

def get_fallback_suggestions(data, mbi_score, mbi_category):
    """Fallback suggestions if AI is not available"""
    suggestions = []
    
    # High priority suggestions based on MBI category
    if mbi_category == "High Risk":
        suggestions.append("⚠️ Your MBI score indicates high burnout risk. Consider speaking with a counselor or mental health professional.")
    
    # Specific suggestions based on input values
    if float(data['stress_level']) > 7:
        suggestions.append("🧘 Your stress level is very high. Practice daily stress management: try meditation, deep breathing, or yoga for 10-15 minutes.")
    
    if float(data['study_hours']) > 10:
        suggestions.append("📚 Reduce study hours and focus on quality over quantity. Use the Pomodoro technique: 25 min study, 5 min break.")
    
    if float(data['screen_time']) > 8:
        suggestions.append("📱 Limit screen time by taking regular breaks (20-20-20 rule: every 20 min, look 20 feet away for 20 seconds).")
    
    if float(data['physical_activity']) < 1:
        suggestions.append("🏃 Increase physical activity to at least 30 minutes daily - even a brisk walk can reduce burnout symptoms significantly.")
    
    if int(data['number_of_courses']) > 6:
        suggestions.append("📖 Consider reducing your course load next semester to achieve better balance and prevent burnout.")
    
    # Add general wellness tip if fewer than 3 suggestions
    if len(suggestions) < 3:
        if mbi_category == "Low Risk":
            suggestions.append("✅ Great job maintaining balance! Continue your healthy habits and check in regularly with yourself.")
        else:
            suggestions.append("💤 Prioritize sleep (7-9 hours nightly) and establish a consistent bedtime routine for better recovery.")
    
    return suggestions[:4]  # Limit to 4 suggestions

if __name__ == '__main__':
    app.run(debug=True)
