from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Configure Gemini AI
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini AI configured successfully")
except Exception as e:
    print(f"Warning: Could not configure Gemini AI: {e}")
    gemini_model = None

# Load the CARRAD MBI trained model
model_path = Path(__file__).parent / "src" / "MBI" / "carrad_mbi_model.joblib"
pipeline = None
feature_cols = []
target_cols = []
burnout_thresholds = {}

try:
    if model_path.exists():
        model_bundle = joblib.load(model_path)
        pipeline = model_bundle.get("pipeline")
        feature_cols = model_bundle.get("feature_cols", [])
        target_cols = model_bundle.get("target_cols", [])
        burnout_thresholds = model_bundle.get("burnout_thresholds", {})
        print("CARRAD MBI Model loaded successfully")
        print(f"Features: {len(feature_cols)}, Targets: {target_cols}")
    else:
        print("CARRAD model file not found")
except Exception as e:
    print(f"Error loading CARRAD model: {e}")
    pipeline = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features_dict = {
            'stud_h': float(data.get('stud_h', 20)),
            'health': int(data.get('health', 3)),
            'psyt': int(data.get('psyt', 0)),
            'cesd': float(data.get('cesd', 15)),
            'stai_t': float(data.get('stai_t', 40)),
            'amsp': float(data.get('amsp', 20)),
            'qcae_cog': float(data.get('qcae_cog', 55)),
            'qcae_aff': float(data.get('qcae_aff', 35)),
            'jspe': float(data.get('jspe', 100)),
            'age': int(data.get('age', 22)),
            'year': int(data.get('year', 3)),
            'sex': int(data.get('sex', 1))
        }
        
        if pipeline:
            features_df = pd.DataFrame([features_dict])
            print(f"Input features: {features_df.iloc[0].to_dict()}")
            
            # Make prediction - returns [mbi_ex, mbi_cy, mbi_ea]
            predictions = pipeline.predict(features_df)[0]
            
            # Clip to valid ranges
            mbi_ex = float(np.clip(predictions[0], 5, 30))
            mbi_cy = float(np.clip(predictions[1], 0, 30))
            mbi_ea = float(np.clip(predictions[2], 10, 36))
            
            print(f"Predictions - Exhaustion: {mbi_ex:.1f}, Cynicism: {mbi_cy:.1f}, Efficacy: {mbi_ea:.1f}")
        else:
            # Fallback if model not loaded
            mbi_ex, mbi_cy, mbi_ea = calculate_mbi_fallback(features_dict)
            print(f"Fallback - Exhaustion: {mbi_ex:.1f}, Cynicism: {mbi_cy:.1f}, Efficacy: {mbi_ea:.1f}")
        
        risk_levels = interpret_burnout(mbi_ex, mbi_cy, mbi_ea)
        high_risk_factors = identify_high_risk_factors(features_dict)
        
        # Generate suggestions with explanations
        suggestions = generate_suggestions_with_why(features_dict, mbi_ex, mbi_cy, mbi_ea, risk_levels, high_risk_factors)
        
        return jsonify({
            'success': True,
            'mbi_ex': round(mbi_ex, 1),
            'mbi_cy': round(mbi_cy, 1),
            'mbi_ea': round(mbi_ea, 1),
            'risk_levels': risk_levels,
            'high_risk_factors': high_risk_factors,
            'suggestions': suggestions
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def calculate_mbi_fallback(features):
    """Fallback calculation based on CARRAD feature weights"""
    # Based on feature importance: cesd (46%), stai_t (11%), stud_h (7%)
    cesd_norm = features['cesd'] / 60.0  # CESD range ~0-60
    stai_norm = features['stai_t'] / 80.0  # STAI range ~20-80
    health_inv = (6 - features['health']) / 5.0  # Invert health (5=best)
    
    mbi_ex = 5 + (cesd_norm * 15 + stai_norm * 8 + health_inv * 2)
    mbi_cy = cesd_norm * 12 + (1 - features['amsp'] / 35.0) * 8 + features['psyt'] * 5
    mbi_ea = 36 - (cesd_norm * 10 + health_inv * 6 + (1 - features['amsp'] / 35.0) * 8)
    return (
        float(np.clip(mbi_ex, 5, 30)),
        float(np.clip(mbi_cy, 0, 30)),
        float(np.clip(mbi_ea, 10, 36))
    )

def interpret_burnout(mbi_ex, mbi_cy, mbi_ea):
    """Categorize each MBI dimension"""
    ex_category = "Low" if mbi_ex <= 13 else "Moderate" if mbi_ex <= 20 else "High"
    cy_category = "Low" if mbi_cy <= 6 else "Moderate" if mbi_cy <= 13 else "High"
    # Efficacy is inverted - lower scores = worse
    ea_category = "High" if mbi_ea >= 28 else "Moderate" if mbi_ea >= 22 else "Low"
    
    # Overall risk based on primary indicator (Exhaustion)
    overall_risk = ex_category
    
    colors = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"}
    
    return {
        "exhaustion": {"score": mbi_ex, "level": ex_category, "color": colors[ex_category]},
        "cynicism": {"score": mbi_cy, "level": cy_category, "color": colors[cy_category]},
        "efficacy": {"score": mbi_ea, "level": ea_category, "color": colors[ea_category]},
        "overall": overall_risk,
        "overall_color": colors[overall_risk]
    }

def identify_high_risk_factors(features):
    """Flag concerning values based on clinical thresholds"""
    factors = []
    
    # Depression (CESD > 16 = depressed, >22 = severely depressed)
    if features['cesd'] >= 22:
        factors.append({"factor": "Depression", "severity": "severe", "value": features['cesd'], "threshold": "≥22"})
    elif features['cesd'] >= 16:
        factors.append({"factor": "Depression", "severity": "moderate", "value": features['cesd'], "threshold": "≥16"})
    
    # Anxiety (STAI-T > 55 = high anxiety)
    if features['stai_t'] >= 55:
        factors.append({"factor": "Anxiety", "severity": "high", "value": features['stai_t'], "threshold": "≥55"})
    
    # Psychological distress
    if features['psyt'] == 1:
        factors.append({"factor": "Psychological Distress", "severity": "present", "value": "Yes", "threshold": "Flag"})
    
    # Poor health
    if features['health'] <= 2:
        factors.append({"factor": "Health Status", "severity": "poor", "value": features['health'], "threshold": "≤2"})
    
    # Low motivation
    if features['amsp'] <= 18:
        factors.append({"factor": "Academic Motivation", "severity": "low", "value": features['amsp'], "threshold": "≤18"})
    
    # Excessive study hours
    if features['stud_h'] >= 50:
        factors.append({"factor": "Study Hours", "severity": "excessive", "value": features['stud_h'], "threshold": "≥50/week"})
    
    # Low empathy (potential professional concern)
    if features['qcae_cog'] <= 45:
        factors.append({"factor": "Cognitive Empathy", "severity": "low", "value": features['qcae_cog'], "threshold": "≤45"})
    
    return factors

def generate_suggestions_with_why(features, mbi_ex, mbi_cy, mbi_ea, risk_levels, high_risk_factors):
    """Generate 5 recommendations with clear explanations of WHY"""
    TARGET_COUNT = 5
    
    if not gemini_model:
        return get_fallback_suggestions_with_why(features, mbi_ex, mbi_cy, mbi_ea, high_risk_factors)[:TARGET_COUNT]
    
    try:
        # Build high-risk summary for prompt
        risk_summary = ", ".join([f"{f['factor']} ({f['severity']})" for f in high_risk_factors]) if high_risk_factors else "None"
        
        prompt = f"""
You are a student wellness counselor. Provide EXACTLY {TARGET_COUNT} numbered recommendations (1-5).

STUDENT BURNOUT PROFILE:
- Exhaustion: {mbi_ex:.1f}/30 ({risk_levels['exhaustion']['level']})
- Cynicism: {mbi_cy:.1f}/30 ({risk_levels['cynicism']['level']})
- Professional Efficacy: {mbi_ea:.1f}/36 ({risk_levels['efficacy']['level']})

INPUT DATA:
- Depression (CESD): {features['cesd']} (clinical cutoff: ≥16)
- Anxiety (STAI): {features['stai_t']} (high: ≥55)
- Study hours/week: {features['stud_h']}
- Health status: {features['health']}/5
- Psychological distress: {'Yes' if features['psyt'] == 1 else 'No'}
- Academic motivation: {features['amsp']}
- Cognitive empathy: {features['qcae_cog']}
- Affective empathy: {features['qcae_aff']}

HIGH-RISK FACTORS DETECTED: {risk_summary}

For each recommendation:
1. Start with the ACTION (1 sentence)
2. Then add "WHY:" followed by the specific reason based on their values (1 sentence citing their exact scores)

Example format:
1. Seek immediate counseling through student health services. WHY: Your depression score of 35 is in the severe range (≥22), indicating significant risk that requires professional intervention.

Focus on the highest-risk areas. Be specific, evidence-based, and compassionate.
"""
        
        response = gemini_model.generate_content(prompt)
        suggestions_text = response.text or ""
        
        # Parse numbered list
        lines = [l.strip() for l in suggestions_text.split('\n') if l.strip()]
        suggestions = []
        
        for line in lines:
            # Skip meta text
            if any(skip in line.lower() for skip in ["here are", "recommendations", "profile"]):
                continue
            # Extract numbered items
            if line[0].isdigit() and ('.' in line[:3] or ')' in line[:3]):
                # Remove number prefix
                content = line.split('.', 1)[-1] if '.' in line[:3] else line.split(')', 1)[-1]
                suggestions.append(content.strip())
        
        # Ensure we have exactly TARGET_COUNT
        if len(suggestions) < TARGET_COUNT:
            fallback = get_fallback_suggestions_with_why(features, mbi_ex, mbi_cy, mbi_ea, high_risk_factors)
            suggestions.extend(fallback[:TARGET_COUNT - len(suggestions)])
        
        return suggestions[:TARGET_COUNT]
        
    except Exception as e:
        print(f"Error generating AI suggestions: {e}")
        return get_fallback_suggestions_with_why(features, mbi_ex, mbi_cy, mbi_ea, high_risk_factors)[:TARGET_COUNT]

def get_fallback_suggestions_with_why(features, mbi_ex, mbi_cy, mbi_ea, high_risk_factors):
    """Rule-based suggestions with WHY explanations"""
    suggestions = []
    
    # Priority 1: Severe depression
    if features['cesd'] >= 22:
        suggestions.append(f"Seek immediate mental health support through campus counseling or crisis services. WHY: Your depression score of {features['cesd']} is in the severe range (≥22), requiring urgent professional intervention.")
    
    # Priority 2: High anxiety
    if features['stai_t'] >= 55:
        suggestions.append(f"Practice daily anxiety management with guided meditation or breathing exercises (10-15 min). WHY: Your anxiety score of {features['stai_t']} exceeds clinical thresholds (≥55), indicating elevated stress that responds well to mindfulness techniques.")
    
    # Priority 3: Moderate depression without severe anxiety
    elif features['cesd'] >= 16 and features['stai_t'] < 55:
        suggestions.append(f"Schedule an appointment with a counselor to discuss mood management strategies. WHY: Your depression score of {features['cesd']} is above the clinical cutoff (≥16), suggesting mild-to-moderate symptoms that benefit from early intervention.")
    
    # Excessive workload
    if features['stud_h'] >= 50:
        suggestions.append(f"Reduce study hours to under 40 hours/week and implement time-blocking strategies. WHY: You're studying {features['stud_h']} hours/week, which exceeds healthy limits (≥50) and strongly correlates with burnout.")
    
    # Poor health
    if features['health'] <= 2:
        suggestions.append(f"Schedule a general health checkup and prioritize basic self-care (sleep, nutrition, exercise). WHY: Your health rating of {features['health']}/5 is poor, and physical health directly impacts mental wellbeing and academic performance.")
    
    # Low motivation with high burnout
    if features['amsp'] <= 18 and mbi_ex >= 20:
        suggestions.append(f"Reconnect with your academic and career goals through mentorship or relevant experiences. WHY: Your academic motivation score of {features['amsp']} is low while exhaustion is high ({mbi_ex}/30), suggesting values misalignment that needs addressing.")
    
    # Psychological distress flag
    if features['psyt'] == 1 and len(suggestions) < 4:
        suggestions.append(f"Join a peer support group to share experiences and coping strategies. WHY: You've indicated psychological distress, and peer support has been shown to reduce isolation and normalize academic challenges.")
    
    # Low empathy with high cynicism
    if features['qcae_cog'] <= 45 and mbi_cy >= 13:
        suggestions.append(f"Engage in reflective practice or community service to rebuild empathy and connection with others. WHY: Your cognitive empathy score ({features['qcae_cog']}) is low and cynicism is elevated ({mbi_cy}/30), which can impact personal and professional relationships.")
    
    # Generic if nothing specific triggered
    if len(suggestions) < 2:
        suggestions.append(f"Maintain a balanced routine with adequate sleep (7-9 hrs), regular exercise, and social connections. WHY: Your exhaustion score of {mbi_ex}/30 indicates burnout risk, which responds well to lifestyle balance and self-care.")
        suggestions.append(f"Set clear boundaries between study and personal time to prevent emotional exhaustion. WHY: Burnout often stems from lack of recovery time; structured work-rest cycles improve both wellbeing and performance.")
    
    return suggestions[:5]

if __name__ == '__main__':
    app.run(debug=True)
