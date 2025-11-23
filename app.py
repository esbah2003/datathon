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
feature_cols = [
    "study_hours",
    "screen_time",
    "stress_level",
    "physical_activity",
    "number_of_courses",
    "sleep_hours",
    "sleep_quality",
    "anxiety_score",
    "depression_score",
    "coping_strategy_score",
    "social_support_score",
    "academic_self_efficacy"
]
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
            'number_of_courses': int(data['number_of_courses']),
            'sleep_hours': float(data['sleep_hours']),
            'sleep_quality': float(data['sleep_quality']),
            'anxiety_score': float(data['anxiety_score']),
            'depression_score': float(data['depression_score']),
            'coping_strategy_score': float(data['coping_strategy_score']),
            'social_support_score': float(data['social_support_score']),
            'academic_self_efficacy': float(data['academic_self_efficacy'])
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
    (study_hours, screen_time, stress_level, physical_activity, number_of_courses,
     sleep_hours, sleep_quality, anxiety_score, depression_score, 
     coping_strategy_score, social_support_score, academic_self_efficacy) = features
    
    # Calculate weighted contribution to burnout (0-54 scale)
    # Mental health is the primary contributor
    stress_contribution = (stress_level / 10.0) * 12  # Max 12 points from stress
    anxiety_contribution = (anxiety_score / 10.0) * 8  # Max 8 points from anxiety
    depression_contribution = (depression_score / 10.0) * 8  # Max 8 points from depression
    
    # Academic factors
    study_contribution = min(study_hours / 16.0, 1.0) * 6  # Max 6 points
    screen_contribution = min(screen_time / 16.0, 1.0) * 4  # Max 4 points
    courses_contribution = min(number_of_courses / 10.0, 1.0) * 4  # Max 4 points
    efficacy_penalty = (1.0 - academic_self_efficacy / 10.0) * 4  # Max 4 points for low efficacy
    
    # Sleep quality impact (poor sleep increases burnout)
    sleep_penalty = 0
    if sleep_hours < 6:
        sleep_penalty += 3
    elif sleep_hours < 7:
        sleep_penalty += 1
    sleep_penalty += (1.0 - sleep_quality / 4.0) * 3  # Max 3 points for poor sleep quality
    
    # Calculate base score
    base_score = (stress_contribution + anxiety_contribution + depression_contribution +
                  study_contribution + screen_contribution + courses_contribution +
                  efficacy_penalty + sleep_penalty)
    
    # Protective factors reduce burnout
    activity_reduction = min(physical_activity / 8.0, 1.0) * 6  # Max 6 point reduction
    coping_reduction = (coping_strategy_score / 10.0) * 4  # Max 4 point reduction
    support_reduction = (social_support_score / 10.0) * 5  # Max 5 point reduction
    
    total_reduction = activity_reduction + coping_reduction + support_reduction
    final_score = max(0, base_score - total_reduction)
    
    # Add bonuses for extreme conditions
    if anxiety_score >= 8 or depression_score >= 8:
        final_score += 4
    if stress_level >= 9:
        final_score += 3
    if sleep_hours < 5:
        final_score += 3
    if social_support_score <= 3:
        final_score += 2
    
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
    """Generate personalized suggestions using Gemini AI ensuring consistent count."""
    TARGET_COUNT = 5
    if not gemini_model:
        # Fallback already can return up to 5; slice to target
        return get_fallback_suggestions(data, mbi_score, mbi_category)[:TARGET_COUNT]

    try:
        prompt = f"""

Provide EXACTLY {TARGET_COUNT} numbered, highly personalized, evidence-based recommendations to reduce student burnout risk.
Number them 1–{TARGET_COUNT}. Do not include any introductory or closing sentences.

FORMAT REQUIREMENTS:
Each recommendation must be 2–3 sentences and follow this structure:
1) A specific action the student can take.
2) A direct explanation referencing their exact values (e.g., “because you sleep 5h…”, “because your stress is 7/10…”).
3) A brief outcome or benefit (“which can improve…”, “which reduces…”).

AVOID generic advice. Every recommendation must tie directly to the user’s values.

USER DATA:
MBI Score: {mbi_score:.1f} / 54 ({mbi_category})
Study hours: {data['study_hours']}
Screen time: {data['screen_time']}
Courses: {data['number_of_courses']}
Sleep: {data['sleep_hours']}h, quality {data['sleep_quality']}/4
Stress: {data['stress_level']}/10
Anxiety: {data['anxiety_score']}/10
Depression: {data['depression_score']}/10
Physical activity: {data['physical_activity']}h/week
Coping strategies: {data['coping_strategy_score']}/10
Social support: {data['social_support_score']}/10
Academic self-efficacy: {data['academic_self_efficacy']}/10

PERSONALIZATION REQUIREMENTS:
- If sleep hours < 7 or sleep quality ≤ 2, include a recommendation referencing BOTH sleep hours and sleep quality.
- If stress/anxiety/depression ≥ 6, include a recommendation targeting mental-health regulation referencing all elevated scores.
- If coping_strategy_score ≤ 4, explain why this creates vulnerability and suggest specific coping improvements.
- If social_support_score ≤ 5, recommend increasing support, referencing the user’s score directly.
- If screen_time > 6h or study_hours > 6h, include a recommendation about workload or screen boundaries, citing the exact hours.
- If academic_self_efficacy ≤ 6, include a recommendation that strengthens planning or confidence.

GOAL:
Provide practical, personalized, and realistic micro-interventions that clearly connect user data → specific cause → recommended action → expected benefit.


"""

        response = gemini_model.generate_content(prompt)
        suggestions_text = response.text or ""

        raw_lines = [l.strip() for l in suggestions_text.split('\n') if l.strip()]
        cleaned = []
        for line in raw_lines:
            # Skip meta/header lines that sometimes appear
            lower = line.lower()
            if any(kw in lower for kw in ["intro", "recommendations", "burnout risk", "here are"]):
                continue
            # Strip leading numbering / bullets
            line = line.lstrip('-* ').strip()
            if line[:2].isdigit():
                # Remove leading number patterns like '1.' or '1)'
                line = line.split('.', 1)[-1] if '.' in line[:4] else line
                line = line.split(')', 1)[-1] if ')' in line[:4] else line
            cleaned.append(line.strip())

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in cleaned:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        # Enforce target count; if short, supplement with fallback
        if len(unique) < TARGET_COUNT:
            fallback = get_fallback_suggestions(data, mbi_score, mbi_category)
            for f in fallback:
                if f not in seen:
                    unique.append(f)
                if len(unique) == TARGET_COUNT:
                    break

        return unique[:TARGET_COUNT]

    except Exception as e:
        print(f"Error generating AI suggestions: {e}")
        return get_fallback_suggestions(data, mbi_score, mbi_category)[:TARGET_COUNT]

def get_fallback_suggestions(data, mbi_score, mbi_category):
    """Fallback suggestions if AI is not available"""
    suggestions = []
    
    # High priority mental health suggestions
    if mbi_category == "High Risk":
        suggestions.append("⚠️ Your MBI score indicates high burnout risk. Consider speaking with a counselor or mental health professional.")
    
    if float(data.get('anxiety_score', 0)) >= 7 or float(data.get('depression_score', 0)) >= 7:
        suggestions.append("🧠 Your anxiety/depression scores are elevated. Please reach out to campus mental health services or a trusted counselor.")
    
    # Sleep-related suggestions
    if float(data.get('sleep_hours', 7)) < 6 or float(data.get('sleep_quality', 3)) <= 2:
        suggestions.append("💤 Prioritize sleep: aim for 7-9 hours nightly with good sleep hygiene (dark room, no screens before bed, consistent schedule).")
    
    # Stress management
    if float(data.get('stress_level', 5)) > 7:
        suggestions.append("🧘 Your stress level is very high. Practice daily stress management: meditation, deep breathing, or yoga for 10-15 minutes.")
    
    # Academic workload
    if float(data.get('study_hours', 0)) > 10 or int(data.get('number_of_courses', 0)) > 6:
        suggestions.append("📚 Reduce academic overload: use the Pomodoro technique (25 min study, 5 min break) and consider dropping a course if possible.")
    
    # Social support
    if float(data.get('social_support_score', 5)) <= 4:
        suggestions.append("🤝 Build your support network: connect with friends, join study groups, or reach out to family regularly.")
    
    # Physical health
    if float(data.get('physical_activity', 0)) < 3:
        suggestions.append("🏃 Increase physical activity to at least 30 minutes daily - exercise significantly reduces burnout symptoms.")
    
    # Coping strategies
    if float(data.get('coping_strategy_score', 5)) <= 4:
        suggestions.append("🛠️ Develop healthy coping strategies: time management, mindfulness, hobbies, and avoiding procrastination.")
    
    # Academic self-efficacy
    if float(data.get('academic_self_efficacy', 5)) <= 4:
        suggestions.append("📖 Build academic confidence: seek tutoring, form study groups, break tasks into smaller steps, and celebrate small wins.")
    
    # Screen time
    if float(data.get('screen_time', 0)) > 8:
        suggestions.append("📱 Limit screen time: take breaks every 20 minutes (20-20-20 rule) and set boundaries for recreational use.")
    
    # Positive reinforcement for low risk
    if len(suggestions) < 3 and mbi_category == "Low Risk":
        suggestions.append("✅ Great job maintaining balance! Continue your healthy habits and check in regularly with yourself.")
    
    return suggestions[:5]  # Limit to 5 suggestions

if __name__ == '__main__':
    app.run(debug=True)
