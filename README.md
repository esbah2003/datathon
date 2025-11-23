# Student Burnout Assessment Dashboard

ML-powered web app for predicting student burnout risk with personalized wellness recommendations.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: Configure Gemini AI
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Train model and run
python src/model/train.py
python app.py
```

Open http://localhost:5000

## Development

```bash
# Generate synthetic training data
python src/mock_data/generate_burnout_data.py
cp burnout_data_scores.csv src/model/

# Train model
python src/model/train.py

# Test predictions
python src/mock_data/test.py
```

## Tech Stack

Flask, scikit-learn, Gemini AI, vanilla JS

## Model

Random Forest Regression on 5 features:
- Study hours, screen time, stress level (1-10)  
- Physical activity hours, number of courses
- Output: Burnout risk score (0-1)

Performance: R² = 0.8944, MAE = 0.076

## Structure

```
app.py              # Flask backend
templates/index.html # Frontend dashboard  
src/model/train.py   # ML training
src/mock_data/       # Data generation
```