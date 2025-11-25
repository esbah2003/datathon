# StressScope - Student Burnout Prediction App

A machine learning powered app that predicts student burnout risk based on lifestyle and academic factors.

## Quick Start

### 1. (Optional) Add API Key

Create a `.env` file in the project and add  Gemini API key for AI suggestions based basned on analyzing patterns:

```bash
GEMINI_API_KEY=your_api_key_here
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements
**If you get any errors with the requirements installation, ensure you are using Python 3.11**

```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

## Features
- **Burnout Risk Assessment**: Predict burnout risk based on study hours, screen time, stress level physical activity, and course load
- **MBI-Compliant Model**: Uses Maslach Burnout Inventory (MBI) standards
- **AI-Powered Suggestions**: Get personalized recommendations via Gemini AI
