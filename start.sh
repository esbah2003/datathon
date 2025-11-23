#!/bin/bash

# Student Burnout Assessment Dashboard - Startup Script

echo " Starting Student Burnout Assessment Dashboard..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo " Installing requirements..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "  No .env file found. Creating from template..."
    cp .env.example .env
    echo " Please edit .env file and add your Gemini API key!"
    echo "   You can get it from: https://makersuite.google.com/app/apikey"
fi

# Train model if it doesn't exist
if [ ! -f "src/model/model.joblib" ]; then
    echo " Training machine learning model..."
    cd src/model
    python train.py
    cd ../..
fi

echo " Starting Flask application..."
echo " Open http://localhost:5000 in your browser"
echo ""
python app.py
