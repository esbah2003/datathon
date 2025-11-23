"""
Quick test script for Gemini AI integration
Run this to verify the API key and connection are working
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_integration():
    print("Testing Gemini AI Integration...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print(" ERROR: GEMINI_API_KEY not found in environment variables")
        print("Make sure you have a .env file with your API key")
        return False
    
    print("✓ API key found")
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model name
        print("✓ Gemini model configured")
        
        # Test prompt with realistic data
        test_prompt = """
        A student has submitted their lifestyle data for burnout assessment. Please analyze this data and provide personalized recommendations:

        STUDENT PROFILE:
        - Study hours per day: 8 hours
        - Screen time per day: 6 hours  
        - Stress level: 7/10
        - Physical activity per day: 1 hours
        - Number of courses: 5 courses
        
        BURNOUT ASSESSMENT RESULT:
        - Calculated burnout risk: 65.0% (0.650 on 0-1 scale)
        
        Based on this student's specific data and their 65.0% burnout risk, please provide 3-4 specific, actionable suggestions to help them improve their wellbeing and reduce burnout risk. Focus on the areas that need the most attention based on their input values and risk level. Keep each suggestion practical and concise (1-2 sentences).
        """
        
        print("\n Testing AI response...")
        response = model.generate_content(test_prompt)
        
        print("✓ AI response received successfully!")
        print("\n Sample Response:")
        print("-" * 30)
        print(response.text)
        print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f" ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_integration()
    
    if success:
        print("\n Gemini integration is working correctly!")
        print("You can now run your Flask app with confidence.")
    else:
        print("\n  Please fix the issues above before running your app.")