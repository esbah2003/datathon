import pandas as pd
import numpy as np

# random seed for reproducablitiy
np.random.seed(67)

# Initialize lists to store data
data = {
    'study_hours': [],
    'screen_time': [],
    'stress_level': [],
    'physical_activity': [],
    'number_of_courses': [],
    'burnout_risk': []
}

# Generate 100 rows for each burnout risk level (0.1-1.0). 
for burnout_score in np.arange(0.1, 1.1, 0.1):
    burnout_score = round(burnout_score, 1)  # Ensure clean decimal values
    
    for _ in range(100):
        # Generate features correlated with burnout risk
        # Higher burnout risk correlates with:
        # - Higher study hours, screen time, stress level, number of courses
        # - Lower physical activity
        
        # Add some noise/variance to make data realistic
        noise = np.random.uniform(-0.15, 0.15)
        adjusted_risk = max(0, min(1, burnout_score + noise))
        
        # Study hours (0-12 hours, increases with burnout)
        study_hours = round(adjusted_risk * 10 + np.random.uniform(0, 2), 1)
        study_hours = max(0, min(12, study_hours))
        
        # Screen time (0-12 hours, increases with burnout)
        screen_time = round(adjusted_risk * 9 + np.random.uniform(0, 3), 1)
        screen_time = max(0, min(12, screen_time))
        
        # Stress level (1-10, increases with burnout)
        stress_level = int(adjusted_risk * 8 + np.random.uniform(1, 3))
        stress_level = max(1, min(10, stress_level))
        
        # Physical activity (0-14 hours/week, decreases with burnout)
        physical_activity = round((1 - adjusted_risk) * 12 + np.random.uniform(0, 2), 1)
        physical_activity = max(0, min(14, physical_activity))
        
        # Number of courses (1-8, increases with burnout)
        number_of_courses = int(adjusted_risk * 6 + np.random.uniform(1, 3))
        number_of_courses = max(1, min(8, number_of_courses))
        
        # Append to data
        data['study_hours'].append(study_hours)
        data['screen_time'].append(screen_time)
        data['stress_level'].append(stress_level)
        data['physical_activity'].append(physical_activity)
        data['number_of_courses'].append(number_of_courses)
        data['burnout_risk'].append(burnout_score)

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the rows to mix burnout levels
df = df.sample(frac=1, random_state=67).reset_index(drop=True)

# Save to CSV
output_file = 'burnout_data_scores.csv'
df.to_csv(output_file, index=False)

print(f"✓ Generated {len(df)} rows of burnout data")
print(f"✓ Saved to {output_file}")
print(f"\nBurnout risk distribution:")
print(df['burnout_risk'].value_counts().sort_index())
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nData summary:")
print(df.describe())
