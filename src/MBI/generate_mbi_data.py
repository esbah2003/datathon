import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(67)
n_samples = 2000  # Increased sample size for more features

# Generate features with realistic distributions
data = {
    # Academic & Lifestyle
    "study_hours": np.random.gamma(3, 2, n_samples),  # Slightly right-skewed, mean ~6
    "screen_time": np.random.gamma(2.5, 2.5, n_samples),  # Mean ~6.25
    "number_of_courses": np.random.randint(2, 9, n_samples),
    
    # Sleep & Rest
    "sleep_hours": np.random.normal(7, 1.5, n_samples),  # Mean 7, realistic variation
    "sleep_quality": np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.25, 0.40, 0.20]),  # Weighted towards fair/good
    
    # Mental Health (correlated with stress)
    "stress_level": np.random.randint(1, 11, n_samples),
}

# Generate correlated anxiety and depression scores (higher stress = higher anxiety/depression)
stress_normalized = (data["stress_level"] - 1) / 9  # Normalize to 0-1
data["anxiety_score"] = np.clip(
    stress_normalized * 10 + np.random.normal(0, 2, n_samples), 1, 10
).astype(int)
data["depression_score"] = np.clip(
    stress_normalized * 8 + np.random.normal(0, 2.5, n_samples), 1, 10
).astype(int)

# Support & Coping (inversely correlated with stress/mental health)
data["physical_activity"] = np.clip(
    np.random.gamma(2, 2, n_samples) * (1.2 - stress_normalized * 0.5), 0, 20
)
data["coping_strategy_score"] = np.clip(
    np.random.normal(6, 2, n_samples) * (1.3 - stress_normalized * 0.4), 1, 10
).astype(int)
data["social_support_score"] = np.clip(
    np.random.normal(6, 2.5, n_samples) * (1.2 - stress_normalized * 0.3), 1, 10
).astype(int)
data["academic_self_efficacy"] = np.clip(
    np.random.normal(6.5, 2, n_samples) * (1.2 - stress_normalized * 0.4), 1, 10
).astype(int)

# Clip study/screen hours to realistic ranges
data["study_hours"] = np.clip(data["study_hours"], 0.5, 16)
data["screen_time"] = np.clip(data["screen_time"], 1, 16)
data["sleep_hours"] = np.clip(data["sleep_hours"], 3, 12)

df = pd.DataFrame(data)

def calculate_mbi_score(row):
    """
    Calculate MBI score based on comprehensive student factors.
    MBI-SS range: 0-54 (Maslach Burnout Inventory for Students)
    - Low burnout: 0-16
    - Moderate burnout: 17-26
    - High burnout: 27-54
    
    Research-based weights reflecting validated burnout predictors.
    """
    # Risk factors (increase burnout)
    mental_health_score = (
        row['stress_level'] * 2.2 +
        row['anxiety_score'] * 1.8 +
        row['depression_score'] * 2.0
    )
    
    academic_load = (
        row['study_hours'] * 1.2 +
        row['screen_time'] * 0.8 +
        row['number_of_courses'] * 1.5
    )
    
    # Sleep penalty (poor sleep increases burnout)
    sleep_penalty = 0
    if row['sleep_hours'] < 6:
        sleep_penalty = 8
    elif row['sleep_hours'] < 7:
        sleep_penalty = 4
    
    sleep_penalty += (4 - row['sleep_quality']) * 2  # Poor quality adds penalty
    
    # Low self-efficacy increases burnout
    efficacy_penalty = (10 - row['academic_self_efficacy']) * 0.8
    
    # Protective factors (decrease burnout)
    protective_factors = (
        row['physical_activity'] * 0.6 +
        row['coping_strategy_score'] * 1.0 +
        row['social_support_score'] * 1.2
    )
    
    # Calculate base score
    base_score = (
        mental_health_score * 0.45 +
        academic_load * 0.25 +
        sleep_penalty * 0.4 +
        efficacy_penalty * 0.3 -
        protective_factors * 0.35
    )
    
    # Add realistic variability
    noise = np.random.normal(0, 2.5)
    final_score = base_score + noise
    
    # Non-linear effects for extreme cases
    if row['anxiety_score'] >= 9 or row['depression_score'] >= 9:
        final_score += 5
    if row['stress_level'] >= 9 and row['sleep_hours'] < 6:
        final_score += 4
    if row['social_support_score'] <= 2:
        final_score += 3
    
    # Clip to MBI range
    return np.clip(final_score, 0, 54)

# apply MBI score calculation
df['mbi_burnout_score'] = df.apply(calculate_mbi_score, axis=1)

# round to 2 decimal place
df['mbi_burnout_score'] = df['mbi_burnout_score'].round(2)

# Sort by burnout score for better visualization
df = df.sort_values('mbi_burnout_score').reset_index(drop=True)

# 
score = df['mbi_burnout_score']

output_path = Path(__file__).parent / "MBIdata_expanded.csv"
df.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print(f"ENHANCED MBI TRAINING DATA GENERATION COMPLETE")
print(f"{'='*70}")
print(f"\nDataset saved to: {output_path}")
print(f"Total samples: {n_samples}")
print(f"\nFeatures included: 12")
print(f"  - Academic: study_hours, screen_time, number_of_courses")
print(f"  - Sleep: sleep_hours, sleep_quality")
print(f"  - Mental Health: stress_level, anxiety_score, depression_score")
print(f"  - Support: physical_activity, coping_strategy_score, social_support_score, academic_self_efficacy")

print(f"\n{'='*70}")
print(f"MBI BURNOUT SCORE DISTRIBUTION")
print(f"{'='*70}")
low_count = len(df[df['mbi_burnout_score'] <= 16])
mod_count = len(df[(df['mbi_burnout_score'] > 16) & (df['mbi_burnout_score'] <= 26)])
high_count = len(df[df['mbi_burnout_score'] > 26])

print(f"Low burnout (0-16):       {low_count:4d} samples ({low_count/n_samples*100:5.1f}%)")
print(f"Moderate burnout (17-26): {mod_count:4d} samples ({mod_count/n_samples*100:5.1f}%)")
print(f"High burnout (27-54):     {high_count:4d} samples ({high_count/n_samples*100:5.1f}%)")

print(f"\nMBI Score Statistics:")
print(f"  Mean:   {df['mbi_burnout_score'].mean():.2f}")
print(f"  Median: {df['mbi_burnout_score'].median():.2f}")
print(f"  Std:    {df['mbi_burnout_score'].std():.2f}")
print(f"  Min:    {df['mbi_burnout_score'].min():.2f}")
print(f"  Max:    {df['mbi_burnout_score'].max():.2f}")
print(f"{'='*70}\n")
