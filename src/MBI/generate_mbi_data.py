import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(67)
n_samples = 1000

# features
data = {
    "study_hours": np.random.uniform(1, 14, n_samples),
    "screen_time": np.random.uniform(2, 15, n_samples),
    "stress_level": np.random.randint(1, 11, n_samples),
    "physical_activity": np.random.uniform(0, 20, n_samples),
    "number_of_courses": np.random.randint(2, 9, n_samples),
}

df = pd.DataFrame(data)

def calculate_mbi_score(row):
    """
    Calculate MBI score based on student factors.
    MBI-SS range: 0-54
    - Low burnout: 0-16
    - Moderate burnout: 17-26
    - High burnout: 27-54
    """
    score = (
        row['study_hours'] * 2.0 +
        row['screen_time'] * 1.5 +
        row['stress_level'] * 3.0 +
        row['number_of_courses'] * 2.5 -
        row['physical_activity'] * 1.0
    )
    
    # normalize to 0-54 range
    normalized = ((score + 20) / 140) * 54
    
    # random noise
    noise = np.random.normal(0, 3)
    final_score = normalized + noise
    
    #  lip to MBI range
    return np.clip(final_score, 0, 54)

# apply MBI score calculation
df['mbi_burnout_score'] = df.apply(calculate_mbi_score, axis=1)

# round to 2 decimal place
df['mbi_burnout_score'] = df['mbi_burnout_score'].round(2)

# Sort by burnout score for better visualization
df = df.sort_values('mbi_burnout_score').reset_index(drop=True)

# 
score = df['mbi_burnout_score']

output_path = Path(__file__).parent / "mbi_burnout_data.csv"
df.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print(f"Low burnout (0-16): {len(df[df['mbi_burnout_score'] <= 16])} samples ({len(df[df['mbi_burnout_score'] <= 16])/n_samples*100:.1f}%)")
print(f"Moderate burnout (17-26): {len(df[(df['mbi_burnout_score'] > 16) & (df['mbi_burnout_score'] <= 26)])} samples ({len(df[(df['mbi_burnout_score'] > 16) & (df['mbi_burnout_score'] <= 26)])/n_samples*100:.1f}%)")
print(f"High burnout (27-54): {len(df[df['mbi_burnout_score'] > 26])} samples ({len(df[df['mbi_burnout_score'] > 26])/n_samples*100:.1f}%)")
