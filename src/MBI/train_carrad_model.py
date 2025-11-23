"""
Train multi-output model to predict MBI burnout dimensions (Exhaustion, Cynicism, Efficacy)
using CARRAD medical student dataset.

Outputs:
- mbi_ex: Exhaustion (5-30) - primary burnout indicator
- mbi_cy: Cynicism (0-30) - depersonalization/detachment
- mbi_ea: Efficacy (10-36) - professional efficacy (higher = better)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load CARRAD dataset
here = Path(__file__).parent
data_path = here / "carrad_dataset.csv"

if not data_path.exists():
    raise SystemExit(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

print("=" * 70)
print("CARRAD MEDICAL STUDENT BURNOUT MODEL TRAINING")
print("=" * 70)
print(f"\nLoaded {len(df)} samples from {data_path.name}")

# Define features based on requirements
feature_cols = [
    "stud_h",      # Study hours per week
    "health",      # Self-reported health (1-5)
    "psyt",        # Psychological distress (0/1 binary)
    "cesd",        # Depression score
    "stai_t",      # Anxiety (trait) score
    "amsp",        # Academic motivation
    "qcae_cog",    # Cognitive empathy
    "qcae_aff",    # Affective empathy
    "jspe",        # Job satisfaction / professional engagement
    "age",         # Age
    "year",        # Year of study (1-6)
    "sex"          # Sex (1=M, 2=F, 3=other)
]

# Target variables (3 MBI dimensions)
target_cols = ["mbi_ex", "mbi_cy", "mbi_ea"]

# Prepare data
X = df[feature_cols].copy()
y = df[target_cols].copy()

# Handle any categorical encoding if needed (sex is already numeric)
print(f"\nFeatures: {len(feature_cols)}")
print(f"Targets: {target_cols}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Create pipeline with multi-output regressor
base_estimator = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiOutputRegressor(base_estimator))
])

print("\nTraining multi-output model...")
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Clip predictions to valid ranges
# mbi_ex: 5-30, mbi_cy: 0-30, mbi_ea: 10-36
y_pred_clipped = np.copy(y_pred)
y_pred_clipped[:, 0] = np.clip(y_pred[:, 0], 5, 30)   # Exhaustion
y_pred_clipped[:, 1] = np.clip(y_pred[:, 1], 0, 30)   # Cynicism
y_pred_clipped[:, 2] = np.clip(y_pred[:, 2], 10, 36)  # Efficacy

# Calculate metrics for each output
print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

metrics = {}
for i, target in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_clipped[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_clipped[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred_clipped[:, i])
    
    metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}
    
    print(f"\n{target.upper()} (Exhaustion)" if target == "mbi_ex" else 
          f"\n{target.upper()} (Cynicism)" if target == "mbi_cy" else
          f"\n{target.upper()} (Efficacy)")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

# Feature importance (from first output - Exhaustion)
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (for Exhaustion prediction)")
print("=" * 70)

# Get feature importance from the first estimator
feature_importance = pipe.named_steps['regressor'].estimators_[0].feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    print(f"  {row['Feature']:15s}: {row['Importance']:.3f}")

# Distribution analysis
print("\n" + "=" * 70)
print("BURNOUT SCORE DISTRIBUTIONS (Test Set)")
print("=" * 70)

for target in target_cols:
    values = y_test[target]
    print(f"\n{target.upper()}:")
    print(f"  Mean: {values.mean():.1f}")
    print(f"  Std:  {values.std():.1f}")
    print(f"  Min:  {values.min():.1f}")
    print(f"  Max:  {values.max():.1f}")

# Define burnout thresholds based on MBI-SS literature
burnout_thresholds = {
    "mbi_ex": {"low": 13, "moderate": 20, "high": 30},  # Exhaustion
    "mbi_cy": {"low": 6, "moderate": 13, "high": 30},   # Cynicism
    "mbi_ea": {"low": 22, "moderate": 28, "high": 36}   # Efficacy (inverted - low is bad)
}

# Save model bundle
model_path = here / "carrad_mbi_model.joblib"
bundle = {
    "pipeline": pipe,
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "burnout_thresholds": burnout_thresholds,
    "metrics": metrics,
    "feature_importance": importance_df.to_dict('records')
}

joblib.dump(bundle, model_path)

print("\n" + "=" * 70)
print(f"Model saved to: {model_path}")
print("=" * 70)
