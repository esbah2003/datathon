"""
Train model to predict MBI burnout scores (0-54 range).
Uses actual MBI thresholds for categories:
- Low burnout: 0-16
- Moderate burnout: 17-26
- High burnout: 27-54
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
here = Path(__file__).parent
data_path = here / "MBIdata_expanded.csv"

if not data_path.exists():
    raise SystemExit(
        "run generate_mbi_data.py first to create the dataset."
    )

df = pd.read_csv(data_path)

print("-" * 60)
print("MBI BURNOUT PREDICTION MODEL TRAINING")
print("-" * 60)
print(f"\nLoaded {len(df)} samples from {data_path.name}")

# features and target
feature_cols = [
    "study_hours",
    "screen_time", 
    "stress_level",
    "physical_activity",
    "number_of_courses"
]

X = df[feature_cols]
y = df["mbi_burnout_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

pipe = Pipeline([("scaler", StandardScaler()),("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1))])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_pred_clipped = np.clip(y_pred, 0, 54) # clip data to stay within the MBI defined ranges

mae = mean_absolute_error(y_test, y_pred_clipped)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
r2 = r2_score(y_test, y_pred_clipped)


# MBI Category accuracy
def get_mbi_category(score):
    """Categorize MBI score using official thresholds."""
    if score <= 16:
        return "Low"
    elif score <= 26:
        return "Moderate"
    else:
        return "High"

y_test_cat = y_test.apply(get_mbi_category).reset_index(drop=True)
y_pred_cat = pd.Series(y_pred_clipped).apply(get_mbi_category).reset_index(drop=True)

category_accuracy = (y_test_cat == y_pred_cat).mean()

print(f"\nMBI Category Accuracy: {category_accuracy:.1%}")

# Distribution in test set
print("\nTest Set MBI Category Distribution:")
print(f"  Low (0-16):       {(y_test <= 16).sum()} samples")
print(f"  Moderate (17-26): {((y_test > 16) & (y_test <= 26)).sum()} samples")
print(f"  High (27-54):     {(y_test > 26).sum()} samples")

# feature importance
feature_importance = pipe.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_cols,'Importance': feature_importance}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for _, row in importance_df.iterrows():
    print(f"  {row['Feature']:20s}: {row['Importance']:.2f}")

# save model
model_path = here / "mbi_model.joblib"
bundle = {
    "pipeline": pipe,
    "feature_cols": feature_cols,
    "mbi_thresholds": {
        "low": 16,
        "moderate": 26,
        "high": 54
    },
    "metrics": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "category_accuracy": category_accuracy
    }
}

joblib.dump(bundle, model_path)