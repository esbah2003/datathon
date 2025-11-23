from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_and_save_model(data_path: Path, model_path: Path) -> dict:
    # Load data
    df = pd.read_csv(data_path)

    feature_names = [
        "study_hours",
        "screen_time",
        "stress_level",
        "physical_activity",
        "number_of_courses",
    ]
    target_name = "burnout_risk"

    X = df[feature_names]
    y = df[target_name]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess: just select and passthrough numeric columns in defined order
    pre = ColumnTransformer(
        transformers=[("num", "passthrough", feature_names)],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = r2_score(y_val, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    # Persist model pipeline
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "feature_names": feature_names,
        "target_name": target_name,
        "metrics": metrics,
    }, model_path)

    return metrics


if __name__ == "__main__":
    here = Path(__file__).parent
    data_path = here / "burnout_data_scores.csv"
    model_path = here / "model.joblib"

    metrics = train_and_save_model(data_path, model_path)

    print("Training complete.")
    print(f"Saved model to: {model_path}")
    print(
        "Metrics (val set):",
        {k: round(v, 4) for k, v in metrics.items()}
    )
