import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("burnout_data.csv")

# Drop risk_score column (mostly empty and computed from other features)
if 'risk_score' in df.columns:
    df = df.drop('risk_score', axis=1)

# Example: if burnout_risk = ["Low", "Medium", "High"]
label_encoder = LabelEncoder()
df["burnout_risk"] = label_encoder.fit_transform(df["burnout_risk"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importances = model.feature_importances_

for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.3f}")

# Save the model and label encoder
joblib.dump(model, "burnout_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
print("\nModel saved as 'burnout_model.pkl'")
print("Label encoder saved as 'label_encoder.pkl'")
print("Feature names saved as 'feature_names.pkl'")
