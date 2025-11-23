import numpy as np
from sklearn.metrics import f1_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from imrahn.test import risk_category,preds


# convert the continuous predictions into categorical labels
y_pred_labels = [risk_category(p) for p in preds]


# give labels for ground truth based on known thresholds
y_true = [
    "High Risk",
    "Medium Risk",
    "Low Risk"
]


# Map the string labels to integers for F1 score calculation
encoder = LabelEncoder()
y_true_encoded = encoder.fit_transform(y_true)
y_pred_encoded = encoder.transform(y_pred_labels)

# compute F1 score
f1=f1_score(y_true_encoded, y_pred_encoded,average='macro') # macro used for multi-class and imbalanced
print("F1 Score:",f1)
