"""
Model training for OptiCrop, matching the notebook pipeline shown in the
project write-up: pandas for loading/EDA, scikit-learn's train_test_split
and LogisticRegression for the classifier, evaluated with confusion_matrix
and classification_report.

Run this once (or whenever Crop_recommendation.csv changes) to regenerate
model.pkl. app.py just loads whatever model.pkl is sitting next to it, so
you can swap in a different algorithm here without touching app.py.
"""
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
data = pd.read_csv('Crop_recommendation.csv')
print(data.head())
print(data.describe())

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Train the classifier
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 5. Save the trained model for app.py
# ---------------------------------------------------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nSaved trained model to model.pkl")
