
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Demonstration training data based on transparent risk rules.
# Replace with real, labelled historical outcomes before production use.

rng = np.random.default_rng(42)
n = 3000

attendance = rng.uniform(40, 100, n)
internal = rng.uniform(35, 100, n)
assignment = rng.uniform(35, 100, n)
previous = rng.uniform(35, 100, n)

df = pd.DataFrame({
    "attendance_percent": attendance,
    "internal_marks": internal,
    "assignment_percent": assignment,
    "previous_performance": previous
})

# Attendance is the main risk indicator.
# Academic performance provides additional context.
academic = (
    0.4 * internal
    + 0.3 * assignment
    + 0.3 * previous
)

score = 0.75 * attendance + 0.25 * academic

def assign_risk(row):
    if row["attendance_percent"] < 65 or row["score"] < 65:
        return "HIGH"
    elif row["attendance_percent"] < 80 or row["score"] < 82:
        return "MEDIUM"
    else:
        return "LOW"

labels = pd.DataFrame({
    "attendance_percent": attendance,
    "score": score
}).apply(assign_risk, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    df,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("THREE-CLASS RISK MODEL")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=["LOW", "MEDIUM", "HIGH"]))

model_path = Path(__file__).resolve().parent / "attendance_risk_model_3class.pkl"

joblib.dump(model, model_path)

print("Model saved to:", model_path)
print("Model classes:", model.classes_)
