import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Sample training data
data = {
    "attendance_percent": [95, 90, 85, 82, 78, 72, 68, 65, 60, 55],
    "internal_marks": [90, 85, 80, 78, 75, 65, 60, 55, 50, 45],
    "assignment_percent": [95, 90, 85, 82, 80, 70, 65, 60, 55, 50],
    "previous_performance": [90, 85, 80, 78, 75, 65, 60, 55, 50, 45],
    "risk": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features
X = df[
    [
        "attendance_percent",
        "internal_marks",
        "assignment_percent",
        "previous_performance"
    ]
]

# Target
y = df["risk"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save trained model
joblib.dump(model, "attendance_risk_model.pkl")

print("Model trained successfully!")
print("Model saved as attendance_risk_model.pkl")