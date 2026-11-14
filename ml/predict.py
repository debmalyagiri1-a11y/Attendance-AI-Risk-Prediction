import joblib

# Load the trained model
model = joblib.load("attendance_risk_model.pkl")

# Student data
attendance = 68
internal_marks = 52
assignment_percent = 60
previous_performance = 55

# Make prediction
prediction = model.predict([[
    attendance,
    internal_marks,
    assignment_percent,
    previous_performance
]])

probability = model.predict_proba([[
    attendance,
    internal_marks,
    assignment_percent,
    previous_performance
]])[0][1]

# Determine risk level
if prediction[0] == 1:
    risk_level = "HIGH"
else:
    risk_level = "LOW"

print("Attendance:", attendance, "%")
print("Internal Marks:", internal_marks)
print("Assignment:", assignment_percent, "%")
print("Previous Performance:", previous_performance)
print("Risk Level:", risk_level)
print("Risk Probability:", round(probability * 100, 2), "%")