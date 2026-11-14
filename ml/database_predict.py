import pandas as pd
import oracledb
import joblib


# ============================================
# ORACLE DATABASE CONNECTION
# ============================================

connection = oracledb.connect(
    user="system",
    password="OracleDB2026",
    dsn="localhost:1521/FREEPDB1"
)

cursor = connection.cursor()

print("========================================")
print("Oracle Database Connected")
print("========================================")


# ============================================
# LOAD ML MODEL
# ============================================

model = joblib.load("attendance_risk_model.pkl")

print("ML Model Loaded Successfully")
print("========================================")


# ============================================
# GET STUDENT DATA FROM ORACLE
# ============================================

query = """
SELECT
    s.STUDENT_ID,
    s.STUDENT_NAME,
    a.ATTENDANCE_PERCENT,
    p.INTERNAL_MARKS,
    p.ASSIGNMENT_PERCENT
FROM STUDENT s

JOIN ATTENDANCE a
    ON s.STUDENT_ID = a.STUDENT_ID

JOIN ACADEMIC_PERFORMANCE p
    ON s.STUDENT_ID = p.STUDENT_ID

ORDER BY s.STUDENT_ID
"""

cursor.execute(query)

students = cursor.fetchall()


# ============================================
# PROCESS EACH STUDENT
# ============================================

for student in students:

    student_id = student[0]
    student_name = student[1]
    attendance = student[2]
    internal_marks = student[3]
    assignment = student[4]

    # Previous performance
    previous_performance = internal_marks

    # ========================================
    # PREPARE ML FEATURES
    # ========================================

    features = [[
        attendance,
        internal_marks,
        assignment,
        previous_performance
    ]]

    feature_names = [
        "attendance_percent",
        "internal_marks",
        "assignment_percent",
        "previous_performance"
    ]

    features_df = pd.DataFrame(
        features,
        columns=feature_names
    )


    # ========================================
    # ML PREDICTION
    # ========================================

    prediction = model.predict(features_df)

    probability = model.predict_proba(features_df)[0][1]


    # ========================================
    # DETERMINE RISK LEVEL
    # ========================================

    if prediction[0] == 1:
        risk_level = "HIGH"
    else:
        risk_level = "LOW"

    risk_probability = round(probability * 100, 2)


    # ========================================
    # CHECK EXISTING PREDICTION
    # ========================================

    cursor.execute(
        """
        SELECT PREDICTION_ID
        FROM RISK_PREDICTION
        WHERE STUDENT_ID = :1
        ORDER BY PREDICTION_ID DESC
        FETCH FIRST 1 ROW ONLY
        """,
        (student_id,)
    )

    existing_prediction = cursor.fetchone()


    # ========================================
    # UPDATE OR INSERT
    # ========================================

    if existing_prediction:

        prediction_id = existing_prediction[0]

        cursor.execute(
            """
            UPDATE RISK_PREDICTION
            SET
                RISK_LEVEL = :1,
                RISK_PROBABILITY = :2,
                PREDICTION_DATE = SYSDATE
            WHERE PREDICTION_ID = :3
            """,
            (
                risk_level,
                risk_probability,
                prediction_id
            )
        )

        action = "Prediction updated in Oracle."

    else:

        cursor.execute(
            """
            SELECT NVL(MAX(PREDICTION_ID), 0) + 1
            FROM RISK_PREDICTION
            """
        )

        prediction_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO RISK_PREDICTION
            (
                PREDICTION_ID,
                STUDENT_ID,
                RISK_LEVEL,
                RISK_PROBABILITY,
                PREDICTION_DATE
            )
            VALUES
            (
                :1,
                :2,
                :3,
                :4,
                SYSDATE
            )
            """,
            (
                prediction_id,
                student_id,
                risk_level,
                risk_probability
            )
        )

        action = "Prediction inserted into Oracle."


    # ========================================
    # DISPLAY RESULT
    # ========================================

    print("----------------------------------------")
    print("Student ID:", student_id)
    print("Student Name:", student_name)
    print("Attendance:", attendance, "%")
    print("Internal Marks:", internal_marks)
    print("Assignment:", assignment, "%")
    print("Risk Level:", risk_level)
    print("Risk Probability:", risk_probability, "%")
    print(action)


# ============================================
# SAVE CHANGES
# ============================================

connection.commit()


# ============================================
# CLOSE CONNECTION
# ============================================

cursor.close()
connection.close()


print("----------------------------------------")
print("All predictions processed successfully!")
print("Oracle database updated.")
print("----------------------------------------")