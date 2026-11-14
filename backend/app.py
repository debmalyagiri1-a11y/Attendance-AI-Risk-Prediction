
import os
import io
import csv
import hmac
import math
import logging
from functools import wraps

import joblib
import pandas as pd
import psycopg2
from psycopg2 import errors
from psycopg2.extras import RealDictCursor

from flask import Flask, jsonify, request
from flask_cors import CORS
from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired,
)
from werkzeug.security import check_password_hash


# ============================================================
# ATTENDANCE AI - NEON POSTGRESQL BACKEND
# Flask + Random Forest + Admin Authentication
# ============================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,"
        "http://localhost:5500,"
        "http://127.0.0.1:5501,"
        "http://localhost:5501",
    ).split(",")
    if origin.strip()
]

CORS(
    app,
    resources={
        r"/*": {"origins": FRONTEND_ORIGINS}
    },
    allow_headers=["Content-Type", "Authorization"],
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# ADMIN AUTHENTICATION
# ============================================================

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
ADMIN_AUTH_SECRET = os.getenv("ADMIN_AUTH_SECRET", "")

TOKEN_TTL_SECONDS = 8 * 60 * 60

if not ADMIN_PASSWORD_HASH:
    raise RuntimeError(
        "ADMIN_PASSWORD_HASH environment variable is missing."
    )

if len(ADMIN_AUTH_SECRET) < 32:
    raise RuntimeError(
        "ADMIN_AUTH_SECRET must contain at least 32 characters."
    )

serializer = URLSafeTimedSerializer(
    ADMIN_AUTH_SECRET,
    salt="attendance-admin-v1",
)


def require_admin(function):
    @wraps(function)
    def protected(*args, **kwargs):
        authorization = request.headers.get(
            "Authorization", ""
        )

        if not authorization.startswith("Bearer "):
            return jsonify(
                error="Admin login required."
            ), 401

        token = authorization[7:].strip()

        if not token:
            return jsonify(
                error="Authentication token missing."
            ), 401

        try:
            payload = serializer.loads(
                token,
                max_age=TOKEN_TTL_SECONDS,
            )
        except SignatureExpired:
            return jsonify(
                error="Session expired. Please log in again."
            ), 401
        except BadSignature:
            return jsonify(
                error="Invalid authentication token."
            ), 401

        if not isinstance(payload, dict):
            return jsonify(
                error="Invalid session."
            ), 401

        valid_role = payload.get("role") == "admin"
        valid_username = hmac.compare_digest(
            str(payload.get("username", "")),
            ADMIN_USERNAME,
        )

        if not valid_role or not valid_username:
            return jsonify(
                error="Administrator access denied."
            ), 403

        return function(*args, **kwargs)

    return protected


@app.post("/auth/login")
def admin_login():
    data = request.get_json(silent=True) or {}

    if not isinstance(data, dict):
        return jsonify(
            error="Invalid login request."
        ), 400

    username = str(data.get("username", ""))
    password = str(data.get("password", ""))

    username_valid = hmac.compare_digest(
        username, ADMIN_USERNAME
    )

    password_valid = check_password_hash(
        ADMIN_PASSWORD_HASH, password
    )

    if not (username_valid and password_valid):
        return jsonify(
            error="Invalid username or password."
        ), 401

    token = serializer.dumps({
        "username": ADMIN_USERNAME,
        "role": "admin",
    })

    return jsonify(
        message="Admin login successful.",
        token=token,
        token_type="Bearer",
        role="admin",
        username=ADMIN_USERNAME,
        expires_in=TOKEN_TTL_SECONDS,
    )


@app.get("/auth/me")
@require_admin
def admin_me():
    return jsonify(
        authenticated=True,
        username=ADMIN_USERNAME,
        role="admin",
    )


# ============================================================
# NEON POSTGRESQL CONNECTION
# ============================================================

def db():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is missing."
        )

    return psycopg2.connect(
        database_url,
        connect_timeout=15,
    )


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE,
    "ml",
    "attendance_risk_model_3class.pkl",
)

model = joblib.load(MODEL_PATH)


def predict(attendance, internal, assignment, previous):
    features = pd.DataFrame([{
        "attendance_percent": float(attendance),
        "internal_marks": float(internal),
        "assignment_percent": float(assignment),
        "previous_performance": float(previous),
    }])

    risk_level = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    class_index = list(model.classes_).index(
        risk_level
    )

    confidence = round(
        float(probabilities[class_index]) * 100,
        2,
    )

    return risk_level, confidence


# ============================================================
# INPUT VALIDATION
# ============================================================

def valid(data):
    if not isinstance(data, dict):
        return False, "Invalid student data."

    required = [
        "student_id",
        "student_name",
        "department",
        "semester",
        "total_classes",
        "classes_attended",
        "internal_marks",
        "assignment_percent",
        "previous_performance",
    ]

    missing = [
        key for key in required
        if data.get(key) in (None, "")
    ]

    if missing:
        return False, "Missing: " + ", ".join(missing)

    if not str(data["student_name"]).strip():
        return False, "Student name is required."

    if not str(data["department"]).strip():
        return False, "Department is required."

    try:
        sid = int(data["student_id"])
        semester = int(data["semester"])
        total = int(data["total_classes"])
        attended = int(data["classes_attended"])

        marks = [
            float(data["internal_marks"]),
            float(data["assignment_percent"]),
            float(data["previous_performance"]),
        ]

    except (ValueError, TypeError, OverflowError):
        return False, "Invalid numeric value."

    if (
        sid <= 0
        or semester <= 0
        or total <= 0
        or attended < 0
        or attended > total
    ):
        return False, "Invalid ID, semester or attendance."

    if any(
        not math.isfinite(value)
        or value < 0
        or value > 100
        for value in marks
    ):
        return False, "Marks must be between 0 and 100."

    return True, ""


# ============================================================
# HELPERS
# ============================================================

def format_date(value):
    return value.strftime("%d-%b-%Y") if value else None


def format_datetime(value):
    return value.strftime("%d-%b-%Y %H:%M") if value else None


def next_id(cursor, table, column):
    # Only internal, fixed table/column names are passed here.
    allowed = {
        ("attendance", "attendance_id"),
        ("academic_performance", "performance_id"),
        ("risk_prediction", "prediction_id"),
    }

    if (table, column) not in allowed:
        raise ValueError("Invalid table or column.")

    # Locks the table to prevent concurrent MAX+1 collisions.
    cursor.execute(
        f"LOCK TABLE {table} IN SHARE ROW EXCLUSIVE MODE"
    )

    cursor.execute(
        f"SELECT COALESCE(MAX({column}), 0) + 1 FROM {table}"
    )

    return int(cursor.fetchone()[0])


def row_to_student(row):
    return {
        "student_id": row["student_id"],
        "student_name": row["student_name"],
        "department": row["department"],
        "semester": row["semester"],
        "email": row["email"],
        "total_classes": row["total_classes"],
        "classes_attended": row["classes_attended"],
        "attendance": float(row["attendance_percent"]),
        "internal_marks": float(row["internal_marks"]),
        "assignment": float(row["assignment_percent"]),
        "previous_result": row["previous_result"],
        "risk_level": row["risk_level"],
        "risk_probability": float(row["risk_probability"]),
        "prediction_date": format_date(
            row["prediction_date"]
        ),
    }


def insert_student(cursor, data):
    sid = int(data["student_id"])
    total = int(data["total_classes"])
    attended = int(data["classes_attended"])

    attendance = round(attended / total * 100, 2)

    internal = float(data["internal_marks"])
    assignment = float(data["assignment_percent"])
    previous = float(data["previous_performance"])

    risk, confidence = predict(
        attendance,
        internal,
        assignment,
        previous,
    )

    previous_result = (
        "Good" if previous >= 75
        else "Average" if previous >= 50
        else "Poor"
    )

    cursor.execute(
        """
        INSERT INTO student (
            student_id,
            student_name,
            email,
            department,
            semester
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            sid,
            str(data["student_name"]).strip(),
            str(data.get("email") or "").strip() or None,
            str(data["department"]).strip(),
            int(data["semester"]),
        ),
    )

    attendance_id = next_id(
        cursor, "attendance", "attendance_id"
    )

    cursor.execute(
        """
        INSERT INTO attendance (
            attendance_id,
            student_id,
            total_classes,
            classes_attended,
            attendance_percent
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            attendance_id,
            sid,
            total,
            attended,
            attendance,
        ),
    )

    performance_id = next_id(
        cursor,
        "academic_performance",
        "performance_id",
    )

    cursor.execute(
        """
        INSERT INTO academic_performance (
            performance_id,
            student_id,
            internal_marks,
            assignment_percent,
            previous_result
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            performance_id,
            sid,
            internal,
            assignment,
            previous_result,
        ),
    )

    prediction_id = next_id(
        cursor, "risk_prediction", "prediction_id"
    )

    cursor.execute(
        """
        INSERT INTO risk_prediction (
            prediction_id,
            student_id,
            risk_level,
            risk_probability,
            prediction_date
        )
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (
            prediction_id,
            sid,
            risk,
            confidence,
        ),
    )

    return {
        "student_id": sid,
        "risk_level": risk,
        "risk_probability": confidence,
    }


# ============================================================
# PUBLIC API HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return jsonify(
        message="AI Student Attendance Risk Management API",
        status="running",
        version="4.0",
        database="Neon PostgreSQL",
        authentication="admin_required",
    )


# ============================================================
# ALL STUDENT PREDICTIONS
# ============================================================

STUDENT_PREDICTIONS_SQL = """
    SELECT
        s.student_id,
        s.student_name,
        s.department,
        s.semester,
        s.email,
        a.total_classes,
        a.classes_attended,
        a.attendance_percent,
        p.internal_marks,
        p.assignment_percent,
        p.previous_result,
        r.risk_level,
        r.risk_probability,
        r.prediction_date
    FROM student s
    JOIN attendance a
        ON s.student_id = a.student_id
    JOIN academic_performance p
        ON s.student_id = p.student_id
    JOIN risk_prediction r
        ON s.student_id = r.student_id
    WHERE r.prediction_id = (
        SELECT MAX(r2.prediction_id)
        FROM risk_prediction r2
        WHERE r2.student_id = s.student_id
    )
"""


@app.get("/predictions")
@require_admin
def predictions():
    try:
        with db() as connection:
            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:
                cursor.execute(
                    STUDENT_PREDICTIONS_SQL
                    + " ORDER BY s.student_id"
                )
                rows = cursor.fetchall()

        return jsonify([
            row_to_student(row) for row in rows
        ])

    except Exception:
        app.logger.exception(
            "Unable to load predictions"
        )
        return jsonify(
            error="Unable to load student predictions."
        ), 500


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/predictions/summary")
@require_admin
def summary():
    try:
        with db() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(
                            CASE WHEN r.risk_level = 'HIGH'
                            THEN 1 ELSE 0 END
                        ), 0),
                        COALESCE(SUM(
                            CASE WHEN r.risk_level = 'MEDIUM'
                            THEN 1 ELSE 0 END
                        ), 0),
                        COALESCE(SUM(
                            CASE WHEN r.risk_level = 'LOW'
                            THEN 1 ELSE 0 END
                        ), 0),
                        ROUND(
                            AVG(a.attendance_percent), 2
                        )
                    FROM student s
                    JOIN attendance a
                        ON a.student_id = s.student_id
                    JOIN risk_prediction r
                        ON r.student_id = s.student_id
                    WHERE r.prediction_id = (
                        SELECT MAX(r2.prediction_id)
                        FROM risk_prediction r2
                        WHERE r2.student_id = s.student_id
                    )
                """)

                row = cursor.fetchone()

        return jsonify(
            total_students=int(row[0] or 0),
            high_risk=int(row[1] or 0),
            medium_risk=int(row[2] or 0),
            low_risk=int(row[3] or 0),
            average_attendance=float(row[4] or 0),
        )

    except Exception:
        app.logger.exception(
            "Unable to load summary"
        )
        return jsonify(
            error="Unable to load dashboard summary."
        ), 500


# ============================================================
# SINGLE STUDENT DETAILS
# ============================================================

@app.get("/student/<int:sid>")
@require_admin
def student(sid):
    try:
        with db() as connection:
            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:
                cursor.execute(
                    STUDENT_PREDICTIONS_SQL
                    + " AND s.student_id = %s",
                    (sid,),
                )
                row = cursor.fetchone()

        if not row:
            return jsonify(
                error="Student not found."
            ), 404

        return jsonify(row_to_student(row))

    except Exception:
        app.logger.exception(
            "Unable to load student"
        )
        return jsonify(
            error="Unable to load student."
        ), 500


# ============================================================
# ADD STUDENT
# ============================================================

@app.post("/students")
@require_admin
def add_student():
    data = request.get_json(silent=True) or {}

    ok, message = valid(data)

    if not ok:
        return jsonify(error=message), 400

    try:
        with db() as connection:
            with connection.cursor() as cursor:
                result = insert_student(cursor, data)

        return jsonify(
            message="Student added and prediction generated.",
            **result,
        ), 201

    except errors.UniqueViolation:
        return jsonify(
            error="Student ID already exists."
        ), 409

    except Exception:
        app.logger.exception(
            "Unable to add student"
        )
        return jsonify(
            error="Unable to add student."
        ), 500


# ============================================================
# DELETE STUDENT
# ============================================================

@app.delete("/student/<int:sid>")
@require_admin
def delete_student(sid):
    try:
        with db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT student_id
                    FROM student
                    WHERE student_id = %s
                    FOR UPDATE
                    """,
                    (sid,),
                )

                if not cursor.fetchone():
                    return jsonify(
                        error="Student not found."
                    ), 404

                for table in (
                    "risk_prediction_history",
                    "risk_prediction",
                    "academic_performance",
                    "attendance",
                    "student",
                ):
                    cursor.execute(
                        f"DELETE FROM {table} "
                        "WHERE student_id = %s",
                        (sid,),
                    )

        return jsonify(
            message=f"Student {sid} deleted."
        )

    except Exception:
        app.logger.exception(
            "Unable to delete student"
        )
        return jsonify(
            error="Unable to delete student."
        ), 500


# ============================================================
# CSV BULK IMPORT
# ============================================================

@app.post("/students/import")
@require_admin
def import_students():
    if "file" not in request.files:
        return jsonify(
            error="Attach a CSV file."
        ), 400

    uploaded = request.files["file"]

    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify(
            error="Only CSV files are supported."
        ), 400

    try:
        content = uploaded.read().decode("utf-8-sig")
        reader = csv.DictReader(
            io.StringIO(content)
        )
        rows = list(reader)

    except (UnicodeError, csv.Error):
        return jsonify(
            error="Unable to read CSV file."
        ), 400

    created = []
    skipped = []
    validation_errors = []

    try:
        with db() as connection:
            with connection.cursor() as cursor:
                for line_number, row in enumerate(
                    rows, start=2
                ):
                    ok, message = valid(row)

                    if not ok:
                        validation_errors.append({
                            "line": line_number,
                            "error": message,
                        })
                        continue

                    sid = int(row["student_id"])

                    # Savepoint isolates each imported student.
                    cursor.execute(
                        "SAVEPOINT import_student"
                    )

                    try:
                        result = insert_student(
                            cursor, row
                        )
                        created.append({
                            "student_id": sid,
                            "risk_level": result["risk_level"],
                        })

                    except errors.UniqueViolation:
                        cursor.execute(
                            "ROLLBACK TO SAVEPOINT import_student"
                        )
                        skipped.append(sid)

                    except Exception:
                        cursor.execute(
                            "ROLLBACK TO SAVEPOINT import_student"
                        )
                        app.logger.exception(
                            "CSV row %s failed",
                            line_number,
                        )
                        validation_errors.append({
                            "line": line_number,
                            "error": "Unable to import this row.",
                        })

                    finally:
                        cursor.execute(
                            "RELEASE SAVEPOINT import_student"
                        )

        return jsonify(
            message="CSV import completed.",
            created=created,
            skipped_existing=skipped,
            errors=validation_errors,
        )

    except Exception:
        app.logger.exception(
            "CSV import failed"
        )
        return jsonify(
            error="CSV import failed."
        ), 500


# ============================================================
# RISK PREDICTION HISTORY
# ============================================================

@app.get("/risk-history/<int:sid>")
@require_admin
def history(sid):
    try:
        with db() as connection:
            with connection.cursor(
                cursor_factory=RealDictCursor
            ) as cursor:
                cursor.execute("""
                    SELECT
                        risk_level,
                        risk_probability,
                        prediction_date,
                        recorded_at,
                        change_type
                    FROM risk_prediction_history
                    WHERE student_id = %s
                    ORDER BY recorded_at DESC, history_id DESC
                """, (sid,))

                rows = cursor.fetchall()

        return jsonify([
            {
                "risk_level": row["risk_level"],
                "risk_probability": float(
                    row["risk_probability"]
                ),
                "prediction_date": format_date(
                    row["prediction_date"]
                ),
                "recorded_at": format_datetime(
                    row["recorded_at"]
                ),
                "change_type": row["change_type"],
            }
            for row in rows
        ])

    except Exception:
        app.logger.exception(
            "Unable to load risk history"
        )
        return jsonify(
            error="Unable to load prediction history."
        ), 500


# ============================================================
# RECALCULATE ALL AI PREDICTIONS
# ============================================================

@app.post("/predictions/recalculate")
@require_admin
def recalculate_predictions():
    try:
        with db() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        s.student_id,
                        a.attendance_percent,
                        p.internal_marks,
                        p.assignment_percent
                    FROM student s
                    JOIN attendance a
                        ON s.student_id = a.student_id
                    JOIN academic_performance p
                        ON s.student_id = p.student_id
                """)

                records = cursor.fetchall()

                updated = 0
                counts = {
                    "LOW": 0,
                    "MEDIUM": 0,
                    "HIGH": 0,
                }

                prediction_id = next_id(
                    cursor,
                    "risk_prediction",
                    "prediction_id",
                ) - 1

                for (
                    sid,
                    attendance,
                    internal,
                    assignment,
                ) in records:
                    # Existing schema stores previous result
                    # as Good/Average/Poor, not numeric score.
                    # Preserve the original internal-marks proxy.
                    previous = internal

                    risk_level, confidence = predict(
                        attendance,
                        internal,
                        assignment,
                        previous,
                    )

                    prediction_id += 1

                    cursor.execute("""
                        INSERT INTO risk_prediction (
                            prediction_id,
                            student_id,
                            risk_level,
                            risk_probability,
                            prediction_date
                        )
                        VALUES (
                            %s, %s, %s, %s,
                            CURRENT_TIMESTAMP
                        )
                    """, (
                        prediction_id,
                        sid,
                        risk_level,
                        confidence,
                    ))

                    counts[risk_level] = (
                        counts.get(risk_level, 0) + 1
                    )
                    updated += 1

        return jsonify(
            status="success",
            students_recalculated=updated,
            risk_distribution=counts,
        )

    except Exception:
        app.logger.exception(
            "Recalculation failed"
        )
        return jsonify(
            error="Unable to recalculate predictions."
        ), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("ATTENDANCE AI - ADMIN BACKEND")
    print("Neon PostgreSQL + Random Forest + Flask")
    print("Admin authentication enabled")
    print("http://127.0.0.1:5000")
    print("=" * 55)

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
