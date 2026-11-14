
import os
import io
import csv
import hmac
import math
import joblib
import oracledb
import pandas as pd

from functools import wraps
from flask import Flask, jsonify, request
from flask_cors import CORS
from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired
)
from werkzeug.security import check_password_hash


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,"
        "http://localhost:5500,"
        "http://127.0.0.1:5501,"
        "http://localhost:5501"
    ).split(",")
    if origin.strip()
]

CORS(
    app,
    resources={
        r"/*": {
            "origins": FRONTEND_ORIGINS
        }
    },
    allow_headers=[
        "Content-Type",
        "Authorization"
    ]
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# ADMIN AUTHENTICATION CONFIGURATION
# ============================================================

ADMIN_USERNAME = os.getenv(
    "ADMIN_USERNAME",
    "admin"
)

ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH",
    ""
)

ADMIN_AUTH_SECRET = os.getenv(
    "ADMIN_AUTH_SECRET",
    ""
)

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
    salt="attendance-admin-v1"
)


# ============================================================
# ADMIN AUTHENTICATION DECORATOR
# ============================================================

def require_admin(function):

    @wraps(function)
    def protected(*args, **kwargs):

        authorization = request.headers.get(
            "Authorization",
            ""
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
                max_age=TOKEN_TTL_SECONDS
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
            ADMIN_USERNAME
        )

        if not valid_role or not valid_username:
            return jsonify(
                error="Administrator access denied."
            ), 403

        return function(*args, **kwargs)

    return protected


# ============================================================
# ADMIN LOGIN
# ============================================================

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
        username,
        ADMIN_USERNAME
    )

    password_valid = check_password_hash(
        ADMIN_PASSWORD_HASH,
        password
    )

    if not (username_valid and password_valid):
        return jsonify(
            error="Invalid username or password."
        ), 401

    token = serializer.dumps({
        "username": ADMIN_USERNAME,
        "role": "admin"
    })

    return jsonify(
        message="Admin login successful.",
        token=token,
        token_type="Bearer",
        role="admin",
        username=ADMIN_USERNAME,
        expires_in=TOKEN_TTL_SECONDS
    )


# ============================================================
# CURRENT ADMIN SESSION
# ============================================================

@app.get("/auth/me")
@require_admin
def admin_me():

    return jsonify(
        authenticated=True,
        username=ADMIN_USERNAME,
        role="admin"
    )


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL = os.path.join(
    BASE,
    "ml",
    "attendance_risk_model_3class.pkl"
)

model = joblib.load(MODEL)


def predict(attendance, internal, assignment, previous):

    X = pd.DataFrame([{
        "attendance_percent": float(attendance),
        "internal_marks": float(internal),
        "assignment_percent": float(assignment),
        "previous_performance": float(previous)
    }])

    risk_level = str(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    class_index = list(model.classes_).index(
        risk_level
    )

    confidence = round(
        float(probabilities[class_index]) * 100,
        2
    )

    return risk_level, confidence


# ============================================================
# ORACLE DATABASE CONNECTION
# ============================================================

def db():

    return oracledb.connect(
        user=os.getenv("ORACLE_USER", "system"),
        password=os.environ["ORACLE_PASSWORD"],
        dsn=os.getenv(
            "ORACLE_DSN",
            "host.docker.internal:1521/FREEPDB1"
        )
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def valid(d):

    if not isinstance(d, dict):
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
        "previous_performance"
    ]

    missing = [
        key for key in required
        if d.get(key) in (None, "")
    ]

    if missing:
        return False, "Missing: " + ", ".join(missing)

    if not str(d["student_name"]).strip():
        return False, "Student name is required."

    if not str(d["department"]).strip():
        return False, "Department is required."

    try:

        sid = int(d["student_id"])
        semester = int(d["semester"])
        total = int(d["total_classes"])
        attended = int(d["classes_attended"])

        marks = [
            float(d["internal_marks"]),
            float(d["assignment_percent"]),
            float(d["previous_performance"])
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
# PUBLIC HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return jsonify(
        message="AI Student Attendance Risk Management API",
        status="running",
        version="3.0",
        authentication="admin_required"
    )


# ============================================================
# ALL STUDENT PREDICTIONS
# ============================================================

@app.get("/predictions")
@require_admin
def predictions():

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        cur.execute("""
            SELECT
                s.STUDENT_ID,
                s.STUDENT_NAME,
                s.DEPARTMENT,
                s.SEMESTER,
                s.EMAIL,
                a.TOTAL_CLASSES,
                a.CLASSES_ATTENDED,
                a.ATTENDANCE_PERCENT,
                p.INTERNAL_MARKS,
                p.ASSIGNMENT_PERCENT,
                p.PREVIOUS_RESULT,
                r.RISK_LEVEL,
                r.RISK_PROBABILITY,
                r.PREDICTION_DATE
            FROM STUDENT s

            JOIN ATTENDANCE a
                ON s.STUDENT_ID = a.STUDENT_ID

            JOIN ACADEMIC_PERFORMANCE p
                ON s.STUDENT_ID = p.STUDENT_ID

            JOIN RISK_PREDICTION r
                ON s.STUDENT_ID = r.STUDENT_ID

            WHERE r.PREDICTION_ID = (
                SELECT MAX(r2.PREDICTION_ID)
                FROM RISK_PREDICTION r2
                WHERE r2.STUDENT_ID = s.STUDENT_ID
            )

            ORDER BY s.STUDENT_ID
        """)

        results = []

        for x in cur.fetchall():

            results.append({
                "student_id": x[0],
                "student_name": x[1],
                "department": x[2],
                "semester": x[3],
                "email": x[4],
                "total_classes": x[5],
                "classes_attended": x[6],
                "attendance": float(x[7]),
                "internal_marks": float(x[8]),
                "assignment": float(x[9]),
                "previous_result": x[10],
                "risk_level": x[11],
                "risk_probability": float(x[12]),
                "prediction_date": (
                    x[13].strftime("%d-%b-%Y")
                    if x[13] else None
                )
            })

        return jsonify(results)

    except Exception:
        app.logger.exception("Unable to load predictions")
        return jsonify(
            error="Unable to load student predictions."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get("/predictions/summary")
@require_admin
def summary():

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        cur.execute("""
            SELECT
                COUNT(*),

                SUM(
                    CASE
                        WHEN r.RISK_LEVEL = 'HIGH'
                        THEN 1 ELSE 0
                    END
                ),

                SUM(
                    CASE
                        WHEN r.RISK_LEVEL = 'MEDIUM'
                        THEN 1 ELSE 0
                    END
                ),

                SUM(
                    CASE
                        WHEN r.RISK_LEVEL = 'LOW'
                        THEN 1 ELSE 0
                    END
                ),

                ROUND(AVG(a.ATTENDANCE_PERCENT), 2)

            FROM STUDENT s

            JOIN ATTENDANCE a
                ON a.STUDENT_ID = s.STUDENT_ID

            JOIN RISK_PREDICTION r
                ON r.STUDENT_ID = s.STUDENT_ID

            WHERE r.PREDICTION_ID = (
                SELECT MAX(r2.PREDICTION_ID)
                FROM RISK_PREDICTION r2
                WHERE r2.STUDENT_ID = s.STUDENT_ID
            )
        """)

        x = cur.fetchone()

        return jsonify(
            total_students=int(x[0] or 0),
            high_risk=int(x[1] or 0),
            medium_risk=int(x[2] or 0),
            low_risk=int(x[3] or 0),
            average_attendance=float(x[4] or 0)
        )

    except Exception:
        app.logger.exception("Unable to load summary")
        return jsonify(
            error="Unable to load dashboard summary."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


# ============================================================
# SINGLE STUDENT DETAILS
# ============================================================

@app.get("/student/<int:sid>")
@require_admin
def student(sid):

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        cur.execute("""
            SELECT
                s.STUDENT_ID,
                s.STUDENT_NAME,
                s.EMAIL,
                s.DEPARTMENT,
                s.SEMESTER,
                a.TOTAL_CLASSES,
                a.CLASSES_ATTENDED,
                a.ATTENDANCE_PERCENT,
                p.INTERNAL_MARKS,
                p.ASSIGNMENT_PERCENT,
                p.PREVIOUS_RESULT,
                r.RISK_LEVEL,
                r.RISK_PROBABILITY,
                r.PREDICTION_DATE

            FROM STUDENT s

            JOIN ATTENDANCE a
                ON a.STUDENT_ID = s.STUDENT_ID

            JOIN ACADEMIC_PERFORMANCE p
                ON p.STUDENT_ID = s.STUDENT_ID

            JOIN RISK_PREDICTION r
                ON r.STUDENT_ID = s.STUDENT_ID

            WHERE s.STUDENT_ID = :1

            AND r.PREDICTION_ID = (
                SELECT MAX(r2.PREDICTION_ID)
                FROM RISK_PREDICTION r2
                WHERE r2.STUDENT_ID = s.STUDENT_ID
            )
        """, (sid,))

        x = cur.fetchone()

        if not x:
            return jsonify(
                error="Student not found."
            ), 404

        return jsonify(
            student_id=x[0],
            student_name=x[1],
            email=x[2],
            department=x[3],
            semester=x[4],
            total_classes=x[5],
            classes_attended=x[6],
            attendance=float(x[7]),
            internal_marks=float(x[8]),
            assignment=float(x[9]),
            previous_result=x[10],
            risk_level=x[11],
            risk_probability=float(x[12]),
            prediction_date=(
                x[13].strftime("%d-%b-%Y")
                if x[13] else None
            )
        )

    except Exception:
        app.logger.exception("Unable to load student")
        return jsonify(
            error="Unable to load student."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


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

    con = None
    cur = None

    try:

        sid = int(data["student_id"])
        total = int(data["total_classes"])
        attended = int(data["classes_attended"])

        attendance = round(
            attended / total * 100,
            2
        )

        internal = float(data["internal_marks"])
        assignment = float(data["assignment_percent"])
        previous = float(data["previous_performance"])

        risk, confidence = predict(
            attendance,
            internal,
            assignment,
            previous
        )

        previous_result = (
            "Good" if previous >= 75
            else "Average" if previous >= 50
            else "Poor"
        )

        con = db()
        cur = con.cursor()

        cur.execute(
            """
            SELECT COUNT(*)
            FROM STUDENT
            WHERE STUDENT_ID = :1
            """,
            (sid,)
        )

        if cur.fetchone()[0]:
            return jsonify(
                error="Student ID already exists."
            ), 409

        cur.execute(
            """
            INSERT INTO STUDENT
            VALUES (:1, :2, :3, :4, :5)
            """,
            (
                sid,
                data["student_name"].strip(),
                data.get("email") or None,
                data["department"].strip(),
                int(data["semester"])
            )
        )

        cur.execute("""
            SELECT NVL(MAX(ATTENDANCE_ID), 0) + 1
            FROM ATTENDANCE
        """)

        attendance_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO ATTENDANCE
            VALUES (:1, :2, :3, :4, :5)
            """,
            (
                attendance_id,
                sid,
                total,
                attended,
                attendance
            )
        )

        cur.execute("""
            SELECT NVL(MAX(PERFORMANCE_ID), 0) + 1
            FROM ACADEMIC_PERFORMANCE
        """)

        performance_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO ACADEMIC_PERFORMANCE
            VALUES (:1, :2, :3, :4, :5)
            """,
            (
                performance_id,
                sid,
                internal,
                assignment,
                previous_result
            )
        )

        cur.execute("""
            SELECT NVL(MAX(PREDICTION_ID), 0) + 1
            FROM RISK_PREDICTION
        """)

        prediction_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO RISK_PREDICTION
            VALUES (:1, :2, :3, :4, SYSDATE)
            """,
            (
                prediction_id,
                sid,
                risk,
                confidence
            )
        )

        con.commit()

        return jsonify(
            message="Student added and prediction generated.",
            student_id=sid,
            risk_level=risk,
            risk_probability=confidence
        ), 201

    except Exception:
        if con:
            con.rollback()

        app.logger.exception("Unable to add student")
        return jsonify(
            error="Unable to add student."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


# ============================================================
# DELETE STUDENT
# ============================================================

@app.delete("/student/<int:sid>")
@require_admin
def delete_student(sid):

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        cur.execute(
            """
            SELECT COUNT(*)
            FROM STUDENT
            WHERE STUDENT_ID = :1
            """,
            (sid,)
        )

        if not cur.fetchone()[0]:
            return jsonify(
                error="Student not found."
            ), 404

        for table in (
            "RISK_PREDICTION",
            "ACADEMIC_PERFORMANCE",
            "ATTENDANCE",
            "STUDENT"
        ):

            cur.execute(
                f"DELETE FROM {table} WHERE STUDENT_ID = :1",
                (sid,)
            )

        con.commit()

        return jsonify(
            message=f"Student {sid} deleted."
        )

    except Exception:
        if con:
            con.rollback()

        app.logger.exception("Unable to delete student")
        return jsonify(
            error="Unable to delete student."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


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

    except (UnicodeError, csv.Error):
        return jsonify(
            error="Unable to read CSV file."
        ), 400

    created = []
    skipped = []
    errors = []

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        for line_number, row in enumerate(reader, 2):

            ok, message = valid(row)

            if not ok:
                errors.append({
                    "line": line_number,
                    "error": message
                })
                continue

            sid = int(row["student_id"])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM STUDENT
                WHERE STUDENT_ID = :1
                """,
                (sid,)
            )

            if cur.fetchone()[0]:
                skipped.append(sid)
                continue

            total = int(row["total_classes"])
            attended = int(row["classes_attended"])

            attendance = round(
                attended / total * 100,
                2
            )

            internal = float(row["internal_marks"])
            assignment = float(row["assignment_percent"])
            previous = float(row["previous_performance"])

            risk, confidence = predict(
                attendance,
                internal,
                assignment,
                previous
            )

            previous_result = (
                "Good" if previous >= 75
                else "Average" if previous >= 50
                else "Poor"
            )

            cur.execute(
                """
                INSERT INTO STUDENT
                VALUES (:1, :2, :3, :4, :5)
                """,
                (
                    sid,
                    row["student_name"].strip(),
                    (row.get("email") or "").strip() or None,
                    row["department"].strip(),
                    int(row["semester"])
                )
            )

            cur.execute("""
                SELECT NVL(MAX(ATTENDANCE_ID), 0) + 1
                FROM ATTENDANCE
            """)

            attendance_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO ATTENDANCE
                VALUES (:1, :2, :3, :4, :5)
                """,
                (
                    attendance_id,
                    sid,
                    total,
                    attended,
                    attendance
                )
            )

            cur.execute("""
                SELECT NVL(MAX(PERFORMANCE_ID), 0) + 1
                FROM ACADEMIC_PERFORMANCE
            """)

            performance_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO ACADEMIC_PERFORMANCE
                VALUES (:1, :2, :3, :4, :5)
                """,
                (
                    performance_id,
                    sid,
                    internal,
                    assignment,
                    previous_result
                )
            )

            cur.execute("""
                SELECT NVL(MAX(PREDICTION_ID), 0) + 1
                FROM RISK_PREDICTION
            """)

            prediction_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO RISK_PREDICTION
                VALUES (:1, :2, :3, :4, SYSDATE)
                """,
                (
                    prediction_id,
                    sid,
                    risk,
                    confidence
                )
            )

            created.append({
                "student_id": sid,
                "risk_level": risk
            })

        con.commit()

        return jsonify(
            message="CSV import completed.",
            created=created,
            skipped_existing=skipped,
            errors=errors
        )

    except Exception:
        if con:
            con.rollback()

        app.logger.exception("CSV import failed")
        return jsonify(
            error="CSV import failed."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


# ============================================================
# RISK PREDICTION HISTORY
# ============================================================

@app.get("/risk-history/<int:sid>")
@require_admin
def history(sid):

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        cur.execute("""
            SELECT
                RISK_LEVEL,
                RISK_PROBABILITY,
                PREDICTION_DATE,
                RECORDED_AT,
                CHANGE_TYPE

            FROM RISK_PREDICTION_HISTORY

            WHERE STUDENT_ID = :1

            ORDER BY RECORDED_AT DESC
        """, (sid,))

        results = []

        for x in cur.fetchall():

            results.append({
                "risk_level": x[0],
                "risk_probability": float(x[1]),
                "prediction_date": (
                    x[2].strftime("%d-%b-%Y")
                    if x[2] else None
                ),
                "recorded_at": (
                    x[3].strftime("%d-%b-%Y %H:%M")
                    if x[3] else None
                ),
                "change_type": x[4]
            })

        return jsonify(results)

    except Exception:
        app.logger.exception("Unable to load risk history")
        return jsonify(
            error="Unable to load prediction history."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


# ============================================================
# RECALCULATE ALL AI PREDICTIONS
# ============================================================

@app.post("/predictions/recalculate")
@require_admin
def recalculate_predictions():

    con = None
    cur = None

    try:

        con = db()
        cur = con.cursor()

        cur.execute("""
            SELECT
                s.STUDENT_ID,
                a.ATTENDANCE_PERCENT,
                p.INTERNAL_MARKS,
                p.ASSIGNMENT_PERCENT

            FROM STUDENT s

            JOIN ATTENDANCE a
                ON s.STUDENT_ID = a.STUDENT_ID

            JOIN ACADEMIC_PERFORMANCE p
                ON s.STUDENT_ID = p.STUDENT_ID
        """)

        records = cur.fetchall()

        updated = 0

        counts = {
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 0
        }

        cur.execute("""
            SELECT NVL(MAX(PREDICTION_ID), 0)
            FROM RISK_PREDICTION
        """)

        next_prediction_id = int(cur.fetchone()[0])

        for sid, attendance, internal, assignment in records:

            # The existing database stores previous performance
            # as Good/Average/Poor, not its original numeric score.
            # Keep the project's existing internal-marks proxy.

            previous = internal

            risk_level, confidence = predict(
                attendance,
                internal,
                assignment,
                previous
            )

            next_prediction_id += 1

            cur.execute("""
                INSERT INTO RISK_PREDICTION (
                    PREDICTION_ID,
                    STUDENT_ID,
                    RISK_LEVEL,
                    RISK_PROBABILITY,
                    PREDICTION_DATE
                )
                VALUES (:1, :2, :3, :4, SYSDATE)
            """, (
                next_prediction_id,
                sid,
                risk_level,
                confidence
            ))

            counts[risk_level] += 1
            updated += 1

        con.commit()

        return jsonify(
            status="success",
            students_recalculated=updated,
            risk_distribution=counts
        )

    except Exception:
        if con:
            con.rollback()

        app.logger.exception("Recalculation failed")
        return jsonify(
            error="Unable to recalculate predictions."
        ), 500

    finally:

        if cur:
            cur.close()

        if con:
            con.close()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("ATTENDANCE AI - ADMIN BACKEND")
    print("Oracle Database + Random Forest + Flask")
    print("Admin authentication enabled")
    print("http://127.0.0.1:5000")
    print("=" * 55)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
