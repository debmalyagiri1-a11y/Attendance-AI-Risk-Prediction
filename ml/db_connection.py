
import os
import psycopg2


def get_connection():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Configure your Neon database URL."
        )

    return psycopg2.connect(
        database_url,
        connect_timeout=15
    )


if __name__ == "__main__":
    try:
        connection = get_connection()

        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            database_name = cursor.fetchone()[0]

        print("Neon PostgreSQL connected successfully!")
        print("Database:", database_name)

        connection.close()

    except Exception as error:
        print("Database connection failed:", error)
