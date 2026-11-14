import oracledb

connection = oracledb.connect(
    user="system",
    password="OracleDB2026",
    dsn="localhost:1521/FREEPDB1"
)

print("Oracle Database connected successfully!")

connection.close()
