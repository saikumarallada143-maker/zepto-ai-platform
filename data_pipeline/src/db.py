"""
db.py
Small helper layer around sqlite3: connect + apply schema.sql.
"""

import sqlite3
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = MODULE_DIR / "data" / "zepto_books.db"
SCHEMA_PATH = MODULE_DIR / "schema.sql"


def get_connection(db_path=DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn):
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
