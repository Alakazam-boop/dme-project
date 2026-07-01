"""
setup_db.py — First thing you run on a fresh install.

Creates the three tables that power the whole system:
  - decision_traces  : one row per patient visit (vitals + what action was taken)
  - outcomes         : did the patient improve or not? linked back to a trace
  - model_versions   : audit log of every time the ML model was retrained

SQLite is fine here — this is a research prototype on 100 patients,
not a hospital production system. No migrations needed for this scale.
"""

import sqlite3
import os

DB_PATH = 'database/dme.db'


def setup_database():
    # Connect (creates the file if it doesn't exist yet)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # decision_traces table
    # This is the heart of the system — every clinical decision that was ever
    # made gets stored here with all the vitals that were available at the time.
    # status starts as 'pending' and flips to 'completed' once an outcome arrives.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS decision_traces (
            trace_id        TEXT PRIMARY KEY,
            patient_id      TEXT NOT NULL,
            timestamp       TEXT,
            heart_rate      REAL,
            systolic_bp     REAL,
            creatinine      REAL,
            wbc             REAL,
            temperature     REAL,
            diagnosis_code  TEXT,
            action          TEXT NOT NULL,
            rationale       TEXT,
            confidence      REAL,
            status          TEXT DEFAULT 'pending'
        )
    ''')

    # outcomes table
    # Linked to decision_traces via trace_id. outcome_value=1 means the patient
    # improved, 0 means they didn't. This pairing is what the ML model learns
    # from — "given these vitals and this action, did it work?"
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            outcome_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id          TEXT NOT NULL,
            patient_id        TEXT NOT NULL,
            outcome_value     INTEGER,
            outcome_timestamp TEXT,
            FOREIGN KEY (trace_id) REFERENCES decision_traces(trace_id)
        )
    ''')

    # model_versions table
    # Every time we retrain, we log the result here. Gives us an audit trail
    # so we can see if accuracy has drifted over time and when we retrained last.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            model_type    TEXT NOT NULL,
            accuracy      REAL,
            n_traces_used INTEGER,
            notes         TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Database created successfully at", DB_PATH)


if __name__ == '__main__':
    setup_database()
