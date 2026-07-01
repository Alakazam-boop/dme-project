"""
decision_logger.py — Writes clinical decisions into the database.

Two main jobs:
  1. log_decision()   : called in real-time when a clinician makes a decision
                        for a patient — stores the vitals snapshot + action taken
  2. load_all_traces(): bulk-loads the processed MIMIC-IV CSV data on first setup,
                        so the system has historical cases to learn from right away

The status column starts at 'pending' for every new trace. Once an outcome
arrives (via outcome_linker.py), it flips to 'completed' — that's when the
trace becomes eligible for ML training.
"""

import sqlite3
import pandas as pd

DB_PATH = 'database/dme.db'


def log_decision(trace_id, patient_id, timestamp, context_features,
                 action, rationale, confidence):
    """Logs a single new decision trace to the database.

    context_features should be a dict with keys like 'heart_rate', 'systolic_bp',
    etc. Any missing keys just store as NULL — the model handles that gracefully.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # INSERT OR REPLACE means if someone accidentally logs the same trace_id
    # twice, we just overwrite rather than crashing with a duplicate key error.
    cursor.execute('''
        INSERT OR REPLACE INTO decision_traces
        (trace_id, patient_id, timestamp, heart_rate, systolic_bp, creatinine,
         wbc, temperature, diagnosis_code, action, rationale, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
    ''', (
        trace_id,
        patient_id,
        str(timestamp),
        context_features.get('heart_rate'),
        context_features.get('systolic_bp'),
        context_features.get('creatinine'),
        context_features.get('wbc'),
        context_features.get('temperature'),
        context_features.get('diagnosis_code'),
        action,
        rationale,
        confidence
    ))

    conn.commit()
    conn.close()
    print(f"Logged new trace {trace_id} for patient {patient_id}.")


def load_all_traces():
    """Bulk-loads the processed MIMIC-IV traces into the database.

    Reads from the two processed CSVs (contexts.csv and decisions.csv),
    merges them on patient_id (and hadm_id if it's there), then inserts every
    row. This is typically run once during setup — the MIMIC data becomes
    the historical memory that the CBR system draws on.
    """
    contexts  = pd.read_csv('data/processed/contexts.csv')
    decisions = pd.read_csv('data/processed/decisions.csv')

    # Try to merge on both patient_id and hadm_id for a cleaner join, but fall
    # back to patient_id alone if hadm_id isn't present in both files.
    merge_keys = ['patient_id']
    if 'hadm_id' in decisions.columns and 'hadm_id' in contexts.columns:
        merge_keys = ['patient_id', 'hadm_id']
    merged = decisions.merge(contexts, on=merge_keys, how='left')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    loaded = 0
    for _, row in merged.iterrows():
        # timestamp_x / timestamp_y come from the merge when both files have
        # a timestamp column — we just take whichever one exists.
        cursor.execute('''
            INSERT OR REPLACE INTO decision_traces
            (trace_id, patient_id, timestamp, heart_rate, systolic_bp,
             creatinine, wbc, temperature, diagnosis_code,
             action, rationale, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        ''', (
            row['trace_id'],
            row['patient_id'],
            str(row.get('timestamp_x', row.get('timestamp', ''))),
            row.get('heart_rate'),
            row.get('systolic_bp'),
            row.get('creatinine'),
            row.get('wbc'),
            row.get('temperature'),
            row.get('diagnosis_code'),
            row['action'],
            row['rationale'],
            row['confidence']
        ))
        loaded += 1

    conn.commit()
    conn.close()
    print(f"Loaded {loaded} decision traces into database.")


def get_all_traces():
    """Simple helper to pull every trace out of the DB as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM decision_traces", conn)
    conn.close()
    return df


if __name__ == '__main__':
    load_all_traces()
    df = get_all_traces()
    print(f"Traces in database: {len(df)}")
    print(df[['trace_id', 'patient_id', 'action', 'status']].head(5))
