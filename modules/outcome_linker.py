"""
outcome_linker.py — Connects patient outcomes back to the decisions that caused them.

This is where the feedback loop closes. After a patient is discharged (or at
some follow-up point), we know whether the clinical action taken actually helped.
link_outcomes() reads that data from a CSV and writes it into the database,
then marks the matching traces as 'completed' so they're eligible for ML training.

The query in get_completed_traces_with_outcomes() is the main data source for
train_classifier() — it only includes traces where we know the outcome.
"""

import os
import sqlite3
import pandas as pd

DB_PATH = 'database/dme.db'


def link_outcomes():
    """Link outcomes to decision traces and mark them completed.

    Looks for a cleaned outcomes CSV first (outcomes_clean.csv) — that one has
    had any messy or ambiguous records removed. Falls back to the raw outcomes.csv
    if no clean version exists. Either way, every matched trace gets its status
    flipped to 'completed' in decision_traces.
    """
    clean_path  = 'data/processed/outcomes_clean.csv'
    normal_path = 'data/processed/outcomes.csv'

    if os.path.exists(clean_path):
        outcomes_df = pd.read_csv(clean_path)
        print(f"Using clean outcomes file: {clean_path}")
    else:
        outcomes_df = pd.read_csv(normal_path)
        print(f"Using standard outcomes file: {normal_path}")

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Unique index on trace_id prevents the same outcome being inserted twice
    # if someone accidentally runs link_outcomes() more than once.
    try:
        cursor.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_trace ON outcomes(trace_id)'
        )
        conn.commit()
    except Exception:
        pass   # index might already exist — that's fine

    linked = 0
    for _, row in outcomes_df.iterrows():
        # INSERT OR IGNORE means we skip duplicates silently rather than crashing.
        cursor.execute('''
            INSERT OR IGNORE INTO outcomes
            (trace_id, patient_id, outcome_value, outcome_timestamp)
            VALUES (?, ?, ?, ?)
        ''', (
            row['trace_id'],
            row['patient_id'],
            int(row['outcome']),
            str(row['outcome_timestamp'])
        ))
        # Mark the corresponding trace as completed so the ML pipeline picks it up.
        cursor.execute('''
            UPDATE decision_traces SET status = 'completed'
            WHERE trace_id = ?
        ''', (row['trace_id'],))
        linked += 1

    conn.commit()
    conn.close()
    print(f"Linked {linked} outcomes. All traces marked completed.")


def get_completed_traces_with_outcomes():
    """Returns completed traces joined with outcomes — deduplicated.

    The subquery on outcomes deduplicates by trace_id (GROUP BY) before the
    join, so even if link_outcomes() was accidentally run twice we don't end
    up with doubled rows going into the ML pipeline.
    """
    conn = sqlite3.connect(DB_PATH)
    query = '''
        SELECT dt.*,
               o.outcome_value
        FROM decision_traces dt
        JOIN (
            SELECT trace_id, outcome_value
            FROM outcomes
            GROUP BY trace_id
        ) o ON dt.trace_id = o.trace_id
        WHERE dt.status = 'completed'
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_pending_traces():
    """Returns all traces that haven't had an outcome linked yet.

    Useful for monitoring how many decisions are still waiting on follow-up data.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM decision_traces WHERE status='pending'", conn
    )
    conn.close()
    return df


if __name__ == '__main__':
    link_outcomes()
    completed = get_completed_traces_with_outcomes()
    print(f"Completed traces ready for ML: {len(completed)}")
    print(f"Outcome distribution:\n{completed['outcome_value'].value_counts()}")
