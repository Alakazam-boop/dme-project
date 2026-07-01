"""
feature_pipeline.py — Turns raw clinical records into ML-ready features.

The pipeline order matters:
  1. Fill missing vitals with column medians (or 0 if column is absent entirely)
  2. Compute derived features from the RAW vitals before scaling distorts them
  3. One-hot encode the diagnosis code
  4. Scale everything with StandardScaler

The scaler is fitted once during training (saved to models/scaler.pkl) and then
loaded read only for every prediction same scaler, same scale, no drift.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

SCALER_PATH  = 'models/scaler.pkl'

# The five raw vitals we always expect to be present (or filled with 0)
FEATURE_COLS = ['heart_rate', 'systolic_bp', 'creatinine', 'wbc', 'temperature']

# These two are computed from the raw vitals before scaling they capture
# clinical relationships that the model can't easily discover on its own.
# shock_index   = heart_rate / systolic_bp rises when perfusion is failing
# creat_x_wbc   = creatinine * WBC a rough proxy for sepsis-driven organ stress
DERIVED_COLS = ['shock_index', 'creat_x_wbc']


def encode_diagnosis(df):
    """One-hot encode the diagnosis_code column into dx_* binary columns.

    Unknown codes at inference time simply produce all-zero dx_ columns,
    which is fine — the model treats them as "no known diagnosis" and relies
    on the vitals instead.
    """
    return pd.get_dummies(df, columns=['diagnosis_code'], prefix='dx')


def scale_features(df, fit=False):
    """Scale FEATURE_COLS + DERIVED_COLS using StandardScaler.

    If fit=True (training run) or the scaler file doesn't exist yet, we fit
    a fresh scaler and save it. For every subsequent call we just load the
    saved scaler and transform — critical for consistency between training
    and inference.

    The scaler is always applied to ALL columns it was fitted on, never a subset,
    because sklearn's transform() requires the exact same column set as fit().
    Any column that was missing at inference time is filled with 0 before scaling.
    """
    scaler = StandardScaler()

    # Make sure every base vital column is present, even if it came in empty
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Scale base vitals plus whichever derived features made it through
    cols_to_scale = FEATURE_COLS + [c for c in DERIVED_COLS if c in df.columns]

    if fit or not os.path.exists(SCALER_PATH):
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler fitted on {cols_to_scale} and saved.")
    else:
        scaler = joblib.load(SCALER_PATH)
        # Use the saved scaler's own column list — it might include columns
        # that aren't in the current df (e.g. derived cols that were missing),
        # so we add them as 0 before transforming.
        fitted_cols = list(scaler.feature_names_in_)
        for col in fitted_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[fitted_cols] = scaler.transform(df[fitted_cols])

    return df


def preprocess(df, fit_scaler=False):
    """Full preprocessing pipeline — call this before any ML step.

    Steps:
      1. Drop rows where ALL vitals are missing (completely empty records)
      2. Fill individual missing vitals with median imputation
      3. Compute shock_index and creat_x_wbc from the raw (unscaled) values
      4. One-hot encode diagnosis codes
      5. Scale everything

    fit_scaler=True should only be set during the initial training run.
    For inference and CBR building, leave it False so we reuse the saved scaler.
    """
    df = df.copy()

    # Drop rows where every vital is missing — these are useless for training
    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns], how='all')

    # Fill individual missing vitals with the column median, or 0 if the
    # column is absent entirely (rare but handled gracefully)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # Derived features must be computed HERE, from the raw vitals, before
    # scaling changes their magnitudes. Computing them after scaling would
    # produce nonsensical values (scaled HR / scaled SBP is not shock index).
    sbp_safe = df['systolic_bp'].clip(lower=1)   # avoid divide-by-zero
    df['shock_index'] = df['heart_rate'] / sbp_safe
    df['creat_x_wbc'] = df['creatinine'] * df['wbc']

    df = encode_diagnosis(df)
    df = scale_features(df, fit=fit_scaler)
    return df


def get_feature_columns(df):
    """Returns all feature column names the model cares about.

    Includes the five base vitals, the two derived features, and every
    diagnosis one-hot column (dx_*) that happened to be in this dataset.
    """
    return [c for c in df.columns
            if c.startswith('dx_') or c in FEATURE_COLS or c in DERIVED_COLS]


if __name__ == '__main__':
    # Quick sanity check — preprocesses the full dataset and saves it
    from modules.outcome_linker import get_completed_traces_with_outcomes
    df        = get_completed_traces_with_outcomes()
    processed = preprocess(df, fit_scaler=True)
    os.makedirs('data/processed', exist_ok=True)
    processed.to_csv('data/processed/processed_traces.csv', index=False)
    print(f"Processed {len(processed)} traces.")
    print(f"Feature columns: {get_feature_columns(processed)}")
