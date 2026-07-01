import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

print("Loading raw MIMIC-IV files...")

patients   = pd.read_csv('data/raw/patients.csv.gz')
admissions = pd.read_csv('data/raw/admissions.csv.gz')
labevents  = pd.read_csv('data/raw/labevents.csv.gz')
diagnoses  = pd.read_csv('data/raw/diagnoses_icd.csv.gz')

print("Loading chartevents (this may take a moment - it's a large file)...")
chartevents = pd.read_csv('data/raw/chartevents.csv.gz',
                           usecols=['subject_id', 'hadm_id', 'itemid',
                                    'charttime', 'valuenum'])

# Step 1: Extract lab values from labevents.
# MIMIC-IV item IDs from labevents
LAB_ITEMS = {
    50912: 'creatinine',
    51301: 'wbc',
}

print("Extracting lab features from labevents...")
labs_filtered = labevents[labevents['itemid'].isin(LAB_ITEMS.keys())].copy()
labs_filtered  = labs_filtered.dropna(subset=['valuenum'])
labs_filtered['feature'] = labs_filtered['itemid'].map(LAB_ITEMS)

labs_pivot = (
    labs_filtered
    .sort_values('charttime')
    .groupby(['subject_id', 'hadm_id', 'feature'])['valuenum']
    .first()
    .unstack('feature')
    .reset_index()
)
print(f"Lab pivot shape: {labs_pivot.shape}")

# Step 2: Extract vitals from chartevents.
# MIMIC-IV item IDs from chartevents
CHART_ITEMS = {
    220045: 'heart_rate',
    220179: 'systolic_bp',
    220050: 'systolic_bp_art',  # arterial line BP — we'll merge with 220179
    223761: 'temperature_f',    # Fahrenheit
    223762: 'temperature_c',    # Celsius
}

print("Extracting vitals from chartevents...")
chart_filtered = chartevents[chartevents['itemid'].isin(CHART_ITEMS.keys())].copy()
chart_filtered  = chart_filtered.dropna(subset=['valuenum'])
chart_filtered['feature'] = chart_filtered['itemid'].map(CHART_ITEMS)

# Merge both systolic BP sources into one column
chart_filtered.loc[chart_filtered['feature'] == 'systolic_bp_art', 'feature'] = 'systolic_bp'

# Convert Fahrenheit temperature to Celsius
mask_f = chart_filtered['feature'] == 'temperature_f'
chart_filtered.loc[mask_f, 'valuenum'] = (
    chart_filtered.loc[mask_f, 'valuenum'] - 32) * 5 / 9
chart_filtered.loc[mask_f, 'feature'] = 'temperature'
chart_filtered.loc[chart_filtered['feature'] == 'temperature_c', 'feature'] = 'temperature'

# Take first recorded value per patient per admission per feature
chart_pivot = (
    chart_filtered
    .sort_values('charttime')
    .groupby(['subject_id', 'hadm_id', 'feature'])['valuenum']
    .first()
    .unstack('feature')
    .reset_index()
)
print(f"Chart pivot shape: {chart_pivot.shape}")
print(f"Chart columns: {chart_pivot.columns.tolist()}")

# Step 3: Get primary diagnosis.
print("Extracting diagnoses...")
primary_dx = (
    diagnoses[diagnoses['seq_num'] == 1]
    [['subject_id', 'hadm_id', 'icd_code']]
    .rename(columns={'icd_code': 'diagnosis_code'})
)

# Step 4: Merge everything together.
print("Merging all tables...")

base = admissions.merge(patients, on='subject_id', how='left')
base = base.merge(labs_pivot,  on=['subject_id', 'hadm_id'], how='left')
base = base.merge(chart_pivot, on=['subject_id', 'hadm_id'], how='left')
base = base.merge(primary_dx,  on=['subject_id', 'hadm_id'], how='left')

print(f"Merged shape: {base.shape}")
print(f"Columns after merge: {base.columns.tolist()}")

# Step 5: Standardise column names.
# Rename to standard names our modules expect
if 'heart_rate' not in base.columns and 'heart_rate' in chart_pivot.columns:
    pass  # already named correctly
if 'systolic_bp' not in base.columns and 'systolic_bp' in chart_pivot.columns:
    pass  # already named correctly

# Step 6: Outcome variable (1 = survived, 0 = did not survive).
base['outcome'] = 1 - base['hospital_expire_flag'].fillna(0).astype(int)

# Step 7: Fill missing values with column medians.
FEATURE_COLS = ['heart_rate', 'systolic_bp', 'creatinine', 'wbc', 'temperature']

for col in FEATURE_COLS:
    if col not in base.columns:
        print(f"WARNING: {col} not found after merge — filling with 0")
        base[col] = np.nan

print("\nMissing values before fill:")
print(base[FEATURE_COLS].isnull().sum())

for col in FEATURE_COLS:
    median_val = base[col].median()
    base[col]  = base[col].fillna(median_val)
    print(f"  {col}: filled NaN with median={median_val:.2f}")

print("\nMissing values after fill:")
print(base[FEATURE_COLS].isnull().sum())

# Step 8: Clean temperature outliers.
# Some Fahrenheit values may not have converted properly
base.loc[base['temperature'] > 45, 'temperature'] = (
    base.loc[base['temperature'] > 45, 'temperature'] - 32) * 5 / 9
base.loc[base['temperature'] > 45, 'temperature'] = 37.0  # fallback

# Step 9: Fill missing diagnosis codes and assign trace IDs.
base['diagnosis_code'] = base['diagnosis_code'].fillna('UNKNOWN')
base = base.reset_index(drop=True)
base['patient_id'] = ['P_ANON_' + str(sid) for sid in base['subject_id']]
base['trace_id']   = ['T_' + str(i).zfill(5) for i in base.index]

# Step 10: Simulate clinical decisions based on real feature values.
# The rules mirror how a junior doctor would triage — highest abnormality first.
def assign_action(row):
    if row['creatinine'] > 2.0:
        return 'refer_specialist'
    elif row['heart_rate'] > 100:
        return 'increase_monitoring'
    elif row['wbc'] > 11:
        return 'prescribe_antibiotic'
    elif row['systolic_bp'] < 90:
        return 'prescribe_vasopressor'
    else:
        return random.choice(['discharge_plan', 'prescribe_diuretic', 'order_imaging'])

def assign_rationale(row):
    if row['creatinine'] > 2.0:
        return 'high_creatinine'
    elif row['heart_rate'] > 100:
        return 'elevated_heart_rate'
    elif row['wbc'] > 11:
        return 'elevated_wbc'
    elif row['systolic_bp'] < 90:
        return 'low_blood_pressure'
    else:
        return 'stable_condition'

base['action']           = base.apply(assign_action, axis=1)
base['rationale']        = base.apply(assign_rationale, axis=1)
base['confidence']       = np.round(np.random.uniform(0.55, 0.95, len(base)), 2)
base['timestamp']        = base['admittime']
base['outcome_timestamp']= base['dischtime']
base['status']           = 'completed'

# Step 11: Save processed files to data/processed/.
print("\nSaving processed files...")
os.makedirs('data/processed', exist_ok=True)

contexts_cols = ['patient_id', 'hadm_id', 'timestamp',
                 'heart_rate', 'systolic_bp', 'creatinine',
                 'wbc', 'temperature', 'diagnosis_code']
base[contexts_cols].to_csv('data/processed/contexts.csv', index=False)

decisions_cols = ['trace_id', 'patient_id', 'hadm_id',
                  'action', 'rationale', 'confidence', 'timestamp']
base[decisions_cols].to_csv('data/processed/decisions.csv', index=False)

outcomes_cols = ['trace_id', 'patient_id', 'outcome', 'outcome_timestamp']
base[outcomes_cols].to_csv('data/processed/outcomes.csv', index=False)

full_cols = ['trace_id', 'patient_id', 'timestamp',
             'heart_rate', 'systolic_bp', 'creatinine',
             'wbc', 'temperature', 'diagnosis_code',
             'action', 'rationale', 'confidence',
             'outcome', 'outcome_timestamp', 'status']
base[full_cols].to_csv('data/processed/full_traces.csv', index=False)

print("\n=== Processing Complete ===")
print(f"Total admissions : {len(base)}")
print(f"Outcome distribution:\n{base['outcome'].value_counts()}")
print(f"Action distribution:\n{base['action'].value_counts()}")
print("\nSample of final data:")
print(base[FEATURE_COLS].describe().round(2))