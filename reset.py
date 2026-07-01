import sqlite3
import os

DB_PATH = 'database/dme.db'

# Wipe database
conn = sqlite3.connect(DB_PATH)
conn.execute('DELETE FROM decision_traces')
conn.execute('DELETE FROM outcomes')
conn.execute('DELETE FROM model_versions')
conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('outcomes','model_versions')")
conn.commit()
conn.close()
print('Database wiped clean.')

# Delete model files
model_files = [
    'models/classifier.pkl',
    'models/scaler.pkl',
    'models/feature_cols.pkl',
    'models/cbr_index.pkl',
    'models/cbr_traces.pkl',
]
for f in model_files:
    if os.path.exists(f):
        os.remove(f)
        print(f'Deleted: {f}')

print('All models deleted.')
print('Ready to retrain from scratch.')