"""
CareerLens AI - Dataset Understanding & Exploration
Phase 2: Dataset Understanding
Purpose: Load all four O*NET datasets, examine their structure, columns, missing values, and data types.
"""

import os
import pandas as pd

# Set display options so pandas displays all columns clearly
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# Path to the data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

datasets = {
    "Occupation Master Dataset": "occupation_data.csv",
    "Software Skills Dataset": "software_skills.csv",
    "Transferable Skills Dataset": "transferable_skills.csv",
    "Training & Experience Dataset": "training_experience.csv"
}

print("=" * 80)
print("                   CAREERLENS AI - PHASE 2 DATASET AUDIT")
print("=" * 80)

for name, filename in datasets.items():
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"\n❌ [MISSING] File not found: {filename}")
        continue
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    print(f"\n📁 Dataset: {name}")
    print(f"   • File Name: {filename}")
    print(f"   • Total Rows (Records): {df.shape[0]:,}")
    print(f"   • Total Columns (Features): {df.shape[1]}")
    print(f"   • Columns: {list(df.columns)}")
    
    # Missing values count per column
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        print(f"   • Missing Values Found:")
        for col, count in cols_with_nulls.items():
            pct = (count / len(df)) * 100
            print(f"     - {col}: {count:,} nulls ({pct:.2f}%)")
    else:
        print(f"   • Missing Values: None (0 missing values)")
        
    print(f"\n   --- Sample 2 Rows ---")
    print(df.head(2).to_string(index=False))
    print("-" * 80)

print("\n✅ Dataset inspection completed successfully!")
