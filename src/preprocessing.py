"""
CareerLens AI - Data Preprocessing & Feature Engineering
Phase 4: Data Preprocessing
Purpose: Clean, normalize, aggregate, and structure occupational and skill datasets
         into a unified format for ML model training and recommendation.
"""

import os
import re
import pandas as pd

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_occupations.csv")
UNIQUE_SKILLS_PATH = os.path.join(DATA_DIR, "unique_skills.csv")

def clean_text(text: str) -> str:
    """
    Cleans raw text while preserving tech-specific tokens like C++, C#, .NET.
    - Replaces newlines and multiple spaces with a single space.
    - Strips leading and trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Normalize multiple whitespace characters into single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def run_preprocessing():
    print("=" * 80)
    print("              CAREERLENS AI - PHASE 4 DATA PREPROCESSING")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Load Raw Datasets
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Loading raw datasets...")
    occ_file = os.path.join(DATA_DIR, "occupation_data.csv")
    soft_file = os.path.join(DATA_DIR, "software_skills.csv")
    trans_file = os.path.join(DATA_DIR, "transferable_skills.csv")
    
    df_occ = pd.read_csv(occ_file)
    df_soft = pd.read_csv(soft_file)
    df_trans = pd.read_csv(trans_file)
    print(f"Loaded: {len(df_occ):,} occupations, {len(df_soft):,} software skills, {len(df_trans):,} transferable skills.")
    
    # -------------------------------------------------------------------------
    # 2. Clean Master Occupation Data
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Cleaning occupation descriptions and titles...")
    df_occ["Title"] = df_occ["Title"].apply(clean_text)
    df_occ["Description"] = df_occ["Description"].apply(clean_text)
    
    # -------------------------------------------------------------------------
    # 3. Aggregate Software Skills per Occupation
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Aggregating software skills per occupation...")
    # Clean skill names and remove duplicates per occupation
    df_soft["Workplace Example"] = df_soft["Workplace Example"].apply(clean_text)
    df_soft_clean = df_soft.dropna(subset=["Workplace Example"]).drop_duplicates(subset=["O*NET-SOC Code", "Workplace Example"])
    
    # Group software skills into a comma-separated string and a list
    soft_agg = df_soft_clean.groupby("O*NET-SOC Code")["Workplace Example"].agg(
        Software_Skills=lambda x: ", ".join(sorted(set(x))),
        Software_Skills_Count=lambda x: len(set(x))
    ).reset_index()

    # Also aggregate Hot Technologies (trending, in-demand industry skills)
    hot_df = df_soft_clean[df_soft_clean["Hot Technology"] == "Y"]
    hot_agg = hot_df.groupby("O*NET-SOC Code")["Workplace Example"].agg(
        Hot_Technologies=lambda x: ", ".join(sorted(set(x)))
    ).reset_index()
    
    # -------------------------------------------------------------------------
    # 4. Aggregate Important Transferable Skills per Occupation
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Aggregating important transferable skills...")
    # We select Transferable Skills where Scale ID is 'IM' (Importance) and Data Value >= 3.0 (Important)
    df_trans_im = df_trans[(df_trans["Scale ID"] == "IM") & (df_trans["Data Value"] >= 3.0)].copy()
    df_trans_im["Element Name"] = df_trans_im["Element Name"].apply(clean_text)
    
    trans_agg = df_trans_im.groupby("O*NET-SOC Code")["Element Name"].agg(
        Transferable_Skills=lambda x: ", ".join(sorted(set(x))),
        Transferable_Skills_Count=lambda x: len(set(x))
    ).reset_index()
    
    # -------------------------------------------------------------------------
    # 5. Merge into Unified Dataset & Build Feature-Rich Representation
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Merging into unified occupational dataset...")
    df_merged = pd.merge(df_occ, soft_agg, on="O*NET-SOC Code", how="left")
    df_merged = pd.merge(df_merged, hot_agg, on="O*NET-SOC Code", how="left")
    df_merged = pd.merge(df_merged, trans_agg, on="O*NET-SOC Code", how="left")
    
    # Fill missing values for occupations that have no software or transferable skills
    df_merged["Software_Skills"] = df_merged["Software_Skills"].fillna("")
    df_merged["Hot_Technologies"] = df_merged["Hot_Technologies"].fillna("")
    df_merged["Software_Skills_Count"] = df_merged["Software_Skills_Count"].fillna(0).astype(int)
    df_merged["Transferable_Skills"] = df_merged["Transferable_Skills"].fillna("")
    df_merged["Transferable_Skills_Count"] = df_merged["Transferable_Skills_Count"].fillna(0).astype(int)
    
    # Construct 'Combined_Features': the composite text representation used by TF-IDF
    # We intentionally include Title twice to give job titles extra lexical weight
    df_merged["Combined_Features"] = (
        df_merged["Title"] + " " +
        df_merged["Title"] + " " +
        df_merged["Description"] + " " +
        df_merged["Software_Skills"] + " " +
        df_merged["Transferable_Skills"]
    ).apply(clean_text)
    
    # Save processed occupations
    df_merged.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Saved processed dataset to: {PROCESSED_DATA_PATH}")
    print(f"   • Total records: {len(df_merged):,}")
    print(f"   • Total columns: {list(df_merged.columns)}")
    
    # Extract and save master list of all unique skills (for frontend autocomplete & matching)
    all_software_skills = set(df_soft_clean["Workplace Example"].unique())
    all_transferable_skills = set(df_trans_im["Element Name"].unique())
    all_unique_skills = sorted(list(all_software_skills.union(all_transferable_skills)))
    
    df_unique_skills = pd.DataFrame({
        "Skill_Name": all_unique_skills,
        "Skill_Type": ["Software" if s in all_software_skills else "Transferable" for s in all_unique_skills]
    })
    df_unique_skills.to_csv(UNIQUE_SKILLS_PATH, index=False)
    print(f"✅ Saved unique skills vocabulary to: {UNIQUE_SKILLS_PATH}")
    print(f"   • Total unique skills cataloged: {len(df_unique_skills):,}")
    
    print("\n" + "=" * 80)
    print("🎉 PHASE 4 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_preprocessing()
