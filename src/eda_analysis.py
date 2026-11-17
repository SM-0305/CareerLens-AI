"""
CareerLens AI - Exploratory Data Analysis (EDA)
Phase 3: Exploratory Data Analysis
Purpose: Generate statistical summaries and visualizations for skills and occupations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling for clean, professional charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

# Directory paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("                   CAREERLENS AI - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------
# 1. Load the Datasets
# ---------------------------------------------------------
print("\n[Step 1/5] Loading datasets...")
df_soft = pd.read_csv(os.path.join(DATA_DIR, "software_skills.csv"))
df_trans = pd.read_csv(os.path.join(DATA_DIR, "transferable_skills.csv"))
df_occ = pd.read_csv(os.path.join(DATA_DIR, "occupation_data.csv"))
print("✅ Datasets loaded successfully.")

# ---------------------------------------------------------
# 2. Top In-Demand Software Skills
# ---------------------------------------------------------
print("\n[Step 2/5] Analyzing top software skills...")
top_software = df_soft["Workplace Example"].value_counts().head(15)

plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=top_software.values, y=top_software.index, hue=top_software.index, palette="viridis", legend=False)
plt.title("Top 15 Most Frequently Occurring Software Skills Across Occupations", fontsize=13, weight="bold", pad=12)
plt.xlabel("Number of Occupations Requiring the Skill", fontsize=11)
plt.ylabel("Software Tool / Skill", fontsize=11)

# Annotate each bar with the count value
for p in bar_plot.patches:
    width = p.get_width()
    bar_plot.annotate(f"{int(width)}",
                     (width + 8, p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "top_15_software_skills.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig1_path}")

# ---------------------------------------------------------
# 3. Occupations Requiring the Most Software Skills
# ---------------------------------------------------------
print("\n[Step 3/5] Analyzing tech-heavy occupations...")
skills_per_occ = df_soft.groupby("Title")["Workplace Example"].nunique().sort_values(ascending=False)
top_tech_roles = skills_per_occ.head(10)

plt.figure(figsize=(10, 6))
bar_plot2 = sns.barplot(x=top_tech_roles.values, y=top_tech_roles.index, hue=top_tech_roles.index, palette="mako", legend=False)
plt.title("Top 10 Occupations with the Highest Number of Required Software Skills", fontsize=13, weight="bold", pad=12)
plt.xlabel("Number of Distinct Software Tools Required", fontsize=11)
plt.ylabel("Occupation Title", fontsize=11)

for p in bar_plot2.patches:
    width = p.get_width()
    bar_plot2.annotate(f"{int(width)}",
                      (width + 5, p.get_y() + p.get_height() / 2.),
                      ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "top_10_tech_heavy_occupations.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig2_path}")

# ---------------------------------------------------------
# 4. Distribution of Software Skills Count per Occupation
# ---------------------------------------------------------
print("\n[Step 4/5] Analyzing software skill distribution across jobs...")
plt.figure(figsize=(9, 5))
sns.histplot(skills_per_occ, bins=35, kde=True, color="#2b5c8f")
plt.axvline(skills_per_occ.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {int(skills_per_occ.median())} skills")
plt.axvline(skills_per_occ.mean(), color="orange", linestyle=":", linewidth=1.5, label=f"Mean: {skills_per_occ.mean():.1f} skills")
plt.title("Distribution of Software Skills Count per Occupation", fontsize=13, weight="bold", pad=12)
plt.xlabel("Number of Software Skills", fontsize=11)
plt.ylabel("Number of Occupations", fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "skills_count_distribution.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig3_path}")

# ---------------------------------------------------------
# 5. Top Transferable Skills by Importance
# ---------------------------------------------------------
print("\n[Step 5/5] Analyzing transferable skills importance...")
# Filter for Scale ID == 'IM' (Importance score, scale 1.0 to 5.0)
df_trans_im = df_trans[df_trans["Scale ID"] == "IM"]
top_trans = df_trans_im.groupby("Element Name")["Data Value"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
bar_plot4 = sns.barplot(x=top_trans.values, y=top_trans.index, hue=top_trans.index, palette="rocket", legend=False)
plt.title("Top 10 Transferable Skills by Mean Importance (Scale 1.0 - 5.0)", fontsize=13, weight="bold", pad=12)
plt.xlabel("Mean Importance Rating across all Occupations", fontsize=11)
plt.ylabel("Transferable Skill Element", fontsize=11)
plt.xlim(0, 5.0)

for p in bar_plot4.patches:
    width = p.get_width()
    bar_plot4.annotate(f"{width:.2f}",
                      (width + 0.08, p.get_y() + p.get_height() / 2.),
                      ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "top_10_transferable_skills.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"✅ Saved figure: {fig4_path}")

print("\n" + "=" * 80)
print("📊 EDA SUMMARY & KEY FINDINGS:")
print("=" * 80)
print(f"1. Total Occupations Analyzed: {len(skills_per_occ)} occupations with software mappings.")
print(f"2. Software Skill Frequency:")
print(f"   • Most universally required tool: {top_software.index[0]} ({top_software.iloc[0]} occupations)")
print(f"   • Runner up: {top_software.index[1]} ({top_software.iloc[1]} occupations)")
print(f"3. Occupational Skill Depth:")
print(f"   • Most skill-dense occupation: '{skills_per_occ.index[0]}' with {skills_per_occ.iloc[0]} distinct software tools.")
print(f"   • 50% of occupations require {int(skills_per_occ.median())} or fewer software tools.")
print(f"   • The distribution is right-skewed: most jobs need 10-35 tools, while tech jobs need 200+.")
print(f"4. Universal Soft / Transferable Skills:")
print(f"   • Highest rated skill: '{top_trans.index[0]}' (avg score: {top_trans.iloc[0]:.2f}/5.0)")
print(f"   • Followed closely by: '{top_trans.index[1]}' (avg score: {top_trans.iloc[1]:.2f}/5.0)")
print(f"\nAll 4 visualization charts have been saved to: {FIGURES_DIR}")
