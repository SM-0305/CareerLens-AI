# 🎯 CareerLens AI — Machine Learning Career Recommendation & Skill-Gap Intelligence

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![O*NET Data](https://img.shields.io/badge/Dataset-O*NET%2028.0-green.svg)](https://www.onetonline.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**CareerLens AI** is an intelligent, end-to-end Machine Learning career recommendation and skill-gap diagnostic platform. Powered by official **O\*NET Occupational Intelligence (1,016 standard occupations and 8,778 recognized industry skills)**, CareerLens AI analyzes a user's technical and transferable abilities, recommends the highest-matching career pathways, calculates precise skill gaps, and generates prioritized, personalized learning roadmaps.

---

## 🚀 Key Features

- **🎯 Hybrid Career Recommendation Engine**: Combines **Direct Skill Overlap Ratio (65%)** and **TF-IDF Cosine Similarity (35%)** to eliminate false-positive keyword matches and deliver highly relevant occupational rankings.
- **🔍 Target Career Skill-Gap Diagnostics**: Select any target dream role (e.g. *Data Scientists* or *Software Developers*) to evaluate your readiness percentage, possessed skills, and missing technical competencies.
- **🔥 Hot Technologies Prioritization**: Automatically surfaces O\*NET's trending, employer-demanded skills (*Docker, AWS, Apache Spark, Git*) rather than obscure tools.
- **🗺️ 4-Step Personalized Learning Roadmap**: Generates an actionable, step-by-step career development plan to bridge the gap.
- **📊 Labor Market Intelligence**: Interactive visual market analysis of skill distributions and tech-heavy occupations across the economy.
- **⚡ Blazing Fast In-Memory Caching**: Serialized model artifacts (`.joblib`) load in milliseconds and deliver real-time recommendations.

---

## 🏛️ System Architecture

```
                                USER SKILL INPUT
              ["Python", "SQL", "Excel", "Statistics", "Data Visualization"]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
Direct Skill Overlap Ratio (65%)                     TF-IDF Vectorizer & Cosine Sim (35%)
(Exact & Substring Matching against                  (20,000 N-Gram Vocabulary & Sublinear
Software & Transferable Abilities)                    Term-Frequency Semantic Matching)
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                             HYBRID MATCH SCORE
                       (Ranked Top-K Occupations)
                                      │
                                      ▼
                             SKILL-GAP DIAGNOSTICS
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
     Skills Possessed                              High-Priority Skills to Learn
(Green UI Badges: Python, SQL)                (Hot Tech Gold Badges: AWS, Docker, Spark)
                                      │
                                      ▼
                     4-STEP PERSONALIZED LEARNING ROADMAP
```

---

## 📂 Project Structure

```bash
CareerLens-AI/
├── app/
│   └── main.py                     # Streamlit multi-tab interactive web dashboard
├── data/
│   ├── occupation_data.csv         # O*NET master occupations catalog (1,016 roles)
│   ├── software_skills.csv         # O*NET software skills & tools (31,821 records)
│   ├── transferable_skills.csv     # O*NET core transferable abilities (45,500 records)
│   ├── training_experience.csv     # Education & experience requirements
│   ├── processed_occupations.csv   # Unified preprocessed occupational dataset
│   └── unique_skills.csv           # Vocabulary catalog of 8,778 unique skills
├── models/
│   ├── tfidf_vectorizer.joblib     # Serialized TF-IDF model vocabulary
│   ├── tfidf_matrix.joblib         # Precomputed 1,016 x 20,000 feature matrix
│   └── occupations_cache.joblib    # Fast in-memory occupational metadata cache
├── notebooks/
│   ├── 01_data_understanding.ipynb # Phase 2: Data loading & initial audit
│   ├── 02_exploratory_data_analysis.ipynb # Phase 3: EDA & statistical distributions
│   ├── 03_data_preprocessing.ipynb # Phase 4: Text cleaning & feature engineering
│   ├── 04_model_development.ipynb  # Phase 5: TF-IDF vectorization & cosine similarity
│   ├── 05_recommendation_and_gap_analysis.ipynb # Phases 6 & 7: Hybrid engine & gap testing
│   └── 06_model_evaluation.ipynb   # Phase 9: Information Retrieval benchmark evaluation
├── reports/
│   ├── figures/                    # High-resolution EDA visualization charts
│   ├── PROJECT_DOCUMENTATION.md    # Comprehensive academic & technical report
│   └── INTERVIEW_VIVA_PREPARATION.md # Exhaustive 25+ Viva Q&A & elevator pitch
├── src/
│   ├── eda_analysis.py             # Automated EDA and chart generation script
│   ├── inspect_datasets.py         # Initial dataset sanity audit script
│   ├── preprocessing.py            # Complete cleaning & feature engineering pipeline
│   ├── recommender.py              # Core CareerRecommender class (hybrid engine)
│   └── train_model.py              # Model training and artifact serialization script
├── tests/
│   └── test_evaluation.py          # Benchmark test suite (P@K, MRR, Edge Cases)
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Python package dependencies
└── README.md                       # Master project overview
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CareerLens-AI.git
cd CareerLens-AI
```

### 2. Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate in PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline (Optional: Artifacts are pre-generated)
```bash
# 1. Clean and structure data
python src/preprocessing.py

# 2. Train and serialize TF-IDF models
python src/train_model.py

# 3. Run validation test suite
python tests/test_evaluation.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/main.py
```
Open your browser to: **`http://localhost:8501`**

---

## 📊 Model Evaluation Results

CareerLens AI was evaluated across **6 industry benchmark personas** using Information Retrieval metrics:

- **Mean Reciprocal Rank (MRR):** **`0.833`** *(Primary target careers are ranked at position #1 or #2)*
- **Mean Precision@5:** **`53.33%`** *(Over half of all top 5 recommendations in every sector are direct domain matches)*
- **Edge-Case Resilience:** 100% pass rate on empty inputs, single-token skills, case variation (`python` vs. `PYTHON`), and out-of-vocabulary terms.

---

## 🚢 Deployment Guide

### Deploying to Streamlit Community Cloud (Free & 1-Click)
1. Push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select your `CareerLens-AI` repository.
4. Set Main file path to: `app/main.py`.
5. Click **"Deploy"**! Your web app will be live on a public URL in 2 minutes.

---

## 📜 Academic Citation & Dataset Acknowledgment
This project utilizes occupational intelligence provided by the **U.S. Department of Labor, Employment and Training Administration (USDOL/ETA) O\*NET 28.0 Database**.

---

## 👩‍💻 Author
- **Developer & Researcher:** Samrudhi
- **Project:** CareerLens AI — End-to-End Machine Learning System
