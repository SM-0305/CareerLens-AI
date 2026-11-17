"""
CareerLens AI - Model Training & Serialization
Phase 5: Machine Learning Model Development
Purpose: Train the Content-Based TF-IDF Vectorizer, generate the feature matrix,
         and serialize model artifacts into the models/ directory.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_occupations.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.joblib")
OCCUPATIONS_CACHE_PATH = os.path.join(MODELS_DIR, "occupations_cache.joblib")

def train_and_save_model():
    print("=" * 80)
    print("           CAREERLENS AI - PHASE 5 MODEL TRAINING & SERIALIZATION")
    print("=" * 80)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Load Processed Dataset
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Loading preprocessed occupations dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at: {DATA_PATH}. Please run src/preprocessing.py first.")
        
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} occupations from {DATA_PATH}")
    
    # -------------------------------------------------------------------------
    # 2. Fit TF-IDF Vectorizer
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Training TF-IDF Vectorizer...")
    # ngram_range=(1, 2) captures both individual words ("python", "sql") 
    # and 2-word phrases ("machine learning", "data science")
    # sublinear_tf=True applies sublinear scaling (1 + log(tf)) to prevent dominant terms from skewing scores
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(df["Combined_Features"].fillna(""))
    print(f"✅ TF-IDF Matrix successfully constructed!")
    print(f"   • Occupations (Rows): {tfidf_matrix.shape[0]:,}")
    print(f"   • Vocabulary Features (Columns): {tfidf_matrix.shape[1]:,}")
    
    # -------------------------------------------------------------------------
    # 3. Serialize (Save) Artifacts
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Serializing model artifacts to disk...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, MATRIX_PATH)
    joblib.dump(df, OCCUPATIONS_CACHE_PATH)
    
    print(f"✅ Saved Vectorizer: {VECTORIZER_PATH} ({os.path.getsize(VECTORIZER_PATH):,} bytes)")
    print(f"✅ Saved TF-IDF Matrix: {MATRIX_PATH} ({os.path.getsize(MATRIX_PATH):,} bytes)")
    print(f"✅ Saved Occupations Cache: {OCCUPATIONS_CACHE_PATH} ({os.path.getsize(OCCUPATIONS_CACHE_PATH):,} bytes)")
    
    # -------------------------------------------------------------------------
    # 4. Model Verification & Test Query
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Verifying model with a test skill query...")
    test_query = "Python, SQL, Machine Learning, Data Visualization, Statistics"
    print(f"Query: '{test_query}'")
    
    # Transform test query using fitted vectorizer
    query_vec = vectorizer.transform([test_query])
    cos_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = cos_sim.argsort()[::-1][:5]
    print("\nTop 5 Content-Based Semantic Matches:")
    for rank, idx in enumerate(top_indices, start=1):
        score = cos_sim[idx] * 100
        title = df.iloc[idx]["Title"]
        code = df.iloc[idx]["O*NET-SOC Code"]
        print(f"  {rank}. [{code}] {title} — Similarity: {score:.2f}%")
        
    print("\n" + "=" * 80)
    print("🎉 PHASE 5 MODEL TRAINING & SERIALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    train_and_save_model()
