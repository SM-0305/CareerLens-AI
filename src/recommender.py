"""
CareerLens AI - Career Recommendation & Skill-Gap Analysis Engine
Phases 6 & 7: Recommendation System & Skill-Gap Analysis Module
Purpose: Core business logic class to match user skills against occupational profiles,
         calculate hybrid match scores, identify missing skills, and suggest learning roadmaps.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CareerRecommender:
    def __init__(self, models_dir: str = None):
        """
        Loads serialized TF-IDF model artifacts and cached occupational dataset.
        """
        if models_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, "models")
            
        self.models_dir = models_dir
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        self.matrix_path = os.path.join(models_dir, "tfidf_matrix.joblib")
        self.cache_path = os.path.join(models_dir, "occupations_cache.joblib")
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.df_occupations = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Helper to deserialize model files from disk."""
        if not (os.path.exists(self.vectorizer_path) and os.path.exists(self.matrix_path) and os.path.exists(self.cache_path)):
            raise FileNotFoundError(
                f"Model artifacts missing in {self.models_dir}. Please run 'python src/train_model.py' first."
            )
            
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.tfidf_matrix = joblib.load(self.matrix_path)
        self.df_occupations = joblib.load(self.cache_path)
        print(f"✅ CareerRecommender loaded: {len(self.df_occupations):,} occupations ready.")

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Standardizes a skill token for clean comparison."""
        cleaned = re.sub(r'\s+', ' ', str(skill)).strip().lower()
        return cleaned

    def recommend(self, user_skills: list, top_n: int = 5) -> list:
        """
        Recommends top N suitable occupations based on the user's skill list.
        
        Algorithm:
        1. TF-IDF Cosine Similarity on Combined Features.
        2. Exact/Substring Skill Overlap Ratio against Software & Transferable Skills.
        3. Hybrid Score = (0.65 * Overlap Ratio) + (0.35 * Cosine Similarity).
        """
        if not user_skills:
            return []

        # Filter out empty or whitespace-only inputs
        clean_user_skills = [s.strip() for s in user_skills if s.strip()]
        if not clean_user_skills:
            return []

        user_text = ", ".join(clean_user_skills)
        user_vector = self.vectorizer.transform([user_text])
        
        # 1. Compute Cosine Similarity
        cosine_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        results = []
        user_skills_norm = [self.normalize_skill(s) for s in clean_user_skills]

        # 2. Iterate through occupations to calculate Direct Skill Overlap & Gaps
        for idx, row in self.df_occupations.iterrows():
            software_skills = [s.strip() for s in str(row["Software_Skills"]).split(",") if s.strip()]
            hot_tech = [s.strip() for s in str(row["Hot_Technologies"]).split(",") if s.strip()]
            transferable_skills = [s.strip() for s in str(row["Transferable_Skills"]).split(",") if s.strip()]
            
            all_req_skills = software_skills + transferable_skills
            all_req_norm = {self.normalize_skill(s): s for s in all_req_skills}
            
            # Check which user skills match this occupation
            matched_skills = []
            for u_orig, u_norm in zip(clean_user_skills, user_skills_norm):
                matched = False
                for req_norm, req_orig in all_req_norm.items():
                    if u_norm == req_norm or u_norm in req_norm or req_norm in u_norm:
                        matched_skills.append(u_orig)
                        matched = True
                        break
                # Also check if user skill appears directly in job title or description
                if not matched and (u_norm in str(row["Title"]).lower() or u_norm in str(row["Description"]).lower()):
                    matched_skills.append(u_orig)
                    
            matched_skills = list(dict.fromkeys(matched_skills))  # preserve order, unique
            overlap_ratio = len(matched_skills) / len(clean_user_skills)
            
            # 3. Hybrid Score Formula
            hybrid_score = (0.65 * overlap_ratio) + (0.35 * cosine_scores[idx])
            
            # 4. Identify Missing Skills (Skills the job requires that the user doesn't have)
            matched_norm_set = {self.normalize_skill(m) for m in matched_skills}
            
            # Identify missing hot technologies first (most valuable to learn!)
            missing_hot = [
                h for h in hot_tech 
                if not any(m in self.normalize_skill(h) or self.normalize_skill(h) in m for m in matched_norm_set)
            ]
            
            # Missing general software skills
            missing_software = [
                s for s in software_skills 
                if not any(m in self.normalize_skill(s) or self.normalize_skill(s) in m for m in matched_norm_set)
            ]
            
            # Priority learning recommendations: top hot tech, then top general software
            priority_to_learn = list(dict.fromkeys(missing_hot[:5] + missing_software[:5]))[:5]

            results.append({
                "soc_code": row["O*NET-SOC Code"],
                "title": row["Title"],
                "description": row["Description"],
                "match_score": round(float(hybrid_score) * 100, 2),
                "overlap_percentage": round(float(overlap_ratio) * 100, 1),
                "cosine_similarity": round(float(cosine_scores[idx]) * 100, 2),
                "matched_skills": matched_skills,
                "matched_count": len(matched_skills),
                "total_user_skills": len(clean_user_skills),
                "missing_skills_count": len(missing_software),
                "recommended_skills_to_learn": priority_to_learn,
                "software_skills_sample": software_skills[:8],
                "transferable_skills": transferable_skills[:5]
            })
            
        # Rank by hybrid match score descending
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:top_n]

    def analyze_skill_gap(self, user_skills: list, target_occupation: str) -> dict:
        """
        Deep-dive Skill-Gap Analysis for a specific target occupation.
        """
        clean_user_skills = [s.strip() for s in user_skills if s.strip()]
        
        # Locate target occupation by SOC code or exact/partial title match
        match = self.df_occupations[
            (self.df_occupations["O*NET-SOC Code"].str.strip() == target_occupation.strip()) |
            (self.df_occupations["Title"].str.lower().str.contains(target_occupation.lower().strip(), regex=False))
        ]
        
        if match.empty:
            return {"error": f"Occupation '{target_occupation}' not found in catalog."}
            
        row = match.iloc[0]
        software_skills = [s.strip() for s in str(row["Software_Skills"]).split(",") if s.strip()]
        hot_tech = [s.strip() for s in str(row["Hot_Technologies"]).split(",") if s.strip()]
        transferable_skills = [s.strip() for s in str(row["Transferable_Skills"]).split(",") if s.strip()]
        
        all_req_skills = software_skills + transferable_skills
        all_req_norm = {self.normalize_skill(s): s for s in all_req_skills}
        user_skills_norm = [self.normalize_skill(s) for s in clean_user_skills]
        
        matched_skills = []
        for u_orig, u_norm in zip(clean_user_skills, user_skills_norm):
            for req_norm, req_orig in all_req_norm.items():
                if u_norm == req_norm or u_norm in req_norm or req_norm in u_norm:
                    matched_skills.append(u_orig)
                    break
                    
        matched_skills = list(dict.fromkeys(matched_skills))
        matched_norm_set = {self.normalize_skill(m) for m in matched_skills}
        
        missing_hot = [
            h for h in hot_tech 
            if not any(m in self.normalize_skill(h) or self.normalize_skill(h) in m for m in matched_norm_set)
        ]
        missing_software = [
            s for s in software_skills 
            if not any(m in self.normalize_skill(s) or self.normalize_skill(s) in m for m in matched_norm_set)
        ]
        
        # Qualification Readiness Score:
        # What percentage of the core occupation's software requirements does the user satisfy?
        total_required = max(len(software_skills), 1)
        qualification_percentage = round((len(matched_skills) / total_required) * 100, 1)
        
        return {
            "soc_code": row["O*NET-SOC Code"],
            "title": row["Title"],
            "description": row["Description"],
            "qualification_percentage": min(qualification_percentage, 100.0),
            "matched_skills": matched_skills,
            "matched_count": len(matched_skills),
            "missing_skills_count": len(missing_software),
            "priority_hot_technologies_to_learn": missing_hot[:8],
            "all_missing_software_sample": missing_software[:15],
            "key_transferable_abilities": transferable_skills
        }

if __name__ == "__main__":
    recommender = CareerRecommender()
    
    test_skills = ["Python", "SQL", "Excel", "Statistics", "Data Visualization"]
    print("\n" + "=" * 80)
    print("DEMO: CAREER RECOMMENDATIONS FOR SKILLS:", test_skills)
    print("=" * 80)
    
    recs = recommender.recommend(test_skills, top_n=3)
    for r in recs:
        print(f"\n🎯 [{r['soc_code']}] {r['title']}")
        print(f"   • Overall Match Score: {r['match_score']}% (Skills Overlap: {r['overlap_percentage']}%, TF-IDF: {r['cosine_similarity']}%)")
        print(f"   • Matched Skills Possessed: {r['matched_skills']}")
        print(f"   • Recommended Skills to Learn: {r['recommended_skills_to_learn']}")
        
    print("\n" + "=" * 80)
    print("DEMO: TARGET SKILL-GAP ANALYSIS FOR 'Data Scientists'")
    print("=" * 80)
    gap = recommender.analyze_skill_gap(test_skills, "Data Scientists")
    print(f"Target Career: {gap['title']}")
    print(f"Skills Already Possessed: {gap['matched_skills']}")
    print(f"High-Priority In-Demand Skills to Learn: {gap['priority_hot_technologies_to_learn']}")
