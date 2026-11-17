"""
CareerLens AI - Model Evaluation & Functional Testing Suite
Phase 9: Model Evaluation and Testing
Purpose: Rigorously evaluate recommendation quality using Information Retrieval metrics
         (Precision@K, Mean Reciprocal Rank - MRR) and functional edge-case tests.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from recommender import CareerRecommender

def run_evaluation():
    print("=" * 80)
    print("           CAREERLENS AI - PHASE 9 MODEL EVALUATION & TESTING")
    print("=" * 80)
    
    recommender = CareerRecommender()
    
    # -------------------------------------------------------------------------
    # 1. Benchmark Test Suite: Ground-Truth Industry Personas
    # -------------------------------------------------------------------------
    benchmark_personas = [
        {
            "persona": "Data Science & Machine Learning",
            "skills": ["Python", "SQL", "Machine Learning", "Statistics", "Tableau", "Pandas"],
            "relevant_domains": [
                "Data Scientist", "Statistic", "Research Scientist", 
                "Quantitative Analyst", "Informatics", "Analytics", "Database Architect"
            ]
        },
        {
            "persona": "Core Software Engineering",
            "skills": ["Java", "C++", "Python", "Git", "Linux", "Software Architecture", "Docker"],
            "relevant_domains": [
                "Software Developer", "Programmer", "Blockchain Engineer", 
                "Systems Engineer", "Architect", "Software Quality", "Computer Systems"
            ]
        },
        {
            "persona": "Web & Front-End Development",
            "skills": ["JavaScript", "HTML", "CSS", "React", "Node.js", "Git", "Web browser software"],
            "relevant_domains": [
                "Web Developer", "Interface Designer", "Software Developer", 
                "Programmer", "Information Technology"
            ]
        },
        {
            "persona": "Database Systems & Architecture",
            "skills": ["SQL", "Oracle", "MySQL", "Database management system software", "Database software"],
            "relevant_domains": [
                "Database Administrator", "Database Architect", "Data Warehousing", 
                "Business Intelligence Analyst", "Systems Analyst"
            ]
        },
        {
            "persona": "Cybersecurity & Infrastructure",
            "skills": ["Network security", "Linux", "Firewalls", "Wireshark", "Information security", "Security software"],
            "relevant_domains": [
                "Security", "Digital Forensics", "Network Architect", 
                "Network Administrator", "Network Support", "Information Security"
            ]
        },
        {
            "persona": "Financial & Business Analysis",
            "skills": ["Financial analysis", "Financial modeling", "Accounting software", "Microsoft Excel", "SAP software"],
            "relevant_domains": [
                "Financial", "Investment Analyst", "Accountant", 
                "Auditor", "Budget Analyst", "Quantitative Analyst", "Risk Specialist"
            ]
        }
    ]
    
    k = 5
    precisions = []
    reciprocal_ranks = []
    
    print(f"\n[Test 1/3] Evaluating Ranking Performance across {len(benchmark_personas)} Industry Benchmarks (K={k})...\n")
    
    for idx, b in enumerate(benchmark_personas, 1):
        recs = recommender.recommend(b["skills"], top_n=k)
        rec_titles = [r["title"] for r in recs]
        
        # Check hits
        hits = 0
        first_hit_rank = None
        for rank, title in enumerate(rec_titles, start=1):
            if any(dom.lower() in title.lower() for dom in b["relevant_domains"]):
                hits += 1
                if first_hit_rank is None:
                    first_hit_rank = rank
                    
        p_at_k = hits / k
        rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        
        precisions.append(p_at_k)
        reciprocal_ranks.append(rr)
        
        print(f"Profile {idx}: {b['persona']}")
        print(f"   • Input Skills: {b['skills']}")
        print(f"   • Top {k} Output: {rec_titles}")
        print(f"   • Precision@{k}: {p_at_k:.2f} ({hits}/{k} domain-relevant)")
        print(f"   • First Relevant Career Rank: #{first_hit_rank} (Reciprocal Rank: {rr:.2f})")
        print("-" * 80)
        
    mean_precision = np.mean(precisions) * 100
    mrr = np.mean(reciprocal_ranks)
    
    print("\n📊 INFORMATION RETRIEVAL BENCHMARK SUMMARY:")
    print(f"   • Mean Precision@{k}: {mean_precision:.2f}% (Average percentage of top 5 recommendations directly within target industry)")
    print(f"   • Mean Reciprocal Rank (MRR): {mrr:.3f} (Values close to 1.0 indicate immediate top-rank placement)")
    
    # -------------------------------------------------------------------------
    # 2. Functional & Robustness Edge-Case Tests
    # -------------------------------------------------------------------------
    print("\n[Test 2/3] Running Functional & Robustness Edge-Case Tests...")
    
    # Edge Case A: Empty Input
    empty_recs = recommender.recommend([])
    assert empty_recs == [], "Edge Case A Failed: Empty input should return empty list."
    print("   ✅ Edge Case A Passed: Empty skills list handled gracefully.")
    
    # Edge Case B: Single Skill
    single_recs = recommender.recommend(["Python"], top_n=3)
    assert len(single_recs) == 3, "Edge Case B Failed: Single skill should return requested top_n."
    print(f"   ✅ Edge Case B Passed: Single skill ('Python') returned valid recommendations ({single_recs[0]['title']}).")
    
    # Edge Case C: Case-Insensitivity
    lower_recs = recommender.recommend(["python", "sql"], top_n=3)
    upper_recs = recommender.recommend(["PYTHON", "SQL"], top_n=3)
    assert lower_recs[0]["title"] == upper_recs[0]["title"], "Edge Case C Failed: Model must be case-insensitive."
    print("   ✅ Edge Case C Passed: Case-insensitivity verified ('python' == 'PYTHON').")
    
    # Edge Case D: Unknown/Nonsense Tokens
    nonsense_recs = recommender.recommend(["RandomNonExistentSkill12345XYZ"], top_n=3)
    assert isinstance(nonsense_recs, list), "Edge Case D Failed: Unknown skills should not crash."
    print("   ✅ Edge Case D Passed: Unseen/Out-of-vocabulary skills handled safely.")
    
    # -------------------------------------------------------------------------
    # 3. Target Skill-Gap Analysis Validation
    # -------------------------------------------------------------------------
    print("\n[Test 3/3] Validating Target Career Skill-Gap Engine...")
    gap_test = recommender.analyze_skill_gap(["Python", "SQL", "Statistics"], "Data Scientists")
    assert "qualification_percentage" in gap_test
    assert "priority_hot_technologies_to_learn" in gap_test
    assert len(gap_test["priority_hot_technologies_to_learn"]) > 0
    print(f"   ✅ Target Gap Test Passed for '{gap_test['title']}':")
    print(f"      - Readiness: {gap_test['qualification_percentage']}%")
    print(f"      - Matched: {gap_test['matched_skills']}")
    print(f"      - Recommended Hot Tech: {gap_test['priority_hot_technologies_to_learn'][:4]}")
    
    print("\n" + "=" * 80)
    print("🎉 ALL PHASE 9 EVALUATION & FUNCTIONAL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
