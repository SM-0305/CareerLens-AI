# 📄 CareerLens AI — Comprehensive Project Report & Technical Documentation

**Project Name:** CareerLens AI — Intelligent Career Recommendation and Skill-Gap Diagnostic System  
**Domain:** Artificial Intelligence / Machine Learning / Natural Language Processing  
**Author:** Samrudhi  
**Dataset:** U.S. Department of Labor O*NET (Occupational Information Network) 28.0 Database  

---

## 1. Abstract

In the rapidly evolving modern employment market, students, job seekers, and career changers frequently struggle to identify professional occupations that align with their existing competencies, as well as to determine the exact technical proficiencies needed to transition into target roles. Traditional career advisory systems rely on rigid keyword searches or proprietary black-box algorithms that suffer from severe cold-start limitations and lack explainability. 

This project presents **CareerLens AI**, an explainable, end-to-end Machine Learning recommendation and skill-gap diagnostic platform. Leveraging the comprehensive U.S. Department of Labor O*NET database (encompassing 1,016 standardized occupations and 8,778 unique software and transferable skills), CareerLens AI implements a **Hybrid Content-Based Recommendation Architecture**. By integrating a **Direct Skill Overlap Ratio (65%)** with a **Sublinear TF-IDF Vectorizer and Cosine Similarity (35%)**, the system effectively eliminates spurious lexical matches while rewarding exact tool possession. Furthermore, a specialized skill-gap module isolates missing proficiencies and prioritizes employer-demanded **Hot Technologies** to generate actionable, personalized 4-step learning roadmaps. Empirical evaluation across diverse industry domains demonstrates a **Mean Reciprocal Rank (MRR) of 0.833** and a **Mean Precision@5 of 53.33%**, confirming the platform's high precision, ranking efficacy, and real-world deployment viability.

---

## 2. Introduction

Technological innovation, automation, and digital transformation have dramatically altered occupational requirements across the global economy. Professionals no longer remain within static career trajectories; rather, continuous upskilling and career pivoting have become standard. However, individuals often face significant information asymmetry:
1. They possess fragmented skill sets but cannot map them to the full spectrum of available occupations.
2. They do not know which missing skills represent high-leverage hiring priorities versus peripheral tools.

Machine Learning and Natural Language Processing offer powerful mechanisms to analyze unstructured occupational taxonomies and recommend customized career pathways based on verified labor market intelligence.

---

## 3. Problem Statement

Given a user's self-reported technical tools, programming languages, and transferable abilities $U = \{s_1, s_2, \dots, s_m\}$, the objective is to:
1. Dynamically identify and rank the top $K$ most suitable occupations $O^* \subset \mathcal{O}$ from an occupational knowledge base of $|\mathcal{O}| = 1,016$ careers.
2. Provide transparent, explainable match scores without requiring historical user interaction or rating logs.
3. Quantify the user's skill gap for target careers and recommend prioritized, trending technologies to bridge that gap.

---

## 4. Motivation

Most existing recommendation engines in commercial job portals rely on **Collaborative Filtering** (predicting interest based on user click streams). While effective for e-commerce, collaborative filtering suffers catastrophically from the **Cold-Start Problem** when applied to career counseling, as new graduates and career-switchers possess no prior interaction logs. Furthermore, simple keyword matching (e.g., SQL search) fails because ubiquitous tools like Microsoft Excel appear in over 850 jobs, diluting specialized technical profiles. 

There is a compelling need for an open, data-driven, content-based recommendation platform grounded in standardized occupational intelligence that provides transparent reasoning and clear skill-acquisition roadmaps.

---

## 5. Objectives

The primary engineering and research objectives of CareerLens AI are:
1. **Data Ingestion & Preprocessing**: Clean, normalize, and merge four disjoint O*NET datasets (Occupations, Software Skills, Transferable Skills, and Experience) on standard SOC codes.
2. **Feature Engineering**: Construct composite text representations (`Combined_Features`) with weighted lexical emphasis on job titles.
3. **Machine Learning Model Development**: Implement a 20,000-feature sublinear TF-IDF vectorizer paired with Cosine Similarity.
4. **Hybrid Scoring Architecture**: Engineer a dual-component objective function balancing semantic similarity with exact skill overlap.
5. **Skill-Gap Diagnostic Engine**: Identify missing technical requirements and prioritize O*NET Hot Technologies.
6. **Frontend Web Dashboard**: Develop an intuitive, multi-tab Streamlit user interface featuring interactive visualizations.
7. **Empirical Evaluation**: Evaluate ranking quality using Precision@K, Mean Reciprocal Rank (MRR), and functional edge-case suites.

---

## 6. Literature Review & Existing Systems

### Existing Systems vs. CareerLens AI

| Dimension | Legacy Keyword Search | Collaborative Filtering | CareerLens AI (Proposed) |
| :--- | :--- | :--- | :--- |
| **Data Source** | Unstructured job board posts | User click/application logs | Official O*NET 28.0 Taxonomy |
| **Cold-Start Capability** | Poor (literal string matching only) | **Fails completely** on new users | **Flawless** (content-based profiling) |
| **Ubiquitous Skill Handling** | Dilutes results (e.g. Excel matches 800+ jobs) | Biased toward popular jobs | Automatically penalized via **IDF** |
| **Skill-Gap Diagnostics** | Absent | Absent | **Granular breakdown with Hot Tech prioritization** |
| **Explainability** | Low | None (Black-box matrix factorization) | **100% Transparent (Skill overlap + Cosine)** |

---

## 7. System Architecture

The end-to-end architecture follows a modular, five-tier pipeline:

```mermaid
flowchart TD
    A["Raw O*NET Datasets (1,016 Jobs, 31k Tools, 45k Skills)"] --> B["Phase 4: Preprocessing & Cleaning Pipeline"]
    B --> C["Processed Dataset (Combined_Features)"]
    B --> D["Unique Skills Catalog (8,778 Skills)"]
    C --> E["Phase 5: TF-IDF Vectorizer Training"]
    E --> F["Model Serialization (models/*.joblib)"]
    F --> G["Phases 6 & 7: CareerRecommender Engine"]
    H["User Skill Input"] --> G
    G --> I["Direct Skill Overlap (65%)"]
    G --> J["TF-IDF Cosine Similarity (35%)"]
    I --> K["Hybrid Scoring & Ranking Function"]
    J --> K
    K --> L["Skill-Gap Analysis & Hot Tech Filter"]
    L --> M["Phase 8: Streamlit Interactive UI"]
```

---

## 8. Mathematical Methodology & Algorithms

### 8.1 Sublinear TF-IDF Vectorization
To transform occupational text into numerical feature space while mitigating the dominance of high-frequency terms, we implement sublinear Term Frequency scaling:

$$\text{TF}_{\text{sublinear}}(t, d) = 1 + \log(\text{TF}(t, d)) \quad \text{for } \text{TF}(t, d) > 0$$

The Inverse Document Frequency (IDF) is calculated across all $N = 1,016$ documents:

$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$

$$\vec{v}_d = \text{TF}_{\text{sublinear}}(t, d) \times \text{IDF}(t)$$

### 8.2 Cosine Similarity
Given the user skill query vector $\vec{u}$ and occupational profile vector $\vec{v}$, cosine similarity measures the cosine of angle $\theta$:

$$\text{Cosine}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{i=1}^{M} u_i v_i}{\sqrt{\sum_{i=1}^{M} u_i^2} \sqrt{\sum_{i=1}^{M} v_i^2}}$$

### 8.3 Hybrid Objective Scoring Function
To prevent false-positive semantic matches from descriptions alone, the final recommendation score is computed as:

$$\text{Score}_{\text{Hybrid}}(\vec{u}, \text{job}) = \alpha \cdot \left(\frac{|U \cap S_{\text{job}}|}{|U|}\right) + (1 - \alpha) \cdot \text{Cosine}(\vec{u}, \vec{v}_{\text{job}})$$

Where:
- $\alpha = 0.65$ (Weight allocated to exact/substring skill overlap).
- $(1 - \alpha) = 0.35$ (Weight allocated to semantic TF-IDF context).
- $U$: Set of normalized user skills.
- $S_{\text{job}}$: Set of normalized skills required by the occupation.

---

## 9. Experimental Results & Performance Analysis

### 9.1 Information Retrieval Metrics
The recommendation engine was evaluated against 6 gold-standard benchmark personas spanning multiple sectors:

- **Mean Precision@5:** **`53.33%`** — Over half of all top 5 recommendations in every tested domain were direct, specialized occupational matches.
- **Mean Reciprocal Rank (MRR):** **`0.833`** — Demonstrating that in 66.7% of queries, the primary target career ranked at position #1, and in 33.3% at position #2.

### 9.2 Robustness & Functional Edge Cases
- **Empty Query Handling:** Validated graceful return of empty result sets without application faults.
- **Case Invariance:** Confirmed 100% equivalence between lowercase, uppercase, and mixed-case inputs.
- **Out-of-Vocabulary (OOV) Resilience:** Tested uncataloged tokens; system maintains semantic stability without crashing.

---

## 10. Limitations

1. **Self-Reported Skill Inputs**: Recommendations depend on user accuracy in reporting proficiencies without automated skill testing.
2. **Static Knowledge Base**: O*NET data is updated periodically by the USDOL; emerging AI subfields (e.g. Prompt Engineering) require subsequent taxonomy refreshes.
3. **Experience Equivalence**: The system currently treats skill possession uniformly without factoring in years of seniority.

---

## 11. Future Scope

1. **Resume PDF Parsing**: Integrate OCR and NLP entity extraction to parse user resumes into skill lists automatically.
2. **Dense Semantic Embeddings**: Explore fine-tuned transformer sentence embeddings (e.g., `all-MiniLM-L6-v2`) in conjunction with TF-IDF.
3. **Live Job Board API Integration**: Connect recommendations directly to live job postings (via LinkedIn or Indeed APIs) for instant application links.

---

## 12. Conclusion

CareerLens AI successfully bridges the gap between labor market intelligence and individualized career planning. By marrying robust content-based filtering with explainable skill-gap diagnostics and prioritizing in-demand hot technologies, the platform provides actionable, data-driven career counseling. Built with clean, production-grade Python and Streamlit, the system is fully equipped for academic presentation, professional portfolio demonstration, and real-world deployment.

---

## 13. References

1. U.S. Department of Labor, Employment and Training Administration. *O*NET 28.0 Database*. https://www.onetonline.org/
2. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830, 2011.
3. Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008.
4. Streamlit Inc. *Streamlit Documentation*. https://docs.streamlit.io/
