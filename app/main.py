"""
CareerLens AI - Streamlit Web Application
Phase 8: Frontend Development
Purpose: Interactive, modern web dashboard for career recommendations, skill-gap analysis,
         and labor market intelligence.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="CareerLens AI — Smart Career & Skill-Gap Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project src to sys.path to import our custom recommendation engine
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from recommender import CareerRecommender

# -----------------------------------------------------------------------------
# Cached Resource: Load Model Once
# -----------------------------------------------------------------------------
@st.cache_resource
def load_recommender():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    return CareerRecommender(models_dir=models_dir)

@st.cache_data
def load_skills_catalog():
    skills_path = os.path.join(PROJECT_ROOT, "data", "unique_skills.csv")
    if os.path.exists(skills_path):
        df_skills = pd.read_csv(skills_path)
        return df_skills["Skill_Name"].dropna().tolist()
    return []

recommender = load_recommender()
skills_catalog = load_skills_catalog()

# -----------------------------------------------------------------------------
# Custom CSS for Modern UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .badge-score {
        background-color: #2563EB;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-matched {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .badge-learn {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/compass.png", width=70)
    st.title("CareerLens AI")
    st.caption("Machine Learning Career & Skill Guidance System")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        ["🎯 Career Recommender", "🔍 Target Skill-Gap Analyzer", "📊 Labor Market Insights", "ℹ️ System Architecture"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")
    st.caption("Click a preset to test sample career profiles:")
    
    preset_data_science = "Python, SQL, Excel, Statistics, Data Visualization, Machine Learning, Tableau"
    preset_web_dev = "JavaScript, React, Node.js, HTML, CSS, Git, Docker, Python"
    preset_business_analyst = "Excel, SQL, Power BI, Business Analysis, Tableau, Communication, Critical Thinking"
    
    if st.button("📊 Data Science Profile"):
        st.session_state["skill_input"] = preset_data_science
    if st.button("💻 Full-Stack Dev Profile"):
        st.session_state["skill_input"] = preset_web_dev
    if st.button("📈 Business Analyst Profile"):
        st.session_state["skill_input"] = preset_business_analyst

    st.markdown("---")
    st.markdown("🔗 **Built with O*NET Intelligence**")
    st.caption("Covering 1,016 occupations and 8,778 industry skills.")

# -----------------------------------------------------------------------------
# Page 1: Career Recommender
# -----------------------------------------------------------------------------
if menu == "🎯 Career Recommender":
    st.markdown('<div class="main-title">🎯 Career Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your existing technical and transferable skills to discover suitable career matches backed by O*NET data and machine learning.</div>', unsafe_allow_html=True)
    
    default_text = st.session_state.get("skill_input", "Python, SQL, Excel, Statistics, Data Visualization")
    
    col_input, col_settings = st.columns([3, 1])
    
    with col_input:
        user_input = st.text_area(
            "Enter your skills (separated by commas):",
            value=default_text,
            height=100,
            help="Type your tools, languages, and abilities, e.g. Python, SQL, Excel, Problem Solving"
        )
        
    with col_settings:
        top_k = st.slider("Max Recommendations", min_value=3, max_value=15, value=5, step=1)
        btn_recommend = st.button("🚀 Recommend Careers", type="primary", use_container_width=True)

    if btn_recommend or user_input:
        skills_list = [s.strip() for s in user_input.split(",") if s.strip()]
        
        if not skills_list:
            st.warning("⚠️ Please enter at least one skill to generate recommendations.")
        else:
            with st.spinner("Analyzing occupational requirements and computing hybrid match scores..."):
                recs = recommender.recommend(skills_list, top_n=top_k)
                
            if not recs:
                st.info("No matching occupations found. Try broadening your skill keywords.")
            else:
                st.success(f"Found {len(recs)} highly suitable career paths for your skill profile!")
                
                # Visual summary bar chart
                df_chart = pd.DataFrame({
                    "Occupation": [r["title"] for r in recs],
                    "Match Score (%)": [r["match_score"] for r in recs]
                }).sort_values(by="Match Score (%)", ascending=True)
                
                st.bar_chart(df_chart.set_index("Occupation"), color="#2563EB")
                
                st.markdown("### 📋 Detailed Occupational Breakdown")
                
                for rank, r in enumerate(recs, 1):
                    with st.expander(f"#{rank} — {r['title']} (Match: {r['match_score']}%)", expanded=(rank <= 2)):
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Overall Match Score", f"{r['match_score']}%")
                        with col_m2:
                            st.metric("Direct Skills Overlap", f"{r['overlap_percentage']}%")
                        with col_m3:
                            st.metric("Semantic Similarity", f"{r['cosine_similarity']}%")
                            
                        st.markdown(f"**O*NET-SOC Code:** `{r['soc_code']}`")
                        st.markdown(f"**Job Overview:** {r['description']}")
                        
                        col_have, col_need = st.columns(2)
                        with col_have:
                            st.markdown("##### ✅ Skills You Possess:")
                            if r["matched_skills"]:
                                for s in r["matched_skills"]:
                                    st.markdown(f'<span class="badge-matched">{s}</span>', unsafe_allow_html=True)
                            else:
                                st.caption("No exact keyword overlap.")
                                
                        with col_need:
                            st.markdown("##### 🚀 Recommended Skills to Learn:")
                            if r["recommended_skills_to_learn"]:
                                for s in r["recommended_skills_to_learn"]:
                                    st.markdown(f'<span class="badge-learn">{s}</span>', unsafe_allow_html=True)
                            else:
                                st.caption("You have covered the primary tools!")
                                
                        st.markdown("---")
                        st.caption(f"Typical Software Tools in Role: {', '.join(r['software_skills_sample'])}...")

# -----------------------------------------------------------------------------
# Page 2: Target Skill-Gap Analyzer
# -----------------------------------------------------------------------------
elif menu == "🔍 Target Skill-Gap Analyzer":
    st.markdown('<div class="main-title">🔍 Target Career Skill-Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Have a specific dream role? Select it below to see your readiness score, possessed skills, and a customized learning roadmap.</div>', unsafe_allow_html=True)
    
    all_titles = recommender.df_occupations["Title"].sort_values().tolist()
    
    col_t1, col_t2 = st.columns([2, 2])
    with col_t1:
        selected_role = st.selectbox(
            "Select Target Occupation:",
            options=all_titles,
            index=all_titles.index("Data Scientists") if "Data Scientists" in all_titles else 0
        )
    with col_t2:
        gap_skills_input = st.text_input(
            "Your Current Skills:",
            value=st.session_state.get("skill_input", "Python, SQL, Excel, Statistics")
        )
        
    btn_gap = st.button("⚡ Run Skill-Gap Analysis", type="primary")
    
    if btn_gap or selected_role:
        gap_skills = [s.strip() for s in gap_skills_input.split(",") if s.strip()]
        gap_result = recommender.analyze_skill_gap(gap_skills, selected_role)
        
        if "error" in gap_result:
            st.error(gap_result["error"])
        else:
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.metric("Readiness Level", f"{gap_result['qualification_percentage']}%")
            with col_g2:
                st.metric("Skills Already Possessed", f"{gap_result['matched_count']}")
            with col_g3:
                st.metric("Total Tool Gaps", f"{gap_result['missing_skills_count']}")
                
            st.progress(min(gap_result["qualification_percentage"] / 100.0, 1.0))
            
            st.markdown(f"### Role Profile: {gap_result['title']} (`{gap_result['soc_code']}`)")
            st.info(gap_result["description"])
            
            col_p, col_l = st.columns(2)
            with col_p:
                st.markdown("#### ✅ Skills Already Possessed")
                if gap_result["matched_skills"]:
                    for s in gap_result["matched_skills"]:
                        st.markdown(f'<span class="badge-matched">{s}</span>', unsafe_allow_html=True)
                else:
                    st.write("None matched yet.")
                    
            with col_l:
                st.markdown("#### 🔥 Priority Hot Technologies to Learn")
                st.caption("Industry-trending tools prioritized by employer demand:")
                if gap_result["priority_hot_technologies_to_learn"]:
                    for s in gap_result["priority_hot_technologies_to_learn"]:
                        st.markdown(f'<span class="badge-learn">{s}</span>', unsafe_allow_html=True)
                else:
                    st.write("No major hot technology gaps!")
                    
            st.markdown("---")
            st.markdown("#### 🗺️ Recommended 4-Step Learning Roadmap")
            hot_sample = gap_result["priority_hot_technologies_to_learn"][:4]
            step_tools = hot_sample if len(hot_sample) >= 4 else (hot_sample + ["Advanced Domain Tools", "System Architecture"])[:4]
            
            st.markdown(f"""
            1. **Step 1: Solidify Fundamentals** — Strengthen your existing core ({', '.join(gap_result['matched_skills'] if gap_result['matched_skills'] else ['Core Languages'])}).
            2. **Step 2: Master Primary Gap Tool** — Learn **{step_tools[0]}** through hands-on project implementations.
            3. **Step 3: Expand Ecosystem Tools** — Add proficiency in **{step_tools[1]}** and **{step_tools[2]}**.
            4. **Step 4: Professional Capstone** — Integrate **{step_tools[3]}** into an end-to-end portfolio project to demonstrate job readiness.
            """)

# -----------------------------------------------------------------------------
# Page 3: Labor Market Insights
# -----------------------------------------------------------------------------
elif menu == "📊 Labor Market Insights":
    st.markdown('<div class="main-title">📊 O*NET Labor Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exploratory visual analysis generated from official occupational research data.</div>', unsafe_allow_html=True)
    
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        f1 = os.path.join(figures_dir, "top_15_software_skills.png")
        if os.path.exists(f1):
            st.image(f1, caption="Most In-Demand Software Skills in the Economy", use_container_width=True)
            
    with col_c2:
        f2 = os.path.join(figures_dir, "top_10_tech_heavy_occupations.png")
        if os.path.exists(f2):
            st.image(f2, caption="Occupations with Highest Skill Density", use_container_width=True)
            
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        f3 = os.path.join(figures_dir, "skills_count_distribution.png")
        if os.path.exists(f3):
            st.image(f3, caption="Distribution of Required Skills per Occupation", use_container_width=True)
            
    with col_c4:
        f4 = os.path.join(figures_dir, "top_10_transferable_skills.png")
        if os.path.exists(f4):
            st.image(f4, caption="Top Rated Transferable Abilities by Importance", use_container_width=True)

# -----------------------------------------------------------------------------
# Page 4: System Architecture
# -----------------------------------------------------------------------------
elif menu == "ℹ️ System Architecture":
    st.markdown('<div class="main-title">ℹ️ System Architecture & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed technical overview of the machine learning pipeline powering CareerLens AI.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏛️ Machine Learning Pipeline Architecture
    
    1. **Data Ingestion & Unification**:
       - Standard Occupational Classification (SOC) catalog of 1,016 occupations.
       - 31,821 software tool records and 45,500 transferable skill ratings merged on `O*NET-SOC Code`.
       - High-value filtering retains transferable skills with Importance score $\\ge 3.0$.
       
    2. **Feature Engineering**:
       - `Combined_Features`: Composite text string combining doubled title weight, job description, aggregated software skills, and transferable abilities.
       - Token preservation of critical programming symbols (`C++`, `C#`, `.NET`).
       
    3. **Content-Based Vectorization (TF-IDF)**:
       - 20,000 top n-grams (unigrams + bigrams) with sublinear term-frequency scaling.
       - Downweights ubiquitous office tools while boosting specialized technical proficiencies.
       
    4. **Hybrid Scoring Function**:
       $$\\text{Score} = (0.65 \\times \\text{Skill Overlap Ratio}) + (0.35 \\times \\text{Cosine Similarity})$$
       - Eliminates false-positive textual matches while rewarding exact tool possession.
       
    5. **Skill-Gap Optimization**:
       - Automatically prioritizes O*NET **Hot Technologies** for gap closure recommendations.
    """)
    
    st.markdown("---")
    st.caption("CareerLens AI — Developed as an End-to-End Machine Learning Recommendation System.")
