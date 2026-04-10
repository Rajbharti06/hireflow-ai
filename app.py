"""
app.py — AI Resume Screener v3.0
===================================
Main Streamlit application — startup-tier upgrade:
  - Top Navigation Bar
  - Multi-AI Provider Support (OpenAI, Claude, Gemini, Perplexity, Grok, NVIDIA, Ollama)
  - Job Summary Panel
  - Candidate Comparison View
  - Score Filters
  - CSV Export

Pipeline: PDF → Text → Batch Embed → Skills Extract → LLM Score → Hybrid Blend → Rank → Display

Run: streamlit run app.py
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
from parser import extract_text_from_pdf
from embedder import get_embedding, get_embeddings_batch
from scorer import (
    compute_embedding_score, compute_skill_score,
    compute_hybrid_score, get_score_breakdown,
    get_score_label, get_score_color, compute_keyword_score
)
from skills_local import (
    compare_skills_local, extract_years_experience, detect_education_level,
    compute_resume_quality_score
)
from interview_gen import generate_interview_questions, format_questions_markdown
from explainer import (
    generate_explanation, extract_skills_analysis,
    get_llm_score, generate_cheap_explanation, AI_BACKEND
)
from utils import extract_candidate_name, get_initials
from database import (
    create_session, save_job, save_result,
    get_sessions, get_results_for_session, delete_session,
    get_user_profile, get_total_usage, increment_user_usage, toggle_shortlist
)
from supabase_client import supabase

if "results" not in st.session_state:
    st.session_state["results"] = []


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global reset ── */
    *, .stApp { font-family: 'Inter', sans-serif !important; }
    .stApp {
        background: radial-gradient(ellipse at 80% 0%, rgba(99,102,241,0.08) 0%, transparent 50%),
                    radial-gradient(ellipse at 20% 80%, rgba(212,175,55,0.05) 0%, transparent 50%),
                    linear-gradient(180deg, #080c14 0%, #0b0f19 40%, #050509 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

    /* ── Streamlit tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(13,17,23,0.6);
        border: 1px solid rgba(48,54,61,0.4);
        border-radius: 14px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #6e7681;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212,175,55,0.2), rgba(212,175,55,0.08)) !important;
        color: #d4af37 !important;
        border: 1px solid rgba(212,175,55,0.3) !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Expander ── */
    .stExpander { border: 1px solid rgba(48,54,61,0.4) !important; border-radius: 12px !important; background: rgba(13,17,23,0.4) !important; }
    .stExpander summary { font-weight: 600 !important; color: #c9d1d9 !important; }

    /* ── Metrics (for st.metric) ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(22,27,34,0.8), rgba(13,17,23,0.9));
        border: 1px solid rgba(48,54,61,0.4);
        border-radius: 14px;
        padding: 1rem 1.25rem;
    }
    [data-testid="metric-container"] label { color: #6e7681 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800 !important; color: #e6edf3 !important; }

    /* ── Landing features grid ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(22,27,34,0.7), rgba(13,17,23,0.8));
        border: 1px solid rgba(48,54,61,0.4);
        border-radius: 14px;
        padding: 1.25rem;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: rgba(212,175,55,0.3); }
    .feature-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .feature-title { font-weight: 700; font-size: 0.9rem; color: #e6edf3; margin-bottom: 4px; }
    .feature-desc { font-size: 0.78rem; color: #6e7681; line-height: 1.5; }

    /* ── Steps ── */
    .steps-row {
        display: flex;
        gap: 0;
        margin: 1.5rem 0;
        align-items: flex-start;
    }
    .step-item {
        flex: 1;
        text-align: center;
        padding: 1rem 0.5rem;
        position: relative;
    }
    .step-item:not(:last-child)::after {
        content: '';
        position: absolute;
        right: 0;
        top: 30px;
        width: 1px;
        height: 30px;
        background: rgba(48,54,61,0.5);
    }
    .step-num {
        width: 36px; height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(212,175,55,0.2), rgba(212,175,55,0.08));
        border: 1px solid rgba(212,175,55,0.3);
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.85rem; color: #d4af37;
        margin-bottom: 0.5rem;
    }
    .step-title { font-size: 0.8rem; font-weight: 700; color: #c9d1d9; }
    .step-desc  { font-size: 0.7rem; color: #6e7681; margin-top: 3px; }

    /* ── Upload zone ── */
    .upload-zone {
        background: linear-gradient(135deg, rgba(22,27,34,0.7), rgba(13,17,23,0.8));
        border: 1px dashed rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s, background 0.2s;
    }
    .upload-zone:hover { border-color: rgba(212,175,55,0.4); }
    .upload-zone-label {
        font-size: 0.8rem; font-weight: 700; color: #c9d1d9;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.6rem;
    }
    .upload-hint { font-size: 0.72rem; color: #484f58; margin-top: 4px; }

    /* ── Result Card ── */
    .result-card {
        background: linear-gradient(135deg, rgba(22,27,34,0.92), rgba(13,17,23,0.97));
        border: 1px solid rgba(48,54,61,0.5);
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
    }
    .result-card::before {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.01), transparent);
        pointer-events: none;
    }
    .result-card:hover {
        border-color: rgba(99,102,241,0.35);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(99,102,241,0.1);
    }

    /* ── Candidate header ── */
    .candidate-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem; }
    .candidate-avatar {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem; font-weight: 800; color: white; flex-shrink: 0;
    }
    .candidate-name { font-size: 1.2rem; font-weight: 700; color: #e6edf3; margin: 0; }
    .candidate-file { font-size: 0.75rem; color: #6e7681; margin: 2px 0 0 0; }

    /* ── Score display ── */
    .score-container { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem; }
    .score-number { font-size: 2.8rem; font-weight: 900; line-height: 1; letter-spacing: -2px; }
    .score-label-text { font-size: 0.82rem; font-weight: 600; color: #8b949e; }
    .score-out-of { font-size: 1rem; color: #484f58; font-weight: 400; }

    /* ── Score bar ── */
    .score-bar-bg {
        width: 100%; height: 6px;
        background: rgba(48,54,61,0.4);
        border-radius: 100px; overflow: hidden; margin-bottom: 1rem;
    }
    .score-bar-fill { height: 100%; border-radius: 100px; }

    /* ── Breakdown ── */
    .breakdown-container { display: flex; gap: 8px; margin-bottom: 0.75rem; }
    .breakdown-item {
        flex: 1;
        background: rgba(13,17,23,0.6);
        border: 1px solid rgba(48,54,61,0.25);
        border-radius: 10px;
        padding: 0.55rem 0.6rem;
        text-align: center;
    }
    .breakdown-label { font-size: 0.58rem; font-weight: 700; color: #484f58; text-transform: uppercase; letter-spacing: 0.5px; }
    .breakdown-value { font-size: 1.05rem; font-weight: 800; margin-top: 2px; }
    .breakdown-weight { font-size: 0.52rem; color: #30363d; margin-top: 1px; }

    /* ── Explanation box ── */
    .explanation-box {
        background: rgba(13,17,23,0.5);
        border: 1px solid rgba(48,54,61,0.3);
        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    }
    .explanation-title { font-size: 0.68rem; font-weight: 700; color: #484f58; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.4rem; }
    .explanation-text { font-size: 0.92rem; color: #c9d1d9; line-height: 1.7; }

    /* ── Skills tags ── */
    .skills-section { margin-top: 0.5rem; }
    .skills-row { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 0.4rem; }
    .skill-tag { display: inline-block; padding: 3px 10px; border-radius: 100px; font-size: 0.72rem; font-weight: 600; }
    .skill-matched { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25); color: #4ade80; }
    .skill-missing  { background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.2); color: #f87171; }
    .skill-extra    { background: rgba(56,139,253,0.08); border: 1px solid rgba(56,139,253,0.2); color: #60a5fa; }
    .skills-label { font-size: 0.65rem; font-weight: 700; color: #484f58; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }

    /* ── Rank badge ── */
    .rank-badge {
        position: absolute; top: 1.25rem; right: 1.5rem;
        width: 34px; height: 34px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.82rem; font-weight: 900; color: white;
    }
    .rank-1 { background: linear-gradient(135deg,#f59e0b,#d97706); box-shadow: 0 0 16px rgba(245,158,11,0.35); }
    .rank-2 { background: linear-gradient(135deg,#94a3b8,#64748b); }
    .rank-3 { background: linear-gradient(135deg,#cd7f32,#92400e); }
    .rank-other { background: rgba(48,54,61,0.7); border: 1px solid rgba(48,54,61,0.5); }

    /* ── Stats row ── */
    .stats-container { display: flex; gap: 10px; margin: 1rem 0; }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(22,27,34,0.8), rgba(13,17,23,0.9));
        border: 1px solid rgba(48,54,61,0.35);
        border-radius: 14px; padding: 1.1rem; text-align: center;
    }
    .stat-value { font-size: 1.8rem; font-weight: 900; color: #e6edf3; letter-spacing: -1px; }
    .stat-label { font-size: 0.68rem; color: #6e7681; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07080f 0%, #0d1117 100%);
        border-right: 1px solid rgba(48,54,61,0.35);
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: #8b949e; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #9a7209, #d4af37) !important;
        color: #030304 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.72rem 1.5rem !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(212,175,55,0.35) !important;
        background: linear-gradient(135deg, #d4af37, #fef08a) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── File uploader ── */
    .stFileUploader > div { border-radius: 12px !important; border-color: rgba(99,102,241,0.2) !important; }

    /* ── Input/select ── */
    .stTextInput > div > input, .stSelectbox > div {
        background: rgba(13,17,23,0.7) !important;
        border-color: rgba(48,54,61,0.5) !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
    }

    /* ── Mode badges ── */
    .mode-badge { display: inline-block; padding: 2px 9px; border-radius: 100px; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.5px; }
    .mode-full  { background: rgba(99,102,241,0.12); color: #818cf8; border: 1px solid rgba(99,102,241,0.25); }
    .mode-cheap { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }

    /* ── Processing ── */
    .processing-text { text-align: center; color: #a78bfa; font-size: 1rem; font-weight: 600; letter-spacing: 0.3px; }

    /* ── Alert/info overrides ── */
    .stAlert { border-radius: 12px !important; }

    /* ── Progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg, #d4af37, #f59e0b) !important; border-radius: 100px !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 100px; }

    /* ── Top comparison card ── */
    .top-card {
        background: linear-gradient(135deg, rgba(22,27,34,0.8), rgba(13,17,23,0.9));
        border-radius: 14px; padding: 1rem 1.1rem; text-align: center;
        transition: transform 0.2s;
    }
    .top-card:hover { transform: translateY(-3px); }

    /* ── Divider ── */
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── OAuth Interceptor ────────────────────────────────────────────────────────
if supabase is not None:
    # 1. Listen for OAuth hash in browser and push to query params
    oauth_js = """
    <script>
        if (window.location.hash.includes("access_token")) {
            const hash = window.location.hash.substring(1);
            const params = new URLSearchParams(hash);
            const accessToken = params.get('access_token');
            const refreshToken = params.get('refresh_token');
            if (accessToken) {
                window.location.href = window.location.origin + window.location.pathname + '?access_token=' + accessToken + '&refresh_token=' + refreshToken;
            }
        }
    </script>
    """
    components.html(oauth_js, height=0, width=0)

    # 2. Process query params from the redirect
    query_params = st.query_params
    if "access_token" in query_params and "refresh_token" in query_params:
        try:
            res = supabase.auth.set_session(query_params["access_token"], query_params["refresh_token"])
            st.session_state.user = res.user
            if hasattr(st, "experimental_set_query_params"):
                st.experimental_set_query_params()
            else:
                st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"OAuth Authentication Failed: {str(e)}")


# ─── Auth Guard ───────────────────────────────────────────────────────────────
if supabase is not None and "user" not in st.session_state:
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 5rem;'>Login to Resume AI</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.rerun()
                except Exception as e:
                    msg = str(e)
                    if "Email not confirmed" in msg:
                        st.error("Login failed: Please confirm your email address first.")
                    elif "Invalid login credentials" in msg:
                        st.error("Login failed: Incorrect email or password.")
                    else:
                        st.error(f"Login failed: {msg}")
        with c2:
            if st.button("Sign Up", use_container_width=True):
                if len(password) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    try:
                        res = supabase.auth.sign_up({"email": email, "password": password})
                        # Try to log in immediately if email confirmation is disabled
                        try:
                            res_login = supabase.auth.sign_in_with_password({"email": email, "password": password})
                            st.session_state.user = res_login.user
                            st.rerun()
                        except Exception:
                            st.success("Sign up successful! Please check your email inbox to confirm your account.")
                    except Exception as e:
                        st.error(f"Signup failed: {str(e)}")

        st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1); margin: 2rem 0;'/>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#8b949e;'>Or continue with</p>", unsafe_allow_html=True)
        
        # Single Sign-On Buttons
        google_url = f"{supabase.supabase_url}/auth/v1/authorize?provider=google"
        github_url = f"{supabase.supabase_url}/auth/v1/authorize?provider=github"
        
        st.markdown(f"""
        <div style="display: flex; gap: 10px; justify-content: center; margin-bottom: 20px;">
            <a href="{google_url}" target="_self" style="flex: 1; display:flex; align-items:center; justify-content:center; padding:10px; border-radius:8px; border:1px solid #30363d; color:white; background:#24292e; text-decoration:none;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" width="16" style="margin-right:8px;">Google
            </a>
            <a href="{github_url}" target="_self" style="flex: 1; display:flex; align-items:center; justify-content:center; padding:10px; border-radius:8px; border:1px solid #30363d; color:white; background:#24292e; text-decoration:none;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="16" style="filter:invert(1); margin-right:8px;">GitHub
            </a>
        </div>
        <p style='text-align:center; font-size:12px; color:#8b949e;'><em>For OAuth to work, ensure you enabled Google/GitHub in your Supabase Auth settings.</em></p>
        """, unsafe_allow_html=True)
        
    st.stop()
elif supabase is None:
    # Running without Supabase — dev/local/open-source mode
    st.sidebar.warning("⚠️ Running in local mode (no Supabase). Auth and usage tracking are disabled. Add SUPABASE_URL and SUPABASE_ANON_KEY to .env to enable them.")
    # Inject a minimal session so downstream code that checks st.session_state.user doesn't crash
    if "user" not in st.session_state:
        import types
        fake_user = types.SimpleNamespace(id="local-dev", email="local@dev")
        st.session_state.user = fake_user


# ─── Top Navigation Bar ───────────────────────────────────────────────────────
_nav_user = st.session_state.get("user")
_nav_email = getattr(_nav_user, "email", None) or (
    _nav_user.get("email") if isinstance(_nav_user, dict) else None
) or ""
_nav_initials = (_nav_email[:2].upper() if _nav_email else "?")

st.markdown(f"""
<div style="
display:flex;
justify-content:space-between;
align-items:center;
padding:10px 20px;
background:rgba(22,27,34,0.8);
border-bottom:1px solid rgba(212,175,55,0.2);
margin-top: -3rem;
margin-bottom: 2rem;
margin-left: -3rem;
margin-right: -3rem;
">
    <div style="font-weight:700; font-size:18px; color:white;">
        ⚡ HireFlow AI
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="color:#8b949e; font-size: 13px;">{_nav_email}</span>
        <div style="width:30px;height:30px;border-radius:50%;
            background:linear-gradient(135deg,rgba(212,175,55,0.3),rgba(212,175,55,0.1));
            border:1px solid rgba(212,175,55,0.35);
            display:flex;align-items:center;justify-content:center;
            font-size:0.72rem;font-weight:800;color:#d4af37;">{_nav_initials}</div>
    </div>
</div>
""", unsafe_allow_html=True)





# ─── Sidebar Config ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Usage")
    total_usage = get_total_usage()
    profile = get_user_profile()
    is_pro = profile.get("is_pro", False) if profile else False
    
    limit = 400 if is_pro else 3
    st.progress(min(total_usage / limit, 1.0))
    st.caption(f"{limit - total_usage} AI-powered evaluations remaining")
    
    if total_usage >= limit:
        if is_pro:
            st.warning("⚠️ Pro monthly limit reached.")
        else:
            st.warning("⚠️ Free limit reached")
            st.markdown(
                """
                <a href="https://your-store.lemonsqueezy.com/checkout/buy/xxxxx" target="_blank" style="text-decoration:none;">
                    <button style="width:100%;padding:10px;border-radius:12px;background:linear-gradient(135deg, #6366f1, #4f46e5);color:white;font-weight:700;border:none;cursor:pointer;transition:all 0.3s ease;">
                        🚀 Upgrade to Pro
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # AI Provider configuration is exclusively managed by environment variables.
    
    st.markdown("---")
    
    # Feature toggles
    enable_skills = st.checkbox(
        "🔍 Skill Analysis", value=True,
        help="Extract and compare skills between JD and resumes"
    )
    
    st.markdown("### 🤖 Analysis Mode")
    premium_limit = 5 if is_pro else 2
    remaining_premium = max(0, premium_limit - total_usage)
    if total_usage < premium_limit:
        cheap_mode = False
        st.success(f"✨ Premium AI — {remaining_premium} premium left")
        st.markdown("""
        <div class="mode-badge mode-full">PREMIUM</div>
        <p style="font-size: 0.75rem; color: #6e7681; margin-top: 4px;">
        Gemma 31B explanations & DeepSeek skill extraction
        </p>
        """, unsafe_allow_html=True)
    else:
        cheap_mode = True
        st.info("🔄 Smart Mode active")
        st.markdown("""
        <div class="mode-badge mode-cheap">SMART</div>
        <p style="font-size: 0.75rem; color: #6e7681; margin-top: 4px;">
        Fast structured extraction. AI reasoning disabled to save cost.
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Session History ──
    st.markdown("### 📂 History")
    sessions = get_sessions(limit=10)
    
    if sessions:
        for session in sessions:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                if st.button(
                    f"📋 {session['title'][:25]}",
                    key=f"hist_{session['id']}",
                    use_container_width=True
                ):
                    old_results = get_results_for_session(session['id'])
                    if old_results:
                        st.session_state["results"] = old_results
                        st.session_state["view_mode"] = "history"
                        st.session_state["job_name"] = session["title"]
                        st.rerun()
            with col_b:
                if st.button("🗑️", key=f"del_{session['id']}"):
                    delete_session(session['id'])
                    st.rerun()
    else:
        st.markdown("""
        <p style="font-size: 0.8rem; color: #484f58; text-align: center;">
        No previous sessions
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # ── Logout ──
    if supabase is not None and st.session_state.get("user"):
        _sb_email = getattr(st.session_state.user, "email", "") or ""
        st.markdown(f"<p style='font-size:0.72rem;color:#6e7681;margin-bottom:4px;'>Signed in as<br><b style='color:#c9d1d9'>{_sb_email}</b></p>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            del st.session_state["user"]
            if "results" in st.session_state:
                del st.session_state["results"]
            st.rerun()

    st.markdown("""
    <div style='text-align: center; color: #484f58; font-size: 0.75rem; margin-top: 0.5rem;'>
        <p>Built with ❤️ for recruiters</p>
        <p>v3.0.0 • HireFlow AI</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ──────────────────────────────────────────────────────────────
if not st.session_state.get("results"):
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 0 1rem 0;">
        <div style="
            display: inline-block;
            background: linear-gradient(135deg, rgba(212,175,55,0.15), rgba(255,215,0,0.05));
            border: 1px solid rgba(212,175,55,0.3);
            border-radius: 100px; padding: 5px 18px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
            color: #d4af37; text-transform: uppercase; margin-bottom: 1.2rem;
        ">AI-Powered Recruitment</div>
        <h1 style="
            font-size: 3rem; font-weight: 900; margin: 0;
            background: linear-gradient(135deg, #ffffff 0%, #fef08a 45%, #d4af37 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            line-height: 1.1; letter-spacing: -1px;
        ">HireFlow AI</h1>
        <p style="color:#8b949e; margin-top:0.75rem; font-size:1.05rem; font-weight:400;">
            Screen 100+ candidates in seconds &mdash; explainable, multi-signal AI scoring.
        </p>
    </div>

    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Hybrid AI Scoring</div>
            <div class="feature-desc">Semantic embeddings + skill overlap + LLM judgment blended into one transparent score</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Zero-Cost Mode</div>
            <div class="feature-desc">Works 100% offline with local ML. No API key required to get started</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎤</div>
            <div class="feature-title">Interview Packs</div>
            <div class="feature-desc">Auto-generates tailored interview questions for every candidate — export as markdown</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Analytics Charts</div>
            <div class="feature-desc">Score distribution histogram and skills-coverage chart for the full candidate pool</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Skills Gap Analysis</div>
            <div class="feature-desc">200+ skill categories — matched, missing, and extra skills for every candidate</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📥</div>
            <div class="feature-title">Export Anywhere</div>
            <div class="feature-desc">Download results as CSV or a full markdown interview report with one click</div>
        </div>
    </div>

    <div style="text-align:center; margin: 0.75rem 0 1.5rem 0;">
        <div class="steps-row">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-title">Upload JD</div>
                <div class="step-desc">Job description PDF</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-title">Upload Resumes</div>
                <div class="step-desc">Up to 20 PDFs</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-title">AI Analyzes</div>
                <div class="step-desc">Multi-signal scoring</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-title">Ranked Results</div>
                <div class="step-desc">Hire with confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Application Body ───────────────────────────────────────────────────

if "results" not in st.session_state or not st.session_state["results"]:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-zone-label">📋 &nbsp;Job Description</div>
            <div class="upload-hint">PDF only &nbsp;·&nbsp; Max 5 MB &nbsp;·&nbsp; Max 15 pages</div>
        </div>
        """, unsafe_allow_html=True)
        job_file = st.file_uploader(
            "Upload JD",
            type=["pdf"],
            key="jd_upload",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-zone-label">📄 &nbsp;Candidate Resumes</div>
            <div class="upload-hint">PDF only &nbsp;·&nbsp; Up to 20 files &nbsp;·&nbsp; Max 5 MB each</div>
        </div>
        """, unsafe_allow_html=True)
        resume_files = st.file_uploader(
            "Upload Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_upload",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if job_file and resume_files:
        MAX_BATCH = 20
        if len(resume_files) > MAX_BATCH:
            st.error(f"⚠️ Maximum {MAX_BATCH} resumes allowed per batch to prevent API overload.")
            st.stop()
            
        mode_label = "⚡ Quick Analyze" if cheap_mode else "🚀 Analyze Candidates"
        
        if st.button(mode_label, use_container_width=True, disabled=(total_usage >= limit)):
            
            # ── Processing Pipeline ──
            with st.spinner(""):
                st.markdown('<p class="processing-text">📋 Parsing job description...</p>', unsafe_allow_html=True)
                try:
                    if job_file.size > 5 * 1024 * 1024:
                        raise ValueError("Security Guard: Job description file exceeds the 5MB maximum limit.")
                    job_text = extract_text_from_pdf(job_file)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                
                st.markdown('<p class="processing-text">🧠 Understanding job requirements...</p>', unsafe_allow_html=True)
                job_emb = get_embedding(job_text, input_type="query")
            
            status_text = st.empty()
            status_text.markdown('<p class="processing-text">📄 Parsing resumes...</p>', unsafe_allow_html=True)
            
            resume_texts = []
            resume_names = []
            resume_filenames = []
            parse_errors = []
            
            remaining = limit - total_usage
            if remaining <= 0:
                st.warning("⚠️ Limit reached. Cannot process more resumes.")
                st.stop()
                
            resume_files = resume_files[:remaining]
            
            for resume_file in resume_files:
                candidate_name = extract_candidate_name(resume_file.name)
                try:
                    if resume_file.size > 5 * 1024 * 1024:
                        raise ValueError("Security Guard: Resume file exceeds the 5MB maximum limit.")
                    text = extract_text_from_pdf(resume_file)
                    resume_texts.append(text)
                    resume_names.append(candidate_name)
                    resume_filenames.append(resume_file.name)
                except ValueError as e:
                    parse_errors.append({
                        "name": candidate_name,
                        "filename": resume_file.name,
                        "error": str(e)
                    })
            
            status_text.markdown('<p class="processing-text">🧠 Generating embeddings (batch)...</p>', unsafe_allow_html=True)
            resume_embeddings = get_embeddings_batch(resume_texts)
            
            results = []
            progress_bar = st.progress(0)
            
            job_name = job_file.name.replace(".pdf", "").replace(".PDF", "")
            session_id = create_session(job_name)
            save_job(session_id, job_text, job_file.name)
            
            for i, (resume_text, resume_emb, name, filename) in enumerate(
                zip(resume_texts, resume_embeddings, resume_names, resume_filenames)
            ):
                progress = (i + 1) / len(resume_texts)
                
                current_usage = get_total_usage()
                
                if current_usage >= limit:
                    st.warning(f"⚠️ Limit of {limit} reached. Skipped remaining candidates.")
                    break
                    
                premium_limit = 5 if is_pro else 2
                
                if current_usage < premium_limit:
                    tier = "premium"     # Gemma 31B + DeepSeek + full scoring
                else:
                    tier = "transition"  # DeepSeek skills + cheap explanation
                
                tier_labels = {"premium": "✨ Premium", "transition": "🔄 Smart"}
                status_text.markdown(
                    f'<p class="processing-text">🔍 {tier_labels.get(tier, "⚡ Fast")} | Scoring {name} ({i+1}/{len(resume_texts)})...</p>',
                    unsafe_allow_html=True
                )
                
                try:
                    emb_score = compute_embedding_score(job_emb, resume_emb)

                    # ── Always-free: experience years + education level ──
                    exp_years = extract_years_experience(resume_text)
                    _, edu_label = detect_education_level(resume_text)

                    # ── Skills extraction ──
                    # Premium/transition: try LLM (DeepSeek) first, fall back to local
                    # Local tier / disabled: use local keyword matching directly
                    skills = None
                    s_score = 50.0
                    if enable_skills:
                        if tier in ("premium", "transition"):
                            skills = extract_skills_analysis(job_text, resume_text)
                            # Fall back to local if API returned nothing useful
                            if not (skills.get("matched_skills") or skills.get("missing_skills")):
                                skills = compare_skills_local(job_text, resume_text)
                        else:
                            # Local mode — free, always works
                            skills = compare_skills_local(job_text, resume_text)
                        s_score = compute_skill_score(skills)
                    else:
                        skills = {"matched_skills": [], "missing_skills": [], "extra_skills": []}

                    # ── LLM/keyword confidence score ──
                    # Premium: Gemma 31B for true AI judgment
                    # All others: keyword overlap (free, better than constant 50)
                    if tier == "premium":
                        l_score = get_llm_score(job_text, resume_text)
                        # If LLM failed or returned default, use keyword score instead
                        if l_score == 50.0:
                            l_score = compute_keyword_score(job_text, resume_text)
                    else:
                        l_score = compute_keyword_score(job_text, resume_text)

                    final_score = compute_hybrid_score(emb_score, s_score, l_score)

                    # ── Explanation ──
                    if tier == "premium":
                        explanation = generate_explanation(job_text, resume_text, final_score)
                    else:
                        explanation = generate_cheap_explanation(final_score, skills, exp_years)

                    quality = compute_resume_quality_score(resume_text)

                    results.append({
                        "name": name,
                        "filename": filename,
                        "score": final_score,
                        "embedding_score": emb_score,
                        "skill_score": s_score,
                        "llm_score": l_score,
                        "explanation": explanation,
                        "skills": skills,
                        "experience_years": exp_years,
                        "education": edu_label,
                        "tier": tier,
                        "quality_score": quality,
                        "resume_text": resume_text,
                    })
                    
                    increment_user_usage(1)
                    
                except Exception as e:
                    parse_errors.append({
                        "name": name,
                        "filename": filename,
                        "error": f"AI Pipeline Error: {str(e)}"
                    })
                    st.toast(f"⚠️ Failed to process {name}")
                    continue
                
                progress_bar.progress(progress)
            
            for err in parse_errors:
                results.append({
                    "name": err["name"],
                    "filename": err["filename"],
                    "score": 0,
                    "embedding_score": 0,
                    "skill_score": 0,
                    "llm_score": 0,
                    "explanation": f"⚠️ Error processing: {err['error']}",
                    "skills": None
                })
            
            progress_bar.empty()
            status_text.empty()
            
            # Primary sort: score descending. Tiebreaker: experience years descending.
            results = sorted(
                results,
                key=lambda x: (round(x["score"], 0), x.get("experience_years", 0)),
                reverse=True
            )
            
            for rank, r in enumerate(results, 1):
                save_result(
                    job_id=session_id,
                    candidate_name=r["name"],
                    filename=r["filename"],
                    score=r["score"],
                    explanation=r["explanation"],
                    skills_data=r.get("skills"),
                    embedding_score=r.get("embedding_score"),
                    skill_score=r.get("skill_score"),
                    llm_score=r.get("llm_score"),
                    rank=rank
                )
            
            st.session_state["results"] = results
            st.session_state["view_mode"] = "new"
            st.session_state["job_name"] = job_name
            st.success("⚡ Top candidates identified instantly")
            time.sleep(1)
            st.rerun()

    elif not job_file:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #484f58;">
            <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📋</p>
            <p style="font-size: 1rem; font-weight: 500;">Upload a job description to get started</p>
        </div>
        """, unsafe_allow_html=True)

    elif not resume_files:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #484f58;">
            <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</p>
            <p style="font-size: 1rem; font-weight: 500;">Upload candidate resumes to analyze</p>
        </div>
        """, unsafe_allow_html=True)


# ─── Display Results Phase ───────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    all_results = st.session_state["results"]
    job_name = st.session_state.get("job_name", "Unknown Job")

    high = len([r for r in all_results if r["score"] >= 75])
    mid  = len([r for r in all_results if 50 <= r["score"] < 75])
    low  = len([r for r in all_results if r["score"] < 50])
    avg  = round(sum(r["score"] for r in all_results) / len(all_results), 1) if all_results else 0

    # ── Job title bar ──
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between;
        background:rgba(22,27,34,0.7); border:1px solid rgba(48,54,61,0.4);
        border-radius:14px; padding:1rem 1.5rem; margin-bottom:1.25rem;">
        <div>
            <div style="font-size:0.68rem;font-weight:700;color:#484f58;text-transform:uppercase;letter-spacing:1.2px;">Screening Results</div>
            <div style="font-size:1.2rem;font-weight:800;color:#e6edf3;margin-top:2px;">{job_name}</div>
        </div>
        <div style="display:flex;gap:20px;text-align:center;">
            <div><div style="font-size:1.5rem;font-weight:900;color:#e6edf3;">{len(all_results)}</div><div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;">Total</div></div>
            <div><div style="font-size:1.5rem;font-weight:900;color:#4ade80;">{high}</div><div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;">Strong</div></div>
            <div><div style="font-size:1.5rem;font-weight:900;color:#f59e0b;">{mid}</div><div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;">Consider</div></div>
            <div><div style="font-size:1.5rem;font-weight:900;color:#f87171;">{low}</div><div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;">Reject</div></div>
            <div><div style="font-size:1.5rem;font-weight:900;color:#d4af37;">{avg}</div><div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:1px;">Avg Score</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Results Tabs ──
    tab_overview, tab_candidates, tab_analytics = st.tabs(
        ["🏆  Overview", "👥  All Candidates", "📈  Analytics"]
    )

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown("<h3 style='color:#e6edf3;font-size:1.1rem;margin:1rem 0 0.75rem 0;'>🥇 Top Candidates</h3>", unsafe_allow_html=True)
        top3 = all_results[:min(3, len(all_results))]
        medals = ["🥇", "🥈", "🥉"]
        comp_cols = st.columns(min(3, len(top3)))
        for idx, r in enumerate(top3):
            color = get_score_color(r["score"])
            matched_count = len(r["skills"].get("matched_skills", [])) if r.get("skills") else 0
            exp_str  = f"💼 {r.get('experience_years',0)}yr  " if r.get("experience_years") else ""
            edu_str  = f"🎓 {r.get('education','')}" if r.get("education") and r["education"] != "Not specified" else ""
            q_score  = r.get("quality_score", 0)
            with comp_cols[idx]:
                st.markdown(f"""
                <div class="top-card" style="border:1px solid {color}44; border-top:3px solid {color};">
                    <div style="font-size:1.4rem;margin-bottom:4px;">{medals[idx]}</div>
                    <div style="font-size:2.2rem;font-weight:900;color:{color};letter-spacing:-2px;">{r['score']}</div>
                    <div style="font-size:0.72rem;color:#8b949e;margin-bottom:6px;">/100</div>
                    <div style="font-weight:700;color:#e6edf3;font-size:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r['name']}</div>
                    <div style="font-size:0.72rem;color:#6e7681;margin-top:6px;">✅ {matched_count} skills matched</div>
                    <div style="font-size:0.7rem;color:#6e7681;margin-top:3px;">{exp_str}{edu_str}</div>
                    <div style="font-size:0.7rem;color:#484f58;margin-top:3px;">📄 Resume quality: {q_score}/100</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-divider' style='margin:1.5rem 0;'></div>", unsafe_allow_html=True)

        # Top 5 skills required vs coverage
        st.markdown("<h3 style='color:#e6edf3;font-size:1.1rem;margin:0 0 0.75rem 0;'>📋 Shortlisted Candidates</h3>", unsafe_allow_html=True)
        shortlisted_overview = [r for r in all_results if r.get("shortlisted")]
        if shortlisted_overview:
            for r in shortlisted_overview:
                color = get_score_color(r["score"])
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    background:rgba(22,27,34,0.6);border:1px solid rgba(48,54,61,0.3);
                    border-left:3px solid {color};border-radius:10px;margin-bottom:6px;">
                    <div style="font-size:1.4rem;font-weight:900;color:{color};min-width:40px;">{r['score']}</div>
                    <div>
                        <div style="font-weight:700;color:#e6edf3;font-size:0.9rem;">{r['name']}</div>
                        <div style="font-size:0.7rem;color:#6e7681;">{r.get('filename','')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#484f58;font-size:0.85rem;'>Star candidates on the Candidates tab to shortlist them here.</p>", unsafe_allow_html=True)

        # Export toolbar
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            df_export = pd.DataFrame([
                {k: v for k, v in r.items() if k not in ("skills", "resume_text")}
                for r in all_results
            ])
            st.download_button("📥 Export CSV", df_export.to_csv(index=False), "resume_results.csv", use_container_width=True)
        with exp_col2:
            md_lines = [f"# Resume Screening Report — {job_name}\n"]
            for r in all_results:
                q = generate_interview_questions(job_name, r.get("skills") or {}, r["score"], r["name"], r.get("experience_years", 0))
                md_lines.append(format_questions_markdown(q, r["name"], r["score"]))
                md_lines.append("---\n")
            st.download_button("📋 Full Report (MD)", "\n".join(md_lines), f"{job_name}_report.md", mime="text/markdown", use_container_width=True)
        with exp_col3:
            if st.button("🔄 New Analysis", use_container_width=True):
                del st.session_state["results"]
                if "job_name" in st.session_state:
                    del st.session_state["job_name"]
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — ALL CANDIDATES
    # ════════════════════════════════════════════════════════════════════
    with tab_candidates:
        # Filter toolbar
        search_query = st.text_input(
            "search",
            placeholder="🔎  Search by name or skill  (e.g. 'Python' or 'Sarah')",
            label_visibility="collapsed",
            key="candidate_search"
        )
        fc1, fc2 = st.columns([3, 1])
        with fc1:
            min_score = st.slider("Minimum score", 0, 100, 0, key="min_score_slider",
                                  help="Drag right to hide weak candidates")
        with fc2:
            st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
            show_top = st.checkbox("🔥 Top Picks only (≥70)", key="show_top_cb")

        # Apply filters
        filtered_results = all_results
        if search_query:
            q_lower = search_query.lower()
            filtered_results = [
                r for r in filtered_results
                if q_lower in r["name"].lower()
                or (r.get("skills") and any(
                    q_lower in s.lower()
                    for s in r["skills"].get("matched_skills", []) + r["skills"].get("extra_skills", [])
                ))
            ]
        filtered_results = [r for r in filtered_results if r["score"] >= min_score]
        if show_top:
            filtered_results = [r for r in filtered_results if r["score"] >= 70]

        st.markdown(f"<p style='color:#6e7681;font-size:0.82rem;margin-bottom:0.75rem;'>Showing {len(filtered_results)} of {len(all_results)} candidates</p>", unsafe_allow_html=True)

        # ── Candidate Cards ──
        for rank, r in enumerate(filtered_results, 1):
            absolute_rank = all_results.index(r) + 1
            score = r["score"]
            color = get_score_color(score)
            label = get_score_label(score)
            initials = get_initials(r["name"])

            emb_s = r.get("embedding_score", score)
            skl_s = r.get("skill_score", 50)
            llm_s = r.get("llm_score", 50)
            exp_years = r.get("experience_years", 0)
            edu_label = r.get("education", "")
            tier_used = r.get("tier", "local")

            quality_score = r.get("quality_score", 0)

            # ── Meta badges ──
            meta_parts = []
            if exp_years:
                meta_parts.append(f'<span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:3px 10px;font-size:0.72rem;color:#818cf8;font-weight:600;">💼 {exp_years}yr exp</span>')
            if edu_label and edu_label != "Not specified":
                meta_parts.append(f'<span style="background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.25);border-radius:20px;padding:3px 10px;font-size:0.72rem;color:#4ade80;font-weight:600;">🎓 {edu_label}</span>')
            if quality_score:
                q_color = "#22c55e" if quality_score >= 70 else "#f59e0b" if quality_score >= 45 else "#ef4444"
                meta_parts.append(f'<span style="background:rgba(34,197,94,0.08);border:1px solid {q_color}44;border-radius:20px;padding:3px 10px;font-size:0.72rem;color:{q_color};font-weight:600;">📄 Resume {quality_score}/100</span>')
            tier_colors = {"premium": ("#f59e0b", "✨ Premium AI"), "transition": ("#818cf8", "🔄 Smart"), "local": ("#6e7681", "⚡ Local")}
            t_color, t_label = tier_colors.get(tier_used, ("#6e7681", "⚡ Local"))
            meta_parts.append(f'<span style="background:rgba(110,118,129,0.1);border:1px solid rgba(110,118,129,0.2);border-radius:20px;padding:3px 10px;font-size:0.72rem;color:{t_color};font-weight:600;">{t_label}</span>')
            meta_html = f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">{"".join(meta_parts)}</div>'

            breakdown_html = f"""
            <div class="breakdown-container">
                <div class="breakdown-item">
                    <div class="breakdown-label">Semantic</div>
                    <div class="breakdown-value" style="color:#818cf8;">{emb_s:.0f}</div>
                    <div class="breakdown-weight">50% weight</div>
                </div>
                <div class="breakdown-item">
                    <div class="breakdown-label">Skills</div>
                    <div class="breakdown-value" style="color:#4ade80;">{skl_s:.0f}</div>
                    <div class="breakdown-weight">30% weight</div>
                </div>
                <div class="breakdown-item">
                    <div class="breakdown-label">Keywords</div>
                    <div class="breakdown-value" style="color:#f59e0b;">{llm_s:.0f}</div>
                    <div class="breakdown-weight">20% weight</div>
                </div>
            </div>
            """

            skills_html = ""
            skills_data = r.get("skills")
            if skills_data and isinstance(skills_data, dict):
                matched = skills_data.get("matched_skills", [])
                missing = skills_data.get("missing_skills", [])
                extra   = skills_data.get("extra_skills", [])
                if matched or missing or extra:
                    skills_html = '<div class="skills-section">'
                    if matched:
                        skills_html += '<div class="skills-label">✅ Matched</div><div class="skills-row">'
                        skills_html += ''.join(f'<span class="skill-tag skill-matched">{s}</span>' for s in matched)
                        skills_html += '</div>'
                    if missing:
                        skills_html += '<div class="skills-label">❌ Missing</div><div class="skills-row">'
                        skills_html += ''.join(f'<span class="skill-tag skill-missing">{s}</span>' for s in missing)
                        skills_html += '</div>'
                    if extra:
                        skills_html += '<div class="skills-label">➕ Additional</div><div class="skills-row">'
                        skills_html += ''.join(f'<span class="skill-tag skill-extra">{s}</span>' for s in extra)
                        skills_html += '</div>'
                    skills_html += '</div>'

            expl_title = "AI Assessment" if score >= 40 else "⚠️ Why Not Selected"
            rank_class = f"rank-{absolute_rank}" if absolute_rank <= 3 else "rank-other"

            st.markdown(f"""
            <div class="result-card" style="border-left:3px solid {color};">
                <div style="position:relative;">
                    <div class="rank-badge {rank_class}">#{absolute_rank}</div>
                </div>
                <div class="candidate-header">
                    <div class="candidate-avatar" style="background:linear-gradient(135deg,{color}88,{color}44);">{initials}</div>
                    <div style="flex:1;">
                        <p class="candidate-name">{r['name']}</p>
                        <p class="candidate-file">{r['filename']}</p>
                    </div>
                </div>
                {meta_html}
                <div class="score-container">
                    <div>
                        <span class="score-number" style="color:{color};">{score}</span>
                        <span class="score-out-of">/100</span>
                    </div>
                    <div class="score-label-text">{label}</div>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{score}%;background:linear-gradient(90deg,{color},{color}66);"></div>
                </div>
                {breakdown_html}
                <div class="explanation-box">
                    <div class="explanation-title">{expl_title}</div>
                    <div class="explanation-text">{r['explanation']}</div>
                </div>
                {skills_html}
            </div>
            """, unsafe_allow_html=True)

            # ── Shortlist toggle ──
            sl_col, _ = st.columns([1, 4])
            with sl_col:
                shortlisted = st.checkbox(
                    "⭐ Shortlist",
                    value=r.get("shortlisted", False),
                    key=f"short_{r.get('id', absolute_rank)}"
                )
                if shortlisted != r.get("shortlisted", False):
                    if "id" in r:
                        toggle_shortlist(r["id"], shortlisted)
                    r["shortlisted"] = shortlisted

            # ── Interview questions expander ──
            with st.expander(f"🎤 Interview Questions — {r['name']}", expanded=False):
                interview_qs = generate_interview_questions(
                    job_name, r.get("skills") or {}, r["score"],
                    r["name"], r.get("experience_years", 0),
                )
                md_pack = format_questions_markdown(interview_qs, r["name"], r["score"])
                iq_col1, iq_col2 = st.columns(2)
                sections_map = [
                    ("quick_screen", "⚡ Quick Screen"),
                    ("technical",    "🔧 Technical Deep Dives"),
                    ("gap_probing",  "🔍 Gap Probing"),
                    ("behavioral",   "🧠 Behavioral (STAR)"),
                    ("culture_fit",  "💡 Culture & Motivation"),
                ]
                for idx_s, (key, title) in enumerate(sections_map):
                    qs = interview_qs.get(key, [])
                    if not qs:
                        continue
                    with (iq_col1 if idx_s % 2 == 0 else iq_col2):
                        st.markdown(f"**{title}**")
                        for qi, q in enumerate(qs, 1):
                            st.markdown(f"{qi}. {q}")
                        st.markdown("")
                st.download_button(
                    "⬇️ Download Interview Pack", md_pack,
                    f"{r['name'].replace(' ','_')}_interview_pack.md",
                    mime="text/markdown", key=f"dl_iq_{absolute_rank}",
                )

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — ANALYTICS
    # ════════════════════════════════════════════════════════════════════
    with tab_analytics:
        try:
            import plotly.graph_objects as go

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                scores = [r["score"] for r in all_results]
                bins_bounds = [0, 40, 60, 75, 90, 101]
                bin_labels  = ["Poor\n<40", "Weak\n40–60", "Moderate\n60–75", "Strong\n75–90", "Exceptional\n≥90"]
                bin_colors  = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#10b981"]
                counts = [0] * 5
                for s in scores:
                    for i in range(5):
                        if bins_bounds[i] <= s < bins_bounds[i + 1]:
                            counts[i] += 1
                            break
                fig_dist = go.Figure(go.Bar(
                    x=bin_labels, y=counts, marker_color=bin_colors,
                    text=counts, textposition="outside",
                    textfont=dict(color="#e6edf3", size=13, family="Inter"),
                ))
                fig_dist.update_layout(
                    title=dict(text="Score Distribution", font=dict(color="#e6edf3", size=14)),
                    paper_bgcolor="rgba(13,17,23,0)", plot_bgcolor="rgba(13,17,23,0)",
                    font=dict(color="#6e7681", size=11),
                    xaxis=dict(tickfont=dict(size=10), gridcolor="rgba(48,54,61,0.2)", showline=False),
                    yaxis=dict(gridcolor="rgba(48,54,61,0.2)", zeroline=False),
                    margin=dict(l=20, r=20, t=50, b=20), height=280,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with chart_col2:
                top8 = all_results[:min(8, len(all_results))]
                names8   = [r["name"].split()[0] for r in top8]
                matched8 = [len(r["skills"].get("matched_skills", [])) if r.get("skills") else 0 for r in top8]
                missing8 = [len(r["skills"].get("missing_skills", [])) if r.get("skills") else 0 for r in top8]
                fig_sk = go.Figure()
                fig_sk.add_trace(go.Bar(name="Matched", x=names8, y=matched8, marker_color="#22c55e", text=matched8, textposition="inside"))
                fig_sk.add_trace(go.Bar(name="Missing",  x=names8, y=missing8, marker_color="#ef4444", text=missing8, textposition="inside"))
                fig_sk.update_layout(
                    title=dict(text="Skills Coverage (Top 8)", font=dict(color="#e6edf3", size=14)),
                    barmode="stack",
                    paper_bgcolor="rgba(13,17,23,0)", plot_bgcolor="rgba(13,17,23,0)",
                    font=dict(color="#6e7681", size=11),
                    xaxis=dict(tickfont=dict(size=10), gridcolor="rgba(48,54,61,0.2)"),
                    yaxis=dict(gridcolor="rgba(48,54,61,0.2)", zeroline=False),
                    legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
                    margin=dict(l=20, r=20, t=60, b=20), height=280,
                )
                st.plotly_chart(fig_sk, use_container_width=True)

            # ── Avg metric breakdown ──
            avg_emb = round(sum(r.get("embedding_score", 0) for r in all_results) / len(all_results), 1)
            avg_skl = round(sum(r.get("skill_score", 0) for r in all_results) / len(all_results), 1)
            avg_llm = round(sum(r.get("llm_score", 0) for r in all_results) / len(all_results), 1)
            fig_avg = go.Figure(go.Bar(
                x=[avg_emb, avg_skl, avg_llm],
                y=["Semantic Match", "Skills Overlap", "Keyword Fit"],
                orientation="h",
                marker_color=["#818cf8", "#22c55e", "#f59e0b"],
                text=[f"{avg_emb:.0f}", f"{avg_skl:.0f}", f"{avg_llm:.0f}"],
                textposition="outside",
                textfont=dict(color="#e6edf3", size=13, family="Inter"),
            ))
            fig_avg.update_layout(
                title=dict(text="Avg Signal Breakdown (all candidates)", font=dict(color="#e6edf3", size=14)),
                paper_bgcolor="rgba(13,17,23,0)", plot_bgcolor="rgba(13,17,23,0)",
                font=dict(color="#6e7681", size=11),
                xaxis=dict(range=[0, 110], gridcolor="rgba(48,54,61,0.2)", showline=False),
                yaxis=dict(gridcolor="rgba(48,54,61,0.2)", tickfont=dict(size=11, color="#c9d1d9")),
                margin=dict(l=20, r=60, t=50, b=20), height=200,
            )
            st.plotly_chart(fig_avg, use_container_width=True)

            # ── Score vs Experience scatter ──
            if any(r.get("experience_years") for r in all_results):
                names_all  = [r["name"] for r in all_results]
                scores_all = [r["score"] for r in all_results]
                exp_all    = [r.get("experience_years", 0) for r in all_results]
                colors_all = [get_score_color(s) for s in scores_all]
                fig_scatter = go.Figure(go.Scatter(
                    x=exp_all, y=scores_all, mode="markers+text",
                    text=[n.split()[0] for n in names_all],
                    textposition="top center",
                    textfont=dict(size=10, color="#8b949e"),
                    marker=dict(size=12, color=colors_all, line=dict(color="rgba(0,0,0,0.3)", width=1)),
                ))
                fig_scatter.update_layout(
                    title=dict(text="Score vs. Experience Years", font=dict(color="#e6edf3", size=14)),
                    paper_bgcolor="rgba(13,17,23,0)", plot_bgcolor="rgba(13,17,23,0)",
                    font=dict(color="#6e7681", size=11),
                    xaxis=dict(title="Years of Experience", gridcolor="rgba(48,54,61,0.2)", titlefont=dict(color="#6e7681")),
                    yaxis=dict(title="Match Score", gridcolor="rgba(48,54,61,0.2)", titlefont=dict(color="#6e7681")),
                    margin=dict(l=40, r=20, t=50, b=40), height=300,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        except ImportError:
            st.info("Install plotly to see charts: `pip install plotly`")
