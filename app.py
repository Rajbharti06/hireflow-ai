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
    get_score_label, get_score_color
)
from explainer import (
    generate_explanation, extract_skills_analysis,
    get_llm_score, generate_cheap_explanation, AI_BACKEND
)
from utils import extract_candidate_name, get_initials
from database import (
    create_session, save_job, save_result,
    get_sessions, get_results_for_session, delete_session,
    get_user_profile, get_total_usage
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
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    *, .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: radial-gradient(circle at top right, #111827 0%, #0b0f19 50%, #050505 100%);
    }
    
    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15), rgba(255, 215, 0, 0.05));
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 100px;
        padding: 6px 18px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: #d4af37;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #fef08a 40%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Upload Cards ── */
    .upload-section {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(20px);
    }
    .upload-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c9d1d9;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }
    .upload-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }

    /* ── Result Card ── */
    .result-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.95));
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .result-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }

    /* ── Candidate Header ── */
    .candidate-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .candidate-avatar {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }
    .candidate-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0;
    }
    .candidate-file {
        font-size: 0.8rem;
        color: #6e7681;
        margin: 0;
    }

    /* ── Score Display ── */
    .score-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .score-number {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label-text {
        font-size: 0.85rem;
        font-weight: 500;
        color: #8b949e;
    }
    .score-out-of {
        font-size: 1rem;
        color: #484f58;
        font-weight: 400;
    }

    /* ── Score Bar ── */
    .score-bar-bg {
        width: 100%;
        height: 8px;
        background: rgba(48, 54, 61, 0.5);
        border-radius: 100px;
        overflow: hidden;
        margin-bottom: 1.25rem;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Score Breakdown ── */
    .breakdown-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .breakdown-item {
        flex: 1;
        background: rgba(13, 17, 23, 0.5);
        border: 1px solid rgba(48, 54, 61, 0.3);
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        text-align: center;
    }
    .breakdown-label {
        font-size: 0.6rem;
        font-weight: 600;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .breakdown-value {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .breakdown-weight {
        font-size: 0.55rem;
        color: #484f58;
        margin-top: 1px;
    }

    /* ── Explanation ── */
    .explanation-box {
        background: rgba(13, 17, 23, 0.6);
        border: 1px solid rgba(48, 54, 61, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .explanation-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .explanation-text {
        font-size: 0.95rem;
        color: #c9d1d9;
        line-height: 1.65;
    }

    /* ── Skills Tags ── */
    .skills-section {
        margin-top: 0.75rem;
    }
    .skills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 0.5rem;
    }
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .skill-matched {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.3);
        color: #56d364;
    }
    .skill-missing {
        background: rgba(248, 81, 73, 0.1);
        border: 1px solid rgba(248, 81, 73, 0.25);
        color: #f85149;
    }
    .skill-extra {
        background: rgba(56, 139, 253, 0.1);
        border: 1px solid rgba(56, 139, 253, 0.25);
        color: #58a6ff;
    }
    .skills-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    /* ── Rank Badge ── */
    .rank-badge {
        position: absolute;
        top: 1rem;
        right: 1.5rem;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 800;
        color: white;
    }
    .rank-1 {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
    }
    .rank-2 {
        background: linear-gradient(135deg, #94a3b8, #64748b);
    }
    .rank-3 {
        background: linear-gradient(135deg, #b45309, #92400e);
    }
    .rank-other {
        background: rgba(48, 54, 61, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.6);
    }

    /* ── Stats Row ── */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(48, 54, 61, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #e6edf3;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6e7681;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 100%);
        border-right: 1px solid rgba(48, 54, 61, 0.4);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e;
    }

    /* ── Process Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #b48505, #d4af37) !important;
        color: #050505 !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.3) !important;
        background: linear-gradient(135deg, #d4af37, #fef08a) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader {
        border-radius: 12px;
    }
    .stFileUploader > div {
        border-radius: 12px !important;
    }

    /* ── Divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 2rem 0;
    }

    /* ── Processing Animation ── */
    .processing-text {
        text-align: center;
        color: #a78bfa;
        font-size: 1rem;
        font-weight: 500;
    }

    /* ── History Card ── */
    .history-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(48, 54, 61, 0.3);
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .history-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(22, 27, 34, 0.8);
    }
    .history-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #c9d1d9;
        margin: 0;
    }
    .history-meta {
        font-size: 0.65rem;
        color: #6e7681;
        margin: 2px 0 0 0;
    }

    /* ── Mode Badge ── */
    .mode-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 100px;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-left: 0.5rem;
    }
    .mode-full {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .mode-cheap {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 100px;
    }
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
st.markdown("""
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
        ⚡ Resume AI
    </div>
    <div style="color:#8b949e; font-size: 14px;">
        Dashboard &nbsp;&bull;&nbsp; Jobs &nbsp;&bull;&nbsp; Analytics
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
    st.markdown("""
    <div style='text-align: center; color: #484f58; font-size: 0.75rem;'>
        <p>Built with ❤️ for recruiters</p>
        <p>v3.0.0 • Startup UX</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ──────────────────────────────────────────────────────────────
# Only show hero header if we are not displaying results
if not st.session_state.get("results"):
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-badge">AI-Powered Recruitment</div>
        <h1 class="hero-title">Resume Screener</h1>
        <p style="text-align:center; color:#8b949e; margin-top:0.5rem; font-size:1.05rem;">
        Screen 100+ candidates in seconds using AI — with explainable scoring.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Application Body ───────────────────────────────────────────────────

# If we don't have results yet, show the upload UI
if "results" not in st.session_state or not st.session_state["results"]:
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="upload-section">
            <div class="upload-label"><span class="upload-icon">📋</span> Job Description</div>
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
        <div class="upload-section">
            <div class="upload-label"><span class="upload-icon">📄</span> Candidate Resumes</div>
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
                
                from database import get_total_usage, increment_user_usage
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
                    
                    # ── Skills extraction (DeepSeek V3.2 — premium/transition) ──
                    skills = None
                    s_score = 50.0
                    if enable_skills and tier in ("premium", "transition"):
                        skills = extract_skills_analysis(job_text, resume_text)
                        s_score = compute_skill_score(skills)
                    else:
                        skills = {"matched_skills": [], "missing_skills": [], "extra_skills": []}
                    
                    # ── LLM confidence score (Gemma 31B — premium only) ──
                    if tier == "premium":
                        l_score = get_llm_score(job_text, resume_text)
                    else:
                        l_score = 50.0
                    
                    final_score = compute_hybrid_score(emb_score, s_score, l_score)
                    
                    # ── AI Explanation (Gemma 31B — premium only) ──
                    if tier == "premium":
                        explanation = generate_explanation(job_text, resume_text, final_score)
                    else:
                        explanation = generate_cheap_explanation(final_score, skills)
                    
                    results.append({
                        "name": name,
                        "filename": filename,
                        "score": final_score,
                        "embedding_score": emb_score,
                        "skill_score": s_score,
                        "llm_score": l_score,
                        "explanation": explanation,
                        "skills": skills,
                        "tier": tier
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
            
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
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
    
    # ── Job Analytics ──
    high = len([r for r in all_results if r["score"] >= 75])
    mid = len([r for r in all_results if 50 <= r["score"] < 75])
    low = len([r for r in all_results if r["score"] < 50])
    avg = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0

    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">{len(all_results)}</div>
            <div class="stat-label">Total Candidates</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#4ade80;">{high}</div>
            <div class="stat-label">Strong Matches</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#f59e0b;">{mid}</div>
            <div class="stat-label">Consider</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#f87171;">{low}</div>
            <div class="stat-label">Reject</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison View (Top 3) ──
    st.markdown("<h3 style='color:#e6edf3; font-size:1.2rem; margin-top:20px;'>🏆 Top Candidates Comparison</h3>", unsafe_allow_html=True)
    top3 = all_results[:3]
    comp_cols = st.columns(3)
    for idx, r in enumerate(top3):
        color = get_score_color(r["score"])
        with comp_cols[idx]:
            # Extracted skill counts
            matched_count = 0
            if r.get("skills"):
                matched_count = len(r["skills"].get("matched_skills", []))
                
            st.markdown(f"""
            <div style="
            background: linear-gradient(135deg, rgba(22,27,34,0.8), rgba(13,17,23,0.9));
            border: 1px solid {color}55;
            border-top: 3px solid {color};
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            ">
                <div style="font-size:24px; font-weight:800; color:{color};">{r['score']}</div>
                <div style="font-weight:700; color:#e6edf3; margin-top:5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{r['name']}</div>
                <div style="font-size:12px; color:#8b949e; margin-top:5px;">Semantic: <span style="color:#c9d1d9;">{r.get('embedding_score', 0):.0f}</span> | Skills: <span style="color:#c9d1d9;">{matched_count} matches</span></div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<hr style='border-color: rgba(99,102,241,0.2); margin: 25px 0;'>", unsafe_allow_html=True)

    # ── Quick Filters & Export Toolbar ──
    tool_col1, tool_col2, tool_col3 = st.columns([2, 1.5, 1])
    
    with tool_col1:
        min_score = st.slider("⚙️ Quick Filter: Minimum Score", 0, 100, 0, help="Drag to filter out lower-scoring candidates")
        
    with tool_col2:
        st.markdown("<div style='margin-top:33px;'></div>", unsafe_allow_html=True)
        show_top = st.checkbox("🔥 Show Top Picks (≥70)")

    with tool_col3:
        df = pd.DataFrame(all_results)
        if 'skills' in df.columns:
            df = df.drop(columns=['skills'])
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "📥 Export to CSV",
            df.to_csv(index=False),
            "resume_results.csv",
            use_container_width=True
        )

    # Apply filter
    filtered_results = [r for r in all_results if r["score"] >= min_score]
    if show_top:
        filtered_results = [r for r in filtered_results if r["score"] >= 70]
    
    st.markdown(f"<p style='color:#8b949e; font-size:14px;'>Showing {len(filtered_results)} of {len(all_results)} candidates</p>", unsafe_allow_html=True)
    
    # ── Candidate Cards ──
    for rank, r in enumerate(filtered_results, 1):
        # We need to maintain absolute rank, not filtered rank
        absolute_rank = all_results.index(r) + 1
        score = r["score"]
        color = get_score_color(score)
        label = get_score_label(score)
        initials = get_initials(r["name"])
        
        emb_s = r.get("embedding_score", score)
        skl_s = r.get("skill_score", 50)
        llm_s = r.get("llm_score", 50)
        
        rank_class = f"rank-{absolute_rank}" if absolute_rank <= 3 else "rank-other"
        
        breakdown_html = f"""
        <div class="breakdown-container">
            <div class="breakdown-item">
                <div class="breakdown-label">Semantic</div>
                <div class="breakdown-value" style="color: #818cf8;">{emb_s:.0f}</div>
                <div class="breakdown-weight">50% weight</div>
            </div>
            <div class="breakdown-item">
                <div class="breakdown-label">Skills</div>
                <div class="breakdown-value" style="color: #4ade80;">{skl_s:.0f}</div>
                <div class="breakdown-weight">30% weight</div>
            </div>
            <div class="breakdown-item">
                <div class="breakdown-label">AI Judge</div>
                <div class="breakdown-value" style="color: #f59e0b;">{llm_s:.0f}</div>
                <div class="breakdown-weight">20% weight</div>
            </div>
        </div>
        """
        
        skills_html = ""
        skills_data = r.get("skills")
        if skills_data and isinstance(skills_data, dict):
            matched = skills_data.get("matched_skills", [])
            missing = skills_data.get("missing_skills", [])
            extra = skills_data.get("extra_skills", [])
            
            if matched or missing or extra:
                skills_html = '<div class="skills-section">'
                if matched:
                    skills_html += '<div class="skills-label">✅ Matched Skills</div><div class="skills-row">'
                    skills_html += ''.join(f'<span class="skill-tag skill-matched">{s}</span>' for s in matched)
                    skills_html += '</div>'
                if missing:
                    skills_html += '<div class="skills-label">❌ Missing Skills</div><div class="skills-row">'
                    skills_html += ''.join(f'<span class="skill-tag skill-missing">{s}</span>' for s in missing)
                    skills_html += '</div>'
                if extra:
                    skills_html += '<div class="skills-label">➕ Additional Skills</div><div class="skills-row">'
                    skills_html += ''.join(f'<span class="skill-tag skill-extra">{s}</span>' for s in extra)
                    skills_html += '</div>'
                skills_html += '</div>'
        
        expl_title = "AI Assessment" if score >= 40 else "⚠️ Why Not Selected"
        
        st.markdown(f"""
        <div class="result-card" style="border-left: 3px solid {color};">
            <div style="position: relative;">
                <div class="rank-badge {rank_class}">#{absolute_rank}</div>
            </div>
            <div class="candidate-header">
                <div class="candidate-avatar" style="background: linear-gradient(135deg, {color}88, {color}44);">
                    {initials}
                </div>
                <div>
                    <p class="candidate-name">{r['name']}</p>
                    <p class="candidate-file">{r['filename']}</p>
                </div>
            </div>
            <div class="score-container">
                <div>
                    <span class="score-number" style="color: {color};">{score}</span>
                    <span class="score-out-of">/100</span>
                </div>
                <div class="score-label-text">{label}</div>
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: {score}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
            </div>
            {breakdown_html}
            <div class="explanation-box">
                <div class="explanation-title">{expl_title}</div>
                <div class="explanation-text">{r['explanation']}</div>
            </div>
            {skills_html}
        </div>
        """, unsafe_allow_html=True)
        
        # ── Shortlist Toggle ──
        rec_col1, rec_col2 = st.columns([1, 4])
        with rec_col1:
            shortlisted = st.checkbox(
                f"⭐ Shortlist",
                value=r.get("shortlisted", False),
                key=f"short_{r.get('id', absolute_rank)}"
            )
            if shortlisted != r.get("shortlisted", False):
                if "id" in r:
                    from database import toggle_shortlist
                    toggle_shortlist(r["id"], shortlisted)
                r["shortlisted"] = shortlisted
    
    # ── Footer Actions ──
    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            del st.session_state["results"]
            if "job_name" in st.session_state:
                del st.session_state["job_name"]
            st.rerun()
