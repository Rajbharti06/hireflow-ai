"""
app.py — HireFlow AI v4.0
============================
AI-powered resume screener for recruiters.

Pipeline: PDF → Text → Batch Embed → Skills Extract → LLM Score → Hybrid Blend → Rank → Display

Features: multi-AI backends · blind mode · pipeline board · JD scan · radar analytics
Run: streamlit run app.py
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from parser import extract_text_from_pdf
from embedder import get_embedding, get_embeddings_batch
from scorer import (
    compute_embedding_score, compute_skill_score,
    compute_hybrid_score, get_score_breakdown,
    get_score_label, get_score_color, compute_keyword_score,
    get_confidence_level
)
from skills_local import (
    compare_skills_local, extract_years_experience, detect_education_level,
    compute_resume_quality_score, extract_skills_local,
)
from interview_gen import generate_interview_questions, format_questions_markdown, _detect_job_type
from explainer import (
    generate_explanation, extract_skills_analysis,
    get_llm_score, generate_cheap_explanation, sanitize_explanation, AI_BACKEND,
    get_ollama_models, get_backend_status, set_ollama_model, reset_backend_failures,
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
if "candidate_stages" not in st.session_state:
    st.session_state["candidate_stages"] = {}   # key: candidate filename → stage label
if "candidate_notes" not in st.session_state:
    st.session_state["candidate_notes"] = {}    # key: candidate filename → note text
if "blind_mode" not in st.session_state:
    st.session_state["blind_mode"] = False
if "score_weights" not in st.session_state:
    st.session_state["score_weights"] = (0.30, 0.55, 0.15)  # (embedding, skills, llm) — matches scorer.py defaults
if "_last_active_backend" not in st.session_state:
    st.session_state["_last_active_backend"] = ""
if "_ollama_model" not in st.session_state:
    st.session_state["_ollama_model"] = ""

# ─── Restore Supabase session on every re-run ────────────────────────────────
# The supabase client is a module-level singleton. Streamlit re-runs the script
# on every interaction, so we re-apply the user's JWT at the top of every run.
def _apply_supabase_session(access_token: str, refresh_token: str) -> str:
    """
    Apply a user's JWT to the shared supabase client so every subsequent
    PostgREST request carries it and Supabase RLS passes.

    Two-layer write:
      1. supabase.options.headers  — used when _postgrest is rebuilt from scratch
      2. supabase.postgrest.auth() — updates the currently cached postgrest client

    Returns the (possibly refreshed) access_token so callers can store it.
    """
    current_token = access_token
    try:
        res = supabase.auth.set_session(access_token, refresh_token)
        # set_session may silently refresh an expiring token
        if res and getattr(res, "session", None) and res.session.access_token:
            current_token = res.session.access_token
    except Exception:
        pass  # token invalid/expired — we still apply whatever we have below

    bearer = f"Bearer {current_token}"
    # Layer 1: options headers (used on next _postgrest rebuild)
    supabase.options.headers["Authorization"] = bearer
    # Layer 2: live postgrest client (supabase.postgrest auto-builds if None)
    supabase.postgrest.auth(current_token)

    return current_token


if supabase and "user" in st.session_state:
    _session_restored = False

    if st.session_state.get("auth_access_token"):
        try:
            _new_tok = _apply_supabase_session(
                st.session_state["auth_access_token"],
                st.session_state.get("auth_refresh_token", ""),
            )
            # Persist refreshed token so the next run uses it
            st.session_state["auth_access_token"] = _new_tok
            _session_restored = True
        except Exception:
            pass

    if not _session_restored:
        # Fallback: recover from supabase's in-memory state (surviving restarts
        # only when the Streamlit worker process wasn't recycled)
        try:
            _existing = supabase.auth.get_session()
            if _existing and getattr(_existing, "access_token", None):
                _tok = _apply_supabase_session(
                    _existing.access_token, _existing.refresh_token
                )
                st.session_state["auth_access_token"] = _tok
                st.session_state["auth_refresh_token"] = _existing.refresh_token
                _session_restored = True
        except Exception:
            pass

    if not _session_restored:
        # Stale session: user object present but no valid tokens available.
        # Clear it so the auth guard below re-shows the login form instead of
        # silently failing with RLS errors on every DB write.
        for _k in ("user", "results", "auth_access_token", "auth_refresh_token"):
            st.session_state.pop(_k, None)


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireFlow AI",
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

    /* ── Pipeline stage badge ── */
    .stage-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 100px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.3px;
    }
    .stage-screening  { background: rgba(110,118,129,0.12); color:#8b949e; border:1px solid rgba(110,118,129,0.25); }
    .stage-shortlist  { background: rgba(212,175,55,0.12);  color:#d4af37; border:1px solid rgba(212,175,55,0.3); }
    .stage-phone      { background: rgba(99,102,241,0.12);  color:#818cf8; border:1px solid rgba(99,102,241,0.25); }
    .stage-technical  { background: rgba(56,139,253,0.12);  color:#60a5fa; border:1px solid rgba(56,139,253,0.25); }
    .stage-offer      { background: rgba(34,197,94,0.12);   color:#4ade80; border:1px solid rgba(34,197,94,0.25); }
    .stage-rejected   { background: rgba(248,81,73,0.08);   color:#f87171; border:1px solid rgba(248,81,73,0.2); }

    /* ── JD scan panel ── */
    .jd-panel {
        background: linear-gradient(135deg, rgba(22,27,34,0.7), rgba(13,17,23,0.8));
        border: 1px solid rgba(99,102,241,0.2);
        border-left: 3px solid rgba(99,102,241,0.5);
        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
    }
    .jd-panel-title { font-size: 0.65rem; font-weight: 700; color: #484f58; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.5rem; }
    .jd-stat { display: inline-block; background: rgba(13,17,23,0.6); border: 1px solid rgba(48,54,61,0.3); border-radius: 8px; padding: 4px 12px; font-size: 0.78rem; color: #c9d1d9; margin: 3px; }
    .jd-stat b { color: #e6edf3; }

    /* ── Notes textarea override ── */
    .stTextArea textarea { background: rgba(13,17,23,0.6) !important; border-color: rgba(48,54,61,0.4) !important; color: #c9d1d9 !important; font-size: 0.85rem !important; border-radius: 10px !important; }

    /* ── Blind mode ── */
    .blind-badge { display:inline-block; background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3); border-radius:100px; padding:2px 10px; font-size:0.65rem; font-weight:700; color:#818cf8; letter-spacing:0.5px; }
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
            _tok = _apply_supabase_session(query_params["access_token"], query_params["refresh_token"])
            st.session_state.user = res.user
            st.session_state["auth_access_token"] = _tok
            st.session_state["auth_refresh_token"] = query_params["refresh_token"]
            if hasattr(st, "experimental_set_query_params"):
                st.experimental_set_query_params()
            else:
                st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"OAuth Authentication Failed: {str(e)}")


# ─── Auth Guard ───────────────────────────────────────────────────────────────
if supabase is not None and "user" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; margin-top:3rem; margin-bottom:1.5rem;'>
        <p style='font-size:2rem; margin:0;'>⚡</p>
        <h2 style='color:white; margin:0.25rem 0 0.25rem;'>HireFlow AI</h2>
        <p style='color:#6e7681; font-size:0.9rem;'>Sign in to your account or create a new one</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        _auth_tab_signin, _auth_tab_signup = st.tabs(["Sign In", "Sign Up"])

    with col2:
        with _auth_tab_signin:
            email = st.text_input("Email", key="si_email")
            password = st.text_input("Password", type="password", key="si_pass")
            if st.button("Sign In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please enter your email and password.")
                else:
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        _tok = _apply_supabase_session(res.session.access_token, res.session.refresh_token)
                        st.session_state.user = res.user
                        st.session_state["auth_access_token"] = _tok
                        st.session_state["auth_refresh_token"] = res.session.refresh_token
                        st.rerun()
                    except Exception as e:
                        msg = str(e)
                        if "Email not confirmed" in msg:
                            st.error("Please confirm your email address first.")
                        elif "Invalid login credentials" in msg:
                            st.error("Incorrect email or password.")
                        else:
                            st.error(f"Sign in failed: {msg}")

        with _auth_tab_signup:
            su_email = st.text_input("Email", key="su_email")
            su_pass = st.text_input("Password (min 8 chars)", type="password", key="su_pass")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not su_email or not su_pass:
                    st.error("Please enter your email and a password.")
                elif len(su_pass) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    try:
                        supabase.auth.sign_up({"email": su_email, "password": su_pass})
                        st.success("✅ Account created! Switch to **Sign In** tab and log in.")
                    except Exception as e:
                        msg = str(e)
                        if "already registered" in msg.lower() or "already been registered" in msg.lower():
                            st.error("Email already registered. Please sign in instead.")
                        else:
                            st.error(f"Sign up failed: {msg}")

        st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1); margin: 1.5rem 0;'/>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.85rem;'>Or continue with</p>", unsafe_allow_html=True)
    with col2:
        # Single Sign-On Buttons
        _redirect = "http%3A%2F%2Flocalhost%3A8501"
        google_url = f"{supabase.supabase_url}/auth/v1/authorize?provider=google&redirect_to={_redirect}"
        github_url = f"{supabase.supabase_url}/auth/v1/authorize?provider=github&redirect_to={_redirect}"

        st.markdown(f"""
        <div style="display: flex; gap: 10px; justify-content: center; margin-bottom: 12px;">
            <a href="{google_url}" target="_self" style="flex: 1; display:flex; align-items:center; justify-content:center; padding:10px; border-radius:8px; border:1px solid #30363d; color:white; background:#24292e; text-decoration:none; font-size:0.9rem;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" width="16" style="margin-right:8px;">Google
            </a>
            <a href="{github_url}" target="_self" style="flex: 1; display:flex; align-items:center; justify-content:center; padding:10px; border-radius:8px; border:1px solid #30363d; color:white; background:#24292e; text-decoration:none; font-size:0.9rem;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="16" style="filter:invert(1); margin-right:8px;">GitHub
            </a>
        </div>
        <p style='text-align:center; font-size:11px; color:#6e7681;'>OAuth requires Google/GitHub enabled in your <a href='https://supabase.com/dashboard' target='_blank' style='color:#d4af37;'>Supabase Auth settings</a>.</p>
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
    st.markdown("### ⚙️ Scoring Weights")
    st.caption("Must sum to 100%")
    w_emb_pct = st.slider("🧠 Semantic", 10, 80, 30, step=5, help="Embedding cosine similarity")
    w_skl_pct = st.slider("🔍 Skills", 5, 75, 55, step=5, help="Skill keyword overlap")
    w_llm_pct = 100 - w_emb_pct - w_skl_pct
    w_llm_pct = max(5, w_llm_pct)
    st.caption(f"⌨️ Keywords (auto): **{w_llm_pct}%**")
    # Normalise so they always sum to 1.0
    _total = w_emb_pct + w_skl_pct + w_llm_pct
    st.session_state["score_weights"] = (w_emb_pct/_total, w_skl_pct/_total, w_llm_pct/_total)

    st.markdown("---")

    # Blind mode + skill analysis toggles
    st.markdown("### 🎛️ Display Options")
    st.session_state["blind_mode"] = st.toggle(
        "🙈 Blind Mode", value=st.session_state["blind_mode"],
        help="Hide candidate names for bias-free screening"
    )
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

    # ── AI Backend Status & Fallback ──
    st.markdown("### 🌐 AI Backend")
    _bs = get_backend_status()
    _primary_label = _bs["primary"].upper()

    # Toast when backend switches (detects change across reruns)
    _current_active = _bs["active"]
    if _current_active and _current_active != st.session_state["_last_active_backend"]:
        if st.session_state["_last_active_backend"]:  # only toast on actual change, not first run
            if _current_active == "ollama":
                st.toast("🦙 API credits exhausted — now using Ollama locally", icon="⚠️")
            else:
                st.toast(f"✅ Switched to {_current_active.title()}", icon="🔄")
        st.session_state["_last_active_backend"] = _current_active

    # Exhausted backends — show warnings
    for _eb in _bs["exhausted"]:
        st.warning(f"⚠️ **{_eb.title()}** credits exhausted", icon="💳")

    # Active backend badge
    if _bs["using_fallback"]:
        st.info(f"🔄 Fallback active → **{_bs['active'].title()}**")
    else:
        st.success(f"✅ **{_primary_label}** connected")

    # Ollama model selector — shown when Ollama is primary OR is the active fallback
    _show_ollama = (
        _bs["primary"] == "ollama"
        or _bs["using_ollama_fallback"]
        or bool(_bs["exhausted"])   # pre-configure while credits are running out
    )
    if _show_ollama:
        st.markdown("**🦙 Ollama Model**")
        _ollama_models = _bs["ollama_models"]
        if _ollama_models:
            # Pre-select previously chosen model if still available
            _prev = st.session_state.get("_ollama_model", "")
            _default_idx = _ollama_models.index(_prev) if _prev in _ollama_models else 0
            _chosen_model = st.selectbox(
                "Select model",
                _ollama_models,
                index=_default_idx,
                key="ollama_model_select",
                label_visibility="collapsed",
            )
            if _chosen_model != st.session_state["_ollama_model"]:
                st.session_state["_ollama_model"] = _chosen_model
                set_ollama_model(_chosen_model)
                st.toast(f"🦙 Ollama model set to {_chosen_model}", icon="✅")
            else:
                set_ollama_model(_chosen_model)  # keep module state in sync on every rerun
        else:
            st.error("Ollama not running", icon="🔴")
            st.caption("Start it with `ollama serve`, then refresh this page.")
            st.markdown(
                "[📦 Get Ollama](https://ollama.com)",
                unsafe_allow_html=False,
            )

    # Reset button — lets user retry after re-entering API keys
    if _bs["exhausted"]:
        if st.button("🔄 Retry API connections", use_container_width=True):
            reset_backend_failures()
            st.session_state["_last_active_backend"] = ""
            st.toast("Retrying all backends…", icon="🔄")
            st.rerun()

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
                        # Sanitize any stale error strings saved by older code versions
                        for _r in old_results:
                            _r["explanation"] = sanitize_explanation(
                                _r.get("explanation", ""),
                                score=_r.get("score", 50),
                                skills_data=_r.get("skills"),
                                experience_years=_r.get("experience_years", 0),
                            )
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
            for _k in ("user", "results", "auth_access_token", "auth_refresh_token"):
                st.session_state.pop(_k, None)
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

    # ── JD Quick Scan ────────────────────────────────────────────────────────
    if job_file:
        _jd_cache_key = f"jd_scan_{job_file.name}_{job_file.size}"
        if _jd_cache_key not in st.session_state:
            try:
                _jd_preview_text = extract_text_from_pdf(job_file)
                job_file.seek(0)
                _jd_skills_req = extract_skills_local(_jd_preview_text)
                _jd_exp_req = extract_years_experience(_jd_preview_text)
                _, _jd_edu_req = detect_education_level(_jd_preview_text)
                _jd_type = _detect_job_type(_jd_preview_text).replace("_", " ").title()
                _word_count = len(_jd_preview_text.split())
                st.session_state[_jd_cache_key] = {
                    "skills": _jd_skills_req,
                    "exp": _jd_exp_req,
                    "edu": _jd_edu_req,
                    "type": _jd_type,
                    "words": _word_count,
                }
            except Exception:
                st.session_state[_jd_cache_key] = None

        _scan = st.session_state.get(_jd_cache_key)
        if _scan:
            _jd_skills_preview = ", ".join(_scan["skills"][:8]) + (f" +{len(_scan['skills'])-8} more" if len(_scan["skills"]) > 8 else "") if _scan["skills"] else "None detected"
            _exp_str = f"{_scan['exp']}+ yrs required" if _scan["exp"] else "Not specified"
            _edu_str = _scan["edu"] if _scan["edu"] != "Not specified" else "Not specified"
            st.markdown(f"""
            <div class="jd-panel">
                <div class="jd-panel-title">📋 JD Quick Scan — {job_file.name}</div>
                <span class="jd-stat">🏷️ Role type: <b>{_scan["type"]}</b></span>
                <span class="jd-stat">📝 <b>{_scan["words"]}</b> words</span>
                <span class="jd-stat">🔧 Skills required: <b>{len(_scan["skills"])}</b></span>
                <span class="jd-stat">💼 Experience: <b>{_exp_str}</b></span>
                <span class="jd-stat">🎓 Education: <b>{_edu_str}</b></span>
                <div style="margin-top:8px;font-size:0.75rem;color:#6e7681;">
                    Key skills detected: <span style="color:#c9d1d9;">{_jd_skills_preview}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if job_file and resume_files:
        MAX_BATCH = 20
        if len(resume_files) > MAX_BATCH:
            st.error(f"⚠️ Maximum {MAX_BATCH} resumes allowed per batch to prevent API overload.")
            st.stop()
            
        mode_label = "⚡ Quick Analyze" if cheap_mode else "🚀 Analyze Candidates"
        
        # ── Duplicate detection ──
        _seen_fnames, _deduped, _dupes = set(), [], []
        for _rf in resume_files:
            _norm = _rf.name.strip().lower()
            if _norm in _seen_fnames:
                _dupes.append(_rf.name)
            else:
                _seen_fnames.add(_norm)
                _deduped.append(_rf)
        if _dupes:
            st.warning(f"⚠️ Duplicate files removed: {', '.join(_dupes)}")
            resume_files = _deduped

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
            if not session_id:
                st.toast("Results won't be saved to history (DB unavailable)", icon="ℹ️")
            save_job(session_id, job_text, job_file.name)
            
            current_usage = total_usage  # snapshot once; incremented locally

            for i, (resume_text, resume_emb, name, filename) in enumerate(
                zip(resume_texts, resume_embeddings, resume_names, resume_filenames)
            ):
                progress = (i + 1) / len(resume_texts)

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

                    _we, _ws, _wl = st.session_state.get("score_weights", (0.30, 0.55, 0.15))
                    final_score = compute_hybrid_score(emb_score, s_score, l_score, _we, _ws, _wl)

                    # ── Explanation ──
                    if tier == "premium":
                        explanation = generate_explanation(
                            job_text, resume_text, final_score,
                            skills_data=skills, experience_years=exp_years
                        )
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
                    current_usage += 1

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
                    "skills": None,
                    "experience_years": 0,
                    "education": "Not specified",
                    "quality_score": 0,
                    "tier": "error",
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
                skills_payload = r.get("skills") or {}
                skills_payload["_meta"] = {
                    "experience_years": r.get("experience_years", 0),
                    "education": r.get("education", "Not specified"),
                    "quality_score": r.get("quality_score", 0),
                    "tier": r.get("tier", "local"),
                }
                save_result(
                    job_id=session_id,
                    candidate_name=r["name"],
                    filename=r["filename"],
                    score=r["score"],
                    explanation=r["explanation"],
                    skills_data=skills_payload,
                    embedding_score=r.get("embedding_score"),
                    skill_score=r.get("skill_score"),
                    llm_score=r.get("llm_score"),
                    rank=rank
                )

            st.session_state["results"] = results
            st.session_state["view_mode"] = "new"
            st.session_state["job_name"] = job_name
            st.toast("⚡ Top candidates identified!")
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
        _blind_ov = st.session_state.get("blind_mode", False)
        for idx, r in enumerate(top3):
            color = get_score_color(r["score"])
            matched_count = len(r["skills"].get("matched_skills", [])) if r.get("skills") else 0
            exp_str  = f"💼 {r.get('experience_years',0)}yr  " if r.get("experience_years") else ""
            edu_str  = f"🎓 {r.get('education','')}" if r.get("education") and r["education"] != "Not specified" else ""
            q_score  = r.get("quality_score", 0)
            _ov_name = f"Candidate #{idx+1}" if _blind_ov else r['name']
            _ov_stage = st.session_state["candidate_stages"].get(r.get("filename", ""), "⚪ Screening")
            _ov_conf_label, _ov_conf_color = get_confidence_level(r["score"])
            with comp_cols[idx]:
                st.markdown(f"""
                <div class="top-card" style="border:1px solid {color}44; border-top:3px solid {color};">
                    <div style="font-size:1.4rem;margin-bottom:4px;">{medals[idx]}</div>
                    <div style="font-size:2.2rem;font-weight:900;color:{color};letter-spacing:-2px;">{r['score']}</div>
                    <div style="font-size:0.72rem;color:#8b949e;margin-bottom:4px;">/100</div>
                    <div style="font-size:0.68rem;font-weight:700;color:{_ov_conf_color};
                        background:{_ov_conf_color}18;border:1px solid {_ov_conf_color}44;
                        border-radius:6px;padding:2px 8px;display:inline-block;margin-bottom:6px;">
                        🎯 {_ov_conf_label} Confidence
                    </div>
                    <div style="font-weight:700;color:#e6edf3;font-size:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_ov_name}</div>
                    <div style="font-size:0.72rem;color:#6e7681;margin-top:6px;">✅ {matched_count} skills matched</div>
                    <div style="font-size:0.7rem;color:#6e7681;margin-top:3px;">{exp_str}{edu_str}</div>
                    <div style="font-size:0.7rem;color:#484f58;margin-top:3px;">📄 Quality: {q_score}/100</div>
                    <div style="margin-top:6px;">{_ov_stage}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-divider' style='margin:1.5rem 0;'></div>", unsafe_allow_html=True)

        # ── Pipeline Board ──
        st.markdown("<h3 style='color:#e6edf3;font-size:1.1rem;margin:0 0 0.75rem 0;'>🗂️ Hiring Pipeline</h3>", unsafe_allow_html=True)
        _pipeline_stages = [
            ("⭐ Shortlisted",         "rgba(212,175,55,0.15)",  "#d4af37"),
            ("📞 Phone Screen",         "rgba(99,102,241,0.15)",  "#818cf8"),
            ("🔧 Technical Interview",  "rgba(56,139,253,0.15)",  "#60a5fa"),
            ("💼 Offer Extended",       "rgba(34,197,94,0.15)",   "#4ade80"),
            ("❌ Rejected",             "rgba(248,81,73,0.08)",   "#f87171"),
        ]
        _stages_dict = st.session_state["candidate_stages"]
        _any_in_pipeline = any(
            v != "⚪ Screening" for v in _stages_dict.values()
        )
        if _any_in_pipeline:
            for _stg_label, _stg_bg, _stg_color in _pipeline_stages:
                _stg_candidates = [
                    r for r in all_results
                    if _stages_dict.get(r.get("filename", ""), "⚪ Screening") == _stg_label
                ]
                if not _stg_candidates:
                    continue
                st.markdown(f"<div style='font-size:0.72rem;font-weight:700;color:#6e7681;text-transform:uppercase;letter-spacing:1px;margin:0.75rem 0 0.4rem 0;'>{_stg_label} &nbsp;·&nbsp; {len(_stg_candidates)}</div>", unsafe_allow_html=True)
                for r in _stg_candidates:
                    _p_color = get_score_color(r["score"])
                    _p_name = f"Candidate #{all_results.index(r)+1}" if _blind_ov else r["name"]
                    _p_note = st.session_state["candidate_notes"].get(r.get("filename",""), "")
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;padding:8px 14px;
                        background:{_stg_bg};border:1px solid {_stg_color}33;
                        border-left:3px solid {_stg_color};border-radius:10px;margin-bottom:5px;">
                        <div style="font-size:1.2rem;font-weight:900;color:{_p_color};min-width:36px;">{r['score']}</div>
                        <div style="flex:1;">
                            <div style="font-weight:700;color:#e6edf3;font-size:0.88rem;">{_p_name}</div>
                            {f'<div style="font-size:0.68rem;color:#6e7681;margin-top:2px;">📝 {_p_note}</div>' if _p_note else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#484f58;font-size:0.85rem;'>Set pipeline stages on the Candidates tab to track progress here.</p>", unsafe_allow_html=True)

        # Export toolbar
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            _stages_map = st.session_state.get("candidate_stages", {})
            _notes_map  = st.session_state.get("candidate_notes", {})
            df_export = pd.DataFrame([
                {
                    **{k: v for k, v in r.items() if k not in ("skills", "resume_text")},
                    "pipeline_stage": _stages_map.get(r.get("filename", ""), "⚪ Screening"),
                    "recruiter_notes": _notes_map.get(r.get("filename", ""), ""),
                    "matched_skills": ", ".join(r["skills"].get("matched_skills", [])) if r.get("skills") else "",
                    "missing_skills": ", ".join(r["skills"].get("missing_skills", [])) if r.get("skills") else "",
                }
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
            placeholder="🔎  Search by name, skill, note, or keyword  (e.g. 'Python' or 'Sarah')",
            label_visibility="collapsed",
            key="candidate_search"
        )
        fc1, fc2, fc3 = st.columns([3, 1, 1])
        with fc1:
            min_score = st.slider("Minimum score", 0, 100, 0, key="min_score_slider",
                                  help="Drag right to hide weak candidates")
        with fc2:
            sort_by = st.selectbox(
                "Sort by", ["Score", "Name", "Experience", "Quality"],
                key="sort_by_select", label_visibility="collapsed"
            )
        with fc3:
            st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
            show_top = st.checkbox("🔥 Top Picks only (≥70)", key="show_top_cb")

        # Apply filters
        filtered_results = list(all_results)
        if search_query:
            q_lower = search_query.lower()
            filtered_results = [
                r for r in filtered_results
                if q_lower in r["name"].lower()
                or (r.get("skills") and any(
                    q_lower in s.lower()
                    for s in r["skills"].get("matched_skills", []) + r["skills"].get("extra_skills", [])
                ))
                or q_lower in (r.get("explanation") or "").lower()
                or q_lower in st.session_state.get("candidate_notes", {}).get(r["filename"], "").lower()
            ]
        filtered_results = [r for r in filtered_results if r["score"] >= min_score]
        if show_top:
            filtered_results = [r for r in filtered_results if r["score"] >= 70]

        # Apply sort
        _sort_key = {
            "Score":      lambda x: x["score"],
            "Name":       lambda x: x["name"].lower(),
            "Experience": lambda x: x.get("experience_years", 0),
            "Quality":    lambda x: x.get("quality_score", 0),
        }.get(sort_by, lambda x: x["score"])
        _sort_rev = sort_by != "Name"
        filtered_results = sorted(filtered_results, key=_sort_key, reverse=_sort_rev)

        st.markdown(f"<p style='color:#6e7681;font-size:0.82rem;margin-bottom:0.75rem;'>Showing {len(filtered_results)} of {len(all_results)} candidates · sorted by {sort_by}</p>", unsafe_allow_html=True)

        # ── Candidate Cards ──
        for rank, r in enumerate(filtered_results, 1):
            absolute_rank = all_results.index(r) + 1
            score = r["score"]
            color = get_score_color(score)
            label = get_score_label(score)
            _blind = st.session_state.get("blind_mode", False)
            display_name = f"Candidate #{absolute_rank}" if _blind else r["name"]
            display_file = "●●●●●●.pdf" if _blind else r["filename"]
            initials = f"C{absolute_rank}" if _blind else get_initials(r["name"])

            emb_s = r.get("embedding_score", score)
            skl_s = r.get("skill_score", 50)
            llm_s = r.get("llm_score", 50)
            exp_years = r.get("experience_years", 0)
            edu_label = r.get("education", "")
            tier_used = r.get("tier", "local")

            quality_score = r.get("quality_score", 0)
            confidence_label, confidence_color = get_confidence_level(score)

            # ── Meta badges ──
            meta_parts = []
            if exp_years:
                meta_parts.append(f'<span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:3px 10px;font-size:0.72rem;color:#818cf8;font-weight:600;">💼 {exp_years}yr exp</span>')
            if edu_label and edu_label != "Not specified":
                meta_parts.append(f'<span style="background:rgba(74,222,128,0.1);border:1px solid rgba(74,222,128,0.25);border-radius:20px;padding:3px 10px;font-size:0.72rem;color:#4ade80;font-weight:600;">🎓 {edu_label}</span>')
            if quality_score:
                q_color = "#22c55e" if quality_score >= 70 else "#f59e0b" if quality_score >= 45 else "#ef4444"
                meta_parts.append(f'<span style="background:rgba(34,197,94,0.08);border:1px solid {q_color}44;border-radius:20px;padding:3px 10px;font-size:0.72rem;color:{q_color};font-weight:600;">📄 Resume {quality_score}/100</span>')
            # Confidence level badge
            meta_parts.append(f'<span style="background:{confidence_color}18;border:1px solid {confidence_color}55;border-radius:20px;padding:3px 10px;font-size:0.72rem;color:{confidence_color};font-weight:700;">🎯 {confidence_label} Confidence</span>')
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

            # ── "Why Not Selected" block for weak/poor candidates ──
            why_not_html = ""
            if score < 60 and skills_data:
                _wn_missing = skills_data.get("missing_skills", [])
                _wn_extra   = skills_data.get("extra_skills",   [])
                _wn_reasons = []
                if _wn_missing:
                    _wn_reasons.append(f"Missing key skills: <b>{', '.join(_wn_missing[:5])}</b>.")
                if emb_s < 50:
                    _wn_reasons.append("Low semantic alignment with the job description.")
                if skl_s < 40:
                    _wn_reasons.append("Skill coverage below the required threshold.")
                if not _wn_reasons:
                    _wn_reasons.append("Overall profile does not align closely with the role requirements.")
                _wn_items = "".join(f'<li style="margin-bottom:4px;">{r_}</li>' for r_ in _wn_reasons)
                why_not_html = f"""
                <div style="margin:10px 0;padding:12px 16px;
                    background:rgba(248,81,73,0.06);
                    border:1px solid rgba(248,81,73,0.25);
                    border-left:3px solid #f85149;
                    border-radius:10px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#f85149;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                        ❌ Why Not Selected
                    </div>
                    <ul style="margin:0;padding-left:18px;font-size:0.8rem;color:#c9d1d9;line-height:1.6;">
                        {_wn_items}
                    </ul>
                </div>"""

            expl_title = "AI Assessment"
            rank_class = f"rank-{absolute_rank}" if absolute_rank <= 3 else "rank-other"

            st.markdown(f"""
            <div class="result-card" style="border-left:3px solid {color};">
                <div style="position:relative;">
                    <div class="rank-badge {rank_class}">#{absolute_rank}</div>
                </div>
                <div class="candidate-header">
                    <div class="candidate-avatar" style="background:linear-gradient(135deg,{color}88,{color}44);">{initials}</div>
                    <div style="flex:1;">
                        <p class="candidate-name">{display_name}{"&nbsp;<span class='blind-badge'>BLIND</span>" if _blind else ""}</p>
                        <p class="candidate-file">{display_file}</p>
                    </div>
                </div>
                {meta_html}
                <div class="score-container">
                    <div>
                        <span class="score-number" style="color:{color};">{score}</span>
                        <span class="score-out-of">/100</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div class="score-label-text">{label}</div>
                        <div style="font-size:0.72rem;font-weight:700;color:{confidence_color};
                            background:{confidence_color}15;border:1px solid {confidence_color}44;
                            border-radius:6px;padding:2px 8px;">
                            {confidence_label} Confidence
                        </div>
                    </div>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{score}%;background:linear-gradient(90deg,{color},{color}66);"></div>
                </div>
                {breakdown_html}
                {why_not_html}
                <div class="explanation-box">
                    <div class="explanation-title">{expl_title}</div>
                    <div class="explanation-text">{r['explanation']}</div>
                </div>
                {skills_html}
            </div>
            """, unsafe_allow_html=True)

            # ── Pipeline Stage Selector ──
            _stage_options = [
                "⚪ Screening", "⭐ Shortlisted", "📞 Phone Screen",
                "🔧 Technical Interview", "💼 Offer Extended", "❌ Rejected"
            ]
            _stage_key = r.get("filename", str(absolute_rank))
            _current_stage = st.session_state["candidate_stages"].get(_stage_key, "⚪ Screening")
            if _current_stage not in _stage_options:
                _current_stage = "⚪ Screening"

            _notes_key = _stage_key
            sc1, sc2 = st.columns([2, 3])
            with sc1:
                _new_stage = st.selectbox(
                    "Pipeline stage",
                    _stage_options,
                    index=_stage_options.index(_current_stage),
                    key=f"stage_{absolute_rank}",
                    label_visibility="collapsed",
                )
                if _new_stage != _current_stage:
                    st.session_state["candidate_stages"][_stage_key] = _new_stage
                    # Sync shortlisted flag for DB
                    _is_shortlisted = _new_stage in ("⭐ Shortlisted", "📞 Phone Screen", "🔧 Technical Interview", "💼 Offer Extended")
                    if "id" in r:
                        toggle_shortlist(r["id"], _is_shortlisted)
                    r["shortlisted"] = _is_shortlisted
            with sc2:
                _note_val = st.session_state["candidate_notes"].get(_notes_key, "")
                _new_note = st.text_input(
                    "Recruiter note",
                    value=_note_val,
                    placeholder="Add a note for this candidate...",
                    key=f"note_{absolute_rank}",
                    label_visibility="collapsed",
                )
                if _new_note != _note_val:
                    st.session_state["candidate_notes"][_notes_key] = _new_note

            # ── Interview questions expander ──
            with st.expander(f"🎤 Interview Questions — {display_name}", expanded=False):
                _iq_cache_key = f"iq_{r['filename']}"
                if _iq_cache_key not in st.session_state:
                    st.session_state[_iq_cache_key] = generate_interview_questions(
                        job_name, r.get("skills") or {}, r["score"],
                        r["name"], r.get("experience_years", 0),
                    )
                interview_qs = st.session_state[_iq_cache_key]
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

            # ── Radar chart: top-5 candidate profiles ──
            top5 = all_results[:min(5, len(all_results))]
            if len(top5) >= 2:
                _blind_an = st.session_state.get("blind_mode", False)
                _radar_categories = ["Semantic", "Skills", "Keywords", "Resume Quality", "Experience"]
                _max_exp = max((r.get("experience_years", 0) for r in all_results), default=1) or 1
                fig_radar = go.Figure()
                _radar_colors = [
                    ("#d4af37", "rgba(212,175,55,0.1)"),
                    ("#818cf8", "rgba(129,140,248,0.1)"),
                    ("#4ade80", "rgba(74,222,128,0.1)"),
                    ("#f59e0b", "rgba(245,158,11,0.1)"),
                    ("#60a5fa", "rgba(96,165,250,0.1)"),
                ]
                for _ri, _rc in enumerate(top5):
                    _rc_name = f"Candidate #{_ri+1}" if _blind_an else _rc["name"].split()[0]
                    _rc_vals = [
                        _rc.get("embedding_score", 0),
                        _rc.get("skill_score", 0),
                        _rc.get("llm_score", 0),
                        _rc.get("quality_score", 0),
                        min(_rc.get("experience_years", 0) / _max_exp * 100, 100),
                    ]
                    _line_c, _fill_c = _radar_colors[_ri % len(_radar_colors)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=_rc_vals + [_rc_vals[0]],
                        theta=_radar_categories + [_radar_categories[0]],
                        fill="toself",
                        name=_rc_name,
                        line=dict(color=_line_c, width=2),
                        fillcolor=_fill_c,
                        opacity=0.9,
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(13,17,23,0)",
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color="#484f58"), gridcolor="rgba(48,54,61,0.3)", linecolor="rgba(48,54,61,0.3)"),
                        angularaxis=dict(tickfont=dict(size=11, color="#c9d1d9"), gridcolor="rgba(48,54,61,0.25)", linecolor="rgba(48,54,61,0.3)"),
                    ),
                    title=dict(text=f"Candidate Profile Radar (Top {len(top5)})", font=dict(color="#e6edf3", size=14)),
                    paper_bgcolor="rgba(13,17,23,0)",
                    font=dict(color="#6e7681", size=11),
                    legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
                    margin=dict(l=40, r=40, t=60, b=60), height=380,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

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
