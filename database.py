from supabase_client import supabase
import streamlit as st
from datetime import datetime
import json


def get_user_id():
    user = st.session_state.get("user")
    return user.id if user else None


# ─── Sessions (Jobs) ─────────────────────────────────────

def create_session(job_title: str):
    user_id = get_user_id()

    res = supabase.table("jobs").insert({
        "user_id": user_id,
        "title": job_title,
        "description": ""
    }).execute()

    return res.data[0]["id"]


def save_job(job_id: str, text: str, filename: str):
    supabase.table("jobs").update({
        "description": text
    }).eq("id", job_id).execute()


def get_sessions(limit=10):
    user_id = get_user_id()

    res = supabase.table("jobs") \\
        .select("*") \\
        .eq("user_id", user_id) \\
        .order("created_at", desc=True) \\
        .limit(limit) \\
        .execute()

    return res.data


# ─── Results ─────────────────────────────────────────────

def save_result(
    job_id,
    candidate_name,
    filename,
    score,
    explanation,
    skills_data=None,
    embedding_score=None,
    skill_score=None,
    llm_score=None,
    rank=None
):
    user_id = get_user_id()
    
    # Optional serialization if skills_data is present
    skills_json = json.dumps(skills_data) if skills_data else None

    supabase.table("results").insert({
        "user_id": user_id,
        "job_id": job_id,
        "resume_id": None,
        "score": score,
        "explanation": explanation,
        "candidate_name": candidate_name,
        "filename": filename,
        "embedding_score": embedding_score,
        "skill_score": skill_score,
        "llm_score": llm_score,
        "rank": rank,
        "skills_json": skills_json,
        "shortlisted": False
    }).execute()


def get_results_for_session(job_id):
    user_id = get_user_id()

    res = supabase.table("results") \
        .select("*") \
        .eq("job_id", job_id) \
        .eq("user_id", user_id) \
        .order("rank", desc=False) \
        .execute()

    for r in res.data:
        if r.get("skills_json"):
            # Supabase returns the dict directly since it's JSONB, but if it's string wrapped, load it
            if isinstance(r["skills_json"], str):
                r["skills"] = json.loads(r["skills_json"])
            else:
                r["skills"] = r["skills_json"]

    return res.data


def toggle_shortlist(result_id, status: bool):
    supabase.table("results").update({
        "shortlisted": status
    }).eq("id", result_id).execute()


def delete_session(job_id):
    supabase.table("results").delete().eq("job_id", job_id).execute()
    supabase.table("jobs").delete().eq("id", job_id).execute()
