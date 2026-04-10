from supabase_client import supabase
import streamlit as st
from datetime import datetime
import json


def get_user_id():
    user = st.session_state.get("user")
    return user.id if user else None


def get_user_profile():
    if not supabase:
        return None
    user_id = get_user_id()
    if not user_id:
        return None
    res = supabase.table("profiles").select("*").eq("id", user_id).execute()
    return res.data[0] if res.data else None


def get_total_usage():
    if not supabase:
        return 0
    profile = get_user_profile()
    if profile and "lifetime_usage" in profile:
        return profile["lifetime_usage"] or 0
    return 0

def increment_user_usage(amount=1):
    if not supabase:
        return
    user_id = get_user_id()
    if not user_id:
        return
    try:
        supabase.rpc("increment_usage", {"uid": user_id, "amount": amount}).execute()
    except Exception as e:
        print(f"Failed to increment usage: {e}")

# ─── Sessions (Jobs) ─────────────────────────────────────

def create_session(job_title: str):
    if not supabase:
        return None
    user_id = get_user_id()
    if not user_id:
        return None

    res = supabase.table("jobs").insert({
        "user_id": user_id,
        "title": job_title,
        "description": ""
    }).execute()

    return res.data[0]["id"]


def save_job(job_id: str, text: str, filename: str):
    if not supabase or not job_id:
        return
    supabase.table("jobs").update({
        "description": text
    }).eq("id", job_id).execute()


def get_sessions(limit=10):
    if not supabase:
        return []
    user_id = get_user_id()
    if not user_id:
        return []

    res = supabase.table("jobs") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
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
    if not supabase or not job_id:
        return
    user_id = get_user_id()
    if not user_id:
        return
    
    # Supabase handles JSON conversion automatically, just pass the dict directly
    skills_json = skills_data

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
    if not supabase or not job_id:
        return []
    user_id = get_user_id()
    if not user_id:
        return []

    res = supabase.table("results") \
        .select("*") \
        .eq("job_id", job_id) \
        .eq("user_id", user_id) \
        .order("rank", desc=False) \
        .execute()

    for r in res.data:
        # Normalise candidate_name → name so app code can use r["name"] everywhere
        if "candidate_name" in r and "name" not in r:
            r["name"] = r["candidate_name"]
        # Deserialise skills_json JSONB → skills dict
        if r.get("skills_json"):
            if isinstance(r["skills_json"], str):
                try:
                    r["skills"] = json.loads(r["skills_json"])
                except Exception:
                    r["skills"] = None
            else:
                r["skills"] = r["skills_json"]

    return res.data


def toggle_shortlist(result_id, status: bool):
    if not supabase or not result_id:
        return
    supabase.table("results").update({
        "shortlisted": status
    }).eq("id", result_id).execute()


def delete_session(job_id):
    if not supabase or not job_id:
        return
    supabase.table("results").delete().eq("job_id", job_id).execute()
    supabase.table("jobs").delete().eq("id", job_id).execute()
