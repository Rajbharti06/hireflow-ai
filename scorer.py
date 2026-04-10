"""
scorer.py — Hybrid Resume Scoring Engine
==========================================
Computes match scores using a HYBRID approach, not just naive cosine similarity.

Three scoring signals are combined:
  1. Embedding similarity (50%) — semantic meaning match
  2. Skill overlap score (30%) — hard skill/keyword coverage
  3. LLM confidence score (20%) — AI judgment call

final_score = 0.50 * embedding_score + 0.30 * skill_score + 0.20 * llm_score

Score interpretation:
  90-100: Exceptional match — candidate is almost tailor-made
  75-89:  Strong match — most key requirements met
  60-74:  Moderate match — some relevant experience
  40-59:  Weak match — limited overlap
  0-39:   Poor match — likely not suitable

Note: Raw cosine similarity for text embeddings typically falls in 0.3-0.9 range.
We apply a scaling function to spread scores across the 0-100 range for better
differentiation between candidates.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ─── Weights for hybrid scoring ──────────────────────────────────────────────
# Skill overlap is the most reliable differentiator for same-domain candidates.
# Embedding similarity (all-MiniLM-L6-v2) clusters in 0.45-0.55 for any
# same-domain pair, providing little differentiation — so we weight it lower.
W_EMBEDDING = 0.30
W_SKILLS    = 0.55
W_LLM       = 0.15


def compute_embedding_score(job_embedding: np.ndarray, resume_embedding: np.ndarray) -> float:
    """
    Compute a match score (0-100) based purely on embedding cosine similarity.
    
    This is the "meaning" signal — captures semantic alignment between
    job description and resume at the conceptual level.
    
    Args:
        job_embedding: Embedding vector for the job description
        resume_embedding: Embedding vector for the resume
    
    Returns:
        Float score between 0 and 100
    """
    raw_score = cosine_similarity(
        job_embedding.reshape(1, -1),
        resume_embedding.reshape(1, -1)
    )[0][0]
    
    return round(scale_score(raw_score), 1)


def compute_skill_score(skills_data: dict) -> float:
    """
    Compute a score (0-100) based on skill overlap between JD and resume.
    
    Formula: matched / (matched + missing) * 100
    Bonus: +5 for each extra relevant skill (capped at +15)
    
    This is the "checklist" signal — did the candidate tick the boxes?
    
    Args:
        skills_data: Dict with 'matched_skills', 'missing_skills', 'extra_skills'
    
    Returns:
        Float score between 0 and 100
    """
    if not skills_data or not isinstance(skills_data, dict):
        return 50.0  # Neutral fallback when skills extraction fails
    
    matched = len(skills_data.get("matched_skills", []))
    missing = len(skills_data.get("missing_skills", []))
    extra = len(skills_data.get("extra_skills", []))
    
    total_required = matched + missing
    
    if total_required == 0:
        return 50.0  # No skills to compare
    
    # Base score: what percentage of required skills does the candidate have?
    base = (matched / total_required) * 100.0
    
    # Bonus for extra relevant skills (shows breadth), capped at 15 points
    bonus = min(extra * 5, 15)
    
    # Don't exceed 100
    return min(round(base + bonus, 1), 100.0)


def compute_hybrid_score(
    embedding_score: float,
    skill_score: float,
    llm_score: float,
    w_embedding: float | None = None,
    w_skills: float | None = None,
    w_llm: float | None = None,
) -> float:
    """
    Compute the final hybrid score from all three signals.

    Custom weights can be passed to override the defaults (must sum to 1.0).
    If any custom weight is None, all three fall back to the module defaults.

    Returns:
        Weighted final score (0-100)
    """
    we = W_EMBEDDING if w_embedding is None else w_embedding
    ws = W_SKILLS    if w_skills   is None else w_skills
    wl = W_LLM       if w_llm     is None else w_llm
    final = we * embedding_score + ws * skill_score + wl * llm_score
    return round(final, 1)


def get_score_breakdown(embedding_score: float, skill_score: float, llm_score: float) -> dict:
    """
    Return a breakdown of how each signal contributed to the final score.
    Useful for transparency in the UI.
    
    Args:
        embedding_score: Semantic similarity score (0-100)
        skill_score: Skill overlap score (0-100)
        llm_score: LLM confidence score (0-100)
    
    Returns:
        Dict with contribution values and percentages
    """
    final = compute_hybrid_score(embedding_score, skill_score, llm_score)
    
    return {
        "final": final,
        "embedding": {
            "raw": embedding_score,
            "weight": W_EMBEDDING,
            "contribution": round(W_EMBEDDING * embedding_score, 1)
        },
        "skills": {
            "raw": skill_score,
            "weight": W_SKILLS,
            "contribution": round(W_SKILLS * skill_score, 1)
        },
        "llm": {
            "raw": llm_score,
            "weight": W_LLM,
            "contribution": round(W_LLM * llm_score, 1)
        }
    }


# ─── Keyword Overlap Score (free, no API) ────────────────────────────────────

_STOP_WORDS = {
    # common English function words
    'the', 'and', 'for', 'with', 'this', 'that', 'will', 'are', 'have',
    'has', 'can', 'must', 'should', 'would', 'could', 'been', 'being',
    'our', 'their', 'your', 'you', 'any', 'all', 'not', 'from', 'such',
    'they', 'them', 'its', 'was', 'but', 'were', 'when', 'how', 'what',
    'who', 'which', 'than', 'then', 'also', 'more', 'some', 'over',
    'into', 'other', 'each', 'both', 'very', 'just', 'about', 'well',
    'work', 'role', 'team', 'join', 'help', 'make', 'use', 'used',
    'using', 'strong', 'good', 'new', 'able', 'great', 'high', 'key',
    # ultra-generic job-description filler words that add no signal
    'skill', 'skills', 'experience', 'requirement', 'requirements',
    'responsible', 'responsibility', 'responsibilities', 'required',
    'preferred', 'desired', 'ability', 'must', 'include', 'including',
    'ensure', 'support', 'provide', 'develop', 'maintain', 'manage',
    'knowledge', 'understanding', 'working', 'relevant', 'related',
    'years', 'degree', 'equivalent', 'position', 'candidate', 'company',
    'business', 'department', 'process', 'processes', 'system', 'systems',
    'solution', 'solutions', 'service', 'services', 'implement',
    'implementation', 'application', 'applications', 'environment',
    'environments', 'across', 'within', 'between', 'against', 'through',
    'perform', 'performing', 'performed', 'create', 'creating', 'report',
    'reporting', 'reports', 'analyze', 'analysis', 'review', 'reviews',
    'identify', 'identified', 'build', 'building', 'design', 'designing',
    'senior', 'junior', 'lead', 'level', 'minimum', 'least', 'plus',
}


def compute_keyword_score(job_text: str, resume_text: str) -> float:
    """
    Measure how many MEANINGFUL (5+ char, non-generic) JD terms appear in
    the resume.  Calibrated so that a genuine domain match reaches 70-90
    without needing 40% raw coverage (which is impossible for real JDs).

    Calibration:
      20% coverage of important JD terms → 80   (good match)
      10% coverage                        → 40   (moderate)
       5% coverage                        → 20   (weak)

    Args:
        job_text: Job description text
        resume_text: Candidate resume text

    Returns:
        Float score 0-100
    """
    def _keywords(text: str) -> set[str]:
        # Only keep tokens ≥5 chars to cut generic short words
        tokens = re.findall(r'\b[a-z][a-z+#./]{4,}\b', text.lower())
        return {t for t in tokens if t not in _STOP_WORDS}

    job_kw = _keywords(job_text)
    if not job_kw:
        return 50.0

    resume_kw = _keywords(resume_text)
    overlap = len(job_kw & resume_kw)
    coverage = overlap / len(job_kw)

    # Scale: 20% coverage → 100 (real JDs rarely exceed 25% overlap even for ideal matches)
    scaled = min(coverage / 0.20, 1.0) * 100.0
    return round(scaled, 1)


# ─── Backward compatibility ─────────────────────────────────────────────────
# Keep the old API working for simple use cases
def compute_score(job_embedding: np.ndarray, resume_embedding: np.ndarray) -> float:
    """Legacy API — returns just the embedding-based score."""
    return compute_embedding_score(job_embedding, resume_embedding)


# ─── Score Scaling ───────────────────────────────────────────────────────────

def scale_score(raw_score: float) -> float:
    """
    Scale raw cosine similarity to a more meaningful 0-100 range.

    Calibrated for sentence-transformers all-MiniLM / NV-Embed style models
    where realistic cross-domain similarity sits in 0.35-0.80.

    Mapping:
      ≤ 0.30  → 0    (completely unrelated)
        0.45  → ~38  (weak overlap)
        0.55  → ~63  (moderate, same domain)
        0.65  → ~88  (strong, direct match)
      ≥ 0.76  → 100  (near-perfect)

    Args:
        raw_score: Raw cosine similarity value (0-1)

    Returns:
        Scaled score 0-100
    """
    min_sim = 0.30  # Floor — truly unrelated documents
    max_sim = 0.76  # Ceiling — achievable with same-domain text

    if raw_score <= min_sim:
        return 0.0
    elif raw_score >= max_sim:
        return 100.0
    else:
        return round(((raw_score - min_sim) / (max_sim - min_sim)) * 100.0, 1)


# ─── Labels & Colors ────────────────────────────────────────────────────────

def get_score_label(score: float) -> str:
    """
    Return a human-readable label for a score.
    
    Args:
        score: Match score (0-100)
    
    Returns:
        Label string like "Strong Match"
    """
    if score >= 90:
        return "🟢 Exceptional Match"
    elif score >= 75:
        return "🟢 Strong Match"
    elif score >= 60:
        return "🟡 Moderate Match"
    elif score >= 40:
        return "🟠 Weak Match"
    else:
        return "🔴 Poor Match"


def get_confidence_level(score: float) -> tuple[str, str]:
    """
    Return a (label, hex_color) confidence level for a match score.

    High   ≥ 80 — strong evidence of fit, act on it
    Medium 60-79 — promising but worth probing gaps
    Low    < 60  — limited fit, treat score with caution
    """
    if score >= 80:
        return "High", "#22c55e"
    elif score >= 60:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#ef4444"


def get_score_color(score: float) -> str:
    """
    Return a hex color for the score tier (for UI rendering).
    
    Args:
        score: Match score (0-100)
    
    Returns:
        Hex color string
    """
    if score >= 90:
        return "#00E676"  # Bright green
    elif score >= 75:
        return "#4CAF50"  # Green
    elif score >= 60:
        return "#FFC107"  # Amber
    elif score >= 40:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red
