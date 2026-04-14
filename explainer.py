"""
explainer.py — Multi-AI Engine
==========================================================
Generates plain English summaries explaining why a candidate is or isn't
a good fit for a role. 

Supported Backends:
  - OpenAI (gpt-4o-mini)
  - Claude (claude-3-haiku-20240307)
  - Gemini (gemini-pro)
  - Perplexity (llama-3-sonar-small-32k-chat)
  - Grok (grok-beta)
  - NVIDIA NIM (meta/llama3-70b-instruct)
  - Ollama (local)

Includes:
  - LLM confidence scoring (0-100) as a hybrid scoring signal
  - Skills extraction with matched/missing/extra
  - "Why rejected" explanation for low-score candidates
  - Cheap mode: skip LLM calls to save cost on large batches
"""

import html as _html_mod
import os
import re
import json
import hashlib
import requests
from openai import OpenAI


# ─── Configuration ───────────────────────────────────────────────────────────
AI_BACKEND = os.environ.get("AI_BACKEND", "openai")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")


# ─── LLM Response Cache ─────────────────────────────────────────────────────
_llm_cache: dict[str, str] = {}

# ─── Sentinels ───────────────────────────────────────────────────────────────
# _API_BLOCKED    — content filter / bad request (try next backend)
# _CREDITS_EXHAUSTED — 401/402/403/429 (mark backend as dead, try next)
_API_BLOCKED       = "__API_BLOCKED__"
_CREDITS_EXHAUSTED = "__CREDITS_EXHAUSTED__"

# ─── Session-scoped backend state ────────────────────────────────────────────
# These persist for the lifetime of the Streamlit server process.
# In practice that means one user session (Streamlit reruns don't re-import).
_exhausted_backends: set[str] = set()   # backends that returned 401/402/403/429
_active_backend:     str      = ""      # last backend that returned a clean response

# Priority order for automatic fallback (first with credentials wins)
_FALLBACK_ORDER = ["nvidia", "openai", "claude", "gemini", "perplexity", "grok", "ollama"]


def _cache_key(job_desc: str, resume_text: str, prompt_type: str) -> str:
    content = f"{prompt_type}:{job_desc[:2000]}:{resume_text[:2000]}"
    return hashlib.md5(content.encode()).hexdigest()


def _is_api_error(text: str) -> bool:
    """
    Return True if the text is an error sentinel, raw API error payload,
    content-filter message, or HTML markup — anything that must NOT be
    shown to users or stored as a candidate explanation.
    """
    if not isinstance(text, str) or not text.strip():
        return True
    if text in (_API_BLOCKED, _CREDITS_EXHAUSTED) or text.startswith("⚠️"):
        return True
    s = text.lstrip()
    # Raw API error JSON — Anthropic, OpenAI, NVIDIA formats
    if (s.startswith('{"type":"error"') or s.startswith('{"error":')
            or s.startswith("{'type': 'error'")):
        return True
    # Anthropic / generic "Output blocked" content-filter message that may
    # have slipped through as a string rather than an HTTP 400 status.
    tl = s.lower()
    if ("output blocked" in tl or "content filtering" in tl
            or "content_filter" in tl or "api error:" in tl
            or "request_id" in tl):
        return True
    # HTML markup — some models return structured HTML instead of plain text
    if s.startswith("<div") or s.startswith("<html") or s.startswith("<p>"):
        return True
    return False


def _strip_html(text: str) -> str:
    """Remove HTML/XML tags and unescape HTML entities, leaving plain text."""
    # Remove all tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    # Unescape entities like &amp; &lt; etc.
    return _html_mod.unescape(clean)


def _sanitize_text(text: str, max_chars: int = 2000) -> str:
    """
    Strip non-ASCII characters and truncate to max_chars.
    Reduces the chance of triggering content filters on API calls.
    """
    cleaned = text.encode("ascii", errors="ignore").decode("ascii")
    # Collapse excessive whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_chars]


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_explanation(job_desc: str, resume_text: str, score: float,
                         skills_data: dict | None = None,
                         experience_years: int = 0) -> str:
    cache_k = _cache_key(job_desc, resume_text, "explanation")
    if cache_k in _llm_cache:
        cached = _llm_cache[cache_k]
        if not _is_api_error(cached):
            return cached

    if score < 40:
        prompt = _build_rejection_prompt(job_desc, resume_text, score)
    else:
        prompt = _build_explanation_prompt(job_desc, resume_text, score)

    result = _route_call(prompt)

    # If the API was blocked or errored, fall back to the free local explanation
    if _is_api_error(result):
        result = generate_cheap_explanation(score, skills_data or {}, experience_years)
    else:
        # Strip any accidental HTML markup that some models return (e.g. Gemma)
        result = _strip_html(result)

    _llm_cache[cache_k] = result
    return result


def get_llm_score(job_desc: str, resume_text: str) -> float:
    cache_k = _cache_key(job_desc, resume_text, "llm_score")
    if cache_k in _llm_cache:
        cached = _llm_cache[cache_k]
        if _is_api_error(cached):
            return 50.0
        try:
            return float(cached)
        except (ValueError, TypeError):
            return 50.0

    prompt = _build_scoring_prompt(job_desc, resume_text)
    result = _route_call(prompt)
    _llm_cache[cache_k] = result

    # Immediately return neutral if API was blocked — don't regex-parse error messages
    if _is_api_error(result):
        return 50.0

    try:
        score = float(result.strip())
        return max(0.0, min(100.0, score))
    except (ValueError, TypeError):
        # Only scan for numbers in genuine (non-error) responses
        numbers = re.findall(r'\b(\d{1,3})\b', result)
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(100.0, score))
        return 50.0


def extract_skills_analysis(job_desc: str, resume_text: str) -> dict:
    cache_k = _cache_key(job_desc, resume_text, "skills")
    if cache_k in _llm_cache:
        cached = _llm_cache[cache_k]
        if not _is_api_error(cached):
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                pass

    prompt = _build_skills_prompt(job_desc, resume_text)

    # Route skills extraction through DeepSeek V3.2 for better JSON output
    backend = os.environ.get("AI_BACKEND", AI_BACKEND).lower()
    if backend == "nvidia":
        response = _call_nvidia_deepseek(prompt)
    else:
        response = _route_call(prompt)

    # If API was blocked, signal caller to use local fallback immediately
    if _is_api_error(response):
        return {"matched_skills": [], "missing_skills": [], "extra_skills": []}

    parsed = _parse_skills_json(response)
    _llm_cache[cache_k] = json.dumps(parsed)
    return parsed


def generate_cheap_explanation(score: float, skills_data: dict, experience_years: int = 0) -> str:
    """
    Generate a structured plain-English explanation without any LLM call.
    Uses skill overlap data and experience years extracted locally.

    Args:
        score: Hybrid match score (0-100)
        skills_data: Dict with matched/missing/extra skill lists
        experience_years: Years of experience extracted from resume (0 = unknown)
    """
    matched = skills_data.get("matched_skills", []) if skills_data else []
    missing = skills_data.get("missing_skills", []) if skills_data else []
    extra   = skills_data.get("extra_skills",   []) if skills_data else []

    parts = []

    # Opening verdict
    if score >= 80:
        parts.append("Excellent alignment with the role requirements.")
    elif score >= 65:
        parts.append("Good alignment with most key requirements.")
    elif score >= 50:
        parts.append("Moderate alignment — some relevant experience present.")
    elif score >= 35:
        parts.append("Limited alignment with the core requirements.")
    else:
        parts.append("Significant mismatch with the role profile.")

    # Experience context
    if experience_years >= 1:
        if score >= 65:
            parts.append(f"Brings {experience_years}+ year{'s' if experience_years != 1 else ''} of relevant experience.")
        else:
            parts.append(f"Has {experience_years}+ year{'s' if experience_years != 1 else ''} of experience, though skill overlap is limited.")

    # Skills detail
    if matched:
        parts.append(f"Matched skills: {', '.join(matched[:5])}.")
    if missing:
        parts.append(f"Missing from profile: {', '.join(missing[:4])}.")
    if extra and score >= 50:
        parts.append(f"Additional strengths: {', '.join(extra[:3])}.")

    # Closing recommendation
    if score >= 70:
        parts.append("Recommended for interview.")
    elif score >= 50:
        parts.append("Worth considering if the pipeline is limited.")
    else:
        parts.append("Not a strong fit for this role.")

    return " ".join(parts)


def sanitize_explanation(
    text: str,
    score: float = 50,
    skills_data: dict | None = None,
    experience_years: int = 0,
) -> str:
    """
    Return a clean explanation safe to display in the UI.

    Replaces API error strings, raw JSON error payloads, empty strings, and
    the _API_BLOCKED sentinel with a locally-generated cheap explanation.
    Call this whenever loading saved explanations from the database so stale
    error strings from older code versions are never shown to users.
    """
    if _is_api_error(text):
        return generate_cheap_explanation(score, skills_data or {}, experience_years)
    return text


# ─── Backend Status & Ollama Utilities ───────────────────────────────────────

def get_ollama_models() -> list[str]:
    """
    Fetch the list of locally available Ollama models.
    Returns an empty list if Ollama is not running or not reachable.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def get_backend_status() -> dict:
    """
    Return current backend status for sidebar display.

    Keys:
      primary             — configured AI_BACKEND
      active              — last backend that returned a clean response
      exhausted           — list of backends that failed with credits errors
      using_fallback      — True when active != primary
      using_ollama_fallback — True when fell back to Ollama from an API backend
      ollama_reachable    — True when Ollama is running and has ≥1 model
    """
    primary = os.environ.get("AI_BACKEND", AI_BACKEND).lower()
    active  = _active_backend or primary
    models  = get_ollama_models()
    return {
        "primary":               primary,
        "active":                active,
        "exhausted":             list(_exhausted_backends),
        "using_fallback":        active != primary,
        "using_ollama_fallback": active == "ollama" and primary != "ollama",
        "ollama_reachable":      len(models) > 0,
        "ollama_models":         models,
    }


def set_ollama_model(model_name: str) -> None:
    """Update the Ollama model used for calls (persists for this server process)."""
    global OLLAMA_MODEL
    OLLAMA_MODEL = model_name


def reset_backend_failures() -> None:
    """
    Clear all tracked backend failures so they are retried next call.
    Call this when the user re-enters an API key or wants to retry.
    """
    global _active_backend
    _exhausted_backends.clear()
    _active_backend = ""


# ─── Prompt Templates ────────────────────────────────────────────────────────

def _build_explanation_prompt(job_desc: str, resume_text: str, score: float) -> str:
    jd = _sanitize_text(job_desc, 1800)
    cv = _sanitize_text(resume_text, 1800)
    return f"""You are an expert recruitment analyst. Match score: {score}/100.

JOB DESCRIPTION:
{jd}

CANDIDATE RESUME:
{cv}

Write a concise assessment (under 100 words):
1. Key strengths aligned with the role
2. Notable gaps or concerns
3. Overall recommendation (interview / consider / pass)

Write in plain English paragraphs. Do not repeat the score."""


def _build_rejection_prompt(job_desc: str, resume_text: str, score: float) -> str:
    jd = _sanitize_text(job_desc, 1500)
    cv = _sanitize_text(resume_text, 1500)
    return f"""Recruitment analysis. Score: {score}/100 (poor fit).

JOB:
{jd}

RESUME:
{cv}

In 70 words: state the primary reason this candidate does not meet the requirements, list the key missing skills, and note any salvageable potential."""


def _build_scoring_prompt(job_desc: str, resume_text: str) -> str:
    jd = _sanitize_text(job_desc, 1500)
    cv = _sanitize_text(resume_text, 1500)
    return f"""Rate this candidate's fit for the job on a scale of 0 to 100.
Consider: skill match, experience level, seniority fit.

JOB:
{jd}

RESUME:
{cv}

Reply with a single integer only (0-100). No explanation."""


def _build_skills_prompt(job_desc: str, resume_text: str) -> str:
    jd = _sanitize_text(job_desc, 1500)
    cv = _sanitize_text(resume_text, 1500)
    return f"""Compare the technical skills in the job description versus the resume.

JOB:
{jd}

RESUME:
{cv}

Return ONLY valid JSON with these exact keys (no markdown, no explanation):
{{"matched_skills": ["skill1"], "missing_skills": ["skill2"], "extra_skills": ["skill3"]}}"""


# ─── Internal Routing & Multi-AI Handlers ─────────────────────────────────────

def _has_credentials(backend: str) -> bool:
    """Return True if the backend has an API key configured (or is local)."""
    key_env = {
        "nvidia":     "NVIDIA_API_KEY",
        "openai":     "OPENAI_API_KEY",
        "claude":     "ANTHROPIC_API_KEY",
        "gemini":     "GEMINI_API_KEY",
        "perplexity": "PPLX_API_KEY",
        "grok":       "GROK_API_KEY",
        "ollama":     None,   # local — no key needed
    }
    env_var = key_env.get(backend)
    return True if env_var is None else bool(os.environ.get(env_var, ""))


def _call_backend(backend: str, prompt: str) -> str:
    """Dispatch to the right handler. Returns a response, _API_BLOCKED, or _CREDITS_EXHAUSTED."""
    dispatch = {
        "nvidia":     _call_nvidia,
        "openai":     _call_openai,
        "claude":     _call_claude,
        "gemini":     _call_gemini,
        "perplexity": _call_perplexity,
        "grok":       _call_grok,
        "ollama":     _call_ollama,
    }
    handler = dispatch.get(backend)
    return handler(prompt) if handler else _API_BLOCKED


def _route_call(prompt: str) -> str:
    """
    Route the prompt through an automatic fallback chain.

    Order:
      1. Configured primary backend (AI_BACKEND env var)
      2. Any other backend with credentials set (in _FALLBACK_ORDER priority)
      3. Ollama (local — always last resort before giving up)

    A backend is skipped permanently within this session if it returns
    _CREDITS_EXHAUSTED (401/402/403/429).  Content-filter errors (_API_BLOCKED)
    are retried on the next backend because local Ollama won't filter content.

    Always returns either a clean text string or _API_BLOCKED.
    Never lets _CREDITS_EXHAUSTED or raw error payloads escape to callers.
    """
    global _active_backend
    primary = os.environ.get("AI_BACKEND", AI_BACKEND).lower()

    # Build chain: primary first, then fallbacks that have credentials
    chain: list[str] = [primary]
    for b in _FALLBACK_ORDER:
        if b != primary and _has_credentials(b) and b not in chain:
            chain.append(b)
    # Ollama is always appended as the final local escape hatch
    if "ollama" not in chain:
        chain.append("ollama")

    for backend in chain:
        if backend in _exhausted_backends:
            continue

        result = _call_backend(backend, prompt)

        if result == _CREDITS_EXHAUSTED:
            # Permanently skip this backend for the rest of the session
            _exhausted_backends.add(backend)
            continue   # try next

        if _is_api_error(result):
            # Content filter or transient error — try the next backend.
            # Local Ollama never content-filters, so it acts as the final escape.
            continue

        # Clean response
        _active_backend = backend
        return result

    # Every backend failed or was skipped
    return _API_BLOCKED


def _call_openai(prompt: str) -> str:
    """OpenAI GPT-4o-mini."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return _CREDITS_EXHAUSTED
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # OpenAI SDK raises typed exceptions — check status if available
        status = getattr(e, "status_code", None)
        if status in (401, 402, 403, 429):
            return _CREDITS_EXHAUSTED
        return _API_BLOCKED


def _call_claude(prompt: str) -> str:
    """Anthropic Claude (claude-3-haiku)."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return _CREDITS_EXHAUSTED
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 300,
                "temperature": 0.3,
                "system": "You are a concise, expert recruitment analyst.",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if response.status_code in (401, 402, 403, 429):
            return _CREDITS_EXHAUSTED
        if 400 <= response.status_code < 500:
            return _API_BLOCKED
        if response.status_code >= 500:
            return _API_BLOCKED
        return response.json()["content"][0]["text"].strip()
    except Exception:
        return _API_BLOCKED


def _call_gemini(prompt: str) -> str:
    """Google Gemini 2.0 Flash."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return _CREDITS_EXHAUSTED
        url = (
            "https://generativelanguage.googleapis.com/v1beta"
            f"/models/gemini-2.0-flash:generateContent?key={api_key}"
        )
        response = requests.post(
            url,
            json={"contents": [{"parts": [{"text": "You are a concise, expert recruitment analyst.\n\n" + prompt}]}]},
            timeout=60,
        )
        if response.status_code in (401, 402, 403, 429):
            return _CREDITS_EXHAUSTED
        if 400 <= response.status_code < 500:
            return _API_BLOCKED
        if response.status_code >= 500:
            return _API_BLOCKED
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return _API_BLOCKED


def _call_perplexity(prompt: str) -> str:
    """Perplexity AI (sonar-small-chat)."""
    try:
        api_key = os.environ.get("PPLX_API_KEY", "")
        if not api_key:
            return _CREDITS_EXHAUSTED
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "sonar-small-chat",
                "messages": [
                    {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=60,
        )
        if response.status_code in (401, 402, 403, 429):
            return _CREDITS_EXHAUSTED
        if 400 <= response.status_code < 500:
            return _API_BLOCKED
        if response.status_code >= 500:
            return _API_BLOCKED
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return _API_BLOCKED


def _call_grok(prompt: str) -> str:
    """xAI Grok (grok-beta)."""
    try:
        api_key = os.environ.get("GROK_API_KEY", "")
        if not api_key:
            return _CREDITS_EXHAUSTED
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=60,
        )
        if response.status_code in (401, 402, 403, 429):
            return _CREDITS_EXHAUSTED
        if 400 <= response.status_code < 500:
            return _API_BLOCKED
        if response.status_code >= 500:
            return _API_BLOCKED
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return _API_BLOCKED


def _call_nvidia_model(prompt: str, model: str, api_key: str) -> str:
    """
    Generic NVIDIA NIM chat completion handler.
    Returns _API_BLOCKED on ANY error — content filter, rate limit, bad key,
    or a 200 OK response whose body is an error object (NVIDIA sometimes does this).
    """
    try:
        if not api_key:
            return _CREDITS_EXHAUSTED

        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=60,
        )

        # Auth/quota errors → exhausted; content filter / bad request → blocked
        if response.status_code in (401, 402, 403, 429):
            return _CREDITS_EXHAUSTED
        if 400 <= response.status_code < 500:
            return _API_BLOCKED

        if response.status_code >= 500:
            return _API_BLOCKED

        body = response.json()

        # Guard: some NVIDIA models return 200 OK but with an error body
        # e.g. {"type":"error","error":{"type":"invalid_request_error",...}}
        if isinstance(body, dict) and body.get("type") == "error":
            return _API_BLOCKED

        # Guard: content_filter finish_reason (model stopped mid-output)
        choices = body.get("choices", [])
        if not choices:
            return _API_BLOCKED
        if choices[0].get("finish_reason") == "content_filter":
            return _API_BLOCKED

        content = choices[0].get("message", {}).get("content", "")
        return content.strip() if content else _API_BLOCKED

    except Exception:
        return _API_BLOCKED


def _call_nvidia(prompt: str) -> str:
    """Gemma 4 31B — premium explanations and LLM scoring."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    return _call_nvidia_model(prompt, "google/gemma-4-31b-it", api_key)


def _call_nvidia_deepseek(prompt: str) -> str:
    """DeepSeek V3.2 — structured JSON output and skills extraction."""
    api_key = os.environ.get("NVIDIA_DEEPSEEK_API_KEY", "")
    if not api_key:
        # Fallback to Gemma if no DeepSeek key
        return _call_nvidia(prompt)
    return _call_nvidia_model(prompt, "deepseek-ai/deepseek-v3.2", api_key)


def _call_ollama(prompt: str) -> str:
    """Local Ollama. Returns _API_BLOCKED on any error."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "You are a concise, expert recruitment analyst.\n\n" + prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300},
            },
            timeout=120,
        )
        if 400 <= response.status_code < 500:
            return _API_BLOCKED
        if response.status_code >= 500:
            return _API_BLOCKED
        return response.json().get("response", _API_BLOCKED)
    except Exception:
        return _API_BLOCKED


# ─── JSON Parsing Helper ────────────────────────────────────────────────────

def _parse_skills_json(response: str) -> dict:
    fallback = {"matched_skills": [], "missing_skills": [], "extra_skills": []}
    if not isinstance(response, str):
        return fallback
    try:
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue
        return json.loads(response)
    except json.JSONDecodeError:
        return fallback
