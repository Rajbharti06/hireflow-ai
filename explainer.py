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

import os
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


def _cache_key(job_desc: str, resume_text: str, prompt_type: str) -> str:
    content = f"{prompt_type}:{job_desc[:2000]}:{resume_text[:2000]}"
    return hashlib.md5(content.encode()).hexdigest()


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_explanation(job_desc: str, resume_text: str, score: float) -> str:
    cache_k = _cache_key(job_desc, resume_text, "explanation")
    if cache_k in _llm_cache:
        return _llm_cache[cache_k]
    
    if score < 40:
        prompt = _build_rejection_prompt(job_desc, resume_text, score)
    else:
        prompt = _build_explanation_prompt(job_desc, resume_text, score)
    
    result = _route_call(prompt)
    _llm_cache[cache_k] = result
    return result


def get_llm_score(job_desc: str, resume_text: str) -> float:
    cache_k = _cache_key(job_desc, resume_text, "llm_score")
    if cache_k in _llm_cache:
        try:
            return float(_llm_cache[cache_k])
        except (ValueError, TypeError):
            return 50.0
    
    prompt = _build_scoring_prompt(job_desc, resume_text)
    result = _route_call(prompt)
    _llm_cache[cache_k] = result
    
    try:
        score = float(result.strip())
        return max(0.0, min(100.0, score))
    except (ValueError, TypeError):
        import re
        numbers = re.findall(r'\b(\d{1,3})\b', str(result))
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(100.0, score))
        return 50.0


def extract_skills_analysis(job_desc: str, resume_text: str) -> dict:
    cache_k = _cache_key(job_desc, resume_text, "skills")
    if cache_k in _llm_cache:
        try:
            return json.loads(_llm_cache[cache_k])
        except (json.JSONDecodeError, TypeError):
            pass
    
    prompt = _build_skills_prompt(job_desc, resume_text)
    
    # Route skills extraction through DeepSeek V3.2 for better JSON output
    backend = os.environ.get("AI_BACKEND", AI_BACKEND).lower()
    if backend == "nvidia":
        response = _call_nvidia_deepseek(prompt)
    else:
        response = _route_call(prompt)
    
    parsed = _parse_skills_json(response)
    _llm_cache[cache_k] = json.dumps(parsed)
    return parsed


def generate_cheap_explanation(score: float, skills_data: dict) -> str:
    matched = skills_data.get("matched_skills", []) if skills_data else []
    missing = skills_data.get("missing_skills", []) if skills_data else []
    extra = skills_data.get("extra_skills", []) if skills_data else []
    
    parts = []
    if score >= 75:
        parts.append("Strong alignment with role requirements.")
    elif score >= 60:
        parts.append("Moderate alignment with the position.")
    elif score >= 40:
        parts.append("Limited alignment with core requirements.")
    else:
        parts.append("Significant mismatch with the role.")
    
    if matched:
        parts.append(f"Key matches: {', '.join(matched[:4])}.")
    if missing:
        parts.append(f"Gaps: {', '.join(missing[:4])}.")
    if extra:
        parts.append(f"Additional strengths: {', '.join(extra[:3])}.")
    
    if score >= 70:
        parts.append("Recommend for interview screening.")
    elif score >= 50:
        parts.append("Consider if pipeline is thin.")
    else:
        parts.append("Likely not a fit for this role.")
    
    return " ".join(parts)


# ─── Prompt Templates ────────────────────────────────────────────────────────

def _build_explanation_prompt(job_desc: str, resume_text: str, score: float) -> str:
    return f"""You are an expert recruitment analyst. Compare this job description with the candidate's resume.
The automated match score is {score}/100.

JOB DESCRIPTION:
{job_desc[:3000]}

CANDIDATE RESUME:
{resume_text[:3000]}

Write a concise, plain English assessment (under 120 words) covering:
1. Key strengths that align with the role
2. Notable gaps or concerns
3. Overall recommendation (consider, interview, or pass)

Be direct and specific. Reference actual skills/experience from the resume. Do NOT repeat the score. Do NOT use bullet points. Write in flowing paragraphs."""


def _build_rejection_prompt(job_desc: str, resume_text: str, score: float) -> str:
    return f"""You are an expert recruitment analyst. This candidate scored poorly ({score}/100) against the job requirements.

JOB DESCRIPTION:
{job_desc[:3000]}

CANDIDATE RESUME:
{resume_text[:3000]}

Explain in 80 words or less:
1. The primary reason this candidate is NOT a fit
2. What specific skills or experience they lack
3. Whether there's any salvageable potential

Be respectful but honest."""


def _build_scoring_prompt(job_desc: str, resume_text: str) -> str:
    return f"""You are a senior recruitment analyst. Score how well this candidate matches the job on a scale of 0-100.
Consider skill match, experience, seniority fit, and red flags.

JOB DESCRIPTION:
{job_desc[:3000]}

CANDIDATE RESUME:
{resume_text[:3000]}

Respond with ONLY a single integer between 0 and 100. Nothing else. No explanation."""


def _build_skills_prompt(job_desc: str, resume_text: str) -> str:
    return f"""Analyze the job description and resume below. Extract and compare key technical skills, tools, and qualifications.

JOB DESCRIPTION:
{job_desc[:3000]}

RESUME:
{resume_text[:3000]}

Return ONLY a JSON object (no markdown, no backticks, no explanation) with these exact keys:
{{
    "matched_skills": ["skill1"],
    "missing_skills": ["skill2"],
    "extra_skills": ["skill3"]
}}
"""


# ─── Internal Routing & Multi-AI Handlers ─────────────────────────────────────

def _route_call(prompt: str) -> str:
    backend = os.environ.get("AI_BACKEND", AI_BACKEND).lower()

    if backend == "claude":
        return _call_claude(prompt)
    elif backend == "gemini":
        return _call_gemini(prompt)
    elif backend == "perplexity":
        return _call_perplexity(prompt)
    elif backend == "grok":
        return _call_grok(prompt)
    elif backend == "nvidia":
        return _call_nvidia(prompt)
    elif backend == "ollama":
        return _call_ollama(prompt)
    else:
        return _call_openai(prompt)


def _call_openai(prompt: str) -> str:
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key: return "⚠️ Missing OpenAI API Key"
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI API error: {str(e)}"


def _call_claude(prompt: str) -> str:
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key: return "⚠️ Missing Anthropic API Key"
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 300,
                "temperature": 0.3,
                "system": "You are a concise, expert recruitment analyst.",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Claude error: {str(e)}"


def _call_gemini(prompt: str) -> str:
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key: return "⚠️ Missing Gemini API Key"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        response = requests.post(
            url,
            json={"contents": [{"parts": [{"text": "You are a concise, expert recruitment analyst.\n\n" + prompt}]}]}
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"


def _call_perplexity(prompt: str) -> str:
    try:
        api_key = os.environ.get("PPLX_API_KEY", "")
        if not api_key: return "⚠️ Missing Perplexity API Key"
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "sonar-small-chat",
                "messages": [
                    {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Perplexity error: {str(e)}"


def _call_grok(prompt: str) -> str:
    try:
        api_key = os.environ.get("GROK_API_KEY", "")
        if not api_key: return "⚠️ Missing Grok API Key"
        
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Grok error: {str(e)}"


def _call_nvidia_model(prompt: str, model: str, api_key: str) -> str:
    """Generic NVIDIA NIM chat completion handler for any hosted model."""
    try:
        if not api_key:
            return f"⚠️ Missing NVIDIA API Key for {model}"
        
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a concise, expert recruitment analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ NVIDIA {model} error: {str(e)}"


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
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "You are a concise, expert recruitment analyst.\n\n" + prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"


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
