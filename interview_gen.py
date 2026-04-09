"""
interview_gen.py — Zero-Cost Interview Question Generator
=========================================================
Generates targeted, structured interview questions for each candidate
based on their skill profile, score tier, and detected gaps.

No API calls. Template-driven with skill-specific depth.
Works 100% offline.

Question categories:
  - technical:    Deep-dives on skills the candidate matched
  - gap_probing:  Non-confrontational questions about missing skills
  - behavioral:   STAR-format situational questions
  - culture_fit:  Motivation and working-style questions
  - quick_screen: 5-minute phone screen questions (always generated)
"""

import random
import re


# ─── Skill-Specific Technical Questions ──────────────────────────────────────
# Maps lowercase skill name → list of questions (2-4 per skill)

_SKILL_QUESTIONS: dict[str, list[str]] = {
    "python": [
        "Walk me through how you'd design a memory-efficient pipeline in Python for processing 10M+ records.",
        "How do you manage dependency and environment isolation across multiple Python projects?",
        "Describe a time you debugged a significant performance bottleneck in a Python application. What tools did you use?",
        "What's your approach to writing testable Python code — how do you structure it for unit vs integration tests?",
    ],
    "javascript": [
        "Explain the JavaScript event loop and how it affects async code execution.",
        "How do you handle state management complexity as a JavaScript application scales?",
        "Walk me through your approach to identifying and fixing memory leaks in a browser application.",
    ],
    "typescript": [
        "How do you use TypeScript's type system to enforce business rules rather than just types?",
        "When would you use generics vs union types vs discriminated unions? Give a real example.",
        "Describe a refactor you've done to migrate a JavaScript codebase to TypeScript.",
    ],
    "react": [
        "Explain how React reconciliation works and how it affects your component design decisions.",
        "Walk me through a complex state management problem you've solved — what tradeoffs did you consider?",
        "How do you approach performance optimization in a React app with many deeply nested components?",
        "Describe your testing strategy for React components — unit, integration, and E2E.",
    ],
    "node.js": [
        "How do you handle CPU-intensive tasks in Node.js without blocking the event loop?",
        "Walk me through your approach to error handling and graceful shutdown in a Node.js server.",
        "How do you manage database connection pooling in a high-concurrency Node.js service?",
    ],
    "django": [
        "How do you approach database query optimization in Django? Walk me through a real case.",
        "Explain how Django's middleware system works and give an example of a custom middleware you've written.",
        "How do you handle long-running background tasks in Django — Celery, channels, something else?",
    ],
    "fastapi": [
        "How do you handle authentication and authorization in a FastAPI application?",
        "Walk me through how you'd structure a FastAPI project for a team of 5+ engineers.",
        "How do you approach API versioning and backwards compatibility in FastAPI?",
    ],
    "flask": [
        "How do you structure a Flask application to scale beyond a single file?",
        "What's your approach to request validation and error handling in Flask?",
    ],
    "aws": [
        "Walk me through the last AWS architecture you designed. What were the key tradeoffs?",
        "How do you approach cost optimization in AWS for a production workload?",
        "How do you handle IAM permissions and least-privilege principles across a multi-service AWS deployment?",
    ],
    "docker": [
        "Walk me through how you'd optimize a Dockerfile for a production Python application.",
        "How do you handle secrets and sensitive config in Docker containers?",
        "Describe a Docker networking problem you solved in a multi-service environment.",
    ],
    "kubernetes": [
        "Walk me through how you'd debug a pod that's in CrashLoopBackOff state.",
        "How do you handle rolling deployments and zero-downtime releases in Kubernetes?",
        "Explain your approach to resource requests and limits — what happens when you get it wrong?",
    ],
    "sql": [
        "Walk me through how you'd diagnose and fix a slow query in a production database.",
        "Explain window functions and give a real use case where they solved something otherwise complex.",
        "How do you approach database schema migrations in a live production system?",
    ],
    "postgresql": [
        "Walk me through how you'd tune a slow PostgreSQL query — what tools and techniques do you use?",
        "How do you use PostgreSQL's EXPLAIN ANALYZE output? What are you looking for?",
        "Describe a time you used PostgreSQL's advanced features (JSONB, CTEs, window functions) to solve a real problem.",
    ],
    "mongodb": [
        "How do you design a MongoDB schema when you know queries will be complex? Walk me through your thought process.",
        "Describe a MongoDB performance problem you encountered and how you solved it.",
        "How do you handle data consistency in MongoDB for operations that would normally require transactions?",
    ],
    "machine learning": [
        "Walk me through how you evaluate whether a model is actually ready for production.",
        "How do you handle class imbalance in a classification problem? Give a specific example.",
        "Describe a model that underperformed expectations. What did you learn from debugging it?",
    ],
    "deep learning": [
        "How do you approach debugging a neural network that's not converging?",
        "Walk me through how you'd design a training pipeline for a new deep learning project.",
        "How do you decide between fine-tuning a pre-trained model vs training from scratch?",
    ],
    "nlp": [
        "Walk me through how you'd build a text classification system from scratch vs using a pre-trained model.",
        "How do you handle noisy, unstructured text data in an NLP pipeline?",
        "Describe the tradeoffs between embedding approaches — TF-IDF, word2vec, BERT-based.",
    ],
    "llm": [
        "How do you evaluate whether an LLM is actually solving the problem vs hallucinating?",
        "Walk me through how you've used prompt engineering to improve model output quality.",
        "How do you approach RAG system design — chunking strategy, retrieval, reranking?",
    ],
    "git": [
        "Describe your team's branching strategy. What problems has it solved or created?",
        "Walk me through a complex merge conflict you resolved. What was the context?",
    ],
    "ci/cd": [
        "Walk me through the CI/CD pipeline you've designed or maintained. What made it reliable?",
        "How do you handle flaky tests in a CI environment without just disabling them?",
    ],
    "agile": [
        "Describe how your team runs sprints — what works well and what you'd change.",
        "How do you handle technical debt in an agile environment where velocity is always the priority?",
    ],
}

# ─── Gap Probing Templates ────────────────────────────────────────────────────

_GAP_TEMPLATES: list[str] = [
    "{skill} is central to how we work here. What's your current exposure to it, and how would you approach getting up to speed quickly?",
    "We use {skill} extensively in this role. Have you worked with it at all — even in a personal project or learning context?",
    "One area we'd expect you to grow into is {skill}. How do you typically approach learning a new technology when you're thrown into it?",
    "The team runs on {skill} day-to-day. If you joined us, how would you plan your first 30 days of getting comfortable with it?",
    "You haven't had as much exposure to {skill} — can you tell me about the closest technology you have used and how you see that experience transferring?",
]

# ─── Behavioral Questions (STAR-format) ──────────────────────────────────────

_BEHAVIORAL: list[str] = [
    "Tell me about the most technically complex problem you've solved. Walk me through your process from initial diagnosis to solution.",
    "Describe a time you had to deliver something under significant time or resource pressure. What did you cut and why?",
    "Tell me about a technical disagreement you had with a teammate or manager. How did you handle it?",
    "Describe a project where something you built failed in production. What happened, what did you do, and what did you change after?",
    "Walk me through a time you significantly improved a system's performance, reliability, or maintainability.",
    "Tell me about a time you had to onboard quickly onto an unfamiliar codebase. What was your strategy?",
    "Describe a situation where you had to say no to a feature request or technical decision. How did you handle it?",
    "Tell me about a time you mentored or helped a less experienced colleague. What was your approach?",
]

# ─── Culture / Fit Questions ──────────────────────────────────────────────────

_CULTURE: list[str] = [
    "What does good engineering culture look like to you — and where does this role fit into that picture?",
    "How do you decide when something is 'good enough' vs when it needs more polish?",
    "Walk me through how you stay current with what's happening in your area of tech.",
    "What's a technical skill or concept you've been exploring recently, just because it interested you?",
    "Describe your ideal working arrangement and how you collaborate best with teammates.",
    "What's drawn you to this type of role rather than a different direction you could have taken?",
]

# ─── Quick Screen (5-min phone screen) ───────────────────────────────────────

_QUICK_SCREEN: list[str] = [
    "In one or two sentences, what's the most impressive technical thing you've built?",
    "What are you most proud of from your last or current role?",
    "What's one thing you'd want us to know about you that's not on your resume?",
    "What are you looking for in your next role that you don't have right now?",
    "Any concerns or questions you'd want to get out of the way early?",
]


# ─── Job-Type Detection ───────────────────────────────────────────────────────

_JOB_TYPE_SIGNALS: dict[str, list[str]] = {
    "backend":    ["backend", "server", "api", "database", "microservice", "rest", "grpc"],
    "frontend":   ["frontend", "ui", "ux", "react", "angular", "vue", "css", "html"],
    "fullstack":  ["fullstack", "full-stack", "full stack", "end-to-end"],
    "data":       ["data scientist", "ml engineer", "machine learning", "data analyst", "analytics"],
    "devops":     ["devops", "sre", "platform", "infrastructure", "kubernetes", "ci/cd"],
    "mobile":     ["mobile", "ios", "android", "react native", "flutter", "swift", "kotlin"],
    "security":   ["security", "cybersecurity", "penetration", "devsecops"],
    "management": ["engineering manager", "tech lead", "staff engineer", "principal"],
}


def _detect_job_type(job_text: str) -> str:
    text_lower = job_text.lower()
    for job_type, signals in _JOB_TYPE_SIGNALS.items():
        if any(s in text_lower for s in signals):
            return job_type
    return "general"


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_interview_questions(
    job_text: str,
    skills_data: dict,
    score: float,
    candidate_name: str = "Candidate",
    experience_years: int = 0,
) -> dict:
    """
    Generate a structured interview pack for a specific candidate.

    Args:
        job_text:         Full job description text
        skills_data:      Dict with matched_skills / missing_skills / extra_skills
        score:            Candidate's overall match score (0-100)
        candidate_name:   Name for personalization
        experience_years: Detected years of experience from resume

    Returns:
        {
            "quick_screen": [...],   # 5 phone-screen Qs (always populated)
            "technical":    [...],   # 3 skill-specific deep dives
            "gap_probing":  [...],   # Up to 3 Qs on missing skills
            "behavioral":   [...],   # 3 STAR-format behavioral Qs
            "culture_fit":  [...],   # 2 culture/motivation Qs
        }
    """
    matched = skills_data.get("matched_skills", []) if skills_data else []
    missing = skills_data.get("missing_skills", []) if skills_data else []

    # ── Quick screen (always 3 questions) ──
    quick = random.sample(_QUICK_SCREEN, min(3, len(_QUICK_SCREEN)))

    # ── Technical: deep-dives on matched skills ──
    technical: list[str] = []
    used_skills = set()

    for skill in matched:
        if len(technical) >= 3:
            break
        skill_lower = skill.lower()
        if skill_lower in _SKILL_QUESTIONS and skill_lower not in used_skills:
            q = random.choice(_SKILL_QUESTIONS[skill_lower])
            technical.append(q)
            used_skills.add(skill_lower)

    # Pad with generic skill-based questions if needed
    for skill in matched:
        if len(technical) >= 3:
            break
        if skill.lower() not in used_skills:
            technical.append(
                f"Walk me through a real project where {skill} was critical to the solution. "
                f"What decisions did you make and what would you do differently now?"
            )
            used_skills.add(skill.lower())

    # Final fallback if no matched skills at all
    if not technical:
        technical = [
            "Walk me through the most technically complex system you've designed or contributed to.",
            "How do you approach learning a new technology stack quickly when thrown into a new project?",
            "What does 'production-quality code' mean to you? Give me a concrete example.",
        ]

    # ── Gap probing: missing skills ──
    gap_probing: list[str] = []
    for skill in missing[:3]:
        template = random.choice(_GAP_TEMPLATES)
        gap_probing.append(template.format(skill=skill))

    # ── Behavioral (3 questions) ──
    behavioral = random.sample(_BEHAVIORAL, min(3, len(_BEHAVIORAL)))

    # ── Culture fit (2 questions) ──
    culture = random.sample(_CULTURE, min(2, len(_CULTURE)))

    return {
        "quick_screen": quick,
        "technical":    technical[:3],
        "gap_probing":  gap_probing,
        "behavioral":   behavioral,
        "culture_fit":  culture,
    }


def format_questions_markdown(
    questions: dict,
    candidate_name: str,
    score: float,
) -> str:
    """Render the interview pack as a clean markdown string for export."""
    sections = {
        "⚡ Quick Screen (5 min phone call)": questions.get("quick_screen", []),
        "🔧 Technical Deep Dives":           questions.get("technical", []),
        "🔍 Gap Probing":                    questions.get("gap_probing", []),
        "🧠 Behavioral (STAR format)":       questions.get("behavioral", []),
        "💡 Culture & Motivation":           questions.get("culture_fit", []),
    }

    lines = [
        f"# Interview Pack — {candidate_name}",
        f"**Match Score:** {score}/100",
        "",
    ]

    for title, qs in sections.items():
        if not qs:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for i, q in enumerate(qs, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    return "\n".join(lines)
