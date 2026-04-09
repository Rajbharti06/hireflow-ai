"""
skills_local.py — Zero-Cost Skill Extraction Engine
=====================================================
Extracts and compares skills, experience, and education level from resume
and job description text using pattern matching only.

No API calls. No cost. Works fully offline.

Used as:
  - Primary skills engine when no API keys are configured
  - Automatic fallback when API calls fail or return empty results
  - Always-on extraction for experience years and education level
"""

import re
from functools import lru_cache


# ─── Skills Database ─────────────────────────────────────────────────────────
# ~200 skills across 10 categories. Add more at the bottom of each list.

_SKILLS_DB: dict[str, list[str]] = {
    "Programming Languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C",
        "Go", "Golang", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
        "R", "MATLAB", "Perl", "Dart", "Haskell", "Elixir", "Clojure",
        "Lua", "Julia", "Bash", "PowerShell", "Groovy", "COBOL", "Fortran",
        "Objective-C", "Assembly",
    ],
    "Web Frontend": [
        "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt", "Gatsby",
        "HTML", "CSS", "Sass", "SCSS", "Tailwind", "Bootstrap", "Material UI",
        "Redux", "Webpack", "Vite", "jQuery", "Ember", "Backbone",
        "WebAssembly", "PWA", "Storybook", "Three.js",
    ],
    "Web Backend": [
        "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
        "Spring", "Rails", "Laravel", "ASP.NET", "NestJS", "GraphQL",
        "REST API", "gRPC", "WebSockets", "OAuth", "JWT", "Celery",
        "Kafka", "RabbitMQ", "Nginx", "Apache",
    ],
    "Data & AI/ML": [
        "Pandas", "NumPy", "SciPy", "Scikit-learn", "TensorFlow", "Keras",
        "PyTorch", "Hugging Face", "Transformers", "LLM", "RAG",
        "Fine-tuning", "LangChain", "LlamaIndex", "Machine Learning",
        "Deep Learning", "NLP", "Computer Vision", "Time Series",
        "XGBoost", "LightGBM", "CatBoost", "Matplotlib", "Seaborn",
        "Plotly", "OpenCV", "NLTK", "spaCy", "Reinforcement Learning",
        "Feature Engineering", "A/B Testing", "Statistics",
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite",
        "Elasticsearch", "Cassandra", "DynamoDB", "Firebase", "Supabase",
        "Oracle", "MSSQL", "Neo4j", "Pinecone", "ChromaDB", "BigQuery",
        "Snowflake", "Redshift", "InfluxDB", "CockroachDB",
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "Ansible", "Jenkins", "GitHub Actions", "GitLab CI",
        "CI/CD", "Linux", "Serverless", "Lambda", "S3", "EC2",
        "CloudFormation", "Helm", "Prometheus", "Grafana", "ArgoCD",
        "Datadog", "New Relic", "Pulumi", "Vagrant",
    ],
    "Data Engineering": [
        "Apache Spark", "Spark", "Hadoop", "Kafka", "Airflow", "dbt",
        "Databricks", "ETL", "Data Pipeline", "Data Warehouse", "Data Lake",
        "Flink", "Storm", "NiFi", "Hive", "Pig",
    ],
    "Tools & Practices": [
        "Git", "Agile", "Scrum", "Kanban", "Jira", "Confluence", "Figma",
        "Postman", "Swagger", "Microservices", "System Design",
        "API Design", "TDD", "BDD", "Unit Testing", "Pytest", "Jest",
        "Selenium", "Cypress", "UNIX", "Shell Scripting", "gRPC",
        "OpenAPI", "Protobuf",
    ],
    "Security": [
        "Cybersecurity", "Penetration Testing", "OWASP", "OAuth2",
        "SSL", "TLS", "Encryption", "Zero Trust", "SOC2", "GDPR",
        "IAM", "SIEM", "Vulnerability Assessment",
    ],
    "Soft Skills": [
        "Communication", "Leadership", "Teamwork", "Problem Solving",
        "Critical Thinking", "Project Management", "Mentoring",
        "Cross-functional", "Stakeholder Management", "Presentation",
        "Negotiation", "Adaptability", "Time Management",
    ],
}

# Build flat lookup: lowercase_skill → original_casing
_SKILL_LOOKUP: dict[str, str] = {}
for _skills in _SKILLS_DB.values():
    for _s in _skills:
        _SKILL_LOOKUP[_s.lower()] = _s

# Pre-compiled patterns for each skill (word-boundary aware)
_SKILL_PATTERNS: dict[str, re.Pattern] = {
    sk: re.compile(r'(?<![a-z+#])' + re.escape(sk) + r'(?![a-z+#])', re.IGNORECASE)
    for sk in _SKILL_LOOKUP
}


# ─── Experience Patterns ─────────────────────────────────────────────────────
_EXP_PATTERNS = [
    re.compile(r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience', re.IGNORECASE),
    re.compile(r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience', re.IGNORECASE),
    re.compile(r'experience\s+(?:of\s+)?(\d+)\+?\s*years?', re.IGNORECASE),
    re.compile(r'(\d+)\+?\s*years?\s+in\s+(?:the\s+)?(?:industry|field|software|tech|it\b)', re.IGNORECASE),
    re.compile(r'(\d+)\+?\s*years?\s+(?:as\s+a?\s+)?(?:senior|junior|lead|principal)', re.IGNORECASE),
]

_EDU_MAP: list[tuple[int, str, list[str]]] = [
    (4, "PhD",        ["phd", "ph.d", "doctorate", "doctoral"]),
    (3, "Master's",   ["master", "master's", "masters", "mba", "m.sc", "m.tech", "mtech", "m.s.", "msc", "me "]),
    (2, "Bachelor's", ["bachelor", "bachelor's", "b.tech", "btech", "b.e.", "b.sc", "bsc", "b.s.", "undergraduate", "b.eng"]),
    (1, "Diploma",    ["diploma", "associate degree", "associate's"]),
]


# ─── Public API ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def extract_skills_local(text: str) -> list[str]:
    """
    Extract all recognised skills from text using keyword matching.
    Fast, offline, zero cost. Cached per unique text.

    Returns list of skills in original casing, deduplicated.
    """
    found = []
    for skill_lower, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text):
            found.append(_SKILL_LOOKUP[skill_lower])
    return found


def compare_skills_local(job_text: str, resume_text: str) -> dict:
    """
    Compare skills between a job description and a resume.
    Returns the same shape as extract_skills_analysis() — fully drop-in compatible.

    {
        "matched_skills": [...],  # in JD and resume
        "missing_skills": [...],  # in JD but not resume
        "extra_skills":   [...],  # in resume but not JD
    }
    """
    job_skills_raw = extract_skills_local(job_text)
    resume_skills_raw = extract_skills_local(resume_text)

    job_set = {s.lower() for s in job_skills_raw}
    resume_set = {s.lower() for s in resume_skills_raw}

    matched = sorted([_SKILL_LOOKUP[s] for s in job_set & resume_set])
    missing = sorted([_SKILL_LOOKUP[s] for s in job_set - resume_set])
    extra   = sorted([_SKILL_LOOKUP[s] for s in resume_set - job_set])

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills":   extra,
    }


def extract_years_experience(text: str) -> int:
    """
    Extract the maximum years of experience mentioned in the text.
    Returns 0 if nothing found.

    Examples matched:
      "5 years of experience" → 5
      "10+ years in the field" → 10
      "experience of 3 years" → 3
    """
    found = []
    for pattern in _EXP_PATTERNS:
        for m in pattern.finditer(text):
            try:
                found.append(int(m.group(1)))
            except (IndexError, ValueError):
                pass
    return max(found) if found else 0


def detect_education_level(text: str) -> tuple[int, str]:
    """
    Detect the highest education level mentioned in text.

    Returns (level: int 0-4, label: str)
      4 → "PhD"
      3 → "Master's"
      2 → "Bachelor's"
      1 → "Diploma"
      0 → "Not specified"
    """
    text_lower = text.lower()
    for level, label, keywords in _EDU_MAP:
        if any(kw in text_lower for kw in keywords):
            return level, label
    return 0, "Not specified"
