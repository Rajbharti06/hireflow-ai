# HireFlow AI

> AI-powered resume screener for recruiters — screen 100+ candidates in seconds with explainable, multi-signal scoring.

![HireFlow AI Login](screenshots/01_login.png)

---

## Overview

HireFlow AI ranks resumes against a job description using a three-signal hybrid scoring engine: semantic embeddings, skill overlap analysis, and LLM judgment. Results are ranked, explained in plain English, and exportable — all without leaving your browser.

Works fully offline (Zero-Cost Mode) with a local ML model and no API key required.

---

## Screenshots

### Dashboard
![HireFlow AI Dashboard](screenshots/02_dashboard.png)

---

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid AI Scoring** | Semantic embeddings + skill overlap + LLM judgment blended into one transparent score |
| **Zero-Cost Mode** | Works 100% offline with local ML — no API key needed |
| **Blind Mode** | Hides candidate names during scoring to reduce unconscious bias |
| **Interview Packs** | Auto-generates tailored interview questions for each candidate — export as markdown |
| **Analytics Charts** | Score distribution histogram and skills-coverage chart for the full candidate pool |
| **Skills Gap Analysis** | Per-candidate matched, missing, and extra skills breakdown |
| **Export Anywhere** | Download results as CSV or a full markdown interview report with one click |
| **Pipeline Board** | Move candidates through Screened → Interview → Offer → Rejected stages |
| **JD Scanner** | Scan any job description to preview required skills before uploading resumes |
| **Radar Charts** | Visual skills radar per candidate (experience, education, quality, tier) |
| **Session History** | Past screenings saved to Supabase — reload any session from the sidebar |
| **Auth & Usage Tracking** | Email/password + OAuth sign-in via Supabase Auth |

---

## Scoring Pipeline

```
PDF → Text → Batch Embed → Skills Extract → LLM Score → Hybrid Blend → Rank → Display
```

### Hybrid Score Formula

```
final_score = W₁ × embedding_similarity + W₂ × skill_overlap + W₃ × llm_confidence
```

Default weights (tunable in sidebar):

| Signal | Default | What it captures |
|--------|---------|-----------------|
| Semantic Similarity | 50% | Conceptual alignment between resume and JD |
| Skill Overlap | 30% | Keyword/skill checklist coverage |
| LLM Confidence | 20% | AI judgment on seniority fit, red flags, trajectory |

Weights are adjustable via sliders in the sidebar — must sum to 100%.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Rajbharti06/hireflow-ai.git
cd hireflow-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your keys (see Configuration below)

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**No API key?** Leave `OPENAI_API_KEY` blank — the app falls back to Zero-Cost Mode using the local `all-MiniLM-L6-v2` sentence-transformer (~80 MB, downloaded on first run).

---

## Configuration

Copy `.env.example` to `.env` and fill in the values you need:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Optional | OpenAI key for GPT-4o-mini LLM scoring and explanations |
| `SUPABASE_URL` | Optional | Supabase project URL — enables auth and session history |
| `SUPABASE_ANON_KEY` | Optional | Supabase anon key — used for auth and read queries |
| `SUPABASE_SERVICE_KEY` | Optional | Service-role key — bypasses RLS for server-side DB writes |

Without Supabase keys the app runs in local mode — auth and history are disabled but all scoring works.

---

## AI Backends

### OpenAI (recommended for best explanations)
- Set `OPENAI_API_KEY` in `.env`
- Uses `gpt-4o-mini` — fast and cost-effective
- Generates rich plain-English explanations and per-candidate LLM scores

### Ollama (free, local, private)
- Install: [ollama.com](https://ollama.com)
- Run: `ollama serve` then `ollama pull mistral`
- Select **Ollama** in the sidebar AI backend dropdown

### Zero-Cost Mode (no API key)
- Uses only the local sentence-transformer for semantic scoring
- Skill scoring and rule-based explanations still run
- LLM score defaults to neutral (50) — no API calls made

---

## Architecture

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit UI — upload, process, display ranked results |
| `parser.py` | PDF → clean text via pdfplumber |
| `embedder.py` | Text → 384-dim vectors via sentence-transformers |
| `scorer.py` | Hybrid scoring engine — embedding + skills + LLM blend |
| `explainer.py` | LLM explanations, skill extraction, AI scoring |
| `skills_local.py` | Rule-based skill extraction, experience/education detection, resume quality scoring |
| `interview_gen.py` | Interview question generation per candidate |
| `database.py` | Supabase persistence — sessions, results, history |
| `supabase_client.py` | Supabase client setup (anon + service-role) |
| `utils.py` | Shared helpers (name extraction, text truncation) |

---

## Database Schema

Requires two tables in Supabase (`supabase_schema.sql` included):

- **`jobs`** — one row per screening session (job title, description, user)
- **`results`** — one row per candidate per session (scores, explanation, skills JSON, shortlist flag)
- **`profiles`** — one row per user (usage tracking)

Row Level Security (RLS) is enforced on all tables. The service-role key (`SUPABASE_SERVICE_KEY`) is used for server-side writes to bypass RLS.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **pdfplumber** — PDF text extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local semantic embeddings
- **scikit-learn** — cosine similarity
- **OpenAI / Ollama** — LLM explanations and scoring
- **Supabase** — auth, PostgreSQL database, RLS
- **Plotly** — analytics charts and radar visualizations
- **pandas** — result table and CSV export

---

## Development

```bash
# Run with auto-reload
streamlit run app.py

# Generate sample PDFs for testing
python generate_pdfs.py
```

---

## License

MIT
