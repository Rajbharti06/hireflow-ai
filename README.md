# 🎯 AI Resume Screener v2.0

> AI-powered candidate screening tool with **hybrid scoring** — scores and ranks resumes against job descriptions using semantic understanding, skill analysis, and AI judgment.

## Pipeline

```
PDF → Text → Batch Embed → Skills Extract → LLM Score → Hybrid Blend → Rank → Display
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
set OPENAI_API_KEY=sk-your-key-here

# 3. Run the app
streamlit run app.py
```

## Architecture

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit UI — upload, process, display ranked results with score breakdowns |
| `parser.py` | PDF → clean text extraction via pdfplumber |
| `embedder.py` | Text → 384-dim semantic vectors via sentence-transformers |
| `scorer.py` | **Hybrid scoring engine** — 50% embedding + 30% skills + 20% LLM |
| `explainer.py` | AI explanations, LLM scoring, skills extraction, rejection reasoning |
| `database.py` | SQLite persistence — sessions, results, and history |
| `utils.py` | Shared helpers |

## Hybrid Scoring (v2.0)

```
final_score = 0.50 × embedding_similarity + 0.30 × skill_overlap + 0.20 × llm_confidence
```

| Signal | Weight | What it captures |
|--------|--------|-----------------|
| Embedding Similarity | 50% | Semantic meaning — conceptual alignment |
| Skill Overlap | 30% | Checklist coverage — did they tick the boxes? |
| LLM Confidence | 20% | AI judgment — seniority fit, red flags, trajectory |

## AI Backends

### OpenAI (recommended for demos)
- Set `OPENAI_API_KEY` environment variable
- Uses `gpt-4o-mini` — fast, cheap, high quality

### Ollama (free, local, private)
- Install: [ollama.com](https://ollama.com)
- Run: `ollama serve` then `ollama pull mistral`
- Select "Ollama" in the sidebar

## Features

### Core
- ✅ PDF parsing with error handling
- ✅ Semantic similarity scoring (not just keyword matching)
- ✅ AI-generated plain English explanations
- ✅ Skill extraction: matched, missing, and extra skills
- ✅ Ranked results from highest to lowest

### v2.0 Upgrades
- ✅ **Hybrid scoring** — 3-signal blend (embedding + skills + LLM)
- ✅ **Score breakdown** — transparent Semantic/Skills/AI Judge contribution per candidate
- ✅ **Batch embeddings** — process all resumes at once (faster)
- ✅ **LLM response caching** — no duplicate API calls
- ✅ **💰 Cheap Mode** — skip LLM calls, use rule-based explanations
- ✅ **"Why not selected"** — rejection-specific explanations for low scorers
- ✅ **SQLite persistence** — sessions survive refresh, browsable history
- ✅ **Session history** — reload past screenings from sidebar

## Cheap Mode

Toggle **💰 Cheap Mode** in the sidebar to:
- Skip LLM explanation calls (saves API cost)
- Use rule-based explanations from skills data
- Default LLM score to 50 (neutral)
- Perfect for screening 50+ resumes on a budget

## Tech Stack

- **Python** — core
- **Streamlit** — UI
- **pdfplumber** — PDF parsing
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **scikit-learn** — cosine similarity
- **OpenAI / Ollama** — LLM explanations + scoring
- **SQLite** — persistence

## Notes

- This is a **POC / demo**, not production software
- Hybrid scoring produces more differentiated results than pure cosine similarity
- First run will download the sentence-transformer model (~80MB)
- LLM calls are cached per JD+resume pair within a session
