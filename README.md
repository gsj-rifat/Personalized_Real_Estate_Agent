# HomeMatch - AI-Powered Personalized Real Estate Agent

`HomeMatch` is a **RAG (Retrieval-Augmented Generation)** application that recommends homes based on buyer preferences.
It started from the **Udacity Generative AI Nanodegree** project outline and was independently refactored into a portfolio-ready app.

---

## Why This Project Matters

- Translates open-ended user preferences into ranked home options and a tailored recommendation.
- Demonstrates practical LLM engineering patterns recruiters look for: modular architecture, security guards, and test coverage.
- Includes a Streamlit interface for quick, interview-friendly demos.

---

## From Course Submission to Portfolio Project

| Area | Course baseline | Portfolio upgrade |
|------|------------------|-------------------|
| Code structure | Monolithic scripts | Layered `src/` architecture with interfaces, adapters, and infrastructure |
| LLM integration | Deprecated import paths | LangChain v0.3 compatible `langchain-openai` and `langchain-community` |
| Configuration | Inline env handling | Fail-fast `pydantic-settings` config |
| Reliability | Assumes data exists | Auto-generates `data/home.csv` when missing |
| Security | Raw prompt inputs | Input sanitization and placeholder-key validation |
| UX | CLI-first | Streamlit multi-step UI with top-3 listing cards |
| Quality | Minimal tests | `pytest` suite for entities, security, and parsing logic |

---

## Core Features

- **RAG pipeline:** ChromaDB retrieval + ChatOpenAI response generation.
- **Structured preference capture:** 5-question flow converted into a `BuyerPreferences` data contract.
- **Top-3 ranked listings:** Streamlit cards with match score badges for transparent recommendations.
- **One-click listing regeneration:** Sidebar action to refresh listing CSV via LLM.
- **Dependency injection:** Central wiring via `src/infrastructure/container.py`.

Architecture diagram: see `docs/ARCHITECTURE.md`.

---

## Quick Demo (Recommended)

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Configure environment
```bash
copy .env.example .env
```
Set `OPENAI_API_KEY` in `.env`.

3. Launch Streamlit UI
```bash
streamlit run src/frontend/app.py
```

Optional CLI entry points:
```bash
python main.py
python HomeMatch.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Core language |
| LangChain 0.3 | LLM orchestration framework |
| langchain-openai | OpenAI chat + embeddings integration |
| langchain-community | CSV loader and Chroma vector store adapters |
| ChromaDB | Vector database for semantic retrieval |
| Pydantic v2 + pydantic-settings | Typed contracts and config validation |
| Streamlit | Recruiter-facing interactive frontend |
| Pytest | Unit testing |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_API_BASE` | No | API base URL (default: `https://api.openai.com/v1`) |
