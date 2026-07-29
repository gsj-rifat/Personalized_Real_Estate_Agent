# HomeMatch — AI-Powered Personalized Real Estate Agent

Portfolio-ready GenAI project built from the **Udacity Generative AI Nanodegree** course outline and refactored into a recruiter-friendly, production-minded Python app.

---

## How It Works

1. **Preference Collection** — The agent asks 5 questions about size, priorities, amenities, transport, and neighborhood type.
2. **Resilient Data Bootstrap** — On startup, the app checks `data/home.csv` and **auto-generates it** if missing.
3. **Semantic Embedding** — Home listings are embedded with `OpenAIEmbeddings` and stored in ChromaDB.
4. **RAG Recommendation** — The buyer's (sanitized) preferences are used as a semantic query; the best matching listing is retrieved and narrated by `ChatOpenAI`.

---

## Project Structure

```
├── directives/              # Architecture SOPs
├── docs/                     # Recruiter-facing architecture docs
├── src/
│   ├── core/
│   │   ├── entities/        # Pydantic data contracts (BuyerPreferences, HomeListing)
│   │   ├── interfaces/      # ABCs (IPreferenceCollector, IRecommendationEngine)
│   │   └── logic/           # Pure business logic (reserved)
│   ├── adapters/            # Concrete implementations
│   │   ├── listing_generator.py  # LLM-based CSV bootstrap
│   │   ├── preference_collector.py   # CLI + Static collectors
│   │   └── recommendation_engine.py  # LangChain RAG engine
│   └── infrastructure/
│       ├── config.py        # Pydantic Settings (all env vars here)
│       └── prompts.py       # System prompts as code
├── data/home.csv            # Home listings (gitignored — generate via notebook)
├── db/                      # ChromaDB vector store (gitignored)
├── HomeMatch.py             # Demo entry point (static preferences)
├── main.py                  # Interactive CLI entry point
├── HomeMatch.ipynb          # Notebook: data generation + exploration
├── requirements.txt
├── .env.example             # Required env var template
└── tests/
```

---

## Portfolio Highlights

- **RAG with LangChain + ChromaDB** to retrieve the best matching listing and generate a concise recommendation.
- **Fail-fast configuration** with `pydantic-settings` so missing/placeholder secrets surface immediately.
- **Dependency injection (DI)** via a lightweight container (`src/infrastructure/container.py`) to keep wiring separate from logic.
- **Async-first interface** (`async def recommend(...)`) and **unit tests** with `pytest` for critical logic.
- **Security hardening**: sanitizes untrusted user preferences before they are inserted into prompts.
- **Streamlit recruiter demo UI** with multi-step preference capture, top-3 property cards, and one-click listing regeneration.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 3. Generate home listings (first time)
If `data/home.csv` is missing, the app will auto-generate it on first run. You can also use `HomeMatch.ipynb` to inspect and regenerate the dataset.

### 4. Run the agent

**Interactive CLI** (asks you 5 questions):
```bash
python main.py
```

**Demo mode** (uses preset preferences):
```bash
python HomeMatch.py
```

**Streamlit UI** (recommended for portfolio demos):
```bash
streamlit run src/frontend/app.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Core language |
| LangChain 0.3 | LLM orchestration framework |
| langchain-openai | OpenAI LLM + Embeddings adapter |
| langchain-community | CSVLoader, ChromaDB adapter |
| ChromaDB | Vector database for semantic search |
| Pydantic v2 + pydantic-settings | Data contracts + env config |
| python-dotenv | Local `.env` loading |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `OPENAI_API_BASE` | No | API base URL (default: `https://api.openai.com/v1`) |
