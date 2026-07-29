# HomeMatch — AI-Powered Personalized Real Estate Agent

An AI assistant that understands buyer preferences through conversation and recommends the best matching home using semantic search and LLMs.

---

## How It Works

1. **Preference Collection** — The agent asks 5 questions about size, priorities, amenities, transport, and neighborhood type.
2. **Semantic Embedding** — Home listings from `data/home.csv` are embedded with `OpenAIEmbeddings` and stored in ChromaDB.
3. **RAG Recommendation** — The buyer's preferences are used as a semantic query; the best matching listing is retrieved and narrated by `ChatOpenAI`.

---

## Project Structure

```
├── directives/              # Architecture SOPs
├── src/
│   ├── core/
│   │   ├── entities/        # Pydantic data contracts (BuyerPreferences, HomeListing)
│   │   ├── interfaces/      # ABCs (IPreferenceCollector, IRecommendationEngine)
│   │   └── logic/           # Pure business logic (reserved)
│   ├── adapters/            # Concrete implementations
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
Open `HomeMatch.ipynb` and run the data generation cell to create `data/home.csv`.

### 4. Run the agent

**Interactive CLI** (asks you 5 questions):
```bash
python main.py
```

**Demo mode** (uses preset preferences):
```bash
python HomeMatch.py
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
