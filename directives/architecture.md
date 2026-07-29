# Architecture Directive — HomeMatch

## Goal
Build a modular, GenAI-native real estate recommendation assistant using LangChain, OpenAI, and ChromaDB.

## 3-Layer Model
| Layer | Location | Responsibility |
|-------|----------|----------------|
| Directive | `directives/` | SOPs, goals, constraints |
| Orchestration | `src/infrastructure/` | Config, DI, Prompts |
| Execution | `src/core/`, `src/adapters/` | Pure logic + concrete implementations |

## Key Constraints
- No `os.getenv` outside `config.py` — all secrets via `pydantic-settings`.
- All I/O adapters are `async def`.
- No file > 150 lines.
- LangChain imports: use `langchain_openai` and `langchain_community` — NOT the deprecated `langchain.*` paths.
- ChromaDB persisted at path from `settings.db_path`.
