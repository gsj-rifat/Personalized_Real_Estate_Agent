"""
main.py — interactive CLI entry point.
Prompts the user with 5 questions then returns a personalized home recommendation.
"""
import asyncio

from src.infrastructure.container import AppContainer


async def main() -> None:
    container = AppContainer.create_cli()
    await container.ensure_home_listings()

    preferences = container.collector.collect()
    result = await container.engine.recommend(preferences)

    print("\nPersonalized Recommendation:\n")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
