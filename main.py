"""
main.py — interactive CLI entry point.
Prompts the user with 5 questions then returns a personalized home recommendation.
"""
import asyncio

from src.adapters.preference_collector import CLIPreferenceCollector
from src.adapters.recommendation_engine import LangChainRecommendationEngine


async def main() -> None:
    collector = CLIPreferenceCollector()
    engine = LangChainRecommendationEngine()

    preferences = collector.collect()
    result = await engine.recommend(preferences)

    print("\nPersonalized Recommendation:\n")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
