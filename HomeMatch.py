"""
HomeMatch — entry point for interactive CLI mode.
Uses StaticPreferenceCollector for demo; swap for CLIPreferenceCollector for live input.
"""
import asyncio

from src.adapters.preference_collector import StaticPreferenceCollector
from src.adapters.recommendation_engine import LangChainRecommendationEngine
from src.core.entities.models import BuyerPreferences


DEMO_PREFERENCES = BuyerPreferences(
    house_size="A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    priorities="A quiet neighborhood, good local schools, and convenient shopping options.",
    amenities="A backyard for gardening, a two-car garage, and a modern energy-efficient heating system.",
    transportation="Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    neighborhood_type="A balance between suburban tranquility and access to urban amenities.",
)


async def main() -> None:
    collector = StaticPreferenceCollector(DEMO_PREFERENCES)
    engine = LangChainRecommendationEngine()

    preferences = collector.collect()
    result = await engine.recommend(preferences)

    print("\nPersonalized Recommendation:\n")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
