from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.adapters.listing_generator import LLMHomeListingGenerator
from src.adapters.preference_collector import CLIPreferenceCollector, StaticPreferenceCollector
from src.adapters.recommendation_engine import LangChainRecommendationEngine
from src.core.entities.models import BuyerPreferences
from src.core.interfaces.base import IListingGenerator, IPreferenceCollector, IRecommendationEngine
from src.infrastructure.config import Settings, get_settings


@dataclass(frozen=True)
class AppContainer:
    settings: Settings
    collector: IPreferenceCollector
    engine: IRecommendationEngine
    listing_generator: IListingGenerator

    @classmethod
    def create_cli(cls) -> "AppContainer":
        settings = get_settings()
        collector = CLIPreferenceCollector()
        engine = LangChainRecommendationEngine(settings=settings)
        listing_generator = LLMHomeListingGenerator(settings=settings)
        return cls(
            settings=settings,
            collector=collector,
            engine=engine,
            listing_generator=listing_generator,
        )

    @classmethod
    def create_demo(cls, preferences: BuyerPreferences) -> "AppContainer":
        settings = get_settings()
        collector = StaticPreferenceCollector(preferences=preferences)
        engine = LangChainRecommendationEngine(settings=settings)
        listing_generator = LLMHomeListingGenerator(settings=settings)
        return cls(
            settings=settings,
            collector=collector,
            engine=engine,
            listing_generator=listing_generator,
        )

    async def ensure_home_listings(self) -> None:
        output_path = Path(self.settings.data_path)
        if output_path.exists():
            return
        await self.listing_generator.generate_home_listings()

