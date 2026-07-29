from abc import ABC, abstractmethod
from src.core.entities.models import BuyerPreferences, RecommendationResult


class IPreferenceCollector(ABC):
    @abstractmethod
    def collect(self) -> BuyerPreferences:
        """Collect buyer preferences through Q&A."""

class IRecommendationEngine(ABC):
    @abstractmethod
    async def recommend(self, preferences: BuyerPreferences) -> RecommendationResult:
        """Return a personalized home recommendation."""


class IListingGenerator(ABC):
    @abstractmethod
    async def generate_home_listings(self) -> None:
        """Generate the home listings CSV used by the recommendation engine."""
