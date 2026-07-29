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
