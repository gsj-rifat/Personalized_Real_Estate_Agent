from src.core.interfaces.base import IPreferenceCollector
from src.core.entities.models import BuyerPreferences
from src.infrastructure.prompts import BUYER_QUESTIONS


class CLIPreferenceCollector(IPreferenceCollector):
    """Collects buyer preferences via CLI prompts."""

    def collect(self) -> BuyerPreferences:
        answers = [input(f"{q}\nAnswer: ").strip() for q in BUYER_QUESTIONS]
        return BuyerPreferences(
            house_size=answers[0],
            priorities=answers[1],
            amenities=answers[2],
            transportation=answers[3],
            neighborhood_type=answers[4],
        )


class StaticPreferenceCollector(IPreferenceCollector):
    """Returns fixed preferences — useful for demos and tests."""

    def __init__(self, preferences: BuyerPreferences) -> None:
        self._preferences = preferences

    def collect(self) -> BuyerPreferences:
        return self._preferences
