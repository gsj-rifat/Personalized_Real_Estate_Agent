from src.adapters.preference_collector import StaticPreferenceCollector
from src.core.entities.models import BuyerPreferences


SAMPLE = BuyerPreferences(
    house_size="3-bedroom",
    priorities="quiet, schools",
    amenities="backyard",
    transportation="bus",
    neighborhood_type="suburban",
)


def test_static_collector_returns_given_preferences():
    collector = StaticPreferenceCollector(SAMPLE)
    result = collector.collect()
    assert result == SAMPLE


def test_static_collector_is_idempotent():
    collector = StaticPreferenceCollector(SAMPLE)
    assert collector.collect() == collector.collect()
