import pytest
from src.core.entities.models import BuyerPreferences, RecommendationResult


def make_preferences(**overrides) -> BuyerPreferences:
    defaults = dict(
        house_size="3-bedroom",
        priorities="quiet, schools, shops",
        amenities="backyard, garage",
        transportation="bus, highway",
        neighborhood_type="suburban",
    )
    return BuyerPreferences(**{**defaults, **overrides})


def test_buyer_preferences_to_query_contains_all_fields():
    prefs = make_preferences()
    query = prefs.to_query()
    assert "3-bedroom" in query
    assert "quiet, schools, shops" in query
    assert "backyard, garage" in query
    assert "bus, highway" in query
    assert "suburban" in query


def test_buyer_preferences_requires_all_fields():
    with pytest.raises(Exception):
        BuyerPreferences()  # type: ignore[call-arg]


def test_recommendation_result_stores_answer():
    result = RecommendationResult(answer="Great home found!", query="find me a home")
    assert result.answer == "Great home found!"
    assert result.query == "find me a home"
