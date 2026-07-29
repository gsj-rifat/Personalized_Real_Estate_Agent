from src.adapters.listing_generator import LLMHomeListingGenerator


def test_extract_csv_content_strips_code_fences() -> None:
    raw = """```csv
Neighborhood,Location
Los Angeles,West Hollywood
```"""
    cleaned = LLMHomeListingGenerator._extract_csv_content(raw)
    assert "```" not in cleaned
    assert "Neighborhood,Location" in cleaned
    assert "Los Angeles,West Hollywood" in cleaned

