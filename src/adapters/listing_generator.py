import re
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.core.interfaces.base import IListingGenerator
from src.infrastructure.config import Settings, get_settings
from src.infrastructure.prompts import LISTING_GENERATION_TEMPLATE


class LLMHomeListingGenerator(IListingGenerator):
    """Generates the home listings CSV using the configured LLM."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = ChatOpenAI(
            model=self._settings.llm_model,
            temperature=0.2,
            max_tokens=min(self._settings.llm_max_tokens, 1200),
            api_key=self._settings.openai_api_key,
        )

    @staticmethod
    def _extract_csv_content(raw: str) -> str:
        """
        Extract raw CSV content from typical LLM wrappers like:
        - ```csv ... ```
        - ``` ... ```
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(csv)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    async def generate_home_listings(self) -> None:
        output_path = Path(self._settings.data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_text = LISTING_GENERATION_TEMPLATE.format(
            topic="Homes",
            attributes="Neighborhood,Location,Bedrooms,Bathrooms,House Size (sqft),Price (k$)",
            rows=20,
        )

        message = await self._llm.ainvoke(prompt_text)
        content = message.content if hasattr(message, "content") else str(message)
        csv_text = self._extract_csv_content(content)
        output_path.write_text(csv_text, encoding="utf-8")

