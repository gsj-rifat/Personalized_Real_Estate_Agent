from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    llm_temperature: float = Field(default=0.3, description="LLM temperature")
    llm_max_tokens: int = Field(default=2000, description="LLM max tokens")
    data_path: str = Field(default="data/home.csv", description="Path to home listings CSV")
    db_path: str = Field(default="db", description="ChromaDB persist directory")

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, v: str) -> str:
        candidate = v.strip()
        placeholder_values = {
            "your-openai-api-key-here",
            "your-openai-api-key",
            "your api key",
            "your api key here",
            "put-your-api-key-here",
            "replace-me",
        }
        if candidate.lower() in placeholder_values:
            raise ValueError(
                "OPENAI_API_KEY appears to be a placeholder. Set a real key in .env or environment variables."
            )
        return candidate


def get_settings() -> Settings:
    """
    Instantiate settings on demand.

    This avoids module import side effects and makes unit tests easier.
    """
    return Settings()

