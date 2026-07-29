from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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


settings = Settings()
