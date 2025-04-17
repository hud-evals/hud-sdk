from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global settings for the HUD SDK.

    This class manages configuration values loaded from environment variables
    and provides global access to settings throughout the application.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    base_url: str = Field(
        default="https://orcstaging.hud.so/hud-gym/api",
        description="Base URL for the HUD API",
        validation_alias="base_url",
    )

    api_key: str | None = Field(
        default=None,
        description="API key for authentication with the HUD API",
        validation_alias="HUD_API_KEY",
    )
    
    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for Anthropic models",
        validation_alias="ANTHROPIC_API_KEY",
    )
    
    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI models",
        validation_alias="OPENAI_API_KEY",
    )

# Create a singleton instance
settings = Settings()


# Add utility functions for backwards compatibility
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
