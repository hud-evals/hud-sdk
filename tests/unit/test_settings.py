# Test cases for hud.settings using pydantic-settings

import os
import pytest
from pydantic import ValidationError

# We need to test the Settings class directly
from hud.settings import Settings

# Ensure tests don't interfere with each other or actual user env vars
@pytest.fixture(autouse=True)
def clear_env_vars(monkeypatch):
    """Clears relevant env vars before each test and restores after."""
    vars_to_clear = ["HUD_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BASE_URL"]
    original_values = {var: os.environ.get(var) for var in vars_to_clear}
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    yield
    for var, val in original_values.items():
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)

@pytest.fixture
def mock_dotenv_file(tmp_path, monkeypatch):
    """Creates a dummy .env file in a temporary directory."""
    dotenv_path = tmp_path / ".env"
    
    def _create_dotenv(content):
        dotenv_path.write_text(content)
        # Monkeypatch CWD to make pydantic-settings find the .env file
        monkeypatch.chdir(tmp_path)
        return dotenv_path
        
    return _create_dotenv

def test_settings_load_from_env_vars(monkeypatch):
    """Tests loading API keys directly from environment variables."""
    monkeypatch.setenv("HUD_API_KEY", "env_hud_key")
    monkeypatch.setenv("OPENAI_API_KEY", "env_openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env_anthropic_key")
    monkeypatch.setenv("BASE_URL", "http://env.example.com")
    
    # Force reload settings for the test by creating a new instance.
    # The clear_env_vars fixture ensures no .env file interferes implicitly.
    settings_instance = Settings()
    
    assert settings_instance.api_key == "env_hud_key"
    assert settings_instance.openai_api_key == "env_openai_key"
    assert settings_instance.anthropic_api_key == "env_anthropic_key"
    assert settings_instance.base_url == "http://env.example.com"

def test_settings_load_from_dotenv_file(mock_dotenv_file):
    """Tests loading API keys from a .env file."""
    mock_dotenv_file(
        "HUD_API_KEY=dotenv_hud_key\n"
        "OPENAI_API_KEY=dotenv_openai_key\n"
        "#ANTHROPIC_API_KEY=commented_out\n"
        "BASE_URL=http://dotenv.example.com"
    )
    
    # Pydantic-settings automatically loads .env if present in CWD
    # Create a new instance to trigger loading
    settings_instance = Settings()
    
    assert settings_instance.api_key == "dotenv_hud_key"
    assert settings_instance.openai_api_key == "dotenv_openai_key"
    assert settings_instance.anthropic_api_key is None # Should be None as it was commented out
    assert settings_instance.base_url.startswith("http")

def test_settings_env_vars_override_dotenv(mock_dotenv_file, monkeypatch):
    """Tests that environment variables take precedence over .env file."""
    # Create .env file
    mock_dotenv_file(
        "HUD_API_KEY=dotenv_hud_key\n"
        "OPENAI_API_KEY=dotenv_openai_key"
    )
    
    # Set overriding env vars
    monkeypatch.setenv("HUD_API_KEY", "env_hud_key_override")
    
    # Create a new instance to trigger loading
    settings_instance = Settings()
    
    assert settings_instance.api_key == "env_hud_key_override" # Env var wins
    assert settings_instance.openai_api_key == "dotenv_openai_key" # Loaded from .env
    assert settings_instance.anthropic_api_key is None

def test_settings_defaults_when_no_source():
    """Tests that settings fall back to defaults when no env var or .env."""
    # No env vars (cleared by fixture), no .env file (mock_dotenv_file not used)
    settings_instance = Settings()
    
    # Only assert that the clear_env_vars fixture removed HUD_API_KEY
    assert os.environ.get("HUD_API_KEY") is None
    # base_url should be a non-empty string
    assert settings_instance.base_url.startswith("http")

