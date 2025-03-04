from __future__ import annotations

from hud.settings import settings
from hud.utils.config import Config


def test_config() -> None:
    config = Config()
    assert config.base_url == settings.base_url
