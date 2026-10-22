import pytest
from pydantic import ValidationError

from ohamlab_agent_lab.config import AppConfig


def test_http_config_requires_url():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "mcp": {"transport": "streamable-http"},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost/v1", "model": "x"},
        })


def test_stdio_requires_command():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "mcp": {"transport": "stdio"},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost/v1", "model": "x"},
        })


def test_valid_config():
    cfg = AppConfig.model_validate({
        "mcp": {"transport": "stdio", "command": "python", "args": ["-m", "server"]},
        "llm": {"provider": "openai-compatible", "base_url": "http://localhost/v1", "model": "tool-model"},
    })
    assert cfg.mcp.transport == "stdio"
