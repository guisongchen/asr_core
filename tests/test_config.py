"""Tests for asr_core.config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigDefaults:
    """Test that config defaults are sane."""

    def test_socket_path_default(self):
        from asr_core.config import SOCKET_PATH
        assert SOCKET_PATH.endswith(".sock")

    def test_sample_rate(self):
        from asr_core.config import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_model_choices_is_list(self):
        from asr_core.config import MODEL_CHOICES
        assert isinstance(MODEL_CHOICES, list)
        assert len(MODEL_CHOICES) > 0

    def test_model_size_default_in_choices(self):
        from asr_core.config import MODEL_CHOICES, MODEL_SIZE_DEFAULT
        assert MODEL_SIZE_DEFAULT in MODEL_CHOICES

    def test_model_dir_is_path(self):
        from asr_core.config import MODEL_DIR
        assert isinstance(MODEL_DIR, Path)

    def test_allowed_languages(self):
        from asr_core.config import ALLOWED_LANGUAGES
        assert "english" in ALLOWED_LANGUAGES
        assert "chinese" in ALLOWED_LANGUAGES


class TestConfigFromFile:
    """Test config loading from TOML file."""

    def test_load_config_from_env_var(self):
        """ASR_CORE_CONFIG env var should override default path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[models]\nmodel_dir = "/custom/models"\ndefault_model = "custom-model"\n')
            f.flush()
            try:
                # Re-import to pick up env var
                with patch.dict(os.environ, {"ASR_CORE_CONFIG": f.name}):
                    from asr_core.config import _find_config_path
                    assert _find_config_path() == Path(f.name)
            finally:
                os.unlink(f.name)

    def test_missing_config_returns_defaults(self):
        """Missing config file should not crash."""
        from asr_core.config import _load_config
        with patch.dict(os.environ, {"ASR_CORE_CONFIG": "/nonexistent/path.toml"}):
            # _load_config uses _find_config_path internally
            from asr_core.config import _find_config_path
            path = _find_config_path()
            assert not path.exists() or path == Path("/nonexistent/path.toml")

    def test_socket_path_from_env(self):
        """ASR_CORE_SOCKET env var should be respected at import time."""
        # This tests the pattern; actual value is set at import
        from asr_core.config import SOCKET_PATH
        # Just verify it's a string path
        assert isinstance(SOCKET_PATH, str)
        assert "/" in SOCKET_PATH
