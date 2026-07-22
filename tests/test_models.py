"""Tests for asr_core.models (Pydantic schemas)."""

import pytest
from pydantic import ValidationError

from asr_core.models import (
    LoadRequest,
    ModelStatus,
    TranscribeRequest,
    TranscribeResponse,
)


class TestModelStatus:
    def test_default_state(self):
        status = ModelStatus()
        assert status.state == "unloaded"
        assert status.model_name is None
        assert status.requests_total == 0

    def test_valid_states(self):
        for state in ("unloaded", "loading", "loaded", "error"):
            status = ModelStatus(state=state)
            assert status.state == state

    def test_invalid_state_rejected(self):
        with pytest.raises(ValidationError):
            ModelStatus(state="broken")


class TestLoadRequest:
    def test_default_model(self):
        req = LoadRequest()
        assert req.model_name == "qwen3-asr-0.6b"

    def test_custom_model(self):
        req = LoadRequest(model_name="Qwen3-ASR-1.7B")
        assert req.model_name == "Qwen3-ASR-1.7B"


class TestTranscribeRequest:
    def test_requires_audio_path(self):
        with pytest.raises(ValidationError):
            TranscribeRequest()

    def test_valid_request(self):
        req = TranscribeRequest(audio_path="/tmp/test.wav")
        assert req.audio_path == "/tmp/test.wav"
        assert req.model_name is None
        assert req.language is None


class TestTranscribeResponse:
    def test_valid_response(self):
        resp = TranscribeResponse(text="hello world")
        assert resp.text == "hello world"
        assert resp.detected_language is None

    def test_full_response(self):
        resp = TranscribeResponse(
            text="hello", detected_language="english", duration_seconds=1.5
        )
        assert resp.duration_seconds == 1.5
