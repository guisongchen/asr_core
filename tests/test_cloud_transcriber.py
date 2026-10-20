"""Tests for the DashScope cloud ASR backend (all HTTP mocked)."""

import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest

from asr_core.cloud_transcriber import (
    CloudTranscriber,
    MAX_BASE64_BYTES,
    _audio_data_uri,
)


@pytest.fixture
def wav_file(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFF" + b"\x00" * 100)
    return str(p)


def _ok_response(text="你好世界", usage=None):
    body = {"choices": [{"message": {"content": text}}]}
    if usage:
        body["usage"] = usage
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = body
    return resp


class TestAudioDataUri:
    def test_wav_mime_and_base64(self, wav_file):
        uri = _audio_data_uri(wav_file)
        assert uri.startswith("data:audio/wav;base64,")
        payload = uri.split(",", 1)[1]
        assert base64.b64decode(payload) == b"RIFF" + b"\x00" * 100

    def test_oversized_audio_rejected(self, tmp_path, monkeypatch):
        p = tmp_path / "big.wav"
        p.write_bytes(b"\x00" * 10)
        monkeypatch.setattr(
            "asr_core.cloud_transcriber.MAX_BASE64_BYTES", 4
        )
        with pytest.raises(RuntimeError, match="too large"):
            _audio_data_uri(str(p))
        assert MAX_BASE64_BYTES > 4  # sanity: real limit untouched elsewhere


class TestLoadLifecycle:
    def test_load_without_api_key_fails(self):
        t = CloudTranscriber(api_key="")
        with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
            t.load()
        assert t.state == "error"

    def test_load_with_api_key(self):
        t = CloudTranscriber(api_key="sk-test")
        t.load()
        assert t.state == "loaded"
        assert t.is_ready
        t.unload()
        assert t.state == "unloaded"

    def test_transcribe_requires_load(self, wav_file):
        t = CloudTranscriber(api_key="sk-test")
        with pytest.raises(RuntimeError, match="not ready"):
            t.transcribe(wav_file)

    def test_no_gpu_memory(self):
        assert CloudTranscriber(api_key="sk-test").gpu_memory_mb() is None


class TestTranscribe:
    def _loaded(self):
        t = CloudTranscriber(api_key="sk-test", base_url="https://example.test/v1")
        t.load()
        return t

    def test_success(self, wav_file):
        t = self._loaded()
        with patch("asr_core.cloud_transcriber.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _ok_response(
                "你好世界", usage={"total_tokens": 42}
            )
            mock_client_cls.return_value = mock_client

            result = t.transcribe(wav_file)

        assert result["text"] == "你好世界"
        call = mock_client.post.call_args
        assert call.args[0] == "https://example.test/v1/chat/completions"
        assert call.kwargs["headers"]["Authorization"] == "Bearer sk-test"
        payload = call.kwargs["json"]
        assert payload["model"] == "qwen3-asr-flash"
        assert payload["asr_options"] == {"enable_itn": True}
        data_uri = payload["messages"][0]["content"][0]["input_audio"]["data"]
        assert data_uri.startswith("data:audio/wav;base64,")

    def test_language_mapped_to_code(self, wav_file):
        t = self._loaded()
        with patch("asr_core.cloud_transcriber.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _ok_response()
            mock_client_cls.return_value = mock_client

            t.transcribe(wav_file, language="chinese")

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["asr_options"]["language"] == "zh"

    def test_unsupported_language_omitted(self, wav_file):
        t = self._loaded()
        with patch("asr_core.cloud_transcriber.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _ok_response()
            mock_client_cls.return_value = mock_client

            t.transcribe(wav_file, language="klingon")

        payload = mock_client.post.call_args.kwargs["json"]
        assert "language" not in payload["asr_options"]

    def test_http_4xx_no_retry(self, wav_file):
        t = self._loaded()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 401
        resp.text = "invalid api key"
        with patch("asr_core.cloud_transcriber.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = resp
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="401"):
                t.transcribe(wav_file)

        assert mock_client.post.call_count == 1

    def test_transport_error_retried_once(self, wav_file):
        t = self._loaded()
        with patch("asr_core.cloud_transcriber.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = [
                httpx.ConnectError("boom"),
                _ok_response("恢复成功"),
            ]
            mock_client_cls.return_value = mock_client

            result = t.transcribe(wav_file)

        assert result["text"] == "恢复成功"
        assert mock_client.post.call_count == 2

    def test_429_retried_then_fails(self, wav_file):
        t = self._loaded()
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 429
        resp.text = "rate limited"
        with patch("asr_core.cloud_transcriber.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = resp
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="429"):
                t.transcribe(wav_file)

        assert mock_client.post.call_count == 2
