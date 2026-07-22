"""Tests for asr_core.audio module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_core.audio import AudioPreprocessor
from asr_core.config import SAMPLE_RATE


@pytest.fixture
def wav_file(tmp_path):
    """Create a temporary WAV file with a sine wave."""
    duration = 1.0  # seconds
    freq = 440.0  # Hz
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), data, SAMPLE_RATE)
    return str(path)


@pytest.fixture
def stereo_wav_file(tmp_path):
    """Create a stereo WAV file."""
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    left = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    data = np.column_stack([left, right])
    path = tmp_path / "stereo.wav"
    sf.write(str(path), data, SAMPLE_RATE)
    return str(path)


@pytest.fixture
def mp3_like_file(tmp_path):
    """Create a file with non-.wav extension (simulates mp3/flac input)."""
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    data = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    # Write as WAV but with .flac extension to test path handling
    path = tmp_path / "audio.flac"
    sf.write(str(path), data, SAMPLE_RATE)
    return str(path)


class TestPreprocessPathHandling:
    """Test that preprocess generates safe output paths."""

    def test_wav_gets_processed_suffix(self, wav_file):
        result = AudioPreprocessor.preprocess(wav_file)
        assert result != wav_file
        assert "_processed.wav" in result
        assert Path(result).exists()

    def test_non_wav_extension_gets_processed_path(self, mp3_like_file):
        """Non-.wav files must NOT overwrite the original."""
        original_content = Path(mp3_like_file).read_bytes()
        result = AudioPreprocessor.preprocess(mp3_like_file)
        # The original must be untouched
        assert Path(mp3_like_file).read_bytes() == original_content
        # Result should be a different file
        assert result != mp3_like_file
        assert "_processed.wav" in result

    def test_processed_file_is_valid_wav(self, wav_file):
        result = AudioPreprocessor.preprocess(wav_file)
        data, sr = sf.read(result)
        assert sr == SAMPLE_RATE
        assert len(data) > 0

    def test_original_not_modified(self, wav_file):
        original_content = Path(wav_file).read_bytes()
        AudioPreprocessor.preprocess(wav_file)
        assert Path(wav_file).read_bytes() == original_content


class TestPreprocessAudio:
    """Test audio processing correctness."""

    def test_output_is_mono(self, stereo_wav_file):
        result = AudioPreprocessor.preprocess(stereo_wav_file)
        data, sr = sf.read(result)
        assert data.ndim == 1  # mono

    def test_output_sample_rate(self, wav_file):
        result = AudioPreprocessor.preprocess(wav_file)
        _, sr = sf.read(result)
        assert sr == SAMPLE_RATE

    def test_resampling(self, tmp_path):
        """Files at different sample rates should be resampled to 16kHz."""
        duration = 0.5
        orig_sr = 44100
        t = np.linspace(0, duration, int(orig_sr * duration), endpoint=False)
        data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = tmp_path / "high_sr.wav"
        sf.write(str(path), data, orig_sr)

        result = AudioPreprocessor.preprocess(str(path))
        out_data, out_sr = sf.read(result)
        assert out_sr == SAMPLE_RATE
        # Duration should be approximately preserved
        expected_samples = int(duration * SAMPLE_RATE)
        assert abs(len(out_data) - expected_samples) < SAMPLE_RATE * 0.01

    def test_normalization_peak(self, wav_file):
        """Output peak should be ~0.891 (-1 dBFS)."""
        result = AudioPreprocessor.preprocess(wav_file)
        data, _ = sf.read(result)
        peak = np.max(np.abs(data))
        assert peak <= 0.9  # allow small tolerance
        assert peak > 0.5  # shouldn't be silent

    def test_nonexistent_file_returns_original(self):
        """Preprocessing a missing file should return the original path."""
        result = AudioPreprocessor.preprocess("/nonexistent/audio.wav")
        assert result == "/nonexistent/audio.wav"


class TestReduceNoise:
    """Test noise reduction doesn't crash on edge cases."""

    def test_silence(self):
        """Noise reduction on silence should not crash."""
        import torch
        waveform = torch.zeros(1, SAMPLE_RATE)
        result = AudioPreprocessor._reduce_noise(waveform)
        assert result.shape == waveform.shape

    def test_short_audio(self):
        """Very short audio should not crash."""
        import torch
        waveform = torch.randn(1, 100)
        result = AudioPreprocessor._reduce_noise(waveform)
        assert result.shape[0] == 1
