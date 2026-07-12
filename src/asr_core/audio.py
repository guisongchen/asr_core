import numpy as np
import soundfile as sf
import torch
import torchaudio

from .config import SAMPLE_RATE


class AudioPreprocessor:
    """Audio preprocessing for better transcription accuracy."""

    @staticmethod
    def preprocess(audio_path: str) -> str:
        try:
            data, sample_rate = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data).t().contiguous()

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=SAMPLE_RATE
                )
                waveform = resampler(waveform)

            peak = torch.max(torch.abs(waveform))
            if peak > 0:
                waveform = waveform / peak * 0.891

            waveform = AudioPreprocessor._reduce_noise(waveform)

            processed_path = audio_path.replace(".wav", "_processed.wav")
            sf.write(processed_path, waveform.squeeze().numpy(), SAMPLE_RATE)
            return processed_path

        except Exception as e:
            print(f"  Warning: Audio preprocessing failed ({e}), using original")
            return audio_path

    @staticmethod
    def _reduce_noise(
        waveform, n_fft=2048, hop_length=512, noise_reduction=0.5
    ):
        try:
            audio_np = waveform.squeeze().numpy()
            stft = torch.stft(
                waveform.squeeze(),
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
            )
            window_size = int(0.05 * SAMPLE_RATE)
            hop_size = window_size // 2
            num_windows = max(1, (len(audio_np) - window_size) // hop_size + 1)
            energies = np.array(
                [
                    np.sum(
                        audio_np[i * hop_size : i * hop_size + window_size] ** 2
                    )
                    for i in range(num_windows)
                ]
            )
            quietest_count = max(1, num_windows // 10)
            quietest_indices = np.argsort(energies)[:quietest_count]
            cols_per_window = window_size // hop_length
            hop_cols = hop_size // hop_length
            noise_cols = []
            for idx in quietest_indices:
                start = idx * hop_cols
                end = start + cols_per_window
                noise_cols.extend(range(start, min(end, stft.shape[1])))
            if noise_cols:
                noise_floor = torch.mean(
                    torch.abs(stft[:, noise_cols]), dim=1, keepdim=True
                )
            else:
                noise_floor = torch.median(torch.abs(stft), dim=1, keepdim=True)[0]

            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            mask = torch.clamp(
                (magnitude - noise_reduction * noise_floor)
                / (magnitude + 1e-10),
                0,
                1,
            )
            mask = mask**0.5
            cleaned_magnitude = magnitude * mask
            cleaned_stft = cleaned_magnitude * torch.exp(1j * phase)
            cleaned = torch.istft(
                cleaned_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                length=len(audio_np),
            )
            return cleaned.unsqueeze(0)
        except Exception:
            return waveform
