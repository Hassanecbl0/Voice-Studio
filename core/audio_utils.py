import os
import shutil
import numpy as np
import soundfile as sf
import torchaudio
import torch

TARGET_SAMPLE_RATE = 24000

def ensure_dir(dir_path: str):
    """Ensure directory exists."""
    os.makedirs(dir_path, exist_ok=True)

def load_audio(file_path: str, target_sr: int = TARGET_SAMPLE_RATE):
    """
    Load an audio file, convert to mono and resample to target_sr.
    Returns: (numpy_array, sample_rate)
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception:
        # Fallback to soundfile
        data, sr = sf.read(file_path)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0).float()
        else:
            waveform = torch.from_numpy(data.T).float()

    # Convert to mono if multichannel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Flatten to 1D numpy
    audio_np = waveform.squeeze(0).numpy()
    return audio_np, target_sr

def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Trim leading and trailing silence from audio waveform."""
    if len(audio) == 0:
        return audio
    
    # Find non-silent indices
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio
    
    start_idx = max(0, non_silent[0])
    end_idx = min(len(audio), non_silent[-1] + 1)
    return audio[start_idx:end_idx]

def preprocess_reference_audio(input_path: str, output_path: str, max_duration_sec: float = 12.0) -> str:
    """
    Preprocess reference voice sample for XTTS-v2:
    - Resample to 24000 Hz mono
    - Trim dead silence
    - Limit to max_duration_sec for optimal conditioning
    - Normalize amplitude
    """
    audio, sr = load_audio(input_path, target_sr=TARGET_SAMPLE_RATE)
    audio = trim_silence(audio)

    # Limit duration if too long (XTTS works best with 3 to 10 seconds reference)
    max_samples = int(max_duration_sec * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Normalize volume
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    sf.write(output_path, audio, sr, subtype='PCM_16')
    return output_path

def get_preset_voices(voices_dir: str) -> dict[str, str]:
    """Return dictionary of voice_name -> filepath for all wav/mp3/flac in voices_dir."""
    if not os.path.isdir(voices_dir):
        return {}
    
    supported_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    presets = {}
    for filename in sorted(os.listdir(voices_dir)):
        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_exts:
            clean_name = name.replace("_", " ").title()
            presets[clean_name] = os.path.join(voices_dir, filename)
    return presets

def save_new_preset(source_audio_path: str, voice_name: str, voices_dir: str) -> str:
    """Clean and save a new reference voice into the voices library."""
    ensure_dir(voices_dir)
    safe_name = "".join(c for c in voice_name if c.isalnum() or c in (" ", "_", "-")).strip()
    safe_name = safe_name.replace(" ", "_")
    if not safe_name:
        safe_name = "custom_voice"
    
    target_path = os.path.join(voices_dir, f"{safe_name}.wav")
    preprocess_reference_audio(source_audio_path, target_path)
    return target_path
