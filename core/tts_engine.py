import os
import time
from typing import Optional, Dict, Any
import torch

# Automatically accept Coqui license agreement for non-commercial research/use
os.environ["COQUI_TOS_AGREED"] = "1"

# In PyTorch 2.6+, torch.load defaults weights_only=True, which breaks Coqui TTS checkpoint loading.
# Monkeypatch torch.load to default weights_only=False for TTS models.
_orig_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

# Fix torchaudio 2.11+ torchcodec requirement by routing through soundfile
import soundfile as sf
import torchaudio

def _soundfile_load(filepath, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(data.T).float()
    return tensor, sr

def _soundfile_save(filepath, src, sample_rate, **kwargs):
    if isinstance(src, torch.Tensor):
        src_np = src.detach().cpu().numpy()
        if src_np.ndim == 2:
            src_np = src_np.T
    else:
        src_np = src
    sf.write(filepath, src_np, sample_rate)

torchaudio.load = _soundfile_load
torchaudio.save = _soundfile_save

SUPPORTED_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}

# Import the singleton cache for XTTS
from .singleton import get_xtts

class XTTSEngine:
    """Wrapper around Coqui XTTS‑v2 for zero‑shot voice cloning.

    Improvements:
    - Uses a **singleton cache** (`get_xtts`) so the model loads only once.
    - Supports the **tiny model** (`tts_models/multilingual/multi-dataset/xtts_v2_tiny`) via the `use_tiny` flag for faster inference.
    - Default synthesis parameters are tuned for speed (`speed=1.2`, `temperature=0.5`, `repetition_penalty=2.0`).
    - GPU usage is automatically selected when available.
    """

    def __init__(self,
                 model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: Optional[str] = None,
                 use_tiny: bool = False):
        # If tiny model requested, override the model name
        if use_tiny:
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2_tiny"
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tiny = use_tiny
        # The actual TTS instance will be obtained via the singleton
        self.tts = None

        if self.device == "cpu":
            # Maximize CPU core utilization for torch
            cpu_threads = os.cpu_count() or 4
            torch.set_num_threads(cpu_threads)

    def _ensure_model(self):
        """Retrieve the cached XTTS instance via the singleton.
        The singleton creates the model on first call and re‑uses it afterwards.
        """
        if self.tts is None:
            # The singleton handles loading with the correct device and model name
            self.tts = get_xtts(self.model_name, self.device)
        return self.tts

    def clone_voice(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
        output_path: Optional[str] = None,
        speed: float = 1.2,
        temperature: float = 0.5,
        repetition_penalty: float = 2.0,
    ) -> Dict[str, Any]:
        """Synthesize speech in the cloned voice given reference audio and text.

        Parameters are tuned for **speed** by default (higher `speed`, lower
        `temperature`, reduced `repetition_penalty`). They can be overridden
        by the caller if higher fidelity is required.
        """
        if not text.strip():
            raise ValueError("Text prompt cannot be empty.")
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker audio file not found: {speaker_wav}")

        tts = self._ensure_model()

        if output_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            outputs_dir = os.path.join(project_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            timestamp = int(time.time())
            output_path = os.path.join(outputs_dir, f"xtts_output_{timestamp}.wav")

        start_time = time.time()

        # Call TTS synthesis with faster defaults
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
            temperature=temperature,
            speed=speed,
            repetition_penalty=repetition_penalty,
        )

        elapsed_time = round(time.time() - start_time, 2)

        return {
            "output_path": output_path,
            "elapsed_sec": elapsed_time,
            "language": language,
            "character_count": len(text),
        }
