import threading
from typing import Optional

# Import the original classes (avoid circular import issues by importing inside functions)

_whisper_instance = None
_whisper_lock = threading.Lock()

_xtts_instance = None
_xtts_lock = threading.Lock()

def get_whisper(model_size: str = "medium", device: str = "cpu", compute_type: str = "int8"):
    """Return a cached WhisperTranscriber instance.

    The first call creates the instance and loads the model. Subsequent calls
    reuse the same object, avoiding re‑loading overhead.
    """
    global _whisper_instance
    with _whisper_lock:
        if _whisper_instance is None:
            # Local import to avoid circular import with ``core.transcriber``
            from .transcriber import WhisperTranscriber
            _whisper_instance = WhisperTranscriber(model_size, device, compute_type)
            _whisper_instance.load_model()
        return _whisper_instance

def get_xtts(model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: Optional[str] = None):
    """Return a cached XTTSEngine instance.

    The model is loaded lazily on first use. ``device`` defaults to CUDA if
    available, otherwise CPU.
    """
    global _xtts_instance
    with _xtts_lock:
        if _xtts_instance is None:
            from .tts_engine import XTTSEngine
            _xtts_instance = XTTSEngine(model_name, device)
            _xtts_instance.load_model()
        return _xtts_instance
