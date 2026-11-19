import os
from typing import Optional, Dict, Any

# Singleton cache for Whisper model
from .singleton import get_whisper

class WhisperTranscriber:
    """Transcribe speech using faster-whisper or openai-whisper as a fallback.

    Improved: default model size set to 'medium' for higher quality. Exposes
    optional beam_size and vad_filter parameters for fine‑tuning. Uses a
    singleton to avoid re‑loading the model on each call, reducing latency.
    """

    def __init__(self, model_size: str = "medium", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        # Model will be loaded lazily via singleton
        self.model = None
        self.backend = None

    def load_model(self):
        """Lazy load a Whisper backend and handle the common ModuleNotFoundError case.

        The model instance is cached globally via ``get_whisper`` so subsequent
        calls share the same loaded model.
        """
        if self.model is None:
            # Use the global singleton cache – it will create and store a model
            # with the requested configuration if it does not already exist.
            return get_whisper(self.model_size, self.device, self.compute_type)
        return self.model

    def transcribe(self, audio_path: str, language: Optional[str] = None, *, beam_size: int = 5, vad_filter: bool = True) -> Dict[str, Any]:
        """Transcribe an audio file into text.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., "en", "fr"). ``None`` lets the model
                auto‑detect.
            beam_size: Beam search size for Faster‑Whisper (higher = better
                accuracy, slower). Default 5.
            vad_filter: Whether to apply voice‑activity‑detection filtering.
                Default True.
        Returns:
            dict with keys: ``text``, ``language``, ``language_probability`` and ``duration``.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self.load_model()

        if self.backend == "faster_whisper":
            segments, info = model.transcribe(
                audio_path,
                language=language if language and language != "auto" else None,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            full_text = " ".join(segment.text.strip() for segment in segments).strip()
            return {
                "text": full_text,
                "language": info.language,
                "language_probability": round(float(info.language_probability), 2),
                "duration": round(float(info.duration), 2),
            }

        # Fallback to openai‑whisper (no beam/VAD support)
        result = model.transcribe(
            audio_path,
            language=language if language and language != "auto" else None,
            fp16=False,
        )
        segments = result.get("segments", [])
        full_text = " ".join((seg.get("text", "") or "").strip() for seg in segments).strip()
        if not full_text:
            full_text = (result.get("text") or "").strip()

        language_name = result.get("language") or "en"
        language_probability = result.get("language_probability", 1.0)
        duration = result.get("duration", 0.0)

        return {
            "text": full_text,
            "language": language_name,
            "language_probability": round(float(language_probability), 2),
            "duration": round(float(duration), 2),
        }
