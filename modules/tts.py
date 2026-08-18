import numpy as np
import torch
import spaces
from omnivoice import OmniVoice

from config import MODEL_TTS

_model = None

TTS_SAMPLE_RATE = 24000


def _load_model():
    global _model
    if _model is None:
        _model = OmniVoice.from_pretrained(
            MODEL_TTS,
            device_map="cuda:0",
            dtype=torch.float16,
        )
    return _model


@spaces.GPU
def synthesize(text: str, language: str = "mi") -> tuple[np.ndarray, int]:
    """Synthesize speech from text using OmniVoice.

    Uses voice design mode (no reference audio needed).
    Returns (audio_array, sample_rate).
    """
    model = _load_model()

    audio = model.generate(
        text=text,
        language=language,
    )

    audio_array = audio[0] if isinstance(audio, list) else audio
    return audio_array, TTS_SAMPLE_RATE
