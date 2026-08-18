import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import spaces

from config import MODEL_ASR, SAMPLE_RATE

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = WhisperProcessor.from_pretrained(MODEL_ASR)
        _model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ASR, torch_dtype=torch.float16
        )
        _model.to("cuda")
    return _processor, _model


@spaces.GPU
def transcribe(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Transcribe audio to text using Whisper fine-tuned for Māori.

    Returns dict with 'text' and 'language' keys.
    """
    processor, model = _load_model()

    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Normalize audio
    if np.abs(audio_array).max() > 1.0:
        audio_array = audio_array / np.abs(audio_array).max()

    input_features = processor(
        audio_array, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(model.device, dtype=torch.float16)

    # Generate with language detection
    predicted_ids = model.generate(
        input_features,
        max_new_tokens=128,
        language="mi",  # Default to Māori
        task="transcribe",
    )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Simple language detection heuristic
    maori_markers = ["kia", "te", "nga", "ngā", "wh", "kōrero", "āe", "kei"]
    text_lower = transcription.lower()
    is_maori = any(marker in text_lower for marker in maori_markers)
    detected_lang = "mi" if is_maori else "en"

    return {"text": transcription.strip(), "language": detected_lang}
