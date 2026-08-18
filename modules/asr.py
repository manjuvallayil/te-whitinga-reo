import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import spaces

from config import MODEL_ASR, SAMPLE_RATE

_processor = None
_model = None
_base_processor = None
_base_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = WhisperProcessor.from_pretrained(MODEL_ASR)
        _model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ASR, torch_dtype=torch.float16
        )
        _model.to("cuda")
    return _processor, _model


def _load_base_model():
    """Load base whisper-small for language detection and English transcription."""
    global _base_processor, _base_model
    if _base_model is None:
        _base_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        _base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", torch_dtype=torch.float16
        )
        _base_model.to("cuda")
    return _base_processor, _base_model


def _detect_language(input_features, model, processor) -> str:
    """Use Whisper's built-in language detection."""
    # Generate just the language token
    decoder_input_ids = torch.tensor([[50258]]).to(model.device)  # <|startoftranscript|>
    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=decoder_input_ids).logits
    # Get predicted language token (tokens 50259-50357 are language tokens)
    lang_logits = logits[0, 0, 50259:50357]
    predicted_lang_idx = lang_logits.argmax().item()
    # Māori is typically around index 80+ in Whisper's language list
    # English is index 0
    if predicted_lang_idx == 0:
        return "en"
    # For simplicity, if not English, assume Māori (since that's our context)
    return "mi"


@spaces.GPU
def transcribe(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Transcribe audio with automatic language detection.

    Uses base whisper-small for language detection.
    Uses fine-tuned whisper-small-mi for Māori transcription.
    Uses base whisper-small for English transcription.

    Returns dict with 'text' and 'language' keys.
    """
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Normalize audio
    if np.abs(audio_array).max() > 1.0:
        audio_array = audio_array / np.abs(audio_array).max()

    # Load base model for language detection
    base_processor, base_model = _load_base_model()

    input_features_base = base_processor(
        audio_array, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(base_model.device, dtype=torch.float16)

    # Detect language using base model
    detected_lang = _detect_language(input_features_base, base_model, base_processor)

    if detected_lang == "mi":
        # Use fine-tuned Māori model
        processor, model = _load_model()
        input_features = processor(
            audio_array, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(model.device, dtype=torch.float16)

        predicted_ids = model.generate(
            input_features,
            max_new_tokens=128,
            language="mi",
            task="transcribe",
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    else:
        # Use base model for English
        predicted_ids = base_model.generate(
            input_features_base,
            max_new_tokens=128,
            language="en",
            task="transcribe",
        )
        transcription = base_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {"text": transcription.strip(), "language": detected_lang}
