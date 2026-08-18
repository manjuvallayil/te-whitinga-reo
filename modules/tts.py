import asyncio
import io
import numpy as np
import soundfile as sf
import edge_tts

TTS_SAMPLE_RATE = 24000

# Voice mapping
VOICES = {
    "mi": "en-NZ-MitchellNeural",  # NZ English voice (closest to Māori prosody)
    "en": "en-NZ-MollyNeural",     # NZ English female voice
}


async def _synthesize_async(text: str, language: str = "mi") -> tuple[np.ndarray, int]:
    """Generate speech using edge-tts."""
    voice = VOICES.get(language, VOICES["en"])

    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    if not audio_bytes:
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32), TTS_SAMPLE_RATE

    # edge-tts returns mp3; decode to numpy
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))

    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    return audio_array.astype(np.float32), sr


def synthesize(text: str, language: str = "mi") -> tuple[np.ndarray, int]:
    """Synthesize speech from text using edge-tts (Microsoft).

    Returns (audio_array, sample_rate).
    """
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_synthesize_async(text, language))
    finally:
        loop.close()
    return result
