import io
import gradio as gr
import numpy as np
import soundfile as sf
import librosa

from modules.asr import transcribe
from modules.cultural_ai import generate_response, RESPONSE_MODES
from modules.tts import synthesize, TTS_SAMPLE_RATE

MODE_COLORS = {
    "TRANSLATE": "#065f46",
    "INTERPRET": "#1e40af",
    "EXPLAIN": "#92400e",
    "PRESERVE": "#5b21b6",
    "ESCALATE": "#991b1b",
}


def process_audio(audio):
    """Full pipeline: ASR → Cultural AI → TTS."""
    if audio is None:
        return "No audio recorded.", "", "", None

    sample_rate, audio_array = audio

    # Convert to float32
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / 32768.0

    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16kHz for Whisper
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    # Step 1: ASR
    asr_result = transcribe(audio_array, sample_rate=16000)
    lang_display = "te reo Māori" if asr_result["language"] == "mi" else "English"
    transcription_text = f"{asr_result['text']}\n\n[Detected: {lang_display}]"

    # Step 2: Cultural AI
    ai_result = generate_response(asr_result["text"], asr_result["language"])
    mode = ai_result["mode"]
    mode_desc = ai_result["mode_description"]
    response_text = f"[{mode}] {mode_desc}\n\n{ai_result['text']}"

    # Step 3: TTS
    tts_lang = "mi" if asr_result["language"] == "mi" else "en"
    audio_out, out_sr = synthesize(ai_result["text"], language=tts_lang)

    return transcription_text, response_text, mode, (out_sr, audio_out)


# --- Gradio UI ---
css = """
.main-title { text-align: center; margin-bottom: 0.5rem; }
.subtitle { text-align: center; color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; }
.mode-badge { font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
"""

with gr.Blocks(
    css=css,
    title="Te Whitinga Reo",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="gray"),
) as demo:

    gr.Markdown(
        "<h1 class='main-title'>Te Whitinga Reo</h1>"
        "<p class='subtitle'>Sovereign Culturally Intelligent AI</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Kōrero mai / Speak",
                sources=["microphone"],
                type="numpy",
            )
            submit_btn = gr.Button("Process", variant="primary", size="lg")

        with gr.Column(scale=1):
            transcription_output = gr.Textbox(
                label="Transcription",
                lines=3,
                interactive=False,
            )
            mode_output = gr.Textbox(
                label="Response Mode",
                lines=1,
                interactive=False,
            )
            response_output = gr.Textbox(
                label="Response",
                lines=5,
                interactive=False,
            )
            audio_output = gr.Audio(
                label="Audio Response",
                type="numpy",
                autoplay=True,
            )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcription_output, response_output, mode_output, audio_output],
    )

    gr.Markdown(
        "<p style='text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 2rem;'>"
        "Te Whitinga Reo — Adaptive AI for Cultural Communication</p>"
    )

if __name__ == "__main__":
    demo.launch()
