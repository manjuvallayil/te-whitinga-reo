import io
import gradio as gr
import numpy as np
import soundfile as sf
import librosa

from modules.asr import transcribe
from modules.cultural_ai import generate_response, RESPONSE_MODES
from modules.tts import synthesize, TTS_SAMPLE_RATE

MODE_COLORS = {
    "TRANSLATE": "#059669",
    "INTERPRET": "#2563eb",
    "EXPLAIN": "#d97706",
    "PRESERVE": "#7c3aed",
    "ESCALATE": "#dc2626",
}

MODE_ICONS = {
    "TRANSLATE": "🔄",
    "INTERPRET": "🌊",
    "EXPLAIN": "💡",
    "PRESERVE": "🛡️",
    "ESCALATE": "⚠️",
}


def _build_mode_display(active_mode: str, justification: str) -> str:
    """Build HTML showing all modes with the active one highlighted."""
    lines = []
    lines.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;'>")

    for mode, description in RESPONSE_MODES.items():
        color = MODE_COLORS[mode]
        icon = MODE_ICONS[mode]
        is_active = mode == active_mode

        if is_active:
            style = (
                f"background: {color}; color: white; "
                "padding: 0.4rem 0.8rem; border-radius: 0.5rem; "
                "font-weight: 700; font-size: 0.85rem; "
                "border: 2px solid transparent;"
            )
        else:
            style = (
                f"background: transparent; color: #9ca3af; "
                "padding: 0.4rem 0.8rem; border-radius: 0.5rem; "
                "font-weight: 400; font-size: 0.85rem; "
                f"border: 1px solid #e5e7eb;"
            )

        lines.append(f"<span style='{style}'>{icon} {mode}</span>")

    lines.append("</div>")

    # Justification note
    if justification:
        active_color = MODE_COLORS.get(active_mode, "#374151")
        lines.append(
            f"<p style='color: {active_color}; font-size: 0.9rem; "
            f"font-style: italic; margin: 0; padding: 0.5rem 0;'>"
            f"↳ {justification}</p>"
        )

    return "\n".join(lines)


def process_audio(audio):
    """Full pipeline: ASR -> Cultural AI -> TTS."""
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
    justification = ai_result.get("justification", "")

    # Build mode display HTML
    mode_html = _build_mode_display(mode, justification)

    # Step 3: TTS
    tts_lang = "mi" if asr_result["language"] == "mi" else "en"
    audio_out, out_sr = synthesize(ai_result["text"], language=tts_lang)

    return transcription_text, mode_html, ai_result["text"], (out_sr, audio_out)


# --- Gradio UI ---
css = """
.main-title { text-align: center; margin-bottom: 0.2rem; }
.subtitle { text-align: center; color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; }
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
            mode_output = gr.HTML(
                label="Cultural Response Mode",
                value=(
                    "<div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>"
                    "<span style='background: transparent; color: #9ca3af; padding: 0.4rem 0.8rem; "
                    "border-radius: 0.5rem; font-size: 0.85rem; border: 1px solid #e5e7eb;'>🔄 TRANSLATE</span>"
                    "<span style='background: transparent; color: #9ca3af; padding: 0.4rem 0.8rem; "
                    "border-radius: 0.5rem; font-size: 0.85rem; border: 1px solid #e5e7eb;'>🌊 INTERPRET</span>"
                    "<span style='background: transparent; color: #9ca3af; padding: 0.4rem 0.8rem; "
                    "border-radius: 0.5rem; font-size: 0.85rem; border: 1px solid #e5e7eb;'>💡 EXPLAIN</span>"
                    "<span style='background: transparent; color: #9ca3af; padding: 0.4rem 0.8rem; "
                    "border-radius: 0.5rem; font-size: 0.85rem; border: 1px solid #e5e7eb;'>🛡️ PRESERVE</span>"
                    "<span style='background: transparent; color: #9ca3af; padding: 0.4rem 0.8rem; "
                    "border-radius: 0.5rem; font-size: 0.85rem; border: 1px solid #e5e7eb;'>⚠️ ESCALATE</span>"
                    "</div>"
                ),
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
        outputs=[transcription_output, mode_output, response_output, audio_output],
    )

    gr.Markdown(
        "<p style='text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 2rem;'>"
        "Te Whitinga Reo — Adaptive AI for Cultural Communication</p>"
    )

if __name__ == "__main__":
    demo.launch()
