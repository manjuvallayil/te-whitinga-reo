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

MODE_DESCRIPTIONS = {
    "TRANSLATE": "Direct equivalence — safe to convert between languages",
    "INTERPRET": "Cultural meaning requires contextual interpretation",
    "EXPLAIN": "Audience needs cultural background to understand",
    "PRESERVE": "Concept must remain in its original language",
    "ESCALATE": "Requires authorised human cultural judgement",
}


def _build_mode_display(active_mode: str, justification: str) -> str:
    """Build HTML showing all modes with the active one highlighted."""
    lines = []
    lines.append("<div style='margin-bottom: 1.5rem;'>")
    lines.append("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem; font-weight: 600;'>Cultural Response Mode</p>")
    lines.append("<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;'>")

    for mode in RESPONSE_MODES:
        color = MODE_COLORS[mode]
        icon = MODE_ICONS[mode]
        is_active = mode == active_mode

        if is_active:
            style = (
                f"background: {color}; color: white; "
                "padding: 0.5rem 1rem; border-radius: 0.5rem; "
                "font-weight: 700; font-size: 0.9rem; "
                "box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);"
            )
        else:
            style = (
                "background: #f9fafb; color: #9ca3af; "
                "padding: 0.5rem 1rem; border-radius: 0.5rem; "
                "font-weight: 400; font-size: 0.9rem; "
                "border: 1px solid #e5e7eb;"
            )

        lines.append(f"<span style='{style}'>{icon} {mode}</span>")

    lines.append("</div>")

    # Active mode description
    if active_mode:
        active_color = MODE_COLORS.get(active_mode, "#374151")
        mode_desc = MODE_DESCRIPTIONS.get(active_mode, "")
        lines.append(
            f"<div style='background: #f9fafb; border-left: 3px solid {active_color}; "
            f"padding: 0.75rem 1rem; border-radius: 0 0.5rem 0.5rem 0; margin-bottom: 0.75rem;'>"
            f"<p style='color: {active_color}; font-weight: 600; font-size: 0.85rem; margin: 0 0 0.25rem 0;'>"
            f"{MODE_ICONS[active_mode]} {active_mode}: {mode_desc}</p>"
        )
        if justification:
            lines.append(
                f"<p style='color: #4b5563; font-size: 0.85rem; font-style: italic; margin: 0;'>"
                f"↳ {justification}</p>"
            )
        lines.append("</div>")

    lines.append("</div>")
    return "\n".join(lines)


def _build_pipeline_status(stage: str) -> str:
    """Build a visual pipeline indicator."""
    stages = [
        ("🎤", "Listen", "asr"),
        ("🧠", "Decide", "decide"),
        ("💬", "Respond", "respond"),
        ("🔊", "Speak", "tts"),
    ]
    parts = []
    parts.append("<div style='display: flex; align-items: center; gap: 0.25rem; margin-bottom: 1rem;'>")
    for icon, label, key in stages:
        is_active = key == stage
        is_done = stages.index((icon, label, key)) < [s[2] for s in stages].index(stage) if stage else False
        if is_active:
            style = "background: #059669; color: white; padding: 0.3rem 0.6rem; border-radius: 1rem; font-size: 0.75rem; font-weight: 600;"
        elif is_done:
            style = "background: #d1fae5; color: #065f46; padding: 0.3rem 0.6rem; border-radius: 1rem; font-size: 0.75rem;"
        else:
            style = "background: #f3f4f6; color: #9ca3af; padding: 0.3rem 0.6rem; border-radius: 1rem; font-size: 0.75rem;"
        parts.append(f"<span style='{style}'>{icon} {label}</span>")
        if key != "tts":
            parts.append("<span style='color: #d1d5db;'>→</span>")
    parts.append("</div>")
    return "".join(parts)


def process_audio(audio):
    """Full pipeline: ASR -> Cultural AI -> TTS."""
    if audio is None:
        return (
            "<p style='color: #dc2626; font-size: 0.9rem;'>No audio recorded. Please record your voice first.</p>",
            "",
            "",
            "",
            None,
        )

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
    transcription_text = f"**{lang_display}:** {asr_result['text']}"

    # Step 2: Cultural AI
    ai_result = generate_response(asr_result["text"], asr_result["language"])
    mode = ai_result["mode"]
    justification = ai_result.get("justification", "")

    # Build mode display HTML
    mode_html = _build_mode_display(mode, justification)

    # Build pipeline status
    pipeline_html = _build_pipeline_status("tts")

    # Step 3: TTS
    tts_lang = "mi" if asr_result["language"] == "mi" else "en"
    audio_out, out_sr = synthesize(ai_result["text"], language=tts_lang)

    return pipeline_html, mode_html, ai_result["text"], transcription_text, (out_sr, audio_out)


# --- Default mode display (all inactive) ---
DEFAULT_MODE_HTML = """
<div style='margin-bottom: 1.5rem;'>
<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem; font-weight: 600;'>Cultural Response Mode</p>
<div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;'>
<span style='background: #f9fafb; color: #9ca3af; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; border: 1px solid #e5e7eb;'>🔄 TRANSLATE</span>
<span style='background: #f9fafb; color: #9ca3af; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; border: 1px solid #e5e7eb;'>🌊 INTERPRET</span>
<span style='background: #f9fafb; color: #9ca3af; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; border: 1px solid #e5e7eb;'>💡 EXPLAIN</span>
<span style='background: #f9fafb; color: #9ca3af; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; border: 1px solid #e5e7eb;'>🛡️ PRESERVE</span>
<span style='background: #f9fafb; color: #9ca3af; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 0.9rem; border: 1px solid #e5e7eb;'>⚠️ ESCALATE</span>
</div>
<p style='color: #9ca3af; font-size: 0.85rem; font-style: italic;'>Awaiting input — the AI will determine the culturally appropriate response mode.</p>
</div>
"""

# --- Gradio UI ---
css = """
.main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.main-header h1 { font-size: 2rem; color: #1f2937; margin: 0; }
.main-header .tagline { color: #6b7280; font-size: 0.95rem; margin-top: 0.25rem; }
.main-header .desc { color: #9ca3af; font-size: 0.8rem; margin-top: 0.5rem; max-width: 600px; margin-left: auto; margin-right: auto; }
footer { display: none !important; }
"""

with gr.Blocks(
    css=css,
    title="Te Whitinga Reo",
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="gray"),
) as demo:

    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>Te Whitinga Reo</h1>
        <p class="tagline">Sovereign Culturally Intelligent AI</p>
        <p class="desc">Speak in te reo Māori or English. The AI determines whether to translate, interpret, explain, preserve, or escalate — based on cultural context, meaning, and sensitivity.</p>
    </div>
    """)

    # Pipeline status
    pipeline_display = gr.HTML(
        value=_build_pipeline_status(""),
        show_label=False,
    )

    # Main layout: Input left, Decision + Response right
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300):
            audio_input = gr.Audio(
                label="Kōrero mai / Speak",
                sources=["microphone"],
                type="numpy",
            )
            submit_btn = gr.Button(
                "Process",
                variant="primary",
                size="lg",
                interactive=True,
            )
            # Collapsible transcription
            with gr.Accordion("Transcription", open=False):
                transcription_output = gr.Markdown(
                    value="*Transcription will appear here after processing.*",
                )

        with gr.Column(scale=3, min_width=400):
            # Cultural Decision Area — the hero
            mode_display = gr.HTML(value=DEFAULT_MODE_HTML)

            # Response
            response_output = gr.Textbox(
                label="Response",
                lines=4,
                interactive=False,
                placeholder="The AI's culturally informed response will appear here...",
            )

            # Audio playback
            audio_output = gr.Audio(
                label="Audio Response",
                type="numpy",
                autoplay=True,
            )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[pipeline_display, mode_display, response_output, transcription_output, audio_output],
    )

    # Footer
    gr.HTML("""
    <div style='text-align: center; padding: 1.5rem 0 0.5rem; border-top: 1px solid #e5e7eb; margin-top: 2rem;'>
        <p style='color: #9ca3af; font-size: 0.75rem; margin: 0;'>Te Whitinga Reo — Adaptive AI for Global Communication, Commerce and Cultural Exchange</p>
        <p style='color: #d1d5db; font-size: 0.7rem; margin-top: 0.25rem;'>Sovereign Culturally Intelligent Agentic AI | Aotearoa New Zealand</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
