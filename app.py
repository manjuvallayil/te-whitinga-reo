import io
import streamlit as st
import numpy as np
import soundfile as sf
import librosa

from modules.asr import transcribe
from modules.cultural_ai import generate_response, RESPONSE_MODES
from modules.tts import synthesize, TTS_SAMPLE_RATE

# --- Page config ---
st.set_page_config(
    page_title="Te Whitinga Reo",
    page_icon="🌿",
    layout="centered",
)

# --- Custom styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .mode-translate { background: #d1fae5; color: #065f46; }
    .mode-interpret { background: #dbeafe; color: #1e40af; }
    .mode-explain { background: #fef3c7; color: #92400e; }
    .mode-preserve { background: #ede9fe; color: #5b21b6; }
    .mode-escalate { background: #fee2e2; color: #991b1b; }
    .response-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .lang-tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        background: #f3f4f6;
        color: #374151;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>Te Whitinga Reo</h1>
    <p>Sovereign Culturally Intelligent AI</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Audio input ---
st.markdown("#### Kōrero mai / Speak")
audio_input = st.audio_input(
    "Record your voice in te reo Māori or English",
    key="audio_recorder",
)

# --- Processing pipeline ---
if audio_input is not None:
    # Read audio bytes into numpy array
    audio_bytes = audio_input.getvalue()
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16kHz for Whisper
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    # Step 1: ASR
    with st.status("Whakarongo ana... / Listening...", expanded=True) as status:
        st.write("Transcribing speech...")
        asr_result = transcribe(audio_array, sample_rate=16000)
        status.update(label="Transcription complete", state="complete")

    # Show transcription
    lang_display = "te reo Māori" if asr_result["language"] == "mi" else "English"
    st.markdown(f"**Transcription** <span class='lang-tag'>{lang_display}</span>",
                unsafe_allow_html=True)
    st.info(asr_result["text"])

    # Step 2: Cultural AI response
    with st.status("Whakaaro ana... / Thinking...", expanded=True) as status:
        st.write("Generating culturally intelligent response...")
        ai_result = generate_response(asr_result["text"], asr_result["language"])
        status.update(label="Response generated", state="complete")

    # Show response with mode badge
    mode = ai_result["mode"]
    mode_class = f"mode-{mode.lower()}"
    mode_desc = ai_result["mode_description"]

    st.markdown(f"""
    <div class="response-card">
        <span class="mode-badge {mode_class}">{mode}</span>
        <small style="color: #6b7280; margin-left: 0.5rem;">{mode_desc}</small>
        <p style="margin-top: 1rem; font-size: 1.05rem; line-height: 1.6;">
            {ai_result['text']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Step 3: TTS
    with st.status("Kōrero ana... / Speaking...", expanded=True) as status:
        st.write("Generating speech...")
        tts_lang = "mi" if asr_result["language"] == "mi" else "en"
        audio_out, out_sr = synthesize(ai_result["text"], language=tts_lang)
        status.update(label="Audio ready", state="complete")

    # Play audio
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_out, out_sr, format="WAV")
    audio_buffer.seek(0)
    st.audio(audio_buffer, format="audio/wav", autoplay=True)

# --- Footer ---
st.divider()
st.markdown(
    "<p style='text-align: center; color: #9ca3af; font-size: 0.8rem;'>"
    "Te Whitinga Reo — Adaptive AI for Cultural Communication"
    "</p>",
    unsafe_allow_html=True,
)
