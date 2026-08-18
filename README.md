---
title: Te Whitinga Reo
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
hardware: zero-a10g
---

# Te Whitinga Reo

**Sovereign Culturally Intelligent Agentic AI** for adaptive communication across te reo Māori and English.

## Overview

Te Whitinga Reo is a prototype demonstrator for culturally intelligent AI that goes beyond simple translation. Rather than automatically translating every interaction, the AI determines the most appropriate culturally and contextually informed action:

| Mode | Description |
|------|-------------|
| **TRANSLATE** | Linguistic and cultural equivalence is sufficiently reliable |
| **INTERPRET** | Literal translation would not adequately preserve meaning or intent |
| **EXPLAIN** | Cultural context is required for the receiving audience |
| **PRESERVE** | A culturally significant term or concept should remain in its original language |
| **ESCALATE** | Ambiguity, cultural risk, or potential consequences require human judgement |

## How It Works

```
User speaks (Māori or English)
        │
        ▼
┌─────────────────┐
│ ASR (Whisper)   │  → Transcribes speech, detects language
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Cultural Intelligence Layer │  → AI decides response mode
│ (BYOL-MRI-4B)              │  → Generates culturally aware response
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│ TTS (OmniVoice) │  → Speaks the response aloud
└─────────────────┘
```

## Tech Stack

| Component | Model | Size | Role |
|-----------|-------|------|------|
| ASR | [TheDevilSaid/whisper-small-mi](https://huggingface.co/TheDevilSaid/whisper-small-mi) | 0.2B | Speech-to-text, fine-tuned for te reo Māori |
| Gen AI | [ai-for-good-lab/byol-mri-4b-merged](https://huggingface.co/ai-for-good-lab/byol-mri-4b-merged) | 4B (4-bit quantized) | Culturally intelligent response generation |
| TTS | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 0.6B | Multilingual text-to-speech (600+ languages) |

## Project Structure

```
te-whitinga-reo/
├── app.py                    # Main Gradio application
├── config.py                 # Model IDs, cultural system prompt, settings
├── modules/
│   ├── asr.py                # Speech recognition (Whisper-small-mi)
│   ├── cultural_ai.py        # Generative AI with cultural decision framework
│   └── tts.py                # Text-to-speech (OmniVoice)
└── requirements.txt          # Python dependencies
```

## Deployment

Hosted on HuggingFace Spaces with ZeroGPU (dynamic Nvidia GPU allocation).

**Cost**: ~$9/month (HuggingFace Pro account for ZeroGPU access).

### To deploy:

```bash
# Clone this repo
git clone https://github.com/manjuvallayil/te-whitinga-reo.git
cd te-whitinga-reo

# Push to HuggingFace Space
huggingface-cli repo create te-whitinga-reo --type space --space-sdk streamlit --space-hardware zero-a10g
git remote add hf https://huggingface.co/spaces/<your-hf-username>/te-whitinga-reo
git push hf main
```

### Local development:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Note: Local development requires a CUDA GPU for the models to load.

## Cultural Intelligence Design

The system prompt encodes the Te Whitinga Reo decision framework. The AI does not simply translate — it assesses:

- **What** is being communicated (linguistic content)
- **How** it should be communicated (cultural form)
- **Whether** it should be communicated (permissions, sensitivity)
- **To whom** and in what relational context

This aligns with the programme's vision of bounded cultural agency, where AI operates within explicit authority, knowledge, and sovereignty constraints.

## Roadmap

- [ ] Fine-tune TTS on [shunyalabs/maori-speech-dataset](https://huggingface.co/datasets/shunyalabs/maori-speech-dataset) for improved Māori voice quality
- [ ] Add WITHHOLD mode for culturally restricted knowledge
- [ ] Conversation history and contextual awareness
- [ ] Reference audio support for voice cloning in TTS
- [ ] Evaluation harness aligned with Cultural AI Assurance framework

## Acknowledgements

Built as part of the Te Whitinga Reo research programme — Sovereign Culturally Intelligent Agentic AI for Aotearoa New Zealand.
