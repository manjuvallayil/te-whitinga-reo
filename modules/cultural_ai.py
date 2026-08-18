import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import spaces

from config import MODEL_GEN_AI, CULTURAL_SYSTEM_PROMPT, MAX_NEW_TOKENS

_tokenizer = None
_model = None

RESPONSE_MODES = {
    "TRANSLATE": "Translating between languages with cultural equivalence",
    "INTERPRET": "Interpreting meaning beyond literal translation",
    "EXPLAIN": "Providing cultural context for understanding",
    "PRESERVE": "Preserving concept in its original language",
    "ESCALATE": "Recommending human cultural expertise",
}


def _load_model():
    global _tokenizer, _model
    if _model is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_GEN_AI)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_GEN_AI,
            quantization_config=quantization_config,
            device_map="auto",
        )
    return _tokenizer, _model


def _parse_response(text: str) -> tuple[str, str, str]:
    """Extract mode, justification, and clean response text."""
    # Extract mode
    mode_pattern = r"\[(TRANSLATE|INTERPRET|EXPLAIN|PRESERVE|ESCALATE)\]"
    match = re.search(mode_pattern, text)
    if match:
        mode = match.group(1)
        remainder = text[match.end():].strip()
    else:
        mode = "INTERPRET"
        remainder = text.strip()

    # Extract justification
    justification = ""
    just_pattern = r"(?i)JUSTIFICATION:\s*(.+?)(?:\n|$)"
    just_match = re.search(just_pattern, remainder)
    if just_match:
        justification = just_match.group(1).strip()
        remainder = remainder[just_match.end():].strip()

    # Clean up artifacts from model output
    # Remove "Line 3:" or "LINE 3:" prefixes
    remainder = re.sub(r"(?i)^line\s*\d+:\s*", "", remainder).strip()
    # Remove any remaining format artifacts
    remainder = re.sub(r"(?i)^response:\s*", "", remainder).strip()

    return mode, justification, remainder


@spaces.GPU
def generate_response(user_text: str, detected_language: str = "mi") -> dict:
    """Generate a culturally intelligent response.

    Returns dict with 'mode', 'mode_description', 'text' keys.
    """
    tokenizer, model = _load_model()

    lang_context = (
        "The user is speaking te reo Māori. Respond in te reo Māori."
        if detected_language == "mi"
        else "The user is speaking English. You MUST respond in English. Use Māori terms only for culturally significant concepts that should not be translated (e.g., mana, kaitiaki, whakapapa). The rest of your response MUST be in English."
    )

    messages = [
        {"role": "user", "content": f"{CULTURAL_SYSTEM_PROMPT}\n\nCRITICAL LANGUAGE INSTRUCTION: {lang_context}\n\nUser: {user_text}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode only the generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    mode, justification, clean_text = _parse_response(response_text)

    return {
        "mode": mode,
        "mode_description": RESPONSE_MODES.get(mode, ""),
        "justification": justification,
        "text": clean_text,
        "raw": response_text,
    }
