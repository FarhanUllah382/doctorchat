"""
model/inference.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  - Building the prompt from user message
  - Generating model response (the 4 jobs: format → tokenize → generate → decode)
  - Emergency keyword detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SYSTEM_PROMPT,
    EMERGENCY_KEYWORDS,
    DISCLAIMER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_NEW_TOKENS,
    REPETITION_PENALTY,
)


def check_emergency(message: str) -> bool:
    """
    Scans message for life-threatening keywords.
    If found → skip model generation, return emergency response instead.
    This runs BEFORE the model — safety first.
    """
    msg_lower = message.lower()
    return any(keyword in msg_lower for keyword in EMERGENCY_KEYWORDS)


def build_prompt(user_message: str) -> str:
    """
    Wraps user message in the same format used during training.
    CRITICAL: this format must exactly match format_medical_prompt() in data/prepare.py.
    If formats differ → model produces garbage output.

    ### Doctor: is left empty → model fills this in during generation.
    """
    return f"""### System:
{SYSTEM_PROMPT}

### Patient:
{user_message}

### Doctor:
"""


def generate_response(
    message   : str,
    model,
    tokenizer,
    temperature    = DEFAULT_TEMPERATURE,
    top_p          = DEFAULT_TOP_P,
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Core generation function. The 4 jobs:

    JOB 1 — FORMAT   : build prompt in training format
    JOB 2 — TOKENIZE : text → token IDs → move to GPU
    JOB 3 — GENERATE : model predicts new tokens
    JOB 4 — DECODE   : new token IDs → human readable text
    """

    # JOB 1 — FORMAT
    prompt = build_prompt(message)

    # JOB 2 — TOKENIZE
    # truncation=True → clips input if > max_length (prevents OOM)
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1024,
    ).to("cuda")

    # JOB 3 — GENERATE
    with torch.no_grad():  # no gradient tracking — inference only
        outputs = model.generate(
            **inputs,
            max_new_tokens    = max_new_tokens,
            temperature       = temperature,
            top_p             = top_p,          # nucleus sampling — considers tokens
                                                 # until cumulative prob reaches top_p
            do_sample         = True,
            repetition_penalty= REPETITION_PENALTY,  # penalises repeating phrases
            pad_token_id      = tokenizer.eos_token_id,
        )

    # JOB 4 — DECODE only new tokens
    # outputs[0]                   = full sequence (input + response tokens)
    # inputs['input_ids'].shape[1] = length of input tokens
    # Slicing from that point → only the generated response
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def get_emergency_response() -> str:
    """Returns a fixed emergency response — no model involved."""
    return (
        "🚨 This sounds like a medical emergency!\n\n"
        "Please call emergency services immediately:\n"
        "• Pakistan: 115 or 1122\n"
        "• Or go to the nearest emergency room right away!\n\n"
        "Do not wait — please seek immediate medical help!"
    )
