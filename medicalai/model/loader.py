"""
model/loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  - Loading base model with 4-bit QLoRA quantization
  - Applying LoRA adapter config (for training)
  - Loading saved adapter weights (for inference)
  - Tokenizer setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_ID,
    ADAPTER_PATH,
    HF_TOKEN,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGETS,
)


def get_bnb_config():
    """
    4-bit NF4 quantization config.
    Makes 8B model fit in free Colab T4 (15GB VRAM).

    load_in_4bit          → store weights as 4-bit instead of 16-bit (4× smaller)
    bnb_4bit_quant_type   → NF4 best matches neural weight distribution
    bnb_4bit_compute_dtype→ compute in bfloat16 for precision during math
    bnb_4bit_use_double_quant → quantize the quantization constants too (free saving)
    """
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )


def load_tokenizer():
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token            = HF_TOKEN,
        trust_remote_code= True,
    )
    # pad_token must be set — prevents silent generation errors
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right padding = stable for training
    return tokenizer


def load_base_model():
    """
    Load base LLaMA model with 4-bit quantization.
    Used as starting point for both training and inference.
    """
    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = get_bnb_config(),
        device_map          = "auto",       # auto-distributes across GPU/CPU
        token               = HF_TOKEN,
        trust_remote_code   = True,
    )
    # Disable KV cache during training — incompatible with gradient checkpointing
    model.config.use_cache = False
    print(f"✅ Base model loaded — {model.num_parameters():,} parameters")
    return model


def apply_lora(model):
    """
    Wraps the base model with LoRA adapter layers for training.

    Steps:
    1. prepare_model_for_kbit_training → enables gradient checkpointing
       and casts non-4bit layers to float32 for stable training
    2. LoraConfig → defines A and B matrix sizes and where to attach them
    3. get_peft_model → freezes all original weights, adds A+B as trainable

    After this: only ~0.6% of parameters are trainable.
    """
    print("Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r              = LORA_R,        # rank — A is (d × r), B is (r × d)
        lora_alpha     = LORA_ALPHA,    # scaling = alpha/r = 32/16 = 2
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        target_modules = LORA_TARGETS,  # 7 projection layers targeted
    )

    model = get_peft_model(model, lora_config)

    # Shows how few parameters actually train
    # Expected output: trainable params: ~20M || all params: ~8B → ~0.6%
    model.print_trainable_parameters()

    return model


def load_model_for_inference():
    """
    Loads fine-tuned model for the Gradio app.
    Loads base model first, then loads saved LoRA adapter weights on top.
    Uses merge_and_unload() to combine W + A×B into single matrix —
    eliminates LoRA overhead at inference time.
    """
    print("Loading model for inference...")

    tokenizer = load_tokenizer()
    model     = load_base_model()

    print(f"Loading LoRA adapters from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Merge adapter weights into base model — faster inference, no overhead
    model = model.merge_and_unload()
    model.eval()  # disable dropout — inference mode

    print("✅ Model ready for inference!")
    return model, tokenizer
