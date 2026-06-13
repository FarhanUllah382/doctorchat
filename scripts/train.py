"""
train.py
--------
Loads the prepared dataset, applies QLoRA fine-tuning to
LLaMA 3.1-8B, then saves the LoRA adapter.

Run after prepare_data.py:
    python scripts/train.py
"""

import time
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Config ───────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.1-8B"
HF_TOKEN   = "hf_your_token_here"   # ← replace with your token
OUTPUT_DIR = "./model/medical-llama3-checkpoints"
ADAPTER_DIR = "./model/medical-llama3-adapter"


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"✅ Model loaded! Parameters: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, train_data, test_data):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=test_data,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    print("=" * 50)
    print("STARTING MEDICAL AI TRAINING")
    print("=" * 50)
    print(f"Model   : LLaMA 3.1 8B")
    print(f"Epochs  : 3")
    print(f"Train   : {len(train_data)} examples")
    print("=" * 50)

    start = time.time()
    trainer.train()
    elapsed = (time.time() - start) / 60
    print(f"\n✅ Training complete! Time: {elapsed:.1f} minutes")

    return model, tokenizer


def save_and_push(model, tokenizer):
    print("Saving LoRA adapter...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"✅ Adapter saved to {ADAPTER_DIR}")

    print("Pushing to HuggingFace Hub...")
    model.push_to_hub("your-username/medical-llama3-chatbot", token=HF_TOKEN)
    tokenizer.push_to_hub("your-username/medical-llama3-chatbot", token=HF_TOKEN)
    print("✅ Model pushed to HuggingFace!")


if __name__ == "__main__":
    train_data = load_from_disk("data/train")
    test_data  = load_from_disk("data/test")

    model, tokenizer = load_model_and_tokenizer()
    model            = apply_lora(model)
    model, tokenizer = train(model, tokenizer, train_data, test_data)
    save_and_push(model, tokenizer)
