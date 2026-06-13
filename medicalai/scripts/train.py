"""
scripts/train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QLoRA Fine-tuning pipeline for Medical AI.

Run order:
  1. python scripts/train.py       ← this file
  2. python evaluation/evaluate.py ← check results
  3. python app.py                 ← launch chatbot

What happens here:
  Step 1 → Load + clean dataset
  Step 2 → Load base model with 4-bit quantization
  Step 3 → Apply LoRA adapters
  Step 4 → Train
  Step 5 → Save adapter weights
  Step 6 → Push to HuggingFace Hub
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login

from config import (
    MODEL_ID,
    ADAPTER_PATH,
    HUB_MODEL_NAME,
    HF_TOKEN,
    TRAINING_ARGS,
)
from data.prepare import prepare_datasets
from model.loader import load_base_model, load_tokenizer, apply_lora


def train():
    print("=" * 55)
    print("MEDICAL AI — QLoRA FINE-TUNING")
    print("=" * 55)

    # ── Auth ──────────────────────────────────────────────────────────────────
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("✅ HuggingFace login successful!")
    else:
        print("⚠️  No HF_TOKEN found. Set via: export HF_TOKEN=your_token")

    # ── STEP 1: Prepare data ──────────────────────────────────────────────────
    print("\n[1/5] Preparing dataset...")
    train_data, test_data = prepare_datasets()

    # ── STEP 2: Load base model ───────────────────────────────────────────────
    print("\n[2/5] Loading base model...")
    tokenizer = load_tokenizer()
    model     = load_base_model()

    # ── STEP 3: Apply LoRA ────────────────────────────────────────────────────
    print("\n[3/5] Applying LoRA adapters...")
    model = apply_lora(model)

    # ── STEP 4: Configure trainer ─────────────────────────────────────────────
    print("\n[4/5] Configuring trainer...")

    # Pull max_seq_length out separately — SFTTrainer takes it directly
    max_seq_length = TRAINING_ARGS.pop("max_seq_length", 1024)

    training_args = TrainingArguments(**TRAINING_ARGS)

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_data,
        eval_dataset       = test_data,
        dataset_text_field = "text",       # column name with formatted prompts
        max_seq_length     = max_seq_length,
        args               = training_args,
    )

    print("✅ Trainer configured!")
    print(f"   Model       : {MODEL_ID}")
    print(f"   Train size  : {len(train_data)}")
    print(f"   Test size   : {len(test_data)}")
    print(f"   Epochs      : {TRAINING_ARGS.get('num_train_epochs', 3)}")

    # ── STEP 5: Train ─────────────────────────────────────────────────────────
    print("\n[5/5] Starting training...")
    print("=" * 55)

    start_time = time.time()
    trainer.train()
    elapsed    = (time.time() - start_time) / 60

    print(f"\n✅ Training complete! — {elapsed:.1f} minutes")

    # ── Save adapter weights locally ──────────────────────────────────────────
    # Saves ONLY the A and B matrices — a few MB, not the full 8B model
    print(f"\nSaving LoRA adapter to: {ADAPTER_PATH}")
    trainer.model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print("✅ Adapter saved locally!")

    # ── Push to HuggingFace Hub ───────────────────────────────────────────────
    print(f"\nPushing to HuggingFace Hub: {HUB_MODEL_NAME}")
    trainer.model.push_to_hub(HUB_MODEL_NAME, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_MODEL_NAME,     token=HF_TOKEN)
    print(f"✅ Model live at: huggingface.co/{HUB_MODEL_NAME}")


if __name__ == "__main__":
    train()
