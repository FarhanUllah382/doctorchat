"""
config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All project settings live here.
Change once → affects everything.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "meta-llama/Llama-3.1-8B"   # base model to fine-tune
ADAPTER_PATH   = "./medical-llama3-adapter"   # where LoRA weights are saved
HUB_MODEL_NAME = "your-username/medical-llama3-chatbot"  # HuggingFace Hub repo

# ── Auth ──────────────────────────────────────────────────────────────────────
# Never hardcode tokens — always read from environment variable
# In Colab: set via login(token=...) or os.environ["HF_TOKEN"] = "..."
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "lavita/ChatDoctor-HealthCareMagic-100k"
DATASET_SPLIT  = "train"
TEST_SIZE      = 0.1     # 10% for evaluation
RANDOM_SEED    = 42

# Dataset cleaning thresholds
MIN_OUTPUT_LEN = 100     # remove low-quality short answers
MAX_INPUT_LEN  = 800     # remove very long inputs (OOM risk)

# ── LoRA Hyperparameters ──────────────────────────────────────────────────────
LORA_R        = 16       # rank — size of A and B adapter matrices
LORA_ALPHA    = 32       # scaling factor — rule: alpha = 2 × r
LORA_DROPOUT  = 0.05     # prevents overfitting on small datasets
LORA_TARGETS  = [        # which weight matrices to apply LoRA to
    "q_proj",            # query projection
    "k_proj",            # key projection
    "v_proj",            # value projection
    "o_proj",            # output projection
    "gate_proj",         # FFN gate
    "up_proj",           # FFN up
    "down_proj",         # FFN down
]
# Note: targeting all 7 layers (vs default q+v only) because
# medical domain needs deeper task adaptation

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_ARGS = dict(
    output_dir                  = "./medical-llama3-checkpoints",
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,    # effective batch size = 2×4 = 8
    gradient_checkpointing      = True, # trades compute for VRAM savings
    optim                       = "paged_adamw_32bit",  # memory-efficient optimizer
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",  # smooth LR decay
    warmup_ratio                = 0.05,      # 5% of steps = warmup
    fp16                        = False,
    bf16                        = True,      # bfloat16 — better than fp16 for LLMs
    logging_steps               = 10,
    save_strategy               = "steps",
    save_steps                  = 100,
    evaluation_strategy         = "steps",
    eval_steps                  = 100,
    load_best_model_at_end      = True,
    report_to                   = "none",    # change to "wandb" for experiment tracking
    seed                        = RANDOM_SEED,
    max_seq_length              = 1024,
)

# ── Generation Defaults ───────────────────────────────────────────────────────
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.9
DEFAULT_MAX_NEW_TOKENS = 400
REPETITION_PENALTY     = 1.1

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional and empathetic medical assistant.
When patients describe symptoms or ask health questions, provide clear,
accurate, and helpful medical information. Always recommend consulting
a qualified doctor for serious conditions. Never provide definitive
diagnoses. Be compassionate and easy to understand."""

# ── Emergency Detection ───────────────────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe",
    "unconscious", "heart attack", "stroke",
    "suicide", "overdose", "not breathing",
    "severe bleeding", "poisoning",
]

DISCLAIMER = (
    "\n\n⚕️ This is AI-generated information only. "
    "Please consult a qualified doctor for proper medical advice."
)
