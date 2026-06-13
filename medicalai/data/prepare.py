"""
data/prepare.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  1. Loading ChatDoctor dataset from HuggingFace
  2. Cleaning — removes duplicates, short answers, long inputs
  3. Formatting — wraps each example in the chat prompt template
  4. Splitting — train / test split

Run this before training to inspect and prepare your data.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
from datasets import load_dataset, Dataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_NAME,
    DATASET_SPLIT,
    TEST_SIZE,
    RANDOM_SEED,
    MIN_OUTPUT_LEN,
    MAX_INPUT_LEN,
    SYSTEM_PROMPT,
)


def load_raw_dataset():
    """
    Downloads ChatDoctor dataset from HuggingFace Hub.
    Dataset has 3 columns: instruction, input, output
    We use 'input' (patient message) and 'output' (doctor response).
    """
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    df      = pd.DataFrame(dataset)

    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    print(f"Total examples     : {len(df)}")
    print(f"Columns            : {df.columns.tolist()}")
    print(f"Missing values     : {df.isnull().sum().sum()}")
    print(f"Duplicate rows     : {df.duplicated().sum()}")
    print(f"Avg input length   : {df['input'].str.len().mean():.0f} chars")
    print(f"Avg output length  : {df['output'].str.len().mean():.0f} chars")
    print("\nSAMPLE EXAMPLE:")
    print("-" * 40)
    print("PATIENT:", df['input'][0][:300])
    print("\nDOCTOR :", df['output'][0][:300])

    return df


def clean_dataset(df):
    """
    Removes low-quality, duplicate, and problematic examples.

    Why each filter exists:
    - Dedup        → model shouldn't memorize repeated examples
    - Short output → very short answers are usually unhelpful
    - Long input   → very long inputs cause OOM on T4 GPU
    """
    print("\nCleaning dataset...")
    print(f"  Original size     : {len(df)}")

    df = df.drop_duplicates(subset=['input'])
    print(f"  After dedup       : {len(df)}")

    df = df[df['output'].str.len() > MIN_OUTPUT_LEN]
    print(f"  After quality filter : {len(df)}")

    df = df[df['input'].str.len() < MAX_INPUT_LEN]
    print(f"  After length filter  : {len(df)}")

    df = df.reset_index(drop=True)
    print(f"  ✅ Final clean size  : {len(df)}")

    print("\nCLEANED SAMPLE:")
    print("PATIENT:", df['input'][0][:200])
    print("DOCTOR :", df['output'][0][:200])

    return df


def format_medical_prompt(example):
    """
    Wraps each example in the chat prompt format the model expects.

    Format:
    ### System:   ← tells model its role
    ### Patient:  ← the user's medical question
    ### Doctor:   ← the expected response (training target)

    This format is also used at inference time — must match exactly.
    """
    prompt = f"""### System:
{SYSTEM_PROMPT}

### Patient:
{example['input']}

### Doctor:
{example['output']}"""

    return {"text": prompt}


def prepare_datasets():
    """
    Full pipeline: load → clean → format → split.
    Returns (train_data, test_data) ready for SFTTrainer.
    """
    # Load and clean
    df       = load_raw_dataset()
    df_clean = clean_dataset(df)

    # Convert to HuggingFace Dataset format
    dataset_clean     = Dataset.from_pandas(df_clean)

    # Apply prompt formatting — removes original columns, adds 'text' column
    dataset_formatted = dataset_clean.map(
        format_medical_prompt,
        remove_columns=dataset_clean.column_names,
    )

    # Train / test split
    dataset_split = dataset_formatted.train_test_split(
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    train_data = dataset_split['train']
    test_data  = dataset_split['test']

    print(f"\nTrain size : {len(train_data)}")
    print(f"Test size  : {len(test_data)}")
    print("\nFormatted example preview:")
    print(train_data[0]['text'][:500])

    return train_data, test_data


if __name__ == "__main__":
    prepare_datasets()
