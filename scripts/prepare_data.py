"""
prepare_data.py
---------------
Downloads and cleans the ChatDoctor dataset, then saves
train/test splits ready for fine-tuning.
"""

from datasets import load_dataset, Dataset
import pandas as pd

SYSTEM_PROMPT = """You are a professional and empathetic medical assistant.
When patients describe symptoms or ask health questions, provide clear,
accurate, and helpful medical information. Always recommend consulting
a qualified doctor for serious conditions. Never provide definitive
diagnoses. Be compassionate and easy to understand."""


def load_raw_dataset():
    print("Loading ChatDoctor dataset...")
    dataset = load_dataset(
        "lavita/ChatDoctor-HealthCareMagic-100k",
        split="train"
    )
    df = pd.DataFrame(dataset)

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
    print("Original size:", len(df))

    df = df.drop_duplicates(subset=['input'])
    print("After dedup:", len(df))

    df = df[df['output'].str.len() > 100]
    print("After quality filter:", len(df))

    df = df[df['input'].str.len() < 800]
    print("After length filter:", len(df))

    df = df.reset_index(drop=True)
    print("✅ Final clean size:", len(df))
    return df


def format_medical_prompt(example):
    prompt = f"""### System:
{SYSTEM_PROMPT}

### Patient:
{example['input']}

### Doctor:
{example['output']}"""
    return {"text": prompt}


def prepare_splits(df_clean):
    dataset_clean = Dataset.from_pandas(df_clean)
    dataset_formatted = dataset_clean.map(
        format_medical_prompt,
        remove_columns=dataset_clean.column_names
    )

    dataset_split = dataset_formatted.train_test_split(
        test_size=0.1,
        seed=42
    )
    train_data = dataset_split['train']
    test_data  = dataset_split['test']

    print(f"Train size : {len(train_data)}")
    print(f"Test size  : {len(test_data)}")
    print("\nFormatted example preview:")
    print(train_data[0]['text'][:500])

    return train_data, test_data


if __name__ == "__main__":
    df        = load_raw_dataset()
    df_clean  = clean_dataset(df)
    train_data, test_data = prepare_splits(df_clean)

    # Save to disk for use in training
    train_data.save_to_disk("data/train")
    test_data.save_to_disk("data/test")
    print("✅ Datasets saved to data/train and data/test")
