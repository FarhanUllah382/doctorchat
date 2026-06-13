"""
evaluate.py
-----------
Loads the saved LoRA adapter and runs a quick inference
test on sample medical questions.

Run after training:
    python evaluation/evaluate.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER_DIR = "./model/medical-llama3-adapter"

SYSTEM_PROMPT = """You are a professional and empathetic medical assistant.
When patients describe symptoms or ask health questions, provide clear,
accurate, and helpful medical information. Always recommend consulting
a qualified doctor for serious conditions. Never provide definitive
diagnoses. Be compassionate and easy to understand."""

TEST_QUESTIONS = [
    "I have had a fever of 102F for 2 days with body aches. What should I do?",
    "What are the early warning signs of diabetes?",
    "My child is 5 years old and has a stomach ache. Any advice?",
    "I have been feeling very anxious and can't sleep. Help?",
]


def load_model():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded!")
    return model, tokenizer


def generate_response(model, tokenizer, question):
    prompt = f"""### System:
{SYSTEM_PROMPT}

### Patient:
{question}

### Doctor:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("=" * 50)
    print("MEDICAL CHATBOT TEST RESULTS")
    print("=" * 50)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\nTest {i}:")
        print(f"Patient : {question}")
        print(f"Doctor  : {generate_response(model, tokenizer, question)}")
        print("-" * 40)
