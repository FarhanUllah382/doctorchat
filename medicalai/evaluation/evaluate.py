"""
evaluation/evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluates the fine-tuned model with:
  - Qualitative test  : runs sample questions, prints answers
  - ROUGE-L           : measures output overlap with reference
  - BERTScore         : semantic similarity (modern standard)
  - Perplexity        : how well model understands medical text

Run after training to confirm the model actually improved.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouge_score import rouge_scorer as rouge_lib
from bert_score import score as bert_score

from model.loader import load_model_for_inference
from model.inference import generate_response


# ── Test questions with reference answers ─────────────────────────────────────
# These are the ground-truth answers we compare the model against.
TEST_CASES = [
    {
        "question" : "I have had a fever of 102F for 2 days with body aches. What should I do?",
        "reference": "A fever of 102F lasting 2 days with body aches may indicate a viral infection like flu. Rest, stay hydrated, and take paracetamol or ibuprofen for fever. Monitor temperature. If fever exceeds 103F, persists beyond 3 days, or you develop breathing difficulty, seek medical attention immediately.",
    },
    {
        "question" : "What are the early warning signs of diabetes?",
        "reference": "Early signs of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, slow-healing wounds, and frequent infections. Type 1 can appear suddenly while Type 2 develops gradually. Consult a doctor for blood sugar testing if you notice these symptoms.",
    },
    {
        "question" : "My child is 5 years old and has a stomach ache. Any advice?",
        "reference": "Stomach aches in children are common and often due to gas, constipation, or mild infection. Ensure hydration and light meals. Watch for warning signs: severe or worsening pain, fever, vomiting, blood in stool, or pain that wakes the child. See a doctor if symptoms worsen or persist beyond 24 hours.",
    },
    {
        "question" : "How can I naturally lower my blood pressure?",
        "reference": "Natural ways to lower blood pressure include reducing sodium intake, regular aerobic exercise, maintaining healthy weight, limiting alcohol, quitting smoking, managing stress through meditation or yoga, eating a DASH diet rich in fruits and vegetables, and getting adequate sleep.",
    },
    {
        "question" : "I have been feeling very anxious and can't sleep. Help?",
        "reference": "Anxiety and insomnia often occur together. Try relaxation techniques like deep breathing, meditation, or progressive muscle relaxation. Maintain a consistent sleep schedule, limit caffeine and screen time before bed. Regular exercise helps significantly. If anxiety is severe or persistent, consult a doctor or mental health professional.",
    },
]


def calculate_perplexity(model, tokenizer, text: str) -> float:
    """
    Lower = better. Model was less 'surprised' by this text.
    Measures how well the model understands medical language.
    """
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def run_qualitative_test(model, tokenizer):
    """Runs sample questions and prints answers for human inspection."""
    print("\n" + "=" * 55)
    print("QUALITATIVE TEST — SAMPLE RESPONSES")
    print("=" * 55)

    for i, case in enumerate(TEST_CASES, 1):
        response = generate_response(case["question"], model, tokenizer)
        print(f"\nTest {i}:")
        print(f"  Patient : {case['question']}")
        print(f"  Doctor  : {response[:300]}...")
        print("-" * 40)


def run_automated_metrics(model, tokenizer):
    """
    Runs BERTScore, ROUGE-L, and Perplexity across all test cases.
    Returns a report dict.
    """
    print("\n" + "=" * 55)
    print("AUTOMATED EVALUATION METRICS")
    print("=" * 55)

    rouge       = rouge_lib.RougeScorer(['rougeL'], use_stemmer=True)
    candidates  = []
    references  = []
    rouge_scores= []
    perplexities= []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"  Evaluating case {i}/{len(TEST_CASES)}...")

        # Generate answer
        answer = generate_response(case["question"], model, tokenizer)
        candidates.append(answer)
        references.append(case["reference"])

        # ROUGE-L — longest common subsequence overlap
        r = rouge.score(case["reference"], answer)
        rouge_scores.append(r['rougeL'].fmeasure)

        # Perplexity on generated answer
        ppl = calculate_perplexity(model, tokenizer, answer)
        perplexities.append(ppl)

    # BERTScore — semantic similarity using BERT embeddings
    # Compares meaning, not exact words — handles synonyms well
    print("\n  Calculating BERTScore...")
    _, _, F1   = bert_score(candidates, references, lang="en", verbose=False)
    bert_f1    = F1.mean().item()

    # Compile report
    report = {
        "bertscore_f1" : round(bert_f1,                           4),
        "rouge_l"      : round(sum(rouge_scores) / len(rouge_scores), 4),
        "perplexity"   : round(sum(perplexities) / len(perplexities), 2),
        "num_examples" : len(TEST_CASES),
    }

    print("\n" + "=" * 55)
    print("FINAL SCORES")
    print("=" * 55)
    print(f"  BERTScore F1  : {report['bertscore_f1']}  (higher is better, max 1.0)")
    print(f"  ROUGE-L       : {report['rouge_l']}  (higher is better, max 1.0)")
    print(f"  Perplexity    : {report['perplexity']}  (lower is better)")
    print("=" * 55)

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Results saved → evaluation/results.json")

    return report


def evaluate():
    """Full evaluation pipeline."""
    print("Loading model for evaluation...")
    model, tokenizer = load_model_for_inference()

    run_qualitative_test(model, tokenizer)
    run_automated_metrics(model, tokenizer)


if __name__ == "__main__":
    evaluate()
