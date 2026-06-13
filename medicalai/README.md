# 🏥 Medical AI Assistant

A fine-tuned LLaMA 3.1 8B medical chatbot built with QLoRA, HuggingFace Transformers, and Gradio.  
Fine-tuned on 100K real patient-doctor conversations from the ChatDoctor dataset.

---

## What it does
- Answers medical questions clearly and empathetically
- Detects emergencies and responds with local emergency numbers
- Fine-tuned with QLoRA — only ~0.6% of parameters trained
- Runs on free Colab T4 GPU via 4-bit quantization
- Full Gradio chat UI

---

## Project Structure

```
medicalai/
│
├── app.py                   # ← RUN THIS to launch the chatbot
├── config.py                # ← ALL settings live here
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── loader.py            # model loading + LoRA setup
│   └── inference.py         # prompt building + response generation
│
├── data/
│   └── prepare.py           # dataset loading, cleaning, formatting
│
├── scripts/
│   └── train.py             # QLoRA fine-tuning pipeline
│
└── evaluation/
    └── evaluate.py          # BERTScore + ROUGE-L + Perplexity
```

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Set HuggingFace token
```bash
export HF_TOKEN=your_token_here
```
Get token: https://huggingface.co/settings/tokens  
Request LLaMA access: https://huggingface.co/meta-llama/Llama-3.1-8B

### 3. Fine-tune (optional — takes ~2 hours on Colab A100)
```bash
python scripts/train.py
```

### 4. Evaluate
```bash
python evaluation/evaluate.py
```

### 5. Launch chatbot
```bash
python app.py
```

---

## Fine-Tuning Details

| Setting | Value |
|---|---|
| Base Model | LLaMA 3.1 8B |
| Dataset | ChatDoctor-HealthCareMagic-100k |
| Method | QLoRA (4-bit + LoRA) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | q, k, v, o, gate, up, down projections |
| Trainable Params | ~0.6% |
| Effective Batch Size | 8 (2 × 4 accumulation steps) |
| Learning Rate | 2e-4 with cosine scheduler |

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| BERTScore F1 | Semantic similarity — handles synonyms |
| ROUGE-L | Longest common subsequence overlap |
| Perplexity | How well model understands medical text |

---

## Emergency Safety
The app detects life-threatening keywords before passing to the model.  
Emergency responses include local numbers (115, 1122 for Pakistan).

---

## Tech Stack
HuggingFace Transformers · PEFT · TRL · BitsAndBytes · Gradio · PyTorch

---

⚠️ **Disclaimer**: This AI provides general health information only.  
Always consult a licensed medical professional for diagnosis and treatment.
