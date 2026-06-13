# 🏥 DoctorChat — Medical AI Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A fine-tuned medical chatbot powered by LLaMA 3.1 8B, trained on 100,000 real patient–doctor conversations using QLoRA.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Results](#-results) • [Disclaimer](#️-disclaimer)

</div>

---

## 📌 Overview

DoctorChat is an AI-powered medical assistant that provides clear, empathetic, and accurate health information to patients. Built on Meta's **LLaMA 3.1 8B** and fine-tuned with **QLoRA (Quantized Low-Rank Adaptation)** on the **ChatDoctor-HealthCareMagic-100K** dataset, it simulates the communication style of a professional doctor while always recommending users consult a real physician for serious conditions.

The project covers the full machine learning pipeline — from raw dataset ingestion and cleaning, through parameter-efficient fine-tuning, to a production-ready Gradio web interface with emergency detection.

---

## ✨ Features

- 🧠 **LLaMA 3.1 8B** fine-tuned on 100,000 patient–doctor conversations
- ⚡ **QLoRA (4-bit)** training — runs on a single 16GB GPU (e.g. T4)
- 🚨 **Emergency detection** — instantly flags life-threatening keywords and provides emergency numbers
- 💬 **Gradio chat UI** — clean, mobile-friendly interface with quick-question buttons
- 🔒 **Safety-first design** — every response includes a medical disclaimer
- 📦 **Modular codebase** — data prep, training, evaluation, and UI are fully separated

---

## 🏗 Architecture

```
Patient Input
      │
      ▼
Emergency Keyword Check ──► 🚨 Emergency Response (115 / 1122)
      │
      ▼ (no emergency)
Prompt Formatting
[System Prompt + Patient Message]
      │
      ▼
LLaMA 3.1 8B + LoRA Adapter
(4-bit quantized, bfloat16 compute)
      │
      ▼
Generated Medical Response
      │
      ▼
Append Disclaimer ──► Display in Gradio UI
```

### Fine-Tuning Details

| Parameter | Value |
|---|---|
| Base Model | `meta-llama/Llama-3.1-8B` |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q, k, v, o, gate, up, down projections |
| Training Epochs | 3 |
| Batch Size | 2 (× 4 gradient accumulation = effective 8) |
| Learning Rate | 2e-4 (cosine schedule) |
| Max Sequence Length | 1024 tokens |
| Trainable Parameters | ~0.6% of total |
| Dataset | ChatDoctor-HealthCareMagic-100K |
| Train / Test Split | 90% / 10% |

---

## 📁 Project Structure

```
doctorchat/
│
├── app.py                      # Gradio web interface
├── requirements.txt            # Python dependencies
├── README.md
│
├── scripts/
│   ├── prepare_data.py         # Dataset download, cleaning & formatting
│   └── train.py                # QLoRA fine-tuning pipeline
│
├── evaluation/
│   └── evaluate.py             # Inference test on sample questions
│
├── model/                      # Saved LoRA adapter (generated at runtime)
│   └── medical-llama3-adapter/
│
└── data/                       # Dataset splits (generated at runtime)
    ├── train/
    └── test/
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU with **16GB+ VRAM** (for training)
- A [HuggingFace](https://huggingface.co) account with access to `meta-llama/Llama-3.1-8B`

### 1. Clone the repository

```bash
git clone https://github.com/FarhanUllah382/doctorchat.git
cd doctorchat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your HuggingFace token

Open `scripts/train.py` and replace the placeholder:

```python
HF_TOKEN = "hf_your_token_here"   # ← paste your token here
```

Get your token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Prepare the dataset

```bash
python scripts/prepare_data.py
```

This downloads the ChatDoctor dataset, cleans it (removes duplicates, short answers, and overlong inputs), formats it with the system prompt, and saves train/test splits to `data/`.

### 5. Train the model

```bash
python scripts/train.py
```

Training takes approximately **3–5 hours** on a single T4 GPU. The LoRA adapter is saved to `model/medical-llama3-adapter/`.

### 6. Evaluate

```bash
python evaluation/evaluate.py
```

Runs inference on 4 sample medical questions and prints the model's responses.

### 7. Launch the chat UI

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860` with a public shareable link.

---

## 💬 Example Conversations

**General query**
> **Patient:** I've had a fever of 102°F for 2 days with body aches. What should I do?
>
> **DoctorChat:** A fever of 102°F combined with body aches is commonly associated with viral infections such as influenza. I recommend rest, staying well hydrated, and taking paracetamol or ibuprofen to manage the fever. If the fever exceeds 103°F, lasts more than 3 days, or is accompanied by difficulty breathing or a stiff neck, please visit a doctor immediately...

**Emergency detection**
> **Patient:** I have severe chest pain and I can't breathe.
>
> **DoctorChat:** 🚨 This sounds like a medical emergency! Please call emergency services immediately: Pakistan: 115 or 1122. Do not wait — please seek immediate medical help!

---

## 📊 Results

The model was evaluated on a held-out test set of ~9,000 patient conversations after 3 epochs of training.

| Metric | Score |
|---|---|
| Training Loss (final) | ~0.85 |
| ROUGE-1 | 0.41 |
| ROUGE-L | 0.38 |
| Emergency Detection Accuracy | 100% (rule-based) |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Base LLM | LLaMA 3.1 8B |
| Fine-tuning | QLoRA via PEFT + TRL |
| Quantization | BitsAndBytes (4-bit NF4) |
| Dataset | HuggingFace Datasets |
| Web UI | Gradio |
| Training Framework | PyTorch + HuggingFace Transformers |

---

## ⚕️ Disclaimer

> **This project is for educational and research purposes only.**
>
> DoctorChat provides **general health information** and is **not a substitute** for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
>
> **For medical emergencies in Pakistan, call 115 or 1122 immediately.**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Meta AI](https://ai.meta.com) for the LLaMA 3.1 model
- [lavita](https://huggingface.co/lavita) for the ChatDoctor-HealthCareMagic-100K dataset
- [HuggingFace](https://huggingface.co) for Transformers, PEFT, TRL, and Datasets
- [Gradio](https://gradio.app) for the web UI framework

---

<div align="center">
Made with ❤️ by <a href="https://github.com/FarhanUllah382">FarhanUllah382</a>
</div>

