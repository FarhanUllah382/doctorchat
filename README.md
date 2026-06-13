# 🏥 DoctorChat — Medical AI Assistant

A medical chatbot fine-tuned on **100K patient conversations** using **QLoRA** on **LLaMA 3.1 8B**.

---

## 📁 Project Structure

```
doctorchat/
├── app.py                   # Gradio web UI
├── requirements.txt         # Python dependencies
├── scripts/
│   ├── prepare_data.py      # Download & clean dataset
│   └── train.py             # QLoRA fine-tuning
├── evaluation/
│   └── evaluate.py          # Test the trained model
├── model/                   # Saved adapter (gitignored)
└── data/                    # Dataset splits (gitignored)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
```bash
python scripts/prepare_data.py
```

### 3. Train the model
> Requires a GPU with 16GB+ VRAM (e.g. T4, A100).  
> Add your HuggingFace token inside `scripts/train.py`.
```bash
python scripts/train.py
```

### 4. Evaluate
```bash
python evaluation/evaluate.py
```

### 5. Launch the chat UI
```bash
python app.py
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Base model | LLaMA 3.1 8B |
| Fine-tuning | QLoRA (4-bit) |
| Dataset | ChatDoctor-HealthCareMagic-100K |
| LoRA rank | 16 |
| Training epochs | 3 |

---

## ⚠️ Disclaimer

This AI provides **general health information only**.  
Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.  
**For emergencies in Pakistan: call 115 or 1122.**
