"""
app.py
------
Gradio web interface for the Medical AI Assistant.
Loads the fine-tuned LoRA adapter and serves a chat UI.

Run:
    python app.py
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Emergency keywords ───────────────────────────────────
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

SYSTEM_PROMPT = """You are a professional and empathetic medical assistant.
When patients describe symptoms or ask health questions, provide clear,
accurate, and helpful medical information. Always recommend consulting
a qualified doctor for serious conditions. Never provide definitive
diagnoses. Be compassionate and easy to understand."""

ADAPTER_DIR = "./model/medical-llama3-adapter"

# ── Load model ───────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
model = AutoModelForCausalLM.from_pretrained(
    ADAPTER_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("✅ Model loaded!")


# ── Helper functions ─────────────────────────────────────
def check_emergency(message: str) -> bool:
    msg_lower = message.lower()
    return any(k in msg_lower for k in EMERGENCY_KEYWORDS)


def generate_response(message: str) -> str:
    prompt = f"""### System:
{SYSTEM_PROMPT}

### Patient:
{message}

### Doctor:
"""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def chat(message, history):
    if not message.strip():
        return "", history

    if check_emergency(message):
        reply = (
            "🚨 This sounds like a medical emergency!\n\n"
            "Please call emergency services immediately:\n"
            "• Pakistan: 115 or 1122\n"
            "• Or go to the nearest emergency room right away!\n\n"
            "Do not wait — please seek immediate medical help!"
        )
    else:
        reply = generate_response(message) + DISCLAIMER

    history.append((message, reply))
    return "", history


# ── Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    title="Medical AI Assistant",
    css="""
    .gradio-container {max-width: 860px !important; margin: auto !important}
    .header-box {text-align: center; padding: 20px 0 10px}
    .disclaimer-box {
        background: #E1F5EE;
        border: 1px solid #9FE1CB;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 13px;
        color: #0F6E56;
        margin-bottom: 10px;
    }
    footer {display: none !important}
    """,
) as demo:

    gr.HTML("""
    <div class="header-box">
        <h1 style="font-size:24px;font-weight:600;color:#0F6E56;margin-bottom:6px">
            🏥 Medical AI Assistant
        </h1>
        <p style="font-size:14px;color:#666;margin:0">
            Powered by LLaMA 3.1 8B • Fine-tuned on 100K Patient Conversations
        </p>
    </div>
    """)

    gr.HTML("""
    <div class="disclaimer-box">
        🛡️ <strong>Disclaimer:</strong> This AI provides general health information only.
        Always consult a qualified healthcare professional for medical advice,
        diagnosis, or treatment.
        <strong>For emergencies call 115 or 1122 immediately.</strong>
    </div>
    """)

    chatbot = gr.Chatbot(
        label="",
        height=450,
        bubble_full_width=False,
        avatar_images=(None, "https://img.icons8.com/color/48/caduceus.png"),
        show_label=False,
        value=[[
            None,
            "Hello! I'm your Medical AI Assistant 👨‍⚕️\n\n"
            "I can help you with:\n"
            "• Understanding symptoms\n"
            "• General health questions\n"
            "• Medication information\n"
            "• Health advice\n\n"
            "How can I help you today?",
        ]],
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Describe your symptoms or ask a health question...",
            show_label=False,
            scale=9,
            container=False,
            lines=1,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    gr.HTML("<p style='font-size:12px;color:#999;margin:8px 0 4px'>Quick questions:</p>")
    with gr.Row():
        q1 = gr.Button("🤒 Fever & body aches",  size="sm", variant="secondary")
        q2 = gr.Button("🩺 Diabetes symptoms",    size="sm", variant="secondary")
        q3 = gr.Button("❤️ Blood pressure",       size="sm", variant="secondary")
        q4 = gr.Button("😴 Sleep problems",       size="sm", variant="secondary")

    with gr.Row():
        gr.HTML("""
        <div style="display:flex;gap:12px;margin-top:12px;width:100%">
            <div style="flex:1;text-align:center;background:#f8fffe;border:1px solid #9FE1CB;border-radius:8px;padding:10px">
                <div style="font-size:20px;font-weight:600;color:#0F6E56">100K+</div>
                <div style="font-size:11px;color:#666">Training conversations</div>
            </div>
            <div style="flex:1;text-align:center;background:#f8fffe;border:1px solid #9FE1CB;border-radius:8px;padding:10px">
                <div style="font-size:20px;font-weight:600;color:#0F6E56">8B</div>
                <div style="font-size:11px;color:#666">Model parameters</div>
            </div>
            <div style="flex:1;text-align:center;background:#f8fffe;border:1px solid #9FE1CB;border-radius:8px;padding:10px">
                <div style="font-size:20px;font-weight:600;color:#0F6E56">QLoRA</div>
                <div style="font-size:11px;color:#666">Fine-tuning method</div>
            </div>
            <div style="flex:1;text-align:center;background:#fff3f3;border:1px solid #F09595;border-radius:8px;padding:10px">
                <div style="font-size:20px;font-weight:600;color:#A32D2D">115</div>
                <div style="font-size:11px;color:#666">Emergency (Pakistan)</div>
            </div>
        </div>
        """)

    send_btn.click(chat, [msg_input, chatbot], [msg_input, chatbot])
    msg_input.submit(chat, [msg_input, chatbot], [msg_input, chatbot])

    q1.click(lambda: "I have fever of 102F with body aches for 2 days", outputs=msg_input)
    q2.click(lambda: "What are the early symptoms of diabetes?",         outputs=msg_input)
    q3.click(lambda: "How can I naturally lower my blood pressure?",     outputs=msg_input)
    q4.click(lambda: "I have trouble sleeping every night. What helps?", outputs=msg_input)


if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
