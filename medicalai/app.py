"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Medical AI Assistant — Gradio Chat UI

Run this to launch the chatbot:
  python app.py

Loads the fine-tuned model once at startup.
Gradio handles UI, routing, and the public share link.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gradio as gr
from model.loader import load_model_for_inference
from model.inference import (
    generate_response,
    check_emergency,
    get_emergency_response,
)
from config import DISCLAIMER


# ── Load model once at startup — not on every request ────────────────────────
print("Loading Medical AI model...")
model, tokenizer = load_model_for_inference()
print("✅ Model loaded!")


# ── Core chat function ────────────────────────────────────────────────────────
def chat(message, history):
    """
    Called by Gradio every time user sends a message.

    Parameters:
        message : str         — what user just typed
        history : list        — [[user, assistant], ...] pairs (Gradio manages this)

    Returns:
        ("", updated_history) — empty string clears the input box
    """
    if not message.strip():
        return "", history

    # Emergency check runs BEFORE model — always prioritize safety
    if check_emergency(message):
        history.append((message, get_emergency_response()))
        return "", history

    # Normal medical response
    response = generate_response(message, model, tokenizer)
    response = response + DISCLAIMER

    history.append((message, response))
    return "", history


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue   = "emerald",
        secondary_hue = "teal",
        neutral_hue   = "slate",
        font          = gr.themes.GoogleFont("Inter"),
    ),
    title="Medical AI Assistant",
    css="""
    .gradio-container { max-width: 860px !important; margin: auto !important }
    .header-box       { text-align: center; padding: 20px 0 10px }
    .disclaimer-box {
        background    : #E1F5EE;
        border        : 1px solid #9FE1CB;
        border-radius : 8px;
        padding       : 10px 16px;
        font-size     : 13px;
        color         : #0F6E56;
        margin-bottom : 10px;
    }
    footer { display: none !important }
    """,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
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

    # ── Disclaimer banner ─────────────────────────────────────────────────────
    gr.HTML("""
    <div class="disclaimer-box">
        🛡️ <strong>Disclaimer:</strong> This AI provides general health information only.
        Always consult a qualified healthcare professional for medical advice,
        diagnosis, or treatment.
        <strong>For emergencies call 115 or 1122 immediately.</strong>
    </div>
    """)

    # ── Chatbot — starts with greeting message ────────────────────────────────
    chatbot = gr.Chatbot(
        label             = "",
        height            = 450,
        bubble_full_width = False,
        avatar_images     = (None, "https://img.icons8.com/color/48/caduceus.png"),
        show_label        = False,
        value             = [[
            None,
            (
                "Hello! I'm your Medical AI Assistant 👨‍⚕️\n\n"
                "I can help you with:\n"
                "• Understanding symptoms\n"
                "• General health questions\n"
                "• Medication information\n"
                "• Health advice\n\n"
                "How can I help you today?"
            ),
        ]],
    )

    # ── Input row ─────────────────────────────────────────────────────────────
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder = "Describe your symptoms or ask a health question...",
            show_label  = False,
            scale       = 9,
            container   = False,
            lines       = 1,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

    # ── Quick question buttons ────────────────────────────────────────────────
    gr.HTML("<p style='font-size:12px;color:#999;margin:8px 0 4px'>Quick questions:</p>")
    with gr.Row():
        q1 = gr.Button("🤒 Fever & body aches",  size="sm", variant="secondary")
        q2 = gr.Button("🩺 Diabetes symptoms",    size="sm", variant="secondary")
        q3 = gr.Button("❤️ Blood pressure",       size="sm", variant="secondary")
        q4 = gr.Button("😴 Sleep problems",       size="sm", variant="secondary")

    # ── Stats bar ─────────────────────────────────────────────────────────────
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

    # ── Wire up events ────────────────────────────────────────────────────────
    send_btn.click(
        fn      = chat,
        inputs  = [msg_input, chatbot],
        outputs = [msg_input, chatbot],
    )
    msg_input.submit(
        fn      = chat,
        inputs  = [msg_input, chatbot],
        outputs = [msg_input, chatbot],
    )

    # Quick buttons fill the input box — user still clicks Send
    q1.click(lambda: "I have fever of 102F with body aches for 2 days", outputs=msg_input)
    q2.click(lambda: "What are the early symptoms of diabetes?",         outputs=msg_input)
    q3.click(lambda: "How can I naturally lower my blood pressure?",     outputs=msg_input)
    q4.click(lambda: "I have trouble sleeping every night. What helps?", outputs=msg_input)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share       = True,         # generates public gradio.live link
        debug       = False,
        show_error  = True,
        server_name = "0.0.0.0",
        server_port = 7860,
    )
