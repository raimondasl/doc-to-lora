import argparse
import os
import sys
from pathlib import Path

# --- Patch gradio_client bool-schema bug (additionalProperties: false) ---
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _patched_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

_orig_get_type = _gc_utils.get_type


def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gc_utils.get_type = _patched_get_type
# --- End patch ---

import gradio as gr
import requests
import torch
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
model_info = {"base_model": "Not loaded", "checkpoint": "Not loaded"}


def get_available_checkpoints():
    trained_d2l_checkpoints = {
        str(path)
        for path in Path().glob("trained_d2l/**/pytorch_model.bin")
        if path.is_file()
    }
    run_output_checkpoints = {
        str(path)
        for path in Path().glob("train_outputs/runs/**/pytorch_model.bin")
        if path.is_file()
    }
    checkpoints = sorted(trained_d2l_checkpoints) + sorted(
        run_output_checkpoints - trained_d2l_checkpoints
    )
    return checkpoints if checkpoints else ["No checkpoints found"]


def load_checkpoint(checkpoint_path: str):
    global model, tokenizer, model_info

    if not checkpoint_path or checkpoint_path == "No checkpoints found":
        raise ValueError("No valid checkpoint found.")

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_flash_attn=True, use_sequence_packing=False
    )
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    model.reset()

    base_model_name = model.base_model.config.name_or_path
    tokenizer = get_tokenizer(base_model_name, train=False)

    # Load custom chat template if available
    if "gemma" in base_model_name.lower():
        template_path = "chat_templates/google/gemma-2-2b-it.jinja"
        if os.path.exists(template_path):
            tokenizer.chat_template = Path(template_path).read_text()

    model_info["base_model"] = base_model_name
    model_info["checkpoint"] = checkpoint_path
    print(f"Loaded: base_model={base_model_name}, checkpoint={checkpoint_path}")


def fetch_url(url: str):
    if not url.strip():
        return "Please enter a URL.", "", None

    try:
        resp = requests.get(url.strip(), timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; doc-to-lora-demo/1.0)"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate if very long (context encoder has limits)
        if len(text) > 20000:
            text = text[:20000] + "\n\n[Truncated]"

        return f"Fetched {len(text)} characters from {url}", text, text
    except Exception as e:
        return f"Error fetching URL: {e}", "", None


def internalize_text(text: str):
    if not text.strip():
        return "No text to internalize.", None

    try:
        model.reset()
        model.internalize(text.strip())
        return f"Internalized {len(text.strip())} characters.", text.strip()
    except Exception as e:
        return f"Error internalizing: {e}", None


def generate_response(chat_history, message: str, ctx: str):
    """Generate responses from both base and adapted models."""
    if not message.strip():
        return chat_history, ""

    if not ctx:
        chat_history.append([message, "Please internalize a context first (fetch a URL or paste text)."])
        return chat_history, ""

    conversation = [{"role": "user", "content": message}]
    chat_ids = tokenizer.apply_chat_template(
        conversation,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate base model response (no internalization)
    model.reset()
    with torch.inference_mode(), torch.amp.autocast(str(device)):
        base_outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
    base_response = tokenizer.decode(
        base_outputs[0][chat_ids.shape[1]:], skip_special_tokens=True
    )

    # Generate adapted model response (with internalization)
    model.internalize(ctx)
    with torch.inference_mode(), torch.amp.autocast(str(device)):
        adapted_outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
    adapted_response = tokenizer.decode(
        adapted_outputs[0][chat_ids.shape[1]:], skip_special_tokens=True
    )

    combined = (
        f"**Base model ({model_info['base_model']}):**\n{base_response}\n\n---\n\n"
        f"**With internalized context (Doc-to-LoRA):**\n{adapted_response}"
    )

    chat_history.append([message, combined])
    return chat_history, ""


def create_demo():
    with gr.Blocks(
        title="Doc-to-LoRA Comparison Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Doc-to-LoRA: Base vs Adapted Model Comparison\n"
            "Compare responses from the base model and the same model "
            "after internalizing a web page via the hypernetwork."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    f"### Model\n"
                    f"**Base model:** {model_info['base_model']}\n\n"
                    f"**Checkpoint:** {model_info['checkpoint']}"
                )
                gr.Markdown("---")
                gr.Markdown("### Context")
                url_input = gr.Textbox(
                    label="Web page URL",
                    placeholder="https://example.com/article",
                    lines=1,
                )
                fetch_btn = gr.Button("Fetch & Internalize", variant="primary")
                fetch_status = gr.Textbox(
                    label="Fetch status", lines=1, interactive=False
                )
                context_preview = gr.Textbox(
                    label="Extracted text (editable - click Internalize to apply changes)",
                    lines=10,
                    interactive=True,
                )
                internalize_btn = gr.Button("Internalize text", variant="secondary")
                internalize_status = gr.Textbox(
                    label="Internalize status", lines=1, interactive=False
                )

            with gr.Column(scale=2):
                gr.Markdown(
                    "### Chat\n"
                    "Each response shows the **base model** answer (no context) "
                    "and the **Doc-to-LoRA adapted** answer (with internalized context) "
                    "side by side."
                )
                chatbot = gr.Chatbot(
                    label="Comparison",
                    height=550,
                    elem_id="chatbot",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="Ask something about the internalized context...",
                        lines=2,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear chat", variant="secondary")

        # Hidden state for internalized context text
        ctx_state = gr.State(value=None)

        # Event handlers
        fetch_btn.click(
            fn=fetch_url,
            inputs=[url_input],
            outputs=[fetch_status, context_preview, ctx_state],
        )
        internalize_btn.click(
            fn=internalize_text,
            inputs=[context_preview],
            outputs=[internalize_status, ctx_state],
        )
        msg.submit(
            fn=generate_response,
            inputs=[chatbot, msg, ctx_state],
            outputs=[chatbot, msg],
        )
        send_btn.click(
            fn=generate_response,
            inputs=[chatbot, msg, ctx_state],
            outputs=[chatbot, msg],
        )
        clear_btn.click(fn=lambda: [], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pytorch_model.bin checkpoint. If not specified, "
             "auto-detects the first available checkpoint.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.environ.get("D2L_CHECKPOINT")
    if checkpoint:
        load_checkpoint(checkpoint)
    else:
        checkpoints = get_available_checkpoints()
        if checkpoints and checkpoints[0] != "No checkpoints found":
            load_checkpoint(checkpoints[0])
        else:
            print("WARNING: No checkpoints found. Use --checkpoint or place "
                  "pytorch_model.bin in trained_d2l/")

    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
