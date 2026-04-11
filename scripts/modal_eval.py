from pathlib import Path

import modal

# --- Image Definition ---
# We build the environment once; Modal caches it.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "curl")
    # Install uv
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    # Copy the repo into the image (run from the repo root)
    .add_local_dir(".", "/app", copy=True)
    .workdir("/app")
    # Install PyTorch with CUDA 12.4 backend
    .run_commands(
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    # Install project deps
    .run_commands("pip install -e .")
    .run_commands("pip install tokenizers==0.21.0")
    # Flash-attention pre-built wheel for Python 3.10 + CUDA 12 + Torch 2.6
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    # Flashinfer
    .run_commands(
        "pip install flashinfer-python==0.2.2 "
        "-i https://flashinfer.ai/whl/cu124/torch2.6"
    )
    # RepoQA benchmark and its dependencies
    # Pin tree-sitter to 0.21.x for tree-sitter-languages compatibility
    .run_commands("pip install tree-sitter==0.21.3 repoqa")
)

app = modal.App("doc-to-lora", image=image)

# Default checkpoint path (relative to /app, after symlink)
DEFAULT_CHECKPOINT = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"

# Persistent volume to store downloaded model checkpoints
volume = modal.Volume.from_name("doc-to-lora-models", create_if_missing=True)
MODEL_DIR = "/models"


# --- Download pretrained D2L weights from HuggingFace ---
@app.function(
    gpu="A100",
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN env var
    timeout=3600,
)
def download_models():
    import subprocess
    subprocess.run([
        "hf", "download",
        "SakanaAI/doc-to-lora",
        "--local-dir", f"{MODEL_DIR}/trained_d2l",
    ], check=True)
    volume.commit()
    print("Models downloaded.")


# --- Run inference ---
@app.function(
    gpu="A100",
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def run_inference(document: str, question: str) -> str:
    import torch
    import sys
    sys.path.insert(0, "/app")

    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    checkpoint_path = f"{MODEL_DIR}/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    model.internalize(document)

    chat = [{"role": "user", "content": question}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
    return tokenizer.decode(outputs[0])


@app.local_entrypoint()
def main(document: str, question: str):
    """Read a local document file and run inference remotely."""
    doc_text = Path(document).read_text(encoding="utf-8")
    result = run_inference.remote(doc_text, question)
    print(result)



# --- Training ---
@app.function(
    gpu="A100-80GB",           # training needs more VRAM
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,             # 24h max
    cpu=8,
    memory=64000,
)
def run_training():
    import subprocess
    subprocess.run(
        ["bash", "scripts/main_exp/1-train.sh"],
        cwd="/app",
        check=True,
    )


# --- RepoQA Evaluation ---
# Persistent volume to store RepoQA results across runs
results_volume = modal.Volume.from_name("doc-to-lora-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    gpu="A100",
    volumes={MODEL_DIR: volume, RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,             # 24h — full eval is slow
)
def run_repoqa(
    languages: str = "",
    code_context_size: int = 16384,
    max_new_tokens: int = 1024,
    max_tasks: int = -1,
):
    """Run RepoQA needle-function-retrieval evaluation.

    Each task is run twice:
      1. Baseline — model.reset() only (no internalize), base LM answers from scratch
      2. D2L     — model.reset() + model.internalize(code_context), then answer

    Results are saved as JSONL + SCORES.json under /results/repoqa/.
    """
    import json
    import os
    import sys

    import torch

    sys.path.insert(0, "/app")

    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    from repoqa.compute_score import compute_score, save_json
    from repoqa.data import get_repoqa_data
    from repoqa.search_needle_function import (
        INSTRUCTION,
        TEMPLATE,
        make_code_context,
        make_task_id,
    )
    from repoqa.utility import topological_sort

    # Parse comma-separated languages (empty string = all languages)
    lang_filter = [l.strip() for l in languages.split(",") if l.strip()] or None

    # ---- Load D2L model ----
    checkpoint_path = f"{MODEL_DIR}/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    # Chat template affixes for chunking (from ctx_to_lora/data/definitions.py)
    ctx_encoder_name = model.ctx_encoder.base_model.name_or_path
    from ctx_to_lora.data.definitions import CTX_AFFIXES
    from ctx_to_lora.data.processing import tokenize_ctx_text
    ctx_affixes = CTX_AFFIXES[ctx_encoder_name]
    ctx_tokenizer = get_tokenizer(ctx_encoder_name)

    # Detect the stop sequence for generation (mirrors HfProvider logic)
    stop_seq = None
    if tokenizer.chat_template:
        _magic_ = "&==NowOrNever==&"
        tail = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}, {"role": "assistant", "content": _magic_}],
            tokenize=False,
        ).split(_magic_)[-1]
        if tail.strip():
            stop_seq = tail

    # Max chunk length for the context encoder (training used 6144)
    MAX_CTX_CHUNK_LEN = 4096

    # ---- Helper: chunked internalize for long contexts ----
    @torch.inference_mode()
    def chunked_internalize(ctx_str: str):
        """Internalize a context string, chunking if it exceeds MAX_CTX_CHUNK_LEN."""
        ctx_ids_raw = tokenize_ctx_text(
            dict(context=[ctx_str]), ctx_tokenizer
        )["ctx_ids"][0]

        if len(ctx_ids_raw) <= MAX_CTX_CHUNK_LEN:
            model.internalize(ctx_str)
            return

        # Split into chunks and add chat template affixes
        from math import ceil
        n_chunks = ceil(len(ctx_ids_raw) / MAX_CTX_CHUNK_LEN)
        avg_len = ceil(len(ctx_ids_raw) / n_chunks)
        chunks = [ctx_ids_raw[i : i + avg_len] for i in range(0, len(ctx_ids_raw), avg_len)]

        prefix = ctx_affixes["prefix"]
        suffix = ctx_affixes["suffix"]
        # First chunk gets suffix only, middle chunks get both, last gets prefix only
        chunks[0] = chunks[0] + suffix
        for i in range(1, len(chunks) - 1):
            chunks[i] = prefix + chunks[i] + suffix
        if len(chunks) > 1:
            chunks[-1] = prefix + chunks[-1]

        # Process each chunk through generate_weights one at a time,
        # then stack into the format combine_lora expects:
        # {module: {"A": [tot_chunks, n_layers, r, dim], "B": [...]}}
        per_chunk_loras = []
        for chunk in chunks:
            chunk_ids = torch.tensor([chunk], device=model.device)
            chunk_mask = torch.ones_like(chunk_ids)
            loras, _ = model.generate_weights(chunk_ids, chunk_mask)
            per_chunk_loras.append(loras)

        # Stack: each loras dict has shape [1, n_layers, r, dim] -> cat to [n_chunks, ...]
        stacked_loras = {}
        for module_name in per_chunk_loras[0]:
            stacked_loras[module_name] = {
                "A": torch.cat([l[module_name]["A"] for l in per_chunk_loras], dim=0),
                "B": torch.cat([l[module_name]["B"] for l in per_chunk_loras], dim=0),
            }

        # Combine chunks into a single LoRA set and internalize
        from ctx_to_lora.modeling.lora_merger import combine_lora
        from ctx_to_lora.modeling.hypernet import apply_lora_to_layers
        combined_loras = combine_lora(
            stacked_loras,
            n_chunks=torch.tensor([n_chunks], device=model.device),
            lora_bias=model.hypernet.get_head_bias()
            if model.hypernet.config.use_bias
            else None,
        )

        # Patch LoRA forward and apply the combined weights
        model.patch_lora_forward()
        n_queries = torch.ones(1, dtype=torch.int32, device=model.device)
        apply_lora_to_layers(
            model.base_model,
            model.hypernet.layer_indices,
            combined_loras,
            n_queries,
            None,
        )
        # Mark as internalized so model.generate() uses the base model path
        model.generated_loras = True

    # ---- Helper: generate a reply for a prompt ----
    @torch.inference_mode()
    def generate_reply(prompt: str) -> str:
        chat = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        input_length = input_ids.size(-1)

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if stop_seq:
            gen_kwargs["stop_strings"] = [stop_seq]
            gen_kwargs["tokenizer"] = tokenizer

        # model.generate() handles both baseline (no LoRAs) and single-chunk
        # internalized cases. For chunked internalize, LoRAs are already applied
        # to the layers, so we call base_model.generate() directly.
        if model.generated_loras is True:
            output_ids = model.base_model.generate(**gen_kwargs)
        else:
            output_ids = model.generate(**gen_kwargs)

        return tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    # ---- Load RepoQA dataset & build tasks ----
    dataset = get_repoqa_data()
    tasks = []
    for lang, repos in dataset.items():
        if lang_filter is not None and lang not in lang_filter:
            continue
        for repo in repos:
            if "needles" not in repo:
                continue
            ordered_paths = topological_sort(repo["dependency"])
            file_content_list = [(p, repo["content"][p]) for p in ordered_paths]
            for i, needle in enumerate(repo["needles"]):
                position_ratio = (i + 0.5) / len(repo["needles"])
                task = {
                    "repo": repo["repo"],
                    "name": needle["name"],
                    "language": lang,
                    "path": needle["path"],
                    "position_ratio": position_ratio,
                    "description": f"\nFunction Description:{needle['description']}\n",
                    "instruction": INSTRUCTION,
                    "template": TEMPLATE,
                }
                ctx_info = make_code_context(
                    needle,
                    file_content_list,
                    position_ratio=position_ratio,
                    code_context_size=code_context_size,
                    language=lang,
                )
                task.update(ctx_info)
                tasks.append(task)

    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    print(f"Total RepoQA tasks: {len(tasks)}")

    # ---- Prepare output dirs ----
    base_dir = os.path.join(RESULTS_DIR, "repoqa", f"ntoken_{code_context_size}")
    os.makedirs(base_dir, exist_ok=True)
    baseline_path = os.path.join(base_dir, "d2l_baseline.jsonl")
    d2l_path = os.path.join(base_dir, "d2l_internalized.jsonl")

    # Resume support: skip already-completed tasks
    def load_done_ids(path):
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    done.add(make_task_id(r["language"], r["repo"], r["name"]))
        return done

    baseline_done = load_done_ids(baseline_path)
    d2l_done = load_done_ids(d2l_path)

    baseline_outputs = []
    d2l_outputs = []
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_outputs = [json.loads(line) for line in f]
    if os.path.exists(d2l_path):
        with open(d2l_path) as f:
            d2l_outputs = [json.loads(line) for line in f]

    # ---- Run evaluation ----
    with open(baseline_path, "a") as f_base, open(d2l_path, "a") as f_d2l:
        for idx, task in enumerate(tasks):
            tid = make_task_id(task["language"], task["repo"], task["name"])
            # Build the prompt from the template
            prompt = ""
            for key in task["template"].split("\n"):
                prompt += task[key]

            print(f"[{idx+1}/{len(tasks)}] {tid}")

            # --- Baseline run: no internalization ---
            if tid not in baseline_done:
                model.reset()
                reply_baseline = generate_reply(prompt)
                result_base = {**task, "output": [reply_baseline]}
                f_base.write(json.dumps(result_base) + "\n")
                f_base.flush()
                baseline_outputs.append(result_base)
                print(f"  baseline done ({len(reply_baseline)} chars)")

            # --- D2L run: internalize the code context ---
            if tid not in d2l_done:
                model.reset()
                chunked_internalize(task["code_context"])
                reply_d2l = generate_reply(prompt)
                result_d2l = {**task, "output": [reply_d2l]}
                f_d2l.write(json.dumps(result_d2l) + "\n")
                f_d2l.flush()
                d2l_outputs.append(result_d2l)
                print(f"  d2l done ({len(reply_d2l)} chars)")

    # ---- Compute scores ----
    print("\n=== Baseline Scores ===")
    baseline_scores = compute_score("d2l_baseline", dataset, baseline_outputs, False)
    baseline_score_path = os.path.join(base_dir, "d2l_baseline-SCORES.json")
    with open(baseline_score_path, "w") as f:
        json.dump(baseline_scores, f)

    print("\n=== D2L (Internalized) Scores ===")
    d2l_scores = compute_score("d2l_internalized", dataset, d2l_outputs, False)
    d2l_score_path = os.path.join(base_dir, "d2l_internalized-SCORES.json")
    with open(d2l_score_path, "w") as f:
        json.dump(d2l_scores, f)

    results_volume.commit()
    print(f"\nResults saved to {base_dir}")