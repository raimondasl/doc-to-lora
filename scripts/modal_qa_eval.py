"""Modal script for D2L QA evaluation (SQuAD, DROP).

Runs a three-way comparison:
  1. No-context baseline  — base model answers without any context
  2. D2L internalized     — context compressed into LoRA weights
  3. Full-context         — context in the prompt (upper bound)

Usage:
  # Quick test (10 samples, SQuAD only)
  modal run scripts/modal_qa_eval.py::run_qa_eval --datasets squad --max-tasks 10

  # Full evaluation
  modal run scripts/modal_qa_eval.py::run_qa_eval --datasets squad,drop --max-tasks 500

  # D2L only, different checkpoint
  modal run scripts/modal_qa_eval.py::run_qa_eval --no-run-baseline --no-run-full-context \
      --checkpoint-name "gemma_2b_d2l/checkpoint-20000"
"""

import modal

# --- Image Definition ---
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_dir(".", "/app", copy=True)
    .workdir("/app")
    .run_commands(
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands("pip install -e .")
    .run_commands("pip install tokenizers==0.21.0")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        "pip install flashinfer-python==0.2.2 "
        "-i https://flashinfer.ai/whl/cu124/torch2.6"
    )
    # Download SQuAD dataset into the image (DROP loads from HF Hub directly)
    .run_commands(
        "HF_HUB_ENABLE_HF_TRANSFER=1 pip install hf_transfer && "
        "huggingface-cli download --repo-type dataset rajpurkar/squad "
        "--local-dir data/raw_datasets/squad"
    )
)

app = modal.App("doc-to-lora-qa-eval", image=image)

# Persistent volumes
model_volume = modal.Volume.from_name("doc-to-lora-models", create_if_missing=True)
results_volume = modal.Volume.from_name("doc-to-lora-qa-results", create_if_missing=True)
MODEL_DIR = "/models"
RESULTS_DIR = "/results"


@app.function(
    gpu="A100",
    volumes={MODEL_DIR: model_volume, RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
)
def run_qa_eval(
    datasets: str = "squad,drop",
    max_tasks: int = 500,
    checkpoint_name: str = "gemma_demo/checkpoint-80000",
    split: str = "test",
    run_baseline: bool = True,
    run_d2l: bool = True,
    run_full_context: bool = True,
):
    """Run three-way QA evaluation: no-context vs D2L vs full-context."""
    import json
    import os
    import sys

    sys.path.insert(0, "/app")
    os.environ["WANDB_MODE"] = "disabled"
    # Limit multiprocessing workers for datasets .map()/.filter() calls.
    # Hardcoded num_proc=16 deadlocks in containers with small /dev/shm.
    os.environ["D2L_NUM_PROC"] = "4"

    from ctx_to_lora.eval_utils import run_eval

    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    checkpoint_path = f"{MODEL_DIR}/trained_d2l/{checkpoint_name}/pytorch_model.bin"

    # Detect base model from checkpoint args.yaml
    import yaml

    checkpoint_dir = os.path.dirname(checkpoint_path)
    run_dir = os.path.dirname(checkpoint_dir)
    args_yaml = os.path.join(run_dir, "args.yaml")
    with open(args_yaml) as f:
        train_args = yaml.unsafe_load(f)
    base_model = train_args.get(
        "model_name_or_path",
        train_args.get("base_model_name_or_path", "google/gemma-2-2b-it"),
    )
    print(f"Base model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Datasets: {ds_list}")
    print(f"Max tasks per dataset: {max_tasks}")
    print(f"Split: {split}")
    print()

    all_results = {}

    # --- Run 1: No-context baseline ---
    if run_baseline:
        print("=" * 60)
        print("RUN 1: No-context baseline")
        print("=" * 60)
        metrics = run_eval(
            model_name_or_path=base_model,
            datasets=ds_list,
            split=split,
            eval_batch_size=1,
            max_test_samples_per_ds=max_tasks,
            max_val_samples_per_ds=max_tasks,
            remove_context=True,
            generative=True,
        )
        all_results["no_context"] = metrics
        print()

    # --- Run 2: D2L internalized ---
    if run_d2l:
        print("=" * 60)
        print("RUN 2: D2L internalized")
        print("=" * 60)
        metrics = run_eval(
            checkpoint_path=checkpoint_path,
            datasets=ds_list,
            split=split,
            eval_batch_size=1,
            max_test_samples_per_ds=max_tasks,
            max_val_samples_per_ds=max_tasks,
            max_ctx_chunk_len=8192,
            generative=True,
        )
        all_results["d2l"] = metrics
        print()

    # --- Run 3: Full-context upper bound ---
    if run_full_context:
        print("=" * 60)
        print("RUN 3: Full-context (upper bound)")
        print("=" * 60)
        metrics = run_eval(
            model_name_or_path=base_model,
            datasets=ds_list,
            split=split,
            eval_batch_size=1,
            max_test_samples_per_ds=max_tasks,
            max_val_samples_per_ds=max_tasks,
            remove_context=False,
            generative=True,
        )
        all_results["full_context"] = metrics
        print()

    # --- Print summary table ---
    print()
    print("=" * 70)
    print("SUMMARY: QA F1 Scores")
    print("=" * 70)

    run_labels = {
        "no_context": "No-context baseline",
        "d2l": "D2L internalized",
        "full_context": "Full-context (upper bound)",
    }

    # Collect F1 scores per dataset per run.
    # run_eval returns: {"test_squad": {"test_squad_qa_f1_score": 0.82, ...}, ...}
    # With remove_context: {"test_squad_no_context": {"test_squad_no_context_qa_f1_score": ...}}
    summary = {}
    for run_name, metrics in all_results.items():
        if metrics is None:
            continue
        for split_key, metric_dict in metrics.items():
            if not isinstance(metric_dict, dict):
                continue
            for ds_name in ds_list:
                if ds_name.replace("/", "_") not in split_key.replace("/", "_"):
                    continue
                for k, v in metric_dict.items():
                    if k.endswith("qa_f1_score"):
                        if ds_name not in summary:
                            summary[ds_name] = {}
                        summary[ds_name][run_name] = v
                        break

    # Print table
    header = f"{'Dataset':<20}"
    for run_name in ["no_context", "d2l", "full_context"]:
        if run_name in all_results:
            header += f"{run_labels[run_name]:<25}"
    print(header)
    print("-" * len(header))

    for ds_name in ds_list:
        row = f"{ds_name:<20}"
        if ds_name in summary:
            for run_name in ["no_context", "d2l", "full_context"]:
                if run_name in all_results:
                    score = summary[ds_name].get(run_name, "N/A")
                    if isinstance(score, float):
                        row += f"{score:<25.4f}"
                    else:
                        row += f"{str(score):<25}"
        else:
            row += "  (no F1 scores found — check raw metrics below)"
        print(row)

    # Also dump raw metrics for debugging
    print()
    print("=" * 70)
    print("RAW METRICS")
    print("=" * 70)
    for run_name, metrics in all_results.items():
        print(f"\n--- {run_labels.get(run_name, run_name)} ---")
        if metrics:
            print(json.dumps(metrics, indent=2, default=str))

    # Save summary to results volume
    summary_path = os.path.join(RESULTS_DIR, "qa_eval_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {"results": all_results, "config": {
                "datasets": ds_list,
                "max_tasks": max_tasks,
                "checkpoint_name": checkpoint_name,
                "base_model": base_model,
                "split": split,
            }},
            f,
            indent=2,
            default=str,
        )
    results_volume.commit()
    print(f"\nResults saved to {summary_path}")
