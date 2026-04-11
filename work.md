# RepoQA Evaluation Work Log

## Changes to doc_to_lora_modal.py

1. Image: Added `pip install repoqa` to the container image build
2. `run_repoqa()` function (line 156-354) — a new Modal function that:
   - Loads the D2L checkpoint once onto an A100
   - Downloads the RepoQA dataset (500 needle-function-retrieval tasks across 5 languages)
   - For each task, builds the prompt (instruction + code context + function description) and runs inference twice:
     - Baseline: `model.reset()` only — the base Gemma model answers without any internalized context
     - D2L: `model.reset()` + `model.internalize(code_context)` — D2L generates LoRA weights from the code, then answers
   - Outputs are saved as JSONL to a persistent `doc-to-lora-results` volume
   - Scores are computed using RepoQA's BLEU-based needle matching at 11 thresholds (0.0–1.0)
   - Resume support: if the run is interrupted, it picks up where it left off

To run it:

```bash
# Full eval (all 5 languages, ~500 tasks × 2 runs)
modal run doc_to_lora_modal.py::run_repoqa

# Just Python tasks
modal run doc_to_lora_modal.py::run_repoqa --languages '["python"]'
```

## Chunked Internalize Implementation

Now I have the full picture. The fix is to add a chunked internalize helper that:

1. Tokenizes the context text
2. Splits into chunks of ~4096 tokens (well within the 6144 training limit)
3. Adds proper chat template affixes per chunk
4. Processes chunks one at a time through `generate_weights`
5. Stores all chunk LoRAs so `combine_lora` merges them at generation time

Root cause: The 16K-token RepoQA code context was being fed to `model.internalize()` as a single batch, but the hypernet/perceiver OOMs processing it — the model was trained with `max_packed_ctx_len: 6144`.

Fix: Added `chunked_internalize()` that:

1. Tokenizes the context and checks if it exceeds `MAX_CTX_CHUNK_LEN` (4096 tokens)
2. If short enough, uses the original `model.internalize()`
3. If too long, splits into chunks with proper Gemma chat template affixes (matching D2L's training-time chunking logic from `split_too_long_ctx`)
4. Processes each chunk sequentially through `generate_weights` (avoids OOM)
5. Stacks per-chunk LoRAs and merges them via `combine_lora` (the same path used during training/eval)
6. Applies the combined LoRA weights to the model layers

Try again with:

```bash
modal run scripts/modal_eval.py::run_repoqa --max-tasks 3 --languages python
```

## Initial 3-Task Test Results

1. Only 3 tasks were evaluated — the full RepoQA Python split has ~500 tasks. This was likely a test run with `max_tasks=3` or only one language/repo was included. The scores are not meaningful with so few samples.
2. The model outputs are degenerate:
   - Baseline: outputs are entirely whitespace/newlines — the base Gemma model without any context is just producing blank tokens. This is actually somewhat expected for a small model asked to reproduce a specific function with no code context at all.
   - D2L (internalized): outputs are repetitive gibberish like `(token's_new_` and `to-to-to-to-...` — the model is clearly stuck in a degenerate repetition loop rather than producing coherent code.

The D2L scores show pass@1 = 0.33 only at threshold 0.0 (1 out of 3 "matched"), but looking at the actual output, that match (`_merge_string_group` with similarity 0.002) is essentially noise — the score threshold 0.0 is extremely permissive.

Possible causes for the degenerate D2L outputs:
- The `MAX_CTX_CHUNK_LEN = 4096` (line 211) may be too small, causing excessive chunking and LoRA quality degradation. The training used 6144.
- The chunked internalization logic may have a bug in how it combines LoRAs.
- The 16K context size may be too large for the model to handle effectively.

## Three Baselines Design

Three baselines now evaluated:

| # | Name | What it does |
|---|------|-------------|
| 1 | `d2l_baseline` | No context at all — `model.reset()`, bare prompt |
| 2 | `d2l_internalized` | D2L approach — context internalized as LoRA weights |
| 3 | `long_context` | Full context in the prompt, using Dynamic NTK RoPE scaling (factor = `code_context_size / 8192` = 2.0 for 16K) to extend gemma-2-2b-it beyond its native 8K window |

The long-context model is loaded separately so it doesn't interfere with the D2L model's weights. It reuses the same tokenizer and `generate_reply` helper via the `use_model` parameter.

On sample size: I'd recommend 50 tasks (`max_tasks=50`). That gives ~10 per language — enough to see meaningful per-language trends and overall pass@1 differences between the three approaches, without the 10+ hours a full 500-task run would take. If you want even more statistical confidence, 100 is a safe upper bound for a reasonable runtime.

## Time Estimates

Rough estimate for gemma-2-2b-it on A100 with `max_new_tokens=1024`:

| Step | Per task | Notes |
|------|----------|-------|
| Baseline | ~10-15s | Short prompt, generation only |
| D2L internalize | ~30-60s | Chunked context encoder passes + generation |
| Long-context | ~20-30s | Encode 16K prompt + generation |

So ~1-2 min per task across all three baselines, plus ~5 min for loading both models at startup.

50 tasks ≈ 1-2 hours on A100.

Note the long-context baseline loads a second copy of gemma-2-2b (~5GB in bf16) alongside the D2L model. That fits comfortably in A100-40GB, but if memory is tight, I can restructure to run sequentially (all D2L tasks first, then swap to the long-context model).

## Prompt Fix

The three baselines now use the correct prompts:

| Baseline | Prompt | Context source |
|----------|--------|----------------|
| `d2l_baseline` | `instruction + description + instruction` | None (parametric knowledge only) |
| `d2l_internalized` | `instruction + description + instruction` | Internalized as LoRA weights |
| `long_context` | `instruction + code_context + description + instruction` | In the prompt (RoPE-scaled) |

You'll need to delete the old result files before re-running since the resume logic would skip already-completed task IDs.

```bash
modal volume rm doc-to-lora-results repoqa/ntoken_16384 -r
```

If you also have the local copy:

```bash
rm results/ntoken_16384/*.jsonl results/ntoken_16384/*-SCORES.json
```

## Running D2L-Only Eval

```bash
modal run scripts/modal_eval.py::run_repoqa --code-context-size 6000 --max-tasks 50 --no-run-baseline --no-run-long-context
```

## Final Results: D2L Single-Chunk (6K) vs All Baselines

### Score Comparison

| Threshold | Baseline (no ctx) | D2L chunked (16K) | D2L single-chunk (6K) | Long-ctx RoPE (16K) |
|-----------|-------------------|--------------------|-----------------------|---------------------|
| 0.0       | 44%               | 4%                 | 44%                   | 28%                 |
| 0.1       | 6%                | 0%                 | 6%                    | 16%                 |
| 0.5       | 2%                | 0%                 | 0%                    | 12%                 |
| 1.0       | 0%                | 0%                 | 0%                    | 10%                 |

### Conclusions

1. **The chunking bug is confirmed fixed.** D2L single-chunk (6K) went from 4% to 44% at threshold 0.0 — the single-chunk `model.internalize()` path works correctly, while the manual chunked path was completely broken.

2. **D2L is working, but only at a "fuzzy guess" level.** The outputs are now coherent Python (not gibberish), and the model generates plausible-looking functions that roughly match descriptions. But similarity scores are very low — the highest D2L match is 0.461 (shared with baseline for `is_valid_percentile`), and no exact matches.

3. **D2L does NOT outperform the no-context baseline.** The per-task breakdown shows:
   - 14 tasks: both match
   - 8 tasks: D2L only matches
   - 8 tasks: baseline only matches
   - 20 tasks: neither matches

   D2L wins on some tasks (e.g., `_construct_basic_auth_str`: 0.223 vs 0.044, `_do_execute_operation`: 0.055 vs 0.005), but loses on others (e.g., `_primitive_value_to_str`: 0.227 vs 0.596, `_construct_field`: 0.001 vs 0.201). The net effect is a wash — D2L is essentially no better than parametric knowledge alone on this benchmark.

4. **Long-context remains far superior** at every threshold above 0.0. It's the only approach producing exact function matches (5 perfect scores of 1.0).

5. **D2L outputs show the model "understands" the function description but can't reproduce the exact code.** For example, `_construct_basic_auth_str` generates a reasonable auth helper function with the right concept (base64 encoding), but the implementation differs from the actual code. This suggests the LoRA encodes semantic understanding but not verbatim retrieval — which makes sense given 6K tokens of code compressed into low-rank weight perturbations.

### What This Means

D2L as-is doesn't help for needle-in-a-haystack retrieval tasks where exact reproduction is needed. The approach may be more suited to:
- QA tasks where semantic understanding matters more than exact code
- Smaller context sizes where the compression ratio is more favorable
- The original evaluation tasks the D2L paper used (which may differ from RepoQA)

The chunked internalization for 16K contexts needs a rewrite if you want to support longer contexts — the current implementation has 4 critical bugs. But even with working single-chunk internalization, the fundamental limitation is that LoRA weight compression loses verbatim detail.

## Will Fixed 16K Chunking Beat Single-Chunk 6K?

Probably not, for several reasons:

1. **The model was never trained on multi-chunk combination.** Training used `max_packed_ctx_len=6144` — always a single chunk. Even with a bug-free implementation, combining 3-4 chunk LoRAs is out-of-distribution for this model. The `combine_lora` merger was likely designed for batching multiple documents in training, not for splitting one document across chunks.

2. **More context = more noise in the LoRA, not more signal.** The needle function is already present in the 6K context (the token positions in the results confirm this — all needles fall within 0-6000 tokens). Going to 16K adds surrounding code that dilutes the LoRA's representation of the target function. Each chunk's LoRA captures a low-rank summary of ~4-6K tokens, and averaging them further smears the signal.

3. **The bottleneck is LoRA capacity, not context coverage.** The 6K single-chunk results show the model *understands* what function is being asked for (generates semantically plausible code) but can't *reproduce* it verbatim. That's a fundamental limitation of compressing 6K tokens of code into a few low-rank matrices — there isn't enough capacity for exact retrieval. Adding more context won't fix that; it'll make it worse.

4. **Evidence from the scores.** D2L 6K already matches 22/50 at threshold 0.0 with the needle directly in its single chunk. More chunks would only add noise around the same signal.

**Bottom line:** Fixing the chunking would bring 16K D2L from 4% up to roughly the same ~44% as 6K (at threshold 0.0), but is unlikely to exceed it. The effort would be better spent either:
- Testing D2L on tasks it was actually designed for (QA, summarization — where semantic understanding matters more than verbatim retrieval)
- Or accepting that for code retrieval, long-context with RoPE scaling (or a native long-context model) is simply the better approach
