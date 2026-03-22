# COSI Ollie Finetuning

QLoRA finetuning of Qwen3 models to create "Ollie," a virtual tour guide for the COSI museum.

## Overview

This codebase finetunes Qwen3-8B or Qwen3-32B using QLoRA + SFT (Supervised Fine-Tuning). Every training example includes the full set of museum reference documents (~21K tokens across 36 files) in the system message alongside Ollie's persona prompt, then trains on multi-turn Visitor/Guide conversations.

## Prerequisites

- **Hardware**: 2x NVIDIA RTX A6000 48GB (8B trained on 1 GPU, 32B needs 2)
- **CUDA**: 12.8
- **Conda environment**: `COSI_finetuning` (Python 3.11)
- **Models**: Cached at `/project/.cache`

## Setup

```bash
conda activate COSI_finetuning
pip install -r requirements.txt
export HF_HOME=/project/.cache
```

**Important**: Always set `HF_HOME=/project/.cache` before running any script so model weights are loaded from the shared cache, not downloaded into your home directory.

## Data Format

Training data is specified per config via the `train_file` field (see [Config Reference](#config-reference)). Each JSON file contains a list of conversation objects:

```json
[
  {
    "Given Documents": ["exhibit_life.txt", "exhibit_dinos.txt"],
    "Visitor Visible Files": ["EXP_exhibit_life.txt"],
    "Visitor Access Mode": "SOME_DOCS",
    "Conversation": {
      "Turn 1": { "Role": "Visitor", "Utterance": "Hi! What's here?" },
      "Turn 2": { "Role": "Guide", "Utterance": "Welcome to COSI!", "Source(s)": ["EXHIBIT: Life"] }
    }
  }
]
```

The data pipeline automatically:
- Sorts turns by number
- Strips "Ollie:" prefixes from Guide utterances
- Filters stage directions (`<...>`) and truncates trailing idle turns
- Filters broken/artifact turns

### Adding New Training Data

1. Place new JSON files in `data/raw/` (or anywhere accessible)
2. Follow the schema above
3. Update `train_file` in your config YAML to point to the new file
4. Run the data pipeline test to verify: `python src/data_processing.py`

## Training

**8B model (single GPU):**
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py --config configs/sft_qwen3_8b.yaml
```

`CUDA_VISIBLE_DEVICES=0` is required for the 8B model. The full document context (~21K tokens) plus conversation turns exceeds the memory budget when split across two GPUs with Liger Kernel's fused loss. Pinning to one GPU avoids cross-device tensor errors and fits comfortably within 48GB.

**32B model (2 GPUs):**
```bash
accelerate launch --num_processes=2 src/train_sft.py --config configs/sft_qwen3_32b.yaml
```

Checkpoints are saved to `/project/Ash/COSI_tour_guide_spring2026_checkpoints/` (configurable via `output_dir` in the YAML).

## Inference

### Interactive mode

Chat with Ollie in the terminal:

```bash
python src/inference.py --adapter /project/Ash/COSI_tour_guide_spring2026_checkpoints/qwen3-8b-sft
```

Type messages as a visitor, and Ollie responds. Type `reset` to clear history, `quit` to exit.

### Batch mode

Run inference on a JSON file (same format as training data). The script replays visitor turns, generates Ollie's responses, and saves results with both the original and generated guide utterances side by side:

```bash
python src/inference.py \
  --adapter /project/Ash/COSI_tour_guide_spring2026_checkpoints/qwen3-8b-sft \
  --input data/raw/CLEANED_some_doc_access__1.json \
  --output results.json
```

Output format:
```json
[
  {
    "conversation_index": 0,
    "turns": [
      {
        "visitor": "Wow, it's so cool in here! ...",
        "original_guide": "Hi there! Welcome to my screen! ...",
        "generated_guide": "Hey! Welcome to COSI! ..."
      }
    ]
  }
]
```

## Config Reference

Key parameters in `configs/sft_qwen3_*.yaml`:

| Parameter | 8B value | 32B value | Description |
|---|---|---|---|
| `model_name` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-32B` | HuggingFace model ID |
| `train_file` | — | — | Path to the training data JSON file |
| `output_dir` | — | — | Where to save checkpoints and final adapter |
| `max_seq_length` | 28000 | 32768 | Max tokens per example — must exceed ~21K (system+docs) |
| `lora_r` | 64 | 64 | LoRA rank |
| `lora_alpha` | 64 | 64 | LoRA scaling factor (should equal `lora_r`) |
| `target_modules` | all projection layers | all projection layers | Which layers get LoRA adapters |
| `load_in_4bit` | true | true | Enable 4-bit quantization (QLoRA) |
| `per_device_train_batch_size` | 1 | 1 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | 16 | Effective batch = this × batch_size × num_gpus |
| `learning_rate` | 2e-4 | 1e-4 | Peak LR |
| `gradient_checkpointing` | true | true | Trade compute for memory |
| `optim` | paged_adamw_8bit | paged_adamw_8bit | Optimizer |

## Training Details (for reference / paper reporting)

### 8B model

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3-8B |
| Method | QLoRA (SFT) |
| LoRA rank / alpha | 64 / 64 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit NF4, double quantization, bfloat16 compute |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 (cosine schedule, 3% warmup) |
| Epochs | 3 |
| Batch size (effective) | 8 (batch 1 × grad accum 8) |
| Max sequence length | 28,000 tokens |
| Context: system + docs | ~20,938 tokens (36 files across general, permanent, and temporary exhibit docs) |
| Hardware | 1× NVIDIA RTX A6000 48GB |
| CUDA | 12.8 |
| Memory optimizations | Gradient checkpointing, Liger Kernel fused cross-entropy (`use_liger_kernel=True` in TRL SFTConfig) |
| Thinking mode | Disabled (`enable_thinking=False` in tokenizer chat template) |

## Extending to DPO

To add DPO (Direct Preference Optimization) training:

1. **Data**: Add a `build_preference_dataset()` function to `src/data_processing.py` that returns examples with `{"prompt": ..., "chosen": ..., "rejected": ...}` format
2. **Training script**: Create `src/train_dpo.py` (similar to `train_sft.py` but using TRL's `DPOTrainer` + `DPOConfig`)
3. **Config**: Create `configs/dpo_qwen3_8b.yaml` with DPO-specific params (`beta`, reference model path)
4. The SFT adapter becomes the starting checkpoint for DPO

## Troubleshooting

**OOM errors on the 8B model:**
- Make sure you're running with `CUDA_VISIBLE_DEVICES=0` (single GPU)
- The full document context is ~21K tokens; `max_seq_length` must be at least 22K — the current setting of 28K gives ~7K of headroom for conversations
- `use_liger_kernel=True` in SFTConfig is required — without it, the backward pass materializes a `[seq_len, vocab_size]` float32 tensor (~12 GB) that causes OOM
- Ensure `gradient_checkpointing: true`

**Cross-device tensor errors with Liger Kernel:**
- Use `CUDA_VISIBLE_DEVICES=0` so `device_map="auto"` maps everything to a single GPU
- Liger's fused loss does not support multi-GPU model parallelism (`device_map="auto"` across multiple devices)

**Slow training:**
- Check that `bf16: true` is set
- Ensure `HF_HOME=/project/.cache` so models load from cache, not re-downloaded

**Import errors:**
- Make sure you run from the project root: `python src/train_sft.py ...`
- Or set `PYTHONPATH=src`

**Model downloads into home directory:**
- You forgot to set `export HF_HOME=/project/.cache` before running
- Delete any accidental downloads and re-run with the env var set

**Qwen3 thinking mode:**
- Qwen3 defaults to thinking mode (generates `<think>...</think>` tokens). This is disabled at inference via `enable_thinking=False` in `apply_chat_template`, and during training via a monkey-patch on the tokenizer in `train_sft.py`. Do not remove these without understanding the implications for training/inference consistency.
