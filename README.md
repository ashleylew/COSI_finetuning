# COSI Ollie Finetuning

QLoRA finetuning of Qwen3 models to create "Ollie," a virtual tour guide for the COSI museum.

## Overview

This codebase finetunes Qwen3-8B or Qwen3-32B using QLoRA + SFT (Supervised Fine-Tuning). Every training example includes the full set of museum reference documents (~25K tokens) in the system message alongside Ollie's persona prompt, then trains on multi-turn Visitor/Guide conversations.

## Prerequisites

- **Hardware**: 2x NVIDIA A6000 48GB (8B fits on 1 GPU, 32B needs 2)
- **CUDA**: 12.5
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
python src/train_sft.py --config configs/sft_qwen3_8b.yaml
```

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

| Parameter | Description |
|---|---|
| `model_name` | HuggingFace model ID |
| `train_file` | Path to the training data JSON file |
| `output_dir` | Where to save checkpoints and final adapter |
| `max_seq_length` | Max tokens per example (16384 for 8B, 32768 for 32B) |
| `lora_r` | LoRA rank (64) |
| `lora_alpha` | LoRA scaling factor (16) |
| `target_modules` | Which layers get LoRA adapters |
| `load_in_4bit` | Enable 4-bit quantization (QLoRA) |
| `per_device_train_batch_size` | Batch size per GPU |
| `gradient_accumulation_steps` | Effective batch = this x batch_size x num_gpus |
| `learning_rate` | Peak LR (2e-4 for 8B, 1e-4 for 32B) |
| `gradient_checkpointing` | Trade compute for memory |
| `optim` | Optimizer (paged_adamw_8bit for memory efficiency) |

## Extending to DPO

To add DPO (Direct Preference Optimization) training:

1. **Data**: Add a `build_preference_dataset()` function to `src/data_processing.py` that returns examples with `{"prompt": ..., "chosen": ..., "rejected": ...}` format
2. **Training script**: Create `src/train_dpo.py` (similar to `train_sft.py` but using TRL's `DPOTrainer` + `DPOConfig`)
3. **Config**: Create `configs/dpo_qwen3_8b.yaml` with DPO-specific params (`beta`, reference model path)
4. The SFT adapter becomes the starting checkpoint for DPO

## Troubleshooting

**OOM errors:**
- Reduce `max_seq_length` (the 8B config uses 16384 which fits on 1 A6000)
- Reduce `per_device_train_batch_size` to 1 (should already be 1)
- Ensure `gradient_checkpointing: true`
- For 32B, must use 2 GPUs with `accelerate launch`

**Slow training:**
- Check that `bf16: true` is set
- Ensure `HF_HOME=/project/.cache` so models load from cache, not re-downloaded

**Import errors:**
- Make sure you run from the project root: `python src/train_sft.py ...`
- Or set `PYTHONPATH=src`

**Model downloads into home directory:**
- You forgot to set `export HF_HOME=/project/.cache` before running
- Delete any accidental downloads and re-run with the env var set
