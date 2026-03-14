"""
SFT training script for COSI Ollie finetuning.

Usage:
  Single GPU (8B):
    python src/train_sft.py --config configs/sft_qwen3_8b.yaml

  Multi-GPU (32B):
    accelerate launch --num_processes=2 src/train_sft.py --config configs/sft_qwen3_32b.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from data_processing import build_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="COSI Ollie SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/sft_qwen3_8b.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"Model: {cfg['model_name']}")
    print(f"Output: {cfg['output_dir']}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # --- LoRA config ---
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Dataset ---
    dataset = build_dataset(train_file=cfg.get("train_file"))
    print(f"Training on {len(dataset)} examples")

    # --- Training config ---
    output_dir = Path(cfg["output_dir"])
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        max_length=cfg["max_seq_length"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg["optim"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to="none",
        remove_unused_columns=False,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        chat_template_kwargs={"enable_thinking": False},
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save ---
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done!")


if __name__ == "__main__":
    main()
