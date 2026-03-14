"""
Inference script for testing finetuned Ollie models.

Usage:
  Interactive mode:
    python src/inference.py --adapter /path/to/adapter

  Batch mode (reads visitor turns from JSON, generates guide responses):
    python src/inference.py --adapter /path/to/adapter --input data.json --output results.json
"""

import argparse
import json

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data_processing import load_documents, load_system_prompt


def load_model(adapter_path: str):
    """Load base model + LoRA adapter in 4-bit."""
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_cfg.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, conversation, max_new_tokens):
    """Generate a single assistant response given a conversation so far."""
    input_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_interactive(model, tokenizer, system_content, max_new_tokens):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("Ollie Interactive Chat")
    print("Type 'quit' or 'exit' to stop, 'reset' to clear history")
    print("=" * 60 + "\n")

    conversation = [{"role": "system", "content": system_content}]

    while True:
        try:
            user_input = input("Visitor: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            conversation = [{"role": "system", "content": system_content}]
            print("[Conversation reset]\n")
            continue

        conversation.append({"role": "user", "content": user_input})
        response = generate_response(model, tokenizer, conversation, max_new_tokens)
        conversation.append({"role": "assistant", "content": response})
        print(f"Ollie: {response}\n")


def run_batch(model, tokenizer, system_content, max_new_tokens, input_path, output_path):
    """Batch inference: read visitor turns from JSON, generate guide responses.

    For each conversation in the input file, replays the visitor turns and
    generates a model response after each one. Saves results with both the
    original guide utterance and the generated one for comparison.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    results = []

    for conv_idx, example in enumerate(data):
        conv_dict = example.get("Conversation", {})
        if not conv_dict:
            continue

        # Sort turns by number
        turn_items = []
        for key, turn in conv_dict.items():
            num = int(key.replace("Turn ", ""))
            turn_items.append((num, turn))
        turn_items.sort(key=lambda x: x[0])

        conversation = [{"role": "system", "content": system_content}]
        conv_result = {
            "conversation_index": conv_idx,
            "turns": [],
        }

        for _, turn in turn_items:
            role = turn["Role"]
            utterance = turn["Utterance"].strip()

            if role == "Visitor":
                conversation.append({"role": "user", "content": utterance})

            elif role == "Guide":
                # Generate model response
                print(f"  Conv {conv_idx}, generating response {len(conv_result['turns']) + 1}...")
                generated = generate_response(
                    model, tokenizer, conversation, max_new_tokens
                )

                conv_result["turns"].append({
                    "visitor": conversation[-1]["content"],
                    "original_guide": utterance,
                    "generated_guide": generated,
                })

                # Continue conversation with the generated response
                conversation.append({"role": "assistant", "content": generated})

        results.append(conv_result)
        print(f"Conversation {conv_idx}: {len(conv_result['turns'])} turns generated")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="COSI Ollie Inference")
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="Path to saved LoRA adapter directory",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input JSON file (same format as training data). "
             "If omitted, runs interactive mode.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save generated responses (required with --input)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Maximum tokens to generate per response",
    )
    args = parser.parse_args()

    if args.input and not args.output:
        parser.error("--output is required when using --input")

    model, tokenizer = load_model(args.adapter)

    system_prompt = load_system_prompt()
    documents = load_documents()
    system_content = (
        system_prompt + "\n\n--- REFERENCE DOCUMENTS ---\n\n" + documents
    )

    if args.input:
        run_batch(model, tokenizer, system_content, args.max_new_tokens,
                  args.input, args.output)
    else:
        run_interactive(model, tokenizer, system_content, args.max_new_tokens)


if __name__ == "__main__":
    main()
