"""
Data processing pipeline for COSI Ollie finetuning.

Handles:
  - Loading museum reference documents from disk
  - Cleaning and filtering conversation turns from JSON training data
  - Building a HuggingFace Dataset in TRL's native messages format
"""

import json
import os
import re
from pathlib import Path

from datasets import Dataset

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where museum .txt documents live
DOCS_BASE = Path("/project/COSI_bot_version3/data_simulation_deepseek/2026_docs")
DOC_DIRS = ["general_docs", "permanent_exhibits", "temporary_exhibits"]

# Training data and prompt paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

_cached_documents: str | None = None


def load_documents() -> str:
    """Read all .txt files from the museum document directories.

    Files are sorted by name within each directory for reproducibility.
    Results are cached so the (identical) document block isn't re-read for
    every training example.

    Returns:
        A single string with all documents concatenated, separated by
        ``--- FILENAME ---`` headers.
    """
    global _cached_documents
    if _cached_documents is not None:
        return _cached_documents

    parts: list[str] = []
    for subdir in DOC_DIRS:
        dir_path = DOCS_BASE / subdir
        if not dir_path.is_dir():
            print(f"Warning: document directory not found: {dir_path}")
            continue
        for filepath in sorted(dir_path.glob("*.txt")):
            text = filepath.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"--- {filepath.name} ---\n{text}")

    _cached_documents = "\n\n".join(parts)
    return _cached_documents


def load_system_prompt() -> str:
    """Read the Ollie persona prompt from system_prompt.txt."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Conversation cleaning
# ---------------------------------------------------------------------------

# Matches turns that are entirely a stage direction: <...>
_STAGE_DIRECTION_RE = re.compile(r"^<[^>]+>$", re.DOTALL)

# Matches broken / artifact turns
_BROKEN_TURN_RE = re.compile(r"^(EXHIBIT:|SOURCE:)\s*", re.IGNORECASE)


def _clean_guide_utterance(text: str) -> str:
    """Strip 'Ollie:' prefixes that appear in some Guide utterances."""
    # Handle multi-line "Ollie: ... \nOllie: ..." by stripping each prefix
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"^Ollie:\s*", "", line)
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def _is_stage_direction(text: str) -> bool:
    """Check if the utterance is purely a stage direction like <...>."""
    return bool(_STAGE_DIRECTION_RE.match(text.strip()))


def _is_broken_turn(text: str) -> bool:
    """Check if the utterance is a broken artifact (e.g. 'EXHIBIT: Life]')."""
    text = text.strip()
    if _BROKEN_TURN_RE.match(text):
        return True
    # Also catch bare references like "EXHIBIT: Life]"
    if text.endswith("]") and "EXHIBIT:" in text:
        return True
    return False


def _parse_conversation(conv_dict: dict) -> list[dict]:
    """Parse a conversation dict into a list of (role, content) turns.

    Turns are sorted by turn number (not dict insertion order), cleaned,
    and filtered. Trailing idle/filler turns after the conversation
    naturally ends are truncated.
    """
    # Sort turns by their numeric key
    turn_items = []
    for key, turn in conv_dict.items():
        num = int(key.replace("Turn ", ""))
        turn_items.append((num, turn))
    turn_items.sort(key=lambda x: x[0])

    messages: list[dict] = []
    for _, turn in turn_items:
        role_raw = turn["Role"]
        utterance = turn["Utterance"].strip()

        # Skip broken turns
        if _is_broken_turn(utterance):
            continue

        # Skip stage directions
        if _is_stage_direction(utterance):
            # Once we hit a stage direction, truncate - everything after is
            # idle filler (the conversation has naturally ended)
            break

        # Map roles
        if role_raw == "Visitor":
            role = "user"
            content = utterance
        elif role_raw == "Guide":
            role = "assistant"
            content = _clean_guide_utterance(utterance)
        else:
            continue

        messages.append({"role": role, "content": content})

    return messages


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def build_dataset(train_file: str | None = None) -> Dataset:
    """Build a HuggingFace Dataset in TRL's messages format.

    Args:
        train_file: Path to a specific JSON training file. If None,
            loads all .json files from data/raw/.

    Each example has a single ``"messages"`` column containing a list of
    ``{"role": ..., "content": ...}`` dicts:
      - One ``system`` message (persona prompt + all reference documents)
      - Alternating ``user`` / ``assistant`` turns from the conversation

    TRL's SFTTrainer natively handles this format: it applies the
    tokenizer's chat_template and masks non-assistant tokens from the loss.
    """
    system_prompt = load_system_prompt()
    documents = load_documents()
    system_content = (
        system_prompt
        + "\n\n--- REFERENCE DOCUMENTS ---\n\n"
        + documents
    )

    # Determine which JSON files to load
    all_conversations: list[list[dict]] = []
    if train_file:
        train_path = Path(train_file)
        if not train_path.is_file():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        json_files = [train_path]
    else:
        json_files = sorted(RAW_DATA_DIR.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {RAW_DATA_DIR}. "
                "Place training data JSON files in data/raw/."
            )

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Each JSON file contains a list of conversation objects
        if isinstance(data, list):
            examples = data
        else:
            examples = [data]

        for example in examples:
            conv_dict = example.get("Conversation", {})
            if not conv_dict:
                continue

            turns = _parse_conversation(conv_dict)
            if len(turns) < 2:
                continue

            # Build full message list: system + conversation turns
            messages = [{"role": "system", "content": system_content}]
            messages.extend(turns)
            all_conversations.append(messages)

    print(f"Loaded {len(all_conversations)} conversations from {len(json_files)} file(s)")

    dataset = Dataset.from_dict({"messages": all_conversations})
    return dataset


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = build_dataset()
    print(f"\nDataset: {ds}")
    print(f"Number of examples: {len(ds)}")

    # Show first example summary
    if len(ds) > 0:
        sample = ds[0]["messages"]
        print(f"\nFirst example: {len(sample)} messages")
        print(f"  System message length: {len(sample[0]['content'])} chars")
        for msg in sample[1:]:
            preview = msg["content"][:80].replace("\n", " ")
            print(f"  [{msg['role']}] {preview}...")
