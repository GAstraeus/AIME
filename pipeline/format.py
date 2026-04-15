"""Step 4: Convert segmented conversations into JSONL training data."""

import argparse
import json
import logging
from pathlib import Path

from pipeline.utils.bedrock import BedrockClient
from pipeline.utils.config import get_config, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are converting a text message conversation into training data for a language model that will learn to text like the user.

Rules:
1. The other person's messages become "user" role. The self messages become "assistant" role.
2. Preserve all text EXACTLY — emoji, typos, slang, capitalization, abbreviations. Do not correct or modify anything.
3. If consecutive messages from the same sender were merged (separated by newlines), keep them merged in a single turn.
4. Multi-turn conversations should be kept as a single training example with alternating user/assistant turns.
5. The conversation MUST start with a "user" turn (the other person's message). If the conversation starts with a "self" message, skip those leading self messages and start from the first "other" message.
6. Skip this conversation entirely if:
   - The user (self/assistant) never responds
   - The conversation is flagged as having too-implicit context
7. It is OK to split one long conversation into multiple training examples if there are natural break points where a sub-conversation is self-contained.
8. Every training example MUST have at least one user turn and one assistant turn.

Respond with ONLY a JSON array of training examples (or an empty array [] to skip):
[
  {
    "conversations": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]"""


def serialize_conversation(conversation: dict) -> str:
    """Serialize a conversation's turns for the LLM."""
    lines = []
    for turn in conversation.get("turns", []):
        role = turn["role"]
        content = turn["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_conversation(
    client: BedrockClient,
    conversation: dict,
    relationship: str,
    contact_name: str,
) -> list[dict]:
    """Convert a single conversation into training examples via Claude."""
    if conversation.get("flagged", False):
        return []

    turns = conversation.get("turns", [])
    if not turns:
        return []

    serialized = serialize_conversation(conversation)
    user_message = (
        f"Contact: {contact_name}\n"
        f"Relationship: {relationship}\n"
        f"Turns: {len(turns)}\n\n"
        f"Conversation:\n\n{serialized}"
    )

    try:
        examples = client.invoke_with_json(
            messages=[{"role": "user", "content": user_message}],
            system=SYSTEM_PROMPT,
        )
    except Exception:
        logger.exception("  Failed to format conversation")
        return []

    if not isinstance(examples, list):
        return []

    # Attach relationship context to each example
    valid_examples = []
    for ex in examples:
        convos = ex.get("conversations", [])
        if not convos:
            continue
        # Validate: must start with user, must have at least one assistant turn
        has_user = any(t["role"] == "user" for t in convos)
        has_assistant = any(t["role"] == "assistant" for t in convos)
        if has_user and has_assistant and convos[0]["role"] == "user":
            valid_examples.append({
                "conversations": convos,
                "context": relationship,
            })

    return valid_examples


def load_progress(progress_path: Path) -> set[str]:
    """Load the set of already-processed contact filenames."""
    if progress_path.exists():
        with open(progress_path) as f:
            return set(json.load(f))
    return set()


def save_progress(progress_path: Path, completed: set[str]):
    """Save the set of completed contact filenames."""
    with open(progress_path, "w") as f:
        json.dump(sorted(completed), f)


def format_all(force: bool = False):
    config = get_config()
    ensure_directories(config)

    processed_dir = config["DATA_PROCESSED_DIR"]
    training_dir = config["DATA_TRAINING_DIR"]
    map_path = config["RELATIONSHIP_MAP_PATH"]
    output_path = training_dir / "messages.jsonl"
    progress_path = training_dir / ".progress.json"

    # Load relationship map
    if not map_path.exists():
        logger.error("Relationship map not found at %s. Run classify.py first.", map_path)
        raise SystemExit(1)

    with open(map_path) as f:
        relationship_map = json.load(f)

    processed_files = sorted(processed_dir.glob("*.json"))
    if not processed_files:
        logger.error("No processed files found in %s. Run segment.py first.", processed_dir)
        raise SystemExit(1)

    # Resumability
    completed = set() if force else load_progress(progress_path)
    if force and output_path.exists():
        output_path.unlink()

    logger.info("Formatting %d contact(s) into training data...", len(processed_files))
    client = BedrockClient()

    total_examples = 0
    contacts_formatted = 0
    skipped = 0

    for filepath in processed_files:
        contact_key = filepath.stem

        if contact_key in completed:
            logger.info("  Skipping %s (already formatted)", contact_key)
            skipped += 1
            continue

        with open(filepath) as f:
            data = json.load(f)

        contact_name = data["contact_name"]
        conversations = data.get("conversations", [])

        # Look up relationship
        contact_info = relationship_map.get(contact_key, {})
        relationship = contact_info.get("relationship", "other")

        if not conversations:
            completed.add(contact_key)
            save_progress(progress_path, completed)
            continue

        logger.info(
            "  Formatting %s (%d conversations, relationship: %s)...",
            contact_name, len(conversations), relationship,
        )

        contact_examples = 0
        with open(output_path, "a") as out_f:
            for i, conversation in enumerate(conversations):
                examples = format_conversation(client, conversation, relationship, contact_name)
                for example in examples:
                    out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    contact_examples += 1

        total_examples += contact_examples
        contacts_formatted += 1
        completed.add(contact_key)
        save_progress(progress_path, completed)
        logger.info("  %s: %d training examples", contact_name, contact_examples)

    logger.info("--- Formatting Summary ---")
    logger.info("Contacts formatted: %d", contacts_formatted)
    logger.info("Skipped (already done): %d", skipped)
    logger.info("Total training examples: %d", total_examples)
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format conversations into JSONL training data")
    parser.add_argument("--force", action="store_true", help="Reformat all contacts from scratch")
    args = parser.parse_args()
    format_all(force=args.force)
