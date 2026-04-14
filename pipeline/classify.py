"""Step 3: LLM-based relationship classification per contact."""

import argparse
import json
import logging
import random
from pathlib import Path

from pipeline.utils.bedrock import BedrockClient
from pipeline.utils.config import get_config, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are classifying the relationship between the user (labeled "self") and another person (labeled "other") based on their text message history.

Classify into exactly one of these categories:
- partner: Romantic partner or spouse
- close_friend: Close personal friend
- family: Family member (parent, sibling, child, etc.)
- colleague: Work contact, professional relationship
- other: Acquaintances, service providers, or anyone who doesn't fit above

Analyze the following signals:
- Terms of endearment, pet names, "love you" → partner
- References to family roles (mom, dad, bro, sis, uncle, etc.) → family
- Work topics, formal tone, meeting references, project discussions → colleague
- High frequency, casual tone, social plans, inside jokes → close_friend
- Low frequency, transactional exchanges → other

When uncertain, default to "other".

Respond with ONLY this JSON:
{
  "relationship": "one_of_the_five_labels",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation of key signals observed"
}"""


def sample_messages(messages: list[dict], max_messages: int = 200) -> list[dict]:
    """Select a representative sample: first 50, last 50, and random from middle."""
    if len(messages) <= max_messages:
        return messages

    first = messages[:50]
    last = messages[-50:]
    middle = messages[50:-50]
    middle_sample = random.sample(middle, min(100, len(middle)))

    # Maintain chronological order
    sampled_indices = set(range(50))
    sampled_indices.update(range(len(messages) - 50, len(messages)))
    sampled_indices.update(
        messages.index(m) if m in messages[50:-50] else i + 50
        for i, m in enumerate(middle)
        if m in middle_sample
    )
    # Simpler: just concatenate and sort isn't needed since we serialise anyway
    return first + middle_sample + last


def serialize_sample(messages: list[dict]) -> str:
    """Serialize a message sample for the LLM."""
    lines = []
    for msg in messages:
        ts = msg.get("timestamp", "?")
        sender = msg["sender"]
        text = msg["text"]
        lines.append(f"[{ts}] {sender}: {text}")
    return "\n".join(lines)


def classify_contact(
    client: BedrockClient,
    contact_name: str,
    messages: list[dict],
) -> dict:
    """Classify the relationship for a single contact."""
    sample = sample_messages(messages)
    serialized = serialize_sample(sample)

    user_message = (
        f"Contact: {contact_name}\n"
        f"Total messages: {len(messages)} (showing sample of {len(sample)})\n\n"
        f"Message history:\n\n{serialized}"
    )

    result = client.invoke_with_json(
        messages=[{"role": "user", "content": user_message}],
        system=SYSTEM_PROMPT,
    )

    return result


def classify_all(force: bool = False):
    config = get_config()
    ensure_directories(config)

    raw_dir = config["DATA_RAW_DIR"]
    map_path = config["RELATIONSHIP_MAP_PATH"]

    # Load existing map to preserve manual edits
    existing_map = {}
    if map_path.exists():
        with open(map_path) as f:
            existing_map = json.load(f)

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        logger.error("No raw files found in %s. Run extract.py first.", raw_dir)
        raise SystemExit(1)

    logger.info("Classifying relationships for %d contact(s)...", len(raw_files))
    client = BedrockClient()

    classified = 0
    skipped = 0
    failed = 0

    for filepath in raw_files:
        contact_key = filepath.stem  # filename without .json

        if contact_key in existing_map and not force:
            logger.info("  Skipping %s (already classified, use --force to reclassify)", contact_key)
            skipped += 1
            continue

        with open(filepath) as f:
            data = json.load(f)

        contact_name = data["contact_name"]
        messages = data["messages"]

        if not messages:
            continue

        logger.info("  Classifying %s (%d messages)...", contact_name, len(messages))

        try:
            result = classify_contact(client, contact_name, messages)
            existing_map[contact_key] = {
                "name": contact_name,
                "handle_id": data["handle_id"],
                "relationship": result.get("relationship", "other"),
                "confidence": result.get("confidence", "unknown"),
                "reasoning": result.get("reasoning", ""),
            }
            classified += 1
            logger.info(
                "  %s → %s (%s)",
                contact_name,
                existing_map[contact_key]["relationship"],
                existing_map[contact_key]["confidence"],
            )
        except Exception:
            logger.exception("  Failed to classify %s", contact_name)
            failed += 1

    # Write the map
    with open(map_path, "w") as f:
        json.dump(existing_map, f, indent=2, ensure_ascii=False)

    logger.info("--- Classification Summary ---")
    logger.info("Classified: %d", classified)
    logger.info("Skipped (existing): %d", skipped)
    logger.info("Failed: %d", failed)
    logger.info("Relationship map written to %s", map_path)

    # Print distribution
    relationships = [v["relationship"] for v in existing_map.values()]
    for label in ("partner", "close_friend", "family", "colleague", "other"):
        count = relationships.count(label)
        logger.info("  %s: %d", label, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify contact relationships")
    parser.add_argument("--force", action="store_true", help="Reclassify all contacts")
    args = parser.parse_args()
    classify_all(force=args.force)
