"""Step 2: LLM-based conversation segmentation."""

import argparse
import json
import logging
from pathlib import Path

from pipeline.utils.bedrock import BedrockClient
from pipeline.utils.config import get_config, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are analyzing a text message history between the user (labeled "self") and another person (labeled "other").

Your task is to segment this history into distinct conversations.

Rules:
1. A new conversation starts when there is a clear topic shift, especially after a significant time gap. A 30+ minute gap with a topic shift is a strong signal. However, a reply to an earlier topic after a gap is NOT a new conversation — use semantic judgment.
2. Merge consecutive messages from the same sender into a single turn. Separate merged messages with newlines within the content field.
3. Preserve all text EXACTLY — emoji, typos, slang, abbreviations, capitalization. Do not correct or modify any message content.
4. Flag conversations where context is too implicit to be useful (e.g., references to a phone call or in-person event with no textual context, or isolated "yeah"/"ok" with no preceding context).
5. Each conversation should ideally have at least one message from each participant. Single-sided conversations should still be included but flagged.

Respond with ONLY a JSON array. Each element:
{
  "turns": [
    {"role": "other", "content": "message text"},
    {"role": "self", "content": "message text"}
  ],
  "flagged": false,
  "flag_reason": null
}"""


def serialize_messages(messages: list[dict]) -> str:
    """Serialize messages into a compact text format for the LLM."""
    lines = []
    for msg in messages:
        ts = msg.get("timestamp", "?")
        sender = msg["sender"]
        text = msg["text"]
        lines.append(f"[{ts}] {sender}: {text}")
    return "\n".join(lines)


def chunk_messages(
    messages: list[dict],
    max_messages_per_chunk: int = 400,
    overlap: int = 20,
) -> list[list[dict]]:
    """Split messages into chunks capped by message count.

    The output JSON duplicates all message text, so we limit by message count
    (not input tokens) to keep the output within max_tokens budget.
    Each chunk overlaps with the previous by `overlap` messages to give the
    LLM context at boundaries for correct conversation splitting.
    """
    if not messages:
        return []

    if len(messages) <= max_messages_per_chunk:
        return [messages]

    chunks = []
    start = 0
    while start < len(messages):
        end = min(start + max_messages_per_chunk, len(messages))
        chunks.append(messages[start:end])
        if end == len(messages):
            break
        start = end - overlap

    return chunks


def segment_chunk(client: BedrockClient, chunk: list[dict], contact_name: str) -> list[dict]:
    """Send one chunk of messages to Claude for segmentation."""
    serialized = serialize_messages(chunk)
    user_message = (
        f"Contact: {contact_name}\n\n"
        f"Message history ({len(chunk)} messages):\n\n{serialized}"
    )

    conversations = client.invoke_with_json(
        messages=[{"role": "user", "content": user_message}],
        system=SYSTEM_PROMPT,
        max_tokens=16384,
    )

    if not isinstance(conversations, list):
        logger.warning("Expected list from segmentation, got %s", type(conversations))
        return []

    return conversations


def deduplicate_across_chunks(chunk_results: list[list[dict]]) -> list[dict]:
    """Merge conversation lists from overlapping chunks.

    For overlapping regions, we keep conversations from the chunk where they
    are more centrally located (not at a boundary).
    """
    if len(chunk_results) <= 1:
        return chunk_results[0] if chunk_results else []

    all_conversations = []
    for i, conversations in enumerate(chunk_results):
        if i == 0:
            # First chunk: keep everything except the very last conversation
            # (it might be split across chunks)
            all_conversations.extend(conversations)
        else:
            # Subsequent chunks: skip the first conversation (likely overlaps
            # with the end of the previous chunk), keep the rest
            if len(conversations) > 1:
                all_conversations.extend(conversations[1:])
            elif conversations:
                # Only one conversation in this chunk — keep it
                all_conversations.extend(conversations)

    return all_conversations


def process_contact(raw_filepath: Path, client: BedrockClient, output_dir: Path) -> bool:
    """Process a single contact's raw messages into segmented conversations."""
    with open(raw_filepath) as f:
        data = json.load(f)

    contact_name = data["contact_name"]
    messages = data["messages"]

    if not messages:
        logger.info("  Skipping %s — no messages", contact_name)
        return False

    logger.info("  Segmenting %s (%d messages)...", contact_name, len(messages))

    chunks = chunk_messages(messages)
    logger.info("  Split into %d chunk(s)", len(chunks))

    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info("  Processing chunk %d/%d (%d messages)...", i + 1, len(chunks), len(chunk))
        conversations = segment_chunk(client, chunk, contact_name)
        chunk_results.append(conversations)
        logger.info("  Chunk %d: %d conversations", i + 1, len(conversations))

    all_conversations = deduplicate_across_chunks(chunk_results)

    output = {
        "handle_id": data["handle_id"],
        "contact_name": contact_name,
        "conversation_count": len(all_conversations),
        "conversations": all_conversations,
    }

    output_path = output_dir / raw_filepath.name
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("  %s: %d conversations written", contact_name, len(all_conversations))
    return True


def segment_all(force: bool = False, limit: int = None):
    config = get_config()
    ensure_directories(config)

    raw_dir = config["DATA_RAW_DIR"]
    processed_dir = config["DATA_PROCESSED_DIR"]

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        logger.error("No raw files found in %s. Run extract.py first.", raw_dir)
        raise SystemExit(1)

    if limit:
        raw_files = raw_files[:limit]

    logger.info("Segmenting %d contact(s)...", len(raw_files))
    client = BedrockClient()

    processed = 0
    skipped = 0
    failed = 0

    for filepath in raw_files:
        output_path = processed_dir / filepath.name
        if output_path.exists() and not force:
            logger.info("  Skipping %s (already processed, use --force to reprocess)", filepath.name)
            skipped += 1
            continue

        try:
            if process_contact(filepath, client, processed_dir):
                processed += 1
        except Exception:
            logger.exception("  Failed to process %s", filepath.name)
            failed += 1

    logger.info("--- Segmentation Summary ---")
    logger.info("Processed: %d", processed)
    logger.info("Skipped (existing): %d", skipped)
    logger.info("Failed: %d", failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment message histories into conversations")
    parser.add_argument("--force", action="store_true", help="Reprocess all contacts")
    parser.add_argument("--limit", type=int, help="Process only the first N contacts")
    args = parser.parse_args()
    segment_all(force=args.force, limit=args.limit)
