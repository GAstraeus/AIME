"""Step 2: LLM-based conversation segmentation."""

import argparse
import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

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
    max_messages_per_chunk: int = 50,
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
        max_tokens=32768,
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


# --- Chunk caching for resumption ---

def get_chunk_cache_dir(processed_dir: Path, contact_stem: str) -> Path:
    return processed_dir / ".chunk_cache" / contact_stem


def save_chunk_result(cache_dir: Path, chunk_index: int, result: list[dict]):
    """Atomically save a single chunk result to the cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"chunk_{chunk_index:04d}.json"
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    tmp.rename(target)


def load_cached_chunks(cache_dir: Path) -> dict[int, list[dict]]:
    """Load all cached chunk results. Returns {index: result_list}."""
    cached = {}
    if not cache_dir.exists():
        return cached
    for f in cache_dir.glob("chunk_*.json"):
        try:
            index = int(f.stem.split("_")[1])
            with open(f) as fh:
                cached[index] = json.load(fh)
        except (ValueError, json.JSONDecodeError, IndexError):
            pass
    return cached


def clear_chunk_cache(cache_dir: Path):
    """Remove chunk cache directory after successful assembly."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


# --- Thread-local BedrockClient ---

_thread_local = threading.local()


def _get_thread_client() -> BedrockClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = BedrockClient()
    return _thread_local.client


def process_contact(
    raw_filepath: Path,
    output_dir: Path,
    progress_bar=None,
    workers: int = 1,
) -> bool:
    """Process a single contact's raw messages into segmented conversations."""
    with open(raw_filepath) as f:
        data = json.load(f)

    contact_name = data["contact_name"]
    messages = data["messages"]

    if not messages:
        return False

    senders = {m["sender"] for m in messages}
    if senders == {"other"}:
        return False

    chunks = chunk_messages(messages)
    cache_dir = get_chunk_cache_dir(output_dir, raw_filepath.stem)
    cached = load_cached_chunks(cache_dir)

    # Advance progress bar past already-cached chunks
    if progress_bar is not None and cached:
        progress_bar.update(len(cached))

    pending = [i for i in range(len(chunks)) if i not in cached]

    failed_chunks = []

    if pending:
        def _process_one(chunk_index: int) -> tuple[int, list[dict]]:
            client = _get_thread_client()
            result = segment_chunk(client, chunks[chunk_index], contact_name)
            save_chunk_result(cache_dir, chunk_index, result)
            return chunk_index, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, i): i for i in pending}
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    idx, result = future.result()
                    cached[idx] = result
                except Exception:
                    logger.warning("Chunk %d failed for %s, will retry on next run", chunk_idx, contact_name)
                    failed_chunks.append(chunk_idx)
                if progress_bar is not None:
                    progress_bar.update(1)

    if failed_chunks:
        logger.warning(
            "%s: %d/%d chunks failed — cached %d. Re-run to retry failed chunks.",
            contact_name, len(failed_chunks), len(chunks), len(cached),
        )
        return False

    # Assemble in order and deduplicate
    chunk_results = [cached[i] for i in range(len(chunks))]
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

    clear_chunk_cache(cache_dir)
    return True


def segment_all(force: bool = False, limit: int = None, workers: int = 3):
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

    # Clear chunk caches when forcing a full reprocess
    if force:
        cache_root = processed_dir / ".chunk_cache"
        if cache_root.exists():
            shutil.rmtree(cache_root)

    logger.info("Segmenting %d contact(s) with %d worker(s)...", len(raw_files), workers)

    # Pre-calculate total and cached chunks for the progress bar
    total_chunks = 0
    total_cached = 0
    skip_set = set()
    for filepath in raw_files:
        output_path = processed_dir / filepath.name
        if output_path.exists() and not force:
            skip_set.add(filepath)
            continue

        with open(filepath) as f:
            data = json.load(f)
        messages = data["messages"]
        if not messages:
            continue
        senders = {m["sender"] for m in messages}
        if senders == {"other"}:
            continue
        n_chunks = len(chunk_messages(messages))
        total_chunks += n_chunks

        cache_dir = get_chunk_cache_dir(processed_dir, filepath.stem)
        total_cached += len(load_cached_chunks(cache_dir))

    processed = 0
    skipped = 0
    failed = 0

    chunk_bar = tqdm(total=total_chunks, initial=total_cached, desc="Segmenting", unit="chunk")

    for filepath in raw_files:
        if filepath in skip_set:
            skipped += 1
            continue

        chunk_bar.set_postfix_str(filepath.stem)
        try:
            if process_contact(filepath, processed_dir, progress_bar=chunk_bar, workers=workers):
                processed += 1
        except Exception:
            logger.exception("  Failed to process %s", filepath.name)
            failed += 1

    chunk_bar.close()

    logger.info("--- Segmentation Summary ---")
    logger.info("Processed: %d", processed)
    logger.info("Skipped (existing): %d", skipped)
    logger.info("Failed: %d", failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment message histories into conversations")
    parser.add_argument("--force", action="store_true", help="Reprocess all contacts")
    parser.add_argument("--limit", type=int, help="Process only the first N contacts")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent chunk workers (default: 3)")
    args = parser.parse_args()
    segment_all(force=args.force, limit=args.limit, workers=args.workers)
