"""Step 1: Extract 1:1 iMessages from chat.db."""

import argparse
import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.utils.config import get_config, ensure_directories
from pipeline.utils.contacts import ContactResolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Apple's Core Data epoch: 2001-01-01 00:00:00 UTC
CORE_DATA_EPOCH_OFFSET = 978307200


def convert_timestamp(mac_ts: int) -> str:
    """Convert a macOS Core Data timestamp to ISO 8601.

    Older macOS versions store seconds since 2001-01-01.
    Newer versions store nanoseconds. We detect by magnitude.
    """
    if mac_ts is None:
        return None
    # Nanoseconds: value > 1e15 (dates after ~2001 in nanoseconds)
    if abs(mac_ts) > 1e15:
        unix_ts = (mac_ts / 1e9) + CORE_DATA_EPOCH_OFFSET
    else:
        unix_ts = mac_ts + CORE_DATA_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def is_automated_sender(handle_id: str) -> bool:
    """Return True if the handle looks like an automated/system sender."""
    if not handle_id:
        return True
    stripped = handle_id.strip()
    # Short codes: purely numeric, fewer than 7 digits, no + prefix
    digits_only = re.sub(r"\D", "", stripped)
    if stripped == digits_only and len(digits_only) < 7:
        return True
    # Known automated patterns
    lower = stripped.lower()
    automated_patterns = ["noreply", "no-reply", "alert", "notify", "verification", "verify"]
    if any(p in lower for p in automated_patterns):
        return True
    return False


def sanitize_filename(name: str) -> str:
    """Sanitize a contact name for use as a filename."""
    cleaned = re.sub(r"[^\w\s-]", "", name)
    cleaned = re.sub(r"[\s]+", "_", cleaned).strip("_").lower()
    return cleaned or "unknown"


def get_group_chat_message_ids(conn: sqlite3.Connection) -> set[int]:
    """Return the set of message ROWIDs that belong to group chats.

    A group chat is any chat with more than one handle in chat_handle_join.
    """
    cursor = conn.execute("""
        SELECT DISTINCT cmj.message_id
        FROM chat_message_join cmj
        WHERE (SELECT COUNT(*) FROM chat_handle_join WHERE chat_id = cmj.chat_id) > 1
    """)
    return {row[0] for row in cursor.fetchall()}


def get_all_handles(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Return all (ROWID, identifier) pairs from the handle table."""
    cursor = conn.execute("SELECT ROWID, id FROM handle")
    return cursor.fetchall()


def decode_attributed_body(data: bytes) -> str | None:
    """Extract plain text from an iMessage attributedBody blob.

    Newer macOS versions store message content as a serialized NSAttributedString
    (streamtyped NSKeyedArchiver format) in the attributedBody column instead of
    the plain text column. The string value is length-prefixed after the NSString
    class marker in the binary payload.
    """
    if not data:
        return None
    try:
        pos = data.find(b"NSString")
        if pos == -1:
            return None
        pos += 8  # skip "NSString"
        pos += 4  # skip class version bytes
        if pos >= len(data):
            return None
        b = data[pos]
        if b == 0x85:  # 4-byte big-endian length follows
            pos += 1
            length = int.from_bytes(data[pos : pos + 4], "big")
            pos += 4
        elif b >= 0x81:  # variable-length: low nibble = byte count
            n = b & 0x0F
            pos += 1
            length = int.from_bytes(data[pos : pos + n], "big")
            pos += n
        else:  # single-byte length
            length = b
            pos += 1
        if length == 0 or pos + length > len(data):
            return None
        return data[pos : pos + length].decode("utf-8", errors="replace")
    except Exception:
        return None


def clean_text(text: str) -> str | None:
    """Strip control characters and binary artifacts from message text.

    attributedBody decoding can leak NSKeyedArchiver fragments (e.g. NSDiction,
    NSString) and control characters into the extracted text.
    """
    # Strip control characters (keep newlines and tabs)
    cleaned = re.sub(r"[^\S \n\t]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Remove leaked NSKeyedArchiver class names
    cleaned = re.sub(r"NSDiction\w*|NSString\w*|NSMutable\w*|NSObject\w*", "", cleaned)
    # Collapse multiple spaces
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned if cleaned else None


def get_messages_for_handle(
    conn: sqlite3.Connection,
    handle_rowid: int,
    group_message_ids: set[int],
) -> list[dict]:
    """Return all 1:1 text messages for a handle, ordered by timestamp.

    Uses handle_id directly on the message table (not chat_message_join)
    to capture all messages. Excludes group chat messages, tapback reactions,
    and duplicate messages (same timestamp + sender + text).

    Falls back to decoding attributedBody when text is NULL or empty.
    """
    cursor = conn.execute("""
        SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me
        FROM message m
        WHERE m.handle_id = ?
          AND (
            (m.text IS NOT NULL AND m.text != '')
            OR m.attributedBody IS NOT NULL
          )
          AND m.associated_message_type = 0
        ORDER BY m.date ASC
    """, (handle_rowid,))

    messages = []
    seen = set()
    for rowid, text, attributed_body, date, is_from_me in cursor.fetchall():
        if rowid in group_message_ids:
            continue
        resolved_text = text if text else decode_attributed_body(attributed_body)
        if not resolved_text:
            continue
        resolved_text = clean_text(resolved_text)
        if not resolved_text:
            continue

        ts = convert_timestamp(date)
        sender = "self" if is_from_me else "other"

        # Deduplicate identical messages (same timestamp + sender + text)
        dedup_key = (ts, sender, resolved_text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        messages.append({
            "timestamp": ts,
            "sender": sender,
            "text": resolved_text,
        })
    return messages


def extract_all():
    config = get_config()
    ensure_directories(config)

    db_path = config["CHAT_DB_PATH"]
    raw_dir = config["DATA_RAW_DIR"]

    logger.info("Opening chat.db at %s", db_path)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as e:
        logger.error(
            "Cannot open chat.db: %s\n"
            "Ensure Full Disk Access is granted in System Settings > "
            "Privacy & Security > Full Disk Access.",
            e,
        )
        raise SystemExit(1)

    logger.info("Resolving contacts...")
    resolver = ContactResolver()

    logger.info("Identifying group chat messages to exclude...")
    group_message_ids = get_group_chat_message_ids(conn)
    logger.info("Found %d group chat messages to exclude", len(group_message_ids))

    handles = get_all_handles(conn)
    logger.info("Found %d handles", len(handles))

    total_messages = 0
    contacts_written = 0
    skipped_automated = 0

    for handle_rowid, handle_id in handles:
        if is_automated_sender(handle_id):
            skipped_automated += 1
            continue

        messages = get_messages_for_handle(conn, handle_rowid, group_message_ids)
        if not messages:
            continue

        contact_name = resolver.resolve(handle_id)
        handle_hash = hashlib.sha256(handle_id.encode()).hexdigest()[:8]
        filename = f"{sanitize_filename(contact_name)}_{handle_hash}.json"

        output = {
            "handle_id": handle_id,
            "contact_name": contact_name,
            "message_count": len(messages),
            "messages": messages,
        }

        output_path = raw_dir / filename
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        total_messages += len(messages)
        contacts_written += 1

    conn.close()

    logger.info("--- Extraction Summary ---")
    logger.info("Contacts extracted: %d", contacts_written)
    logger.info("Automated senders skipped: %d", skipped_automated)
    logger.info("Total messages: %d", total_messages)
    logger.info("Output directory: %s", raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 1:1 iMessages from chat.db")
    parser.parse_args()
    extract_all()
