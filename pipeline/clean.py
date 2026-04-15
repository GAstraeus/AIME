"""Post-format cleanup: remove empty assistant turns and redact PII."""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

from pipeline.utils.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# PII patterns
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
ADDRESS_RE = re.compile(
    r"\d+\s+\w+\s+(?:Road|Street|Ave|Avenue|Drive|Blvd|Boulevard|Lane|Court|Ct|Way|Place|Rd|St|Dr|Ln)\b",
    re.IGNORECASE,
)
SSN_RE = re.compile(r"\d{3}-\d{2}-\d{4}")
CC_RE = re.compile(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}")


def redact_pii(text: str) -> tuple[str, dict]:
    """Redact PII from text. Returns (cleaned_text, counts_dict)."""
    counts = {"emails": 0, "addresses": 0, "ssns": 0, "credit_cards": 0}

    text, n = SSN_RE.subn("[redacted]", text)
    counts["ssns"] += n

    text, n = CC_RE.subn("[redacted]", text)
    counts["credit_cards"] += n

    text, n = EMAIL_RE.subn("[email]", text)
    counts["emails"] += n

    text, n = ADDRESS_RE.subn("[address]", text)
    counts["addresses"] += n

    return text, counts


def clean_example(example: dict, redact: bool) -> tuple[dict | None, dict]:
    """Clean a single training example.

    Returns (cleaned_example_or_None, stats_dict).
    """
    stats = {
        "empty_turns_removed": 0,
        "examples_dropped": 0,
        "pii": {"emails": 0, "addresses": 0, "ssns": 0, "credit_cards": 0},
    }

    convos = example.get("conversations", [])

    # Remove empty assistant turns (and their preceding user turn)
    cleaned_turns = []
    i = 0
    while i < len(convos):
        turn = convos[i]
        if turn["role"] == "assistant" and not turn.get("content", "").strip():
            stats["empty_turns_removed"] += 1
            # Also drop the preceding user turn if it exists
            if cleaned_turns and cleaned_turns[-1]["role"] == "user":
                cleaned_turns.pop()
            i += 1
            continue
        cleaned_turns.append(turn)
        i += 1

    # Check if any valid assistant turns remain
    has_assistant = any(t["role"] == "assistant" and t.get("content", "").strip() for t in cleaned_turns)
    has_user = any(t["role"] == "user" for t in cleaned_turns)
    if not has_assistant or not has_user:
        stats["examples_dropped"] = 1
        return None, stats

    # Ensure it starts with a user turn
    while cleaned_turns and cleaned_turns[0]["role"] != "user":
        cleaned_turns.pop(0)
    if not cleaned_turns:
        stats["examples_dropped"] = 1
        return None, stats

    # Redact PII
    if redact:
        for turn in cleaned_turns:
            content = turn.get("content", "")
            cleaned_content, counts = redact_pii(content)
            turn["content"] = cleaned_content
            for k, v in counts.items():
                stats["pii"][k] += v

    example["conversations"] = cleaned_turns
    return example, stats


def clean_all(dry_run: bool = False, preserve_pii: bool = False):
    config = get_config()
    training_file = config["DATA_TRAINING_DIR"] / "messages.jsonl"

    if not training_file.exists():
        logger.error("Training file not found at %s. Run format.py first.", training_file)
        raise SystemExit(1)

    with open(training_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d examples from %s", len(examples), training_file)

    redact = not preserve_pii
    total_stats = {
        "empty_turns_removed": 0,
        "examples_dropped": 0,
        "pii": {"emails": 0, "addresses": 0, "ssns": 0, "credit_cards": 0},
    }

    cleaned = []
    for example in examples:
        result, stats = clean_example(example, redact=redact)
        total_stats["empty_turns_removed"] += stats["empty_turns_removed"]
        total_stats["examples_dropped"] += stats["examples_dropped"]
        for k, v in stats["pii"].items():
            total_stats["pii"][k] += v
        if result is not None:
            cleaned.append(result)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  CLEANUP {'(DRY RUN)' if dry_run else 'REPORT'}")
    print(f"{'=' * 50}")
    print(f"\n  Before: {len(examples)} examples")
    print(f"  After:  {len(cleaned)} examples")
    print(f"  Dropped: {total_stats['examples_dropped']}")
    print(f"  Empty assistant turns removed: {total_stats['empty_turns_removed']}")
    if redact:
        pii = total_stats["pii"]
        total_pii = sum(pii.values())
        print(f"\n  PII redacted: {total_pii} total")
        print(f"    Emails: {pii['emails']}")
        print(f"    Addresses: {pii['addresses']}")
        print(f"    SSNs: {pii['ssns']}")
        print(f"    Credit cards: {pii['credit_cards']}")
    else:
        print(f"\n  PII redaction: skipped (--preserve-pii)")
    print(f"{'=' * 50}\n")

    if dry_run:
        logger.info("Dry run — no changes written.")
        return

    # Backup original
    backup_path = training_file.with_suffix(".jsonl.bak")
    shutil.copy2(training_file, backup_path)
    logger.info("Backup saved to %s", backup_path)

    # Write cleaned data
    with open(training_file, "w") as f:
        for example in cleaned:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info("Cleaned data written to %s", training_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean training data: remove empty turns and redact PII")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without modifying the file")
    parser.add_argument("--preserve-pii", action="store_true", help="Skip PII redaction")
    args = parser.parse_args()
    clean_all(dry_run=args.dry_run, preserve_pii=args.preserve_pii)
