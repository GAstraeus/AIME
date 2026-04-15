"""Step 5: Review and validate the training dataset."""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from pipeline.utils.bedrock import BedrockClient
from pipeline.utils.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIVERSITY_SYSTEM_PROMPT = """You are reviewing a sample of training data for a text-style fine-tuning project. The model will learn to replicate a specific person's texting style.

Evaluate the sample on these dimensions:

1. Topic diversity: Is the data skewed toward one type of conversation (e.g., all logistics/planning, all greetings)? List the top topic categories you observe and their rough proportions.
2. Emotional range: Are emotional, humorous, serious, and casual exchanges all represented?
3. Response variety: Does the assistant show a range of response styles (short vs. long, emoji vs. text, formal vs. casual)?
4. Anomalies: Flag any examples that seem out of character, contain potentially sensitive information (passwords, financial details), or look like they should have been filtered (automated messages, corrupted text).
5. Relationship consistency: Do the relationship labels seem accurate based on the conversation content?

Respond with ONLY this JSON:
{
  "topic_distribution": {"logistics": 0.3, "social": 0.2, "...": 0.0},
  "emotional_range_score": 7,
  "response_variety_score": 8,
  "anomalies": ["description of any anomalous examples"],
  "relationship_accuracy": "high",
  "overall_assessment": "brief paragraph",
  "recommendations": ["suggestion 1", "suggestion 2"]
}"""


def load_training_data(training_file: Path) -> list[dict]:
    """Load all training examples from the JSONL file."""
    examples = []
    with open(training_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def compute_stats(examples: list[dict]) -> dict:
    """Compute automated statistics on the training dataset."""
    if not examples:
        return {"total": 0}

    relationship_counts = Counter()
    turns_per_example = []
    assistant_lengths_by_rel = {}
    empty_assistant = []

    for i, ex in enumerate(examples):
        rel = ex.get("context", "unknown")
        relationship_counts[rel] += 1

        convos = ex.get("conversations", [])
        turns_per_example.append(len(convos))

        if rel not in assistant_lengths_by_rel:
            assistant_lengths_by_rel[rel] = []

        for turn in convos:
            if turn["role"] == "assistant":
                content = turn.get("content", "")
                assistant_lengths_by_rel[rel].append(len(content))
                if not content.strip():
                    empty_assistant.append(i)

    # Compute averages
    avg_turns = sum(turns_per_example) / len(turns_per_example) if turns_per_example else 0
    avg_lengths = {}
    for rel, lengths in assistant_lengths_by_rel.items():
        avg_lengths[rel] = sum(lengths) / len(lengths) if lengths else 0

    return {
        "total": len(examples),
        "relationship_distribution": dict(relationship_counts.most_common()),
        "avg_turns_per_example": round(avg_turns, 1),
        "min_turns": min(turns_per_example) if turns_per_example else 0,
        "max_turns": max(turns_per_example) if turns_per_example else 0,
        "avg_assistant_response_length_by_relationship": {
            k: round(v, 1) for k, v in avg_lengths.items()
        },
        "empty_assistant_turns": len(empty_assistant),
        "empty_assistant_indices": empty_assistant[:20],  # first 20
    }


def check_diversity(client: BedrockClient, examples: list[dict], sample_size: int = 50) -> dict:
    """Send a random sample to Claude for diversity analysis."""
    sample = random.sample(examples, min(sample_size, len(examples)))

    serialized_lines = []
    for i, ex in enumerate(sample):
        convos = ex.get("conversations", [])
        rel = ex.get("context", "?")
        turns_str = " | ".join(f'{t["role"]}: {t["content"]}' for t in convos)
        serialized_lines.append(f"[{i+1}] (context: {rel}) {turns_str}")

    serialized = "\n\n".join(serialized_lines)

    user_message = (
        f"Training data sample ({len(sample)} examples from {len(examples)} total):\n\n"
        f"{serialized}"
    )

    return client.invoke_with_json(
        messages=[{"role": "user", "content": user_message}],
        system=DIVERSITY_SYSTEM_PROMPT,
    )


def print_report(stats: dict, diversity: dict = None):
    """Print a formatted console report."""
    print("\n" + "=" * 60)
    print("  TRAINING DATA REVIEW REPORT")
    print("=" * 60)

    print(f"\n  Total training examples: {stats['total']}")
    print(f"  Average turns per example: {stats['avg_turns_per_example']}")
    print(f"  Turns range: {stats['min_turns']} – {stats['max_turns']}")

    print("\n  Relationship distribution:")
    for rel, count in stats.get("relationship_distribution", {}).items():
        pct = (count / stats["total"] * 100) if stats["total"] else 0
        print(f"    {rel}: {count} ({pct:.1f}%)")

    print("\n  Avg assistant response length (chars) by relationship:")
    for rel, length in stats.get("avg_assistant_response_length_by_relationship", {}).items():
        print(f"    {rel}: {length}")

    if stats["empty_assistant_turns"]:
        print(f"\n  WARNING: {stats['empty_assistant_turns']} examples with empty assistant turns")

    if diversity:
        print("\n" + "-" * 60)
        print("  LLM DIVERSITY ANALYSIS")
        print("-" * 60)

        print(f"\n  Emotional range score: {diversity.get('emotional_range_score', '?')}/10")
        print(f"  Response variety score: {diversity.get('response_variety_score', '?')}/10")
        print(f"  Relationship accuracy: {diversity.get('relationship_accuracy', '?')}")

        print("\n  Topic distribution:")
        for topic, proportion in diversity.get("topic_distribution", {}).items():
            print(f"    {topic}: {proportion}")

        anomalies = diversity.get("anomalies", [])
        if anomalies:
            print(f"\n  Anomalies ({len(anomalies)}):")
            for a in anomalies:
                print(f"    - {a}")

        assessment = diversity.get("overall_assessment", "")
        if assessment:
            print(f"\n  Assessment: {assessment}")

        recommendations = diversity.get("recommendations", [])
        if recommendations:
            print("\n  Recommendations:")
            for r in recommendations:
                print(f"    - {r}")

    print("\n" + "=" * 60)


def review_all(skip_llm: bool = False):
    config = get_config()
    training_file = config["DATA_TRAINING_DIR"] / "messages.jsonl"

    if not training_file.exists():
        logger.error("Training file not found at %s. Run format.py first.", training_file)
        raise SystemExit(1)

    logger.info("Loading training data from %s...", training_file)
    examples = load_training_data(training_file)

    if not examples:
        logger.error("Training file is empty.")
        raise SystemExit(1)

    logger.info("Computing stats on %d examples...", len(examples))
    stats = compute_stats(examples)

    diversity = None
    if not skip_llm:
        logger.info("Running LLM diversity check...")
        client = BedrockClient()
        try:
            diversity = check_diversity(client, examples)
        except Exception:
            logger.exception("LLM diversity check failed, skipping")

    print_report(stats, diversity)

    # Save flagged examples if any
    if stats["empty_assistant_turns"]:
        flagged_path = config["DATA_TRAINING_DIR"] / "flagged.json"
        flagged = [examples[i] for i in stats["empty_assistant_indices"]]
        with open(flagged_path, "w") as f:
            json.dump(flagged, f, indent=2, ensure_ascii=False)
        logger.info("Flagged examples saved to %s", flagged_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review and validate training dataset")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the LLM diversity check")
    args = parser.parse_args()
    review_all(skip_llm=args.skip_llm)
