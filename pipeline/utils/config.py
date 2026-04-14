import os
from pathlib import Path


def get_project_root() -> Path:
    """Walk up from this file to find the repo root (contains CLAUDE.md)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no CLAUDE.md found)")


def get_config() -> dict:
    """Return all resolved configuration from env vars with sensible defaults."""
    root = get_project_root()
    return {
        "CHAT_DB_PATH": os.environ.get(
            "CHAT_DB_PATH",
            str(Path.home() / "Library" / "Messages" / "chat.db"),
        ),
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "BEDROCK_MODEL_ID": os.environ.get(
            "BEDROCK_MODEL_ID",
            "global.anthropic.claude-opus-4-5-20251101-v1:0",
        ),
        "MAX_TOKENS": int(os.environ.get("MAX_TOKENS", "4096")),
        "PROJECT_ROOT": root,
        "DATA_RAW_DIR": root / "data" / "raw",
        "DATA_PROCESSED_DIR": root / "data" / "processed",
        "DATA_TRAINING_DIR": root / "data" / "training",
        "CONTACTS_DIR": root / "contacts",
        "RELATIONSHIP_MAP_PATH": root / "contacts" / "relationship_map.json",
    }


def ensure_directories(config: dict = None):
    """Create output directories if they don't exist."""
    if config is None:
        config = get_config()
    for key in ("DATA_RAW_DIR", "DATA_PROCESSED_DIR", "DATA_TRAINING_DIR", "CONTACTS_DIR"):
        config[key].mkdir(parents=True, exist_ok=True)
