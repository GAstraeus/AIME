# AIME

Fine-tune a Gemma 4 model on your personal iMessage history to create a model that texts like you.

## What It Does

Exports your iMessage conversations from macOS, processes them through an LLM-powered pipeline (Claude via AWS Bedrock), and produces a JSONL training dataset. The dataset is used to fine-tune Gemma 4 31B with QLoRA. The model learns your texting style — including slang, emoji, typos, and tone — and adapts based on who you're talking to.

## Pipeline

```
chat.db (macOS)
    |
[extract] --> data/raw/{contact}.json
    |
[segment] --> data/processed/{contact}.json
    |
[classify] --> contacts/relationship_map.json
    |
[format] --> data/training/messages.jsonl
    |
[review] --> console report + flagged.json
    |
[clean] --> data/training/messages.jsonl (cleaned)
    |
[finetune] --> output/ (LoRA adapter)
```

## Requirements

- macOS with iMessage history (`~/Library/Messages/chat.db`)
- Python 3.10+
- AWS account with Bedrock access (Claude model enabled)
- Cloud GPU for training (H100 recommended via RunPod or Lambda Labs)

## Setup

```bash
# Clone and set up virtual environment
git clone <repo-url> && cd aime
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Grant Full Disk Access to your terminal
# System Settings > Privacy & Security > Full Disk Access > add Terminal/iTerm

# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_DB_PATH` | `~/Library/Messages/chat.db` | Path to iMessage database |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `global.anthropic.claude-opus-4-5-20251101-v1:0` | Claude model ID or inference profile |
| `MAX_TOKENS` | `4096` | Default max tokens for LLM responses |

## Running the Pipeline

### Step 1: Extract Messages

```bash
python -m pipeline.extract
```

Reads `chat.db`, extracts all 1:1 conversations (skips group chats), resolves contact names via macOS Contacts, and filters automated senders (short codes, no-reply numbers). Decodes both the `text` column and `attributedBody` blobs for full message coverage.

**Output:** One JSON file per contact in `data/raw/`

### Step 2: Segment Conversations

```bash
python -m pipeline.segment
```

Claude segments each contact's message history into distinct conversations based on topic shifts and time gaps. Merges consecutive messages from the same sender into single turns.

| Flag | Description |
|------|-------------|
| `--force` | Clear chunk cache and reprocess all contacts |
| `--limit N` | Process only the first N contacts |
| `--workers W` | Number of concurrent Bedrock workers (default: 3) |

**Output:** Segmented conversation files in `data/processed/`

### Step 3: Classify Relationships

```bash
python -m pipeline.classify
```

Claude classifies each contact's relationship based on a message sample. Categories: `partner`, `close_friend`, `family`, `colleague`, `other`.

| Flag | Description |
|------|-------------|
| `--force` | Reclassify all contacts (overwrites manual edits) |

**Output:** `contacts/relationship_map.json` — human-readable and manually editable. Review this before proceeding.

### Step 4: Format Training Data

```bash
python -m pipeline.format
```

Claude converts segmented conversations into training pairs. Maps the other person to `user` role and you to `assistant` role. Attaches relationship context.

| Flag | Description |
|------|-------------|
| `--force` | Reformat all contacts from scratch |

**Output:** `data/training/messages.jsonl`

### Step 5: Review

```bash
python -m pipeline.review
```

Prints stats (example count, relationship distribution, avg turns, response lengths) and runs an LLM diversity check on a sample.

| Flag | Description |
|------|-------------|
| `--skip-llm` | Skip the Claude diversity check, run stats only |

**Output:** Console report. Saves flagged examples to `data/training/flagged.json`.

### Step 6: Clean

```bash
python -m pipeline.clean --dry-run   # preview first
python -m pipeline.clean              # apply
```

Removes empty assistant turns and redacts PII (emails, addresses, SSNs, credit card numbers). Creates a backup at `messages.jsonl.bak`.

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview changes without modifying the file |
| `--preserve-pii` | Skip PII redaction |

## Training

Training runs on a cloud GPU, not locally.

```bash
# On your cloud GPU instance (H100)
pip install -r requirements-training.txt
python -m training.finetune
```

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-4-31b` |
| Technique | QLoRA (4-bit) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Max sequence length | 2048 |

All hyperparameters are overridable via CLI flags (run `python -m training.finetune --help`).

**Output:** LoRA adapter saved to `./output/`

## Training Data Format

```json
{"conversations": [{"role": "user", "content": "you coming tonight?"}, {"role": "assistant", "content": "yeah be there at 8"}], "context": "close_friend"}
```

Multi-turn:

```json
{"conversations": [
  {"role": "user", "content": "you eating yet"},
  {"role": "assistant", "content": "nah not yet"},
  {"role": "user", "content": "want to grab something"},
  {"role": "assistant", "content": "yeah give me 20"}
], "context": "close_friend"}
```

## Resumability

| Step | Resumable | Mechanism | Force flag |
|------|-----------|-----------|------------|
| extract | No | Overwrites all (fast, no LLM) | — |
| segment | Yes | Chunk cache in `.chunk_cache/` | `--force` |
| classify | Yes | Skips contacts already in map | `--force` |
| format | Yes | Progress tracked in `.progress.json` | `--force` |
| review | No | Read-only | — |
| clean | No | Creates `.bak` backup | — |