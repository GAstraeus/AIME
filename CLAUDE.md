# CLAUDE.md

## Project Overview
This project fine-tunes a Gemma 4 model on personal iMessage data to create a model that replicates the user's texting style, context-aware by relationship type.

## Project Scope
**This repo covers Project 1 only: data pipeline + model fine-tuning.**
Auto-reply integration is a separate future project and is out of scope here.

## Tech Stack
- **Language**: Python
- **Message Source**: macOS `~/Library/Messages/chat.db` (SQLite)
- **LLM for pipeline**: Claude via AWS Bedrock (no budget constraint, use highest quality model available)
- **Fine-tuning model**: `google/gemma-4-31b` (maximum quality)
- **Inference model**: `google/gemma-4-e4b` (local on-device use after training)
- **Fine-tuning framework**: Unsloth + Hugging Face TRL
- **Training technique**: QLoRA
- **Cloud GPU**: RunPod or Lambda Labs (H100 recommended for 31B)
- **Output format**: JSONL

## Repo Structure
```
/
├── CLAUDE.md
├── spec.md
├── README.md
├── data/
│   ├── raw/                  # Raw extracted messages from chat.db
│   ├── processed/            # LLM-cleaned conversation chunks
│   └── training/             # Final JSONL ready for fine-tuning
├── contacts/
│   └── relationship_map.json # Contact ID → relationship type mapping
├── pipeline/
│   ├── extract.py            # Step 1: Pull messages from chat.db
│   ├── segment.py            # Step 2: LLM conversation segmentation
│   ├── classify.py           # Step 3: LLM relationship classification
│   ├── format.py             # Step 4: LLM JSONL formatting
│   └── review.py             # Step 5: Sanity check output
├── training/
│   └── finetune.py           # Unsloth fine-tuning script
└── requirements.txt
```

## Key Decisions & Rationale

### Model Strategy
- Fine-tune on **Gemma 4 31B** for maximum quality — this is the primary trained model
- Deploy/run locally using **Gemma 4 E4B** — either via distillation from 31B or independent fine-tune using the same JSONL dataset
- Both can be trained independently at any time using the same dataset
- The JSONL dataset is the core asset — models are interchangeable

### LLM Pipeline via AWS Bedrock
All LLM calls in the pipeline use Claude via AWS Bedrock, not the Anthropic API directly. Use `boto3` with the `bedrock-runtime` client. Always use the highest quality Claude model available on Bedrock.

```python
import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.invoke_model(
    modelId="anthropic.claude-opus-4-5",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "your prompt here"}]
    })
)

result = json.loads(response["body"].read())
text = result["content"][0]["text"]
```

### Group Chats
Omitted from this phase. Multi-speaker format is complex and 1:1 data is sufficient for style training. Group chat support planned for a future iteration.

### Emoji & Short Replies
Keep all of them. Emoji-only replies, short acknowledgements, and informal responses are valid style signals and should be preserved.

### Relationship Context
Each training example is tagged with a relationship type. The model is trained to produce contextually appropriate responses based on who it is talking to.

Relationship categories:
- `partner`
- `close_friend`
- `family`
- `colleague`
- `other` (covers acquaintances, unknown contacts, and anyone who doesn't fit the above)

Contact → relationship mapping is stored in `contacts/relationship_map.json`. The LLM infers relationship type automatically during the pipeline based on message history. The file is human-readable and manually editable for corrections.

### What to Filter
Only remove messages that are not the user communicating:
- Automated texts (2FA codes, bank alerts, delivery notifications)
- Attachment-only messages with no text or emoji
- Corrupted or unreadable messages

Do NOT filter:
- Emoji-only messages
- Very short replies ("ok", "lol", "👍")
- Typos or informal language
- Slang

## Training Data Format
```json
{"conversations": [{"role": "user", "content": "you coming tonight?"}, {"role": "assistant", "content": "yeah be there at 8"}], "context": "close_friend"}
{"conversations": [{"role": "user", "content": "Can you review the doc when you get a chance?"}, {"role": "assistant", "content": "Sure, I'll take a look this afternoon"}], "context": "colleague"}
```

Multi-turn example:
```json
{"conversations": [
  {"role": "user", "content": "you eating yet"},
  {"role": "assistant", "content": "nah not yet"},
  {"role": "user", "content": "want to grab something"},
  {"role": "assistant", "content": "yeah give me 20"}
], "context": "close_friend"}
```

## Pipeline Stages
1. **Extract** — Pull all 1:1 messages from `chat.db` with timestamps and sender info
2. **Segment** — LLM groups messages into coherent conversations using context + time gaps
3. **Classify** — LLM infers relationship type per contact from full message history
4. **Format** — LLM converts conversations into clean JSONL training pairs
5. **Review** — Sanity check: count, diversity check, flag anomalies

## Running the Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python pipeline/extract.py
python pipeline/segment.py
python pipeline/classify.py
python pipeline/format.py
python pipeline/review.py

# Fine-tune
python training/finetune.py
```

## Environment Variables
```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
CHAT_DB_PATH=/Users/yourname/Library/Messages/chat.db
```

## Future Work (Out of Scope Here)
- Fine-tune E4B independently (can reuse same JSONL dataset at any time)
- Distillation from 31B → E4B
- Group chat support
- Auto-reply app integration
- Periodic retraining as new messages accumulate
- Relationship map UI for manual corrections
