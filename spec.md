# spec.md — Personal Texting Style Fine-Tune

## Goal
Fine-tune a Gemma 4 model on personal iMessage data to produce a model that:
- Responds in the user's authentic texting style
- Adapts tone based on the relationship with the recipient
- Preserves natural language including slang, emoji, typos, and short replies

---

## Model Strategy

### Fine-Tuning Model
`google/gemma-4-31b` — used for training. Maximum quality, best results.
- Requires cloud GPU (H100 recommended — RunPod or Lambda Labs)
- Estimated training time: 2–6 hours on H100 depending on dataset size
- QLoRA via Unsloth to manage memory efficiently

### Inference / Local Model
`google/gemma-4-e4b` — used for local on-device inference after training.
- Can be fine-tuned independently using the exact same JSONL dataset at any time
- Or populated via distillation from the 31B trained model
- Runs locally on Mac (~2.5GB quantized)

### Key Principle
The JSONL dataset is the core asset. Models are interchangeable. Fine-tune 31B first for quality, revisit E4B for local deployment whenever ready.

---

## LLM Pipeline

### Provider
All LLM calls use **Claude via AWS Bedrock** (`bedrock-runtime`). Not the Anthropic API directly.

### Model
Use the highest quality Claude model available on Bedrock at time of running. No cost constraint — prioritize quality.

### Auth
Configure via standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`).

### Client Pattern
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

---

## Phase 1: Data Extraction

### Source
- macOS `~/Library/Messages/chat.db`
- SQLite database, accessible via Python `sqlite3`
- Requires Full Disk Access granted to Terminal / IDE in System Settings

### Scope
- **Include**: 1:1 iMessage and SMS conversations only
- **Exclude**: Group chats (deferred to future phase)

### Output
- Raw messages per contact exported to `data/raw/`
- Each file contains: timestamp, sender (self or other), message body, contact ID

### Filter Criteria (extraction stage)
- Skip any thread with more than 2 participants
- Skip messages with no text body and no emoji (attachment-only)
- Skip automated/system senders (identified by pattern matching: no replies, short codes, etc.)

---

## Phase 2: LLM Conversation Segmentation

### Model
Claude via AWS Bedrock

### Task
For each contact's full message history, the LLM:
1. Segments the history into distinct conversations
2. Merges consecutive messages from the same sender into single turns
3. Preserves emoji, slang, and informal language exactly
4. Flags conversations where context is too implicit to be useful (e.g. references a phone call that just happened)

### Grouping Logic
- Time gap alone is not sufficient — use semantic context
- The LLM determines conversation boundaries intelligently
- Suggested heuristic (override-able by LLM): 30+ minute gap with topic shift = new conversation

### Output
- Structured conversation chunks saved to `data/processed/`
- Each chunk: list of turns with sender role and message content

---

## Phase 3: Relationship Classification

### Model
Claude via AWS Bedrock

### Task
For each contact, the LLM reads a sample of message history and classifies the relationship into one of:

| Label | Description |
|---|---|
| `partner` | Romantic partner / spouse |
| `close_friend` | Close personal friend |
| `family` | Family member |
| `colleague` | Work contact, professional relationship |
| `other` | Acquaintances, unknown contacts, or anyone who doesn't fit the above |

### Output
- `contacts/relationship_map.json`
- Format: `{ "contact_id": "relationship_label" }`
- Human-reviewable and manually editable

---

## Phase 4: JSONL Formatting

### Model
Claude via AWS Bedrock

### Task
For each conversation chunk, the LLM:
1. Converts it into one or more prompt/response training pairs
2. Ensures the user is always `assistant`, the other person is always `user`
3. Attaches the relationship context label
4. Skips or merges exchanges where the user's response is not substantive enough to be a useful training signal
5. Handles multi-turn conversations naturally

### Quality Criteria (LLM-assessed)
- Does the exchange reflect genuine communication style?
- Is the response attributable to a clear prompt?
- Would this pair teach the model something meaningful about style or tone?

### Output Format
```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "context": "relationship_label"}
```

- Saved to `data/training/messages.jsonl`
- One training example per line

---

## Phase 5: Review & Validation

### Automated checks
- Total example count
- Distribution across relationship types (flag heavy imbalance)
- Average turns per conversation
- Average response length per relationship type
- Flag any examples where assistant turn is empty

### LLM diversity check (Claude via Bedrock)
Pass a sample and ask:
- Is the dataset too skewed toward one topic type (e.g. logistics)?
- Are emotional, humorous, and opinionated exchanges represented?
- Are there any examples that seem out of character or anomalous?

### Output
- Console report with stats
- Optional: flagged examples saved for manual review

---

## Phase 6: Fine-Tuning (31B)

### Base Model
`google/gemma-4-31b`

### Hardware
- Cloud GPU: H100 (80GB) via RunPod or Lambda Labs
- Estimated time: 2–6 hours depending on dataset size
- Do NOT use local Mac for 31B training

### Framework
- **Unsloth** for efficient training
- **Hugging Face TRL** `SFTTrainer`
- **QLoRA** for memory-efficient fine-tuning

### Training Config (starting point)
```python
max_seq_length = 2048
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
learning_rate = 2e-4
num_train_epochs = 3
per_device_train_batch_size = 4
```

### System Prompt
```
You are texting as the user. Match their natural style, tone, and language exactly.
Adapt your communication style based on the relationship context provided.
```

### Output
- LoRA adapter saved locally
- Can be merged with base model for standalone deployment

---

## Future: E4B Fine-Tuning or Distillation

Two options — both use the same JSONL dataset:

**Option A — Independent fine-tune:**
Fine-tune `google/gemma-4-e4b` directly on the JSONL dataset using the same pipeline.
Faster, cheaper, but lower ceiling than 31B.

**Option B — Distillation from 31B:**
Use the fine-tuned 31B as a teacher model. Generate responses with it, then train E4B on those outputs.
More effort but E4B punches above its weight by learning from a stronger signal.

---

## Data Philosophy

| Keep | Remove |
|---|---|
| Emoji-only replies | Automated system messages |
| Short replies ("ok", "lol", "👍") | Attachment-only messages (no text, no emoji) |
| Typos and informal spelling | Corrupted/unreadable messages |
| Slang and abbreviations | |
| All authentic user communication | |

The goal is authenticity, not cleanliness. The model should learn exactly how the user communicates — not a sanitized version.

---

## Out of Scope (This Phase)
- E4B fine-tuning / distillation (future)
- Group chat support (future)
- Auto-reply app / Messages integration (separate project)
- Real-time inference API
- Relationship map UI
- Periodic retraining

---

## Success Criteria
- The fine-tuned 31B model, given a message and a relationship context, responds in a way that is indistinguishable from how the user would actually reply
- Tone shifts appropriately between relationship types
- Emoji and informal language usage matches user patterns
