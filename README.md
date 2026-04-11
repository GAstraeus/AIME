# AIME

Fine-tune a Gemma 4 model on your personal iMessage history to create a model that texts like you.

## What it does
Exports your iMessage conversations from macOS, runs them through an LLM-powered cleaning pipeline, and produces a JSONL training dataset used to fine-tune Gemma 4 31B. The model learns your texting style and adapts its tone based on who you're talking to — spouse, close friend, family, colleague, or other.

## Requirements
- Mac with iMessage history
- Python 3.10+
- LLM API access (for pipeline)
- Cloud GPU for training

## Status
🚧 In progress
