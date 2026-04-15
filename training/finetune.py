"""Step 6: Fine-tune Gemma 4 31B on the training dataset using QLoRA via Unsloth.

This script is intended to run on a cloud GPU (H100 80GB recommended).
Do NOT run this on a local Mac.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default hyperparameters from spec
DEFAULTS = {
    "base_model": "google/gemma-4-31b",
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "optim": "adamw_8bit",
    "output_dir": "./output",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are texting as the user. Match their natural style, tone, and language exactly. "
    "Adapt your communication style based on the relationship context: {context}"
)


def format_example_to_chat(example: dict, tokenizer) -> str:
    """Convert a training example into Gemma 4 chat template format."""
    context = example.get("context", "other")
    system_msg = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    chat = [{"role": "system", "content": system_msg}]
    for turn in example["conversations"]:
        role = "model" if turn["role"] == "assistant" else turn["role"]
        chat.append({"role": role, "content": turn["content"]})

    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)


def load_dataset(training_file: str, tokenizer):
    """Load the JSONL training data into a HuggingFace Dataset."""
    from datasets import Dataset

    examples = []
    with open(training_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    formatted = [format_example_to_chat(ex, tokenizer) for ex in examples]

    logger.info("Loaded %d training examples", len(formatted))
    return Dataset.from_dict({"text": formatted})


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 on iMessage data")
    parser.add_argument("--training-file", default="data/training/messages.jsonl")
    parser.add_argument("--base-model", default=DEFAULTS["base_model"])
    parser.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    parser.add_argument("--lora-r", type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=DEFAULTS["lora_dropout"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["num_train_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["per_device_train_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    args = parser.parse_args()

    # Import heavy dependencies here (not at top) so the script can be
    # imported/inspected without requiring GPU libraries installed
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig

    logger.info("Loading base model: %s", args.base_model)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    logger.info("Applying LoRA adapters (r=%d, alpha=%d)", args.lora_r, args.lora_alpha)
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    logger.info("Loading training dataset from %s", args.training_file)
    dataset = load_dataset(args.training_file, tokenizer)

    training_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=DEFAULTS["warmup_ratio"],
        weight_decay=DEFAULTS["weight_decay"],
        optim=DEFAULTS["optim"],
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_config,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
