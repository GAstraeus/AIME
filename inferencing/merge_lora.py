"""Merge the trained LoRA adapter into the base model.

Produces a standalone HuggingFace model in 16-bit, ready for GGUF conversion.
Run on the GPU box where training happened — the merged E4B model is ~8-12GB.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", default="./output", help="Path to LoRA adapter directory")
    parser.add_argument("--output", default="./output-merged", help="Output directory for merged model")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    from unsloth import FastModel

    logger.info("Loading adapter from: %s", args.adapter)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,  # Must load in 16-bit for clean merge
    )

    logger.info("Merging LoRA weights into base model...")
    model.save_pretrained_merged(
        args.output,
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info("Merged model saved to: %s", args.output)


if __name__ == "__main__":
    main()
