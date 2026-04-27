"""Export the fine-tuned model to GGUF format for Ollama / llama.cpp.

Uses Unsloth's bundled llama.cpp to merge + convert + quantize in one call.
If this path breaks (sometimes happens for brand-new architectures), fall back
to manual conversion — see README-deployment.md.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF for Ollama")
    parser.add_argument("--adapter", default="./output", help="Path to LoRA adapter directory")
    parser.add_argument("--output", default="./gguf", help="Output directory for GGUF file")
    parser.add_argument("--quant", default="q4_k_m",
                        help="Quantization method: q4_k_m (recommended), q5_k_m, q8_0, f16")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    from unsloth import FastModel

    logger.info("Loading adapter from: %s", args.adapter)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )

    logger.info("Merging + converting to GGUF (quant=%s)...", args.quant)
    model.save_pretrained_gguf(
        args.output,
        tokenizer,
        quantization_method=args.quant,
    )

    logger.info("GGUF saved to: %s", args.output)
    logger.info("Look for a file like: %s/unsloth.%s.gguf", args.output, args.quant.upper())


if __name__ == "__main__":
    main()
