"""Interactive inference with the fine-tuned LoRA adapter."""

import argparse

SYSTEM_PROMPT_TEMPLATE = (
    "You are texting as the user. Match their natural style, tone, and language exactly. "
    "Adapt your communication style based on the relationship context: {context}"
)

CONTEXTS = ["partner", "close_friend", "family", "colleague", "other"]


def main():
    parser = argparse.ArgumentParser(description="Chat with your fine-tuned model")
    parser.add_argument("--base-model", default="google/gemma-4-31b")
    parser.add_argument("--adapter", default="./output", help="Path to LoRA adapter")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--context", default="close_friend", choices=CONTEXTS,
                        help="Relationship context (default: close_friend)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    from unsloth import FastModel

    print(f"Loading LoRA adapter from: {args.adapter}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model.eval()

    system_msg = SYSTEM_PROMPT_TEMPLATE.format(context=args.context)
    print(f"\nContext: {args.context}")
    print("Type a message (or 'quit' to exit, '/context <type>' to switch, '/clear' to reset)\n")

    conversation = []

    while True:
        try:
            user_input = input("them> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "/clear":
            conversation = []
            print("Conversation cleared.\n")
            continue
        if user_input.lower().startswith("/context "):
            new_ctx = user_input.split(maxsplit=1)[1].strip()
            if new_ctx in CONTEXTS:
                args.context = new_ctx
                system_msg = SYSTEM_PROMPT_TEMPLATE.format(context=new_ctx)
                conversation = []
                print(f"Switched to: {new_ctx} (conversation reset)\n")
            else:
                print(f"Unknown context. Choose from: {', '.join(CONTEXTS)}\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Build prompt in Gemma chat format
        parts = [f"<start_of_turn>system\n{system_msg}<end_of_turn>"]
        for turn in conversation:
            role = "model" if turn["role"] == "assistant" else turn["role"]
            parts.append(f"<start_of_turn>{role}\n{turn['content']}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        prompt = "\n".join(parts)

        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.split("<end_of_turn>")[0].strip()

        conversation.append({"role": "assistant", "content": response})
        print(f" you> {response}\n")


if __name__ == "__main__":
    main()
