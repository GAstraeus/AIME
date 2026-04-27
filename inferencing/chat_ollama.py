"""Interactive chat client backed by a local Ollama server.

Assumes `ollama create aime -f Modelfile` has been run and the Ollama daemon
is listening on http://localhost:11434.
"""

import argparse
import json
import urllib.error
import urllib.request

SYSTEM_PROMPT_TEMPLATE = (
    "You are texting as the user. Match their natural style, tone, and language exactly. "
    "Adapt your communication style based on the relationship context: {context}"
)

CONTEXTS = ["partner", "close_friend", "family", "colleague", "other"]


def ollama_chat(host: str, model: str, messages: list[dict], temperature: float) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with your fine-tuned model via Ollama")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="aime", help="Ollama model name")
    parser.add_argument("--context", default="close_friend", choices=CONTEXTS)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    context = args.context
    system_msg = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    print(f"Context: {context}")
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
                context = new_ctx
                system_msg = SYSTEM_PROMPT_TEMPLATE.format(context=context)
                conversation = []
                print(f"Switched to: {context} (conversation reset)\n")
            else:
                print(f"Unknown context. Choose from: {', '.join(CONTEXTS)}\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": system_msg}] + conversation

        try:
            response = ollama_chat(args.host, args.model, messages, args.temperature)
        except urllib.error.URLError as e:
            print(f"Error talking to Ollama at {args.host}: {e}\n")
            conversation.pop()
            continue

        conversation.append({"role": "assistant", "content": response})
        print(f" you> {response}\n")


if __name__ == "__main__":
    main()
