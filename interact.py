"""
Interactive chat with a trained WorldLLM model.

Usage:
    python interact.py
    python interact.py --checkpoint checkpoints/best.pt
    python interact.py --temperature 0.5 --top_k 10

Type messages as the CLIENT. The model generates OUTPUT responses.
The conversation state accumulates across turns.
Type 'reset' to clear the conversation, 'quit' to exit.
"""

import argparse
import torch
from model import WorldLLM
from config import ModelConfig
from vocabulary import (
    tokenize, detokenize,
    SOS_ID, EOS_ID, OUTPUT_ID, CLIENT_ID, PAD_ID,
    ID_TO_WORD, WORD_TO_ID,
)


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = WorldLLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Loaded model from {checkpoint_path} (step {step}, val_loss {val_loss})")
    print(f"Parameters: {model.count_parameters():,}")
    return model, config


def build_conversation_tokens(turns, current_client_msg):
    """
    Build the full token sequence for the conversation so far,
    ending with OUTPUT: so the model generates the response.
    """
    parts = []

    # Previous turns
    for client_msg, output_msg in turns:
        parts.append(f"CLIENT:\n{client_msg}\n\nOUTPUT:\n{output_msg}")

    # Current turn: client message + OUTPUT: prefix
    parts.append(f"CLIENT:\n{current_client_msg}\n\nOUTPUT:")

    full_text = "\n\n".join(parts)
    ids = tokenize(full_text, add_special=True)
    # Remove <eos> at the end — we want the model to continue generating
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return ids


def generate_response(model, token_ids, config, device, temperature=0.8, top_k=20, max_tokens=40):
    """Generate model response tokens until CLIENT: or <eos>."""
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop to max_seq_len
            ids = input_tensor if input_tensor.size(1) <= config.max_seq_len else input_tensor[:, -config.max_seq_len:]

            logits = model(ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # Stop on <eos> or CLIENT: (model tries to start a new turn)
            if token_id == EOS_ID or token_id == CLIENT_ID:
                break

            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Chat with WorldLLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, config = load_model(args.checkpoint, device)
    print(f"Device: {device}")
    print()
    print("WorldLLM Interactive Chat")
    print("=" * 40)
    print("You are the CLIENT. Type messages describing")
    print("who has what, transfers, or ask questions.")
    print()
    print("Commands: 'reset' to clear, 'quit' to exit")
    print("=" * 40)
    print()

    turns = []  # List of (client_msg, output_msg)

    while True:
        try:
            user_input = input("CLIENT:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "reset":
            turns = []
            print("[conversation reset]\n")
            continue

        token_ids = build_conversation_tokens(turns, user_input)

        response_ids = generate_response(
            model, token_ids, config, device,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        response_text = detokenize(response_ids, strip_special=True)
        if not response_text.strip():
            response_text = "..."

        print(f"\nOUTPUT:\n{response_text}\n")

        turns.append((user_input, response_text))


if __name__ == "__main__":
    main()
