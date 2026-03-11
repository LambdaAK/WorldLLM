"""Quick non-interactive test: run a few conversations through the model."""

import torch
from model import TinyGPT
from vocabulary import tokenize, detokenize, EOS_ID, CLIENT_ID
from interact import build_conversation_tokens, generate_response

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
config = checkpoint["config"]
model = TinyGPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model: epoch {checkpoint.get('epoch','?')}, val_loss {checkpoint.get('val_loss','?')}\n")


def chat(messages):
    history = []
    for msg in messages:
        token_ids = build_conversation_tokens(history, msg)
        response_ids = generate_response(model, token_ids, config, device, temperature=0.1, top_k=5)
        reply = detokenize(response_ids, strip_special=True).strip() or "..."
        print(f"CLIENT: {msg}")
        print(f"OUTPUT: {reply}\n")
        history.append((msg, reply))
    print("=" * 40 + "\n")


print("--- Conversation 1 ---")
chat([
    "Alice has the ball.",
    "Bob has the key.",
    "Who has the ball?",
    "Alice gives the ball to Bob.",
    "Who has the ball?",
])

print("--- Conversation 2 ---")
chat([
    "Charlie has the clock.",
    "Alice has the book.",
    "Does Charlie have the book?",
    "Who has what?",
])

print("--- Conversation 3 ---")
chat([
    "Bob has the ball and the key.",
    "What does Bob have?",
    "How many things does Bob have?",
    "Bob gives the key to Alice.",
    "What does Bob have?",
    "How many things does Alice have?",
])

print("--- Conversation 4 ---")
chat([
    "Alice has the ball.",
    "Bob has the key.",
    "Charlie has the clock.",
    "Alice gives the ball to Bob.",
    "Bob gives the ball to Charlie.",
    "Who has the ball?",
])

print("--- Conversation 5 ---")
chat([
    "Alice has the ball.",
    "Alice has the key.",
    "Alice has the clock.",
    "How many things does Alice have?",
    "Who has what?",
    "Does Alice have the book?",
])
