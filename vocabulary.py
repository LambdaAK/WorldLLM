"""
Limited vocabulary for the possession-tracking transformer.
~150 words covering people, objects, actions, questions, and grammar.
"""

from typing import Optional

# Special tokens (must be first)
# CLIENT: and OUTPUT: are message separators for the conversation format
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "CLIENT:", "OUTPUT:"]

# People (10)
PEOPLE = [
    "Alice", "Bob", "Carol", "Dave", "Eve",
    "Frank", "Grace", "Henry", "Ivy", "Jack",
]

# Objects (30)
OBJECTS = [
    "ball", "banana", "apple", "book", "key", "hat", "cup",
    "pen", "phone", "bag", "box", "chair", "table",
    "car", "bike", "dog", "cat", "flower", "ring",
    "coin", "card", "letter", "gift",     "toy", "stick", "rope", "bell", "lamp", "clock",
]

# Verbs for possession/transfer
VERBS = [
    "has", "have", "had", "gives", "give", "gave",
    "takes", "take", "took", "gets", "get", "got",
    "receives", "receive", "received", "puts", "put",
    "passes", "pass", "passed", "goes", "go", "went",
]

# Question words (8)
QUESTION_WORDS = [
    "who", "what", "where", "which", "whose",
    "how", "does", "do", "did",
]

# Pronouns (8)
PRONOUNS = [
    "it", "she", "he", "they", "her", "him", "them",
    "its",
]

# Articles and determiners (6)
DETERMINERS = [
    "a", "an", "the", "some", "this", "that",
]

# Prepositions (10)
PREPOSITIONS = [
    "to", "from", "with", "for", "at", "in", "on",
    "by", "of", "into",
]

# Conjunctions and connectors (10)
CONNECTORS = [
    "and", "then", "but", "so", "or", "because",
    "after", "before", "when", "now",
]

# Common words
COMMON = [
    "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "here", "there",
    "yes", "no", "not", "all", "both",
    "one", "two", "first", "last",
    "answer", "question", "scenario", "based", "following",
    "Okay", "got",  # for "Okay, got it."
]

# Punctuation (handled as tokens)
PUNCTUATION = [".", "?", "!"]

# Build full vocabulary
def _build_vocab() -> list[str]:
    parts = [
        SPECIAL_TOKENS,
        PEOPLE,
        OBJECTS,
        VERBS,
        QUESTION_WORDS,
        PRONOUNS,
        DETERMINERS,
        PREPOSITIONS,
        CONNECTORS,
        COMMON,
        PUNCTUATION,
    ]
    seen = set()
    vocab = []
    for part in parts:
        for w in part:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                vocab.append(w)
    return vocab


VOCAB = _build_vocab()
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}
# Add lowercase aliases for case-insensitive lookup
for w in list(WORD_TO_ID):
    w_lower = w.lower()
    if w_lower not in WORD_TO_ID:
        WORD_TO_ID[w_lower] = WORD_TO_ID[w]
ID_TO_WORD = {i: w for i, w in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# Token indices
PAD_ID = WORD_TO_ID["<pad>"]
SOS_ID = WORD_TO_ID["<sos>"]
EOS_ID = WORD_TO_ID["<eos>"]
UNK_ID = WORD_TO_ID["<unk>"]
CLIENT_ID = WORD_TO_ID["CLIENT:"]
OUTPUT_ID = WORD_TO_ID["OUTPUT:"]


def tokenize(text: str, add_special: bool = True) -> list[int]:
    """
    Tokenize a string into token IDs.
    Unknown words are mapped to <unk>.
    """
    words = text.strip().split()
    ids = []
    if add_special:
        ids.append(SOS_ID)
    for w in words:
        # Normalize: lowercase for lookup, but we store canonical form
        w_clean = w.strip(".,?!")
        if not w_clean:
            continue
        token_id = WORD_TO_ID.get(w_clean, UNK_ID)
        if token_id == UNK_ID:
            token_id = WORD_TO_ID.get(w_clean.lower(), UNK_ID)
        ids.append(token_id)
    if add_special:
        ids.append(EOS_ID)
    return ids


def detokenize(ids: list[int], strip_special: bool = True) -> str:
    """Convert token IDs back to a string."""
    words = []
    for i in ids:
        if strip_special and i in (PAD_ID, SOS_ID, EOS_ID):
            continue
        words.append(ID_TO_WORD.get(i, "<unk>"))
    return " ".join(words)


def is_valid_sentence(text: str) -> tuple[bool, list[str]]:
    """
    Check if all words in text are in vocabulary.
    Returns (all_valid, list_of_unknown_words).
    """
    words = text.strip().split()
    unknown = []
    for w in words:
        w_clean = w.strip(".,?!").lower()
        if w_clean and w_clean not in WORD_TO_ID:
            unknown.append(w_clean)
    return len(unknown) == 0, unknown


def get_vocab_stats() -> dict:
    """Return stats about the vocabulary."""
    return {
        "vocab_size": VOCAB_SIZE,
        "people": len(PEOPLE),
        "objects": len(OBJECTS),
        "verbs": len(VERBS),
    }
