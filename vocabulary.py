"""
Closed vocabulary for the possession-tracking transformer.

Defines a fixed ~137-token vocabulary covering people, objects, verbs,
question words, and basic grammar. Because the vocabulary is small and
known ahead of time, we use simple whitespace + punctuation splitting
rather than subword tokenization (BPE/SentencePiece).

Exports:
    VOCAB, WORD_TO_ID, ID_TO_WORD, VOCAB_SIZE — vocabulary mappings
    PAD_ID, SOS_ID, EOS_ID, UNK_ID, CLIENT_ID, OUTPUT_ID — special token IDs
    tokenize / detokenize — string <-> token ID conversion
    is_valid_sentence — check if all words are in-vocabulary
"""

from typing import Optional

# Special tokens (must be first)
# CLIENT: and OUTPUT: are message separators for the conversation format
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "CLIENT:", "OUTPUT:"]

# People
PEOPLE = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

# Objects
OBJECTS = ["ball", "key", "clock", "book", "hat", "ring", "coin", "lamp", "pen", "cup"]

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
    "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "first", "last",
    "many", "things", "none", "nothing",
    "anyone", "more", "than",
    "answer", "question", "scenario", "based", "following",
    "Okay", "got",
]

# Math: digits, operators, two-digit results (for addition/subtraction chains)
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
MATH_OPERATORS = ["+", "-"]
MATH_RESULTS = [str(n) for n in range(10, 37)]  # "10" through "36"

# Punctuation (handled as tokens)
PUNCTUATION = [".", "?", "!", ","]

# Build full vocabulary
def _build_vocab() -> list[str]:
    """Assemble the full vocabulary list, deduplicating by lowercase form."""
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
        DIGITS,
        MATH_OPERATORS,
        MATH_RESULTS,
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


_PUNCT = set(".,?!")


def _split_punct(word: str) -> list[str]:
    """Split trailing punctuation off a word: 'ball.' -> ['ball', '.']"""
    if not word:
        return []
    tokens = []
    # strip leading punctuation (rare but safe)
    i = 0
    while i < len(word) and word[i] in _PUNCT:
        tokens.append(word[i])
        i += 1
    core = []
    j = len(word)
    while j > i and word[j - 1] in _PUNCT:
        j -= 1
    if i < j:
        tokens.append(word[i:j])
    for k in range(j, len(word)):
        tokens.append(word[k])
    return tokens


def tokenize(text: str, add_special: bool = True) -> list[int]:
    """Convert a string to a list of token IDs.

    Splits on whitespace, separates trailing punctuation into its own token,
    and wraps with <sos>/<eos> when add_special is True. Words not in the
    vocabulary are mapped to <unk> (case-insensitive fallback is attempted).
    """
    words = text.strip().split()
    ids = []
    if add_special:
        ids.append(SOS_ID)
    for w in words:
        for part in _split_punct(w):
            token_id = WORD_TO_ID.get(part, UNK_ID)
            if token_id == UNK_ID:
                token_id = WORD_TO_ID.get(part.lower(), UNK_ID)
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
        for part in _split_punct(w):
            if part.lower() not in WORD_TO_ID:
                unknown.append(part)
    return len(unknown) == 0, unknown


def get_vocab_stats() -> dict:
    """Return stats about the vocabulary."""
    return {
        "vocab_size": VOCAB_SIZE,
        "people": len(PEOPLE),
        "objects": len(OBJECTS),
        "verbs": len(VERBS),
    }
