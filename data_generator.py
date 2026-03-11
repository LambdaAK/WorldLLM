"""
Synthetic data generator for possession-tracking conversations.

Produces multi-turn CLIENT:/OUTPUT: conversations where people are assigned
objects, transfer them, and answer questions. Each conversation maintains a
ground-truth PossessionState so answers are always correct.

Supports: simple possession ("Alice has the ball"), compound possession
("Alice has the ball and the key"), transfers ("Alice gives the ball to Bob"),
multi-hop transfers, five question types (who_has, what_has, yes_no,
who_has_what, how_many), and mixed arithmetic (single-digit addition/subtraction
chains, e.g. "What is 1 + 2 + 4?" -> "7.", "What is 3 - 7?" -> "- 4.").

Usage:
    python data_generator.py --train 200000 --val 2000 --test 2000 --outdir data
    python data_generator.py --preview 5
"""

import argparse
import os
import random
from typing import Iterator, List, Optional, Tuple
from vocabulary import PEOPLE, OBJECTS, is_valid_sentence

ACK = "Got it."
CONVERSATION_TURN_SEP = "\n\n"
CLIENT_PREFIX = "CLIENT:\n"
OUTPUT_PREFIX = "OUTPUT:\n"
CONVERSATION_SEPARATOR = "\n\n---\n\n"

NUMBER_WORDS = {
    0: "none", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

QUESTION_TYPES = [
    "who_has", "what_has", "yes_no", "who_has_what", "how_many",
    "who_doesnt_have", "anyone_has", "comparison",
]


class PossessionState:
    """Ground-truth tracker for object ownership (person -> set of objects).

    Used during data generation to produce correct answers for questions.
    Supports give, transfer, and various query operations.
    """

    def __init__(self):
        self.holders: dict = {}

    def give(self, person: str, obj: str) -> None:
        self.holders[person] = self.holders.get(person, set()) | {obj}

    def transfer(self, from_person: str, to_person: str, obj: str) -> bool:
        if from_person not in self.holders or obj not in self.holders[from_person]:
            return False
        self.holders[from_person].discard(obj)
        self.holders[to_person] = self.holders.get(to_person, set()) | {obj}
        return True

    def who_has(self, obj: str) -> Optional[str]:
        for p, objs in self.holders.items():
            if obj in objs:
                return p
        return None

    def what_does_have(self, person: str) -> List[str]:
        return sorted(self.holders.get(person, set()))

    def count(self, person: str) -> int:
        return len(self.holders.get(person, set()))

    def all_possessions(self) -> List[Tuple[str, List[str]]]:
        result = []
        for p in sorted(self.holders.keys()):
            objs = sorted(self.holders.get(p, set()))
            if objs:
                result.append((p, objs))
        return result


def _format_objects(objects: List[str]) -> str:
    """['ball'] -> 'the ball', ['ball', 'key'] -> 'the ball and the key'"""
    if len(objects) == 1:
        return f"the {objects[0]}"
    return " and ".join(f"the {obj}" for obj in objects)


def format_conversation(turns: List[Tuple[str, str]]) -> str:
    parts = []
    for client_msg, output_msg in turns:
        parts.append(
            f"{CLIENT_PREFIX}{client_msg.strip()}"
            f"{CONVERSATION_TURN_SEP}"
            f"{OUTPUT_PREFIX}{output_msg.strip()}"
        )
    return CONVERSATION_TURN_SEP.join(parts)


def _format_people(names: List[str]) -> str:
    """['Alice'] -> 'Alice', ['Alice', 'Bob'] -> 'Alice and Bob'"""
    if len(names) == 1:
        return names[0]
    return " and ".join(names)


def _build_question(q_type: str, state: PossessionState,
                    people: List[str], objects: List[str]
                    ) -> Optional[Tuple[str, str]]:
    """Build a (question, answer) pair for the given type."""

    if q_type == "who_has":
        held = [o for o in objects if state.who_has(o) is not None]
        if not held:
            return None
        obj = random.choice(held)
        holder = state.who_has(obj)
        return (f"Who has the {obj}?", f"{holder} has the {obj}.")

    if q_type == "what_has":
        person = random.choice(people)
        things = state.what_does_have(person)
        q = f"What does {person} have?"
        if things:
            return (q, f"{_format_objects(things)}.")
        return (q, "nothing.")

    if q_type == "yes_no":
        person = random.choice(people)
        obj = random.choice(OBJECTS)
        q = f"Does {person} have the {obj}?"
        if obj in state.what_does_have(person):
            return (q, "Yes.")
        return (q, "No.")

    if q_type == "who_has_what":
        possessions = state.all_possessions()
        if not possessions:
            return None
        parts = []
        for person, objs in possessions:
            parts.append(f"{person} has {_format_objects(objs)}.")
        return ("Who has what?", " ".join(parts))

    if q_type == "how_many":
        person = random.choice(people)
        c = state.count(person)
        word = NUMBER_WORDS.get(c, str(c))
        return (f"How many things does {person} have?", f"{word}.")

    if q_type == "who_doesnt_have":
        held = [o for o in objects if state.who_has(o) is not None]
        if not held:
            return None
        obj = random.choice(held)
        non_holders = sorted(p for p in people if obj not in state.what_does_have(p))
        if not non_holders:
            return None
        verb = "does not" if len(non_holders) == 1 else "do not"
        return (
            f"Who does not have the {obj}?",
            f"{_format_people(non_holders)} {verb} have the {obj}.",
        )

    if q_type == "anyone_has":
        obj = random.choice(OBJECTS)
        holder = state.who_has(obj)
        q = f"Does anyone have the {obj}?"
        if holder is not None:
            return (q, f"Yes, {holder} has the {obj}.")
        return (q, "No.")

    if q_type == "comparison":
        if len(people) < 2:
            return None
        a, b = random.sample(people, 2)
        q = f"Does {a} have more things than {b}?"
        if state.count(a) > state.count(b):
            return (q, "Yes.")
        return (q, "No.")

    return None


_POSSESSION_TEMPLATES = [
    lambda p, obj: f"{p} has the {obj}.",
    lambda p, obj: f"{p} gets the {obj}.",
    lambda p, obj: f"{p} receives the {obj}.",
    lambda p, obj: f"{p} takes the {obj}.",
]

_COMPOUND_POSSESSION_TEMPLATES = [
    lambda p, objs: f"{p} has {_format_objects(objs)}.",
    lambda p, objs: f"{p} gets {_format_objects(objs)}.",
    lambda p, objs: f"{p} receives {_format_objects(objs)}.",
]


def _phrase_possession(person: str, objs: List[str]) -> List[Tuple[str, str]]:
    """Generate phrased possession statement(s) for one person getting objects."""
    if len(objs) >= 2 and random.random() < 0.4:
        tpl = random.choice(_COMPOUND_POSSESSION_TEMPLATES)
        return [(tpl(person, objs), ACK)]
    tpl = random.choice(_POSSESSION_TEMPLATES)
    return [(tpl(person, obj), ACK) for obj in objs]


def _add_possession(state: PossessionState, turns: List[Tuple[str, str]],
                     people: List[str], objects_pool: List[str]) -> List[str]:
    """Give one person 1-2 new objects. Returns list of newly assigned objects."""
    person = random.choice(people)
    available = [o for o in objects_pool
                 if state.who_has(o) is None and o not in [
                     o2 for p in state.holders for o2 in state.holders.get(p, set())]]
    if not available:
        return []
    count = min(random.choice([1, 1, 1, 2]), len(available))
    objs = sorted(random.sample(available, count))
    for obj in objs:
        state.give(person, obj)
    turns.extend(_phrase_possession(person, objs))
    return objs


_TRANSFER_TEMPLATES = [
    lambda g, r, obj: f"{g} gives the {obj} to {r}.",
    lambda g, r, obj: f"{g} passes the {obj} to {r}.",
    lambda g, r, obj: f"{r} takes the {obj} from {g}.",
    lambda g, r, obj: f"{r} gets the {obj} from {g}.",
]


def _add_transfer(state: PossessionState, turns: List[Tuple[str, str]],
                  people: List[str]) -> bool:
    """Transfer one object between two people. Returns True if successful."""
    givers = [p for p in people if state.what_does_have(p)]
    if not givers:
        return False
    giver = random.choice(givers)
    obj = random.choice(state.what_does_have(giver))
    receivers = [p for p in people if p != giver]
    if not receivers:
        return False
    receiver = random.choice(receivers)
    state.transfer(giver, receiver, obj)
    tpl = random.choice(_TRANSFER_TEMPLATES)
    turns.append((tpl(giver, receiver, obj), ACK))
    return True


def _add_question(state: PossessionState, turns: List[Tuple[str, str]],
                  people: List[str], objects: List[str]) -> bool:
    """Ask a question about current state. Returns True if successful."""
    q_type = random.choice(QUESTION_TYPES)
    qa = _build_question(q_type, state, people, objects)
    if qa is None:
        return False
    turns.append(qa)
    return True


def _evaluate_math(nums: List[int], ops: List[str]) -> int:
    """Evaluate expression left-to-right. nums[0] op[0] nums[1] op[1] nums[2] ..."""
    result = nums[0]
    for i, op in enumerate(ops):
        if op == "+":
            result += nums[i + 1]
        else:
            result -= nums[i + 1]
    return result


def _generate_math_turn() -> Tuple[str, str]:
    """Generate a math question and answer. Single-digit operands, 2-4 operations, left-to-right."""
    num_ops = random.randint(1, 4)  # 1 op = two numbers, 4 ops = five numbers
    nums = [random.randint(0, 9) for _ in range(num_ops + 1)]
    ops = [random.choice(["+", "-"]) for _ in range(num_ops)]

    result = _evaluate_math(nums, ops)

    parts = [str(nums[0])]
    for i, op in enumerate(ops):
        parts.append(op)
        parts.append(str(nums[i + 1]))
    expr = " ".join(parts)
    question = f"What is {expr}?"

    if result >= 0:
        answer = f"{result}."
    else:
        answer = f"- {abs(result)}."

    return (question, answer)


def _add_math(turns: List[Tuple[str, str]]) -> bool:
    """Add a math question turn. Always succeeds."""
    turns.append(_generate_math_turn())
    return True


def generate_conversation_example() -> Optional[Tuple[List[Tuple[str, str]], PossessionState]]:
    """Generate a single randomized conversation.

    Structure:
      Phase 1 — Initial possessions: each sampled person gets at least one object.
      Phase 2 — Interleaved actions: random mix of questions, transfers, and
                additional possessions (0-4 extra actions).
      Phase 3 — 80% chance of a final question to close the conversation.

    Returns None if the conversation would be too short (< 2 turns).
    """
    num_people = random.randint(1, len(PEOPLE))
    people = random.sample(PEOPLE, num_people)
    objects = list(OBJECTS)

    state = PossessionState()
    turns: List[Tuple[str, str]] = []
    assigned_objects: List[str] = []

    # Phase 1: initial possessions (each person gets at least 1 object)
    assignment: dict = {p: [] for p in people}
    num_initial = random.randint(num_people, min(num_people * 2, len(objects)))
    shuffled = random.sample(objects, num_initial)
    for i, p in enumerate(people):
        assignment[p].append(shuffled[i])
    for obj in shuffled[num_people:]:
        assignment[random.choice(people)].append(obj)

    for person in people:
        objs = sorted(assignment[person])
        for obj in objs:
            state.give(person, obj)
            assigned_objects.append(obj)
        turns.extend(_phrase_possession(person, objs))

    # Phase 2: interleaved actions (50% math, 20% questions, 20% transfers, 10% possessions)
    num_extra_actions = random.randint(0, 4)
    for _ in range(num_extra_actions):
        r = random.random()
        if r < 0.50:
            _add_math(turns)
        elif r < 0.70:
            _add_question(state, turns, people, assigned_objects)
        elif r < 0.90 and num_people >= 2:
            _add_transfer(state, turns, people)
        else:
            new = _add_possession(state, turns, people, objects)
            assigned_objects.extend(new)

    # Phase 3: 80% chance of ending with a final question
    if random.random() < 0.8:
        _add_question(state, turns, people, assigned_objects)

    if len(turns) < 2:
        return None

    return (turns, state)


def generate_dataset(
    n: int = 10_000,
    seed: Optional[int] = None,
    **kwargs,
) -> Iterator:
    """Yield n formatted conversation strings, skipping any with out-of-vocabulary words."""
    if seed is not None:
        random.seed(seed)

    count = 0
    attempts = 0
    max_attempts = n * 40

    while count < n and attempts < max_attempts:
        attempts += 1

        result = generate_conversation_example()
        if result is None:
            continue

        turns, _ = result
        all_valid = True
        for client_msg, output_msg in turns:
            for text in [client_msg, output_msg]:
                valid, unknown = is_valid_sentence(text)
                if not valid:
                    all_valid = False
                    break
            if not all_valid:
                break
        if all_valid:
            count += 1
            yield format_conversation(turns)


def generate_and_save(
    output_path: str,
    n: int = 10_000,
    seed: int = 42,
    **kwargs,
) -> None:
    """Generate n conversations and write them to a file, separated by '---'."""
    with open(output_path, "w") as f:
        first = True
        for item in generate_dataset(n=n, seed=seed):
            if not first:
                f.write(CONVERSATION_SEPARATOR)
            f.write(item)
            first = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TinyGPT data splits")
    parser.add_argument("--train", type=int, default=20_000)
    parser.add_argument("--val", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=2_000)
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    if args.preview > 0:
        print("Sample examples:\n")
        for i, conv in enumerate(generate_dataset(n=args.preview, seed=42)):
            print(f"--- Example {i + 1} ---")
            print(conv)
            print()
    else:
        from vocabulary import VOCAB_SIZE, get_vocab_stats

        os.makedirs(args.outdir, exist_ok=True)

        splits = [
            ("train", args.train, 42),
            ("val", args.val, 123),
            ("test", args.test, 456),
        ]

        stats = get_vocab_stats()
        print(f"Vocabulary: {VOCAB_SIZE} tokens ({stats['people']} people, "
              f"{stats['objects']} objects, {stats['verbs']} verbs)")
        print()

        for name, n, seed in splits:
            path = os.path.join(args.outdir, f"{name}.txt")
            print(f"Generating {name}: {n} examples (seed={seed})...", end=" ", flush=True)
            generate_and_save(path, n=n, seed=seed)
            size_kb = os.path.getsize(path) / 1024
            print(f"done -> {path} ({size_kb:.1f} KB)")

        print(f"\nAll splits saved to {args.outdir}/")
