"""
Synthetic data generator for possession-tracking.
Supports: compound possession, transfers, multi-hop transfers,
yes/no questions, "who has what?", and counting.
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

NUMBER_WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}

QUESTION_TYPES = ["who_has", "what_has", "yes_no", "who_has_what", "how_many"]


class PossessionState:
    """Tracks who has what. person -> set of objects."""

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

    return None


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
    if count >= 2 and random.random() < 0.4:
        turns.append((f"{person} has {_format_objects(objs)}.", ACK))
    else:
        for obj in objs:
            turns.append((f"{person} has the {obj}.", ACK))
    return objs


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
    turns.append((f"{giver} gives the {obj} to {receiver}.", ACK))
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


def generate_conversation_example() -> Optional[Tuple[List[Tuple[str, str]], PossessionState]]:
    """
    Generate a conversation as a sequence of interleaved events:
    possession statements, transfers, and questions at any point.
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
        if len(objs) >= 2 and random.random() < 0.4:
            turns.append((f"{person} has {_format_objects(objs)}.", ACK))
        else:
            for obj in objs:
                turns.append((f"{person} has the {obj}.", ACK))

    # Phase 2: interleaved actions (questions, transfers, more possessions)
    num_extra_actions = random.randint(0, 4)
    for _ in range(num_extra_actions):
        r = random.random()
        if r < 0.45:
            _add_question(state, turns, people, assigned_objects)
        elif r < 0.75 and num_people >= 2:
            _add_transfer(state, turns, people)
        elif r < 0.90:
            new = _add_possession(state, turns, people, objects)
            assigned_objects.extend(new)
        else:
            if num_people >= 2:
                _add_transfer(state, turns, people)

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
