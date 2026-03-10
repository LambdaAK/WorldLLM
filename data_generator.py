"""
Synthetic data generator for possession-tracking.
Generates (context, question, answer) triples by simulating state changes.

Features:
- Multi-hop chains (Alice -> Bob -> Carol -> Dave)
- Red herrings (irrelevant transfers to distract)
- Multiple objects per person
- Richer verbs (passes, receives, takes)
- Grammar consistency (always use articles)
"""

import random
from typing import Iterator, List, Optional, Tuple
from vocabulary import (
    PEOPLE,
    OBJECTS,
    is_valid_sentence,
)


# --- Grammar: always use articles. Use "a" for first mention, "the" for known. ---

POSSESSION_TEMPLATES = [
    "{person} has a {object}.",
    "{person} has the {object}.",
]

# Multiple objects: "Alice has a ball and a banana."
POSSESSION_MULTI_TEMPLATES = [
    "{person} has {art1} {obj1} and {art2} {obj2}.",
    "{person} has the {obj1} and the {obj2}.",
    "{person} has {art1} {obj1}, {art2} {obj2}, and {art3} {obj3}.",
]

# Transfer templates - richer verb variety. Use {article} for grammar consistency.
TRANSFER_GIVE_TEMPLATES = [
    "{person} gives {article} {object} to {person2}.",
]
TRANSFER_PASS_TEMPLATES = [
    "{person} passes {article} {object} to {person2}.",
]
TRANSFER_RECEIVE_TEMPLATES = [
    "{person2} receives {article} {object} from {person}.",
]
TRANSFER_TAKE_TEMPLATES = [
    "{person2} takes {article} {object} from {person}.",
]
TRANSFER_IT_TEMPLATES = [
    "{person} gives it to {person2}.",
    "{person} passes it to {person2}.",
    "{person2} receives it from {person}.",
    "{person2} takes it from {person}.",
]

# All transfer template groups (for random selection)
TRANSFER_TEMPLATE_GROUPS = [
    TRANSFER_GIVE_TEMPLATES,
    TRANSFER_PASS_TEMPLATES,
    TRANSFER_RECEIVE_TEMPLATES,
    TRANSFER_TAKE_TEMPLATES,
]

QUESTION_TEMPLATES_WHO_HAS = [
    "Who has the {object}?",
    "Who has a {object}?",
]

QUESTION_TEMPLATES_WHAT_HAS = [
    "What does {person} have?",
    "What has {person}?",
]

ANSWER_TEMPLATES_WHO = [
    "{person} has it.",
    "{person} has the {object}.",
    "{person} has a {object}.",
    "{person}.",
]
ANSWER_TEMPLATES_WHAT = [
    "{person} has the {object}.",
    "{person} has {article} {object}.",
    "The {object}.",
    "{article_cap} {object}.",
]
# For multiple objects
ANSWER_TEMPLATES_WHAT_MULTI = [
    "{person} has the {obj1} and the {obj2}.",
    "{person} has a {obj1} and a {obj2}.",
]

# Acknowledgments for state updates (model says "got it" or echoes the state)
ACK_TEMPLATES = [
    "Okay, got it.",
    "Got it.",
]
# When True, sometimes OUTPUT echoes the client's state update instead of just "Got it"
ACK_CAN_ECHO = True


def _pick(items: list, k: int, exclude: Optional[set] = None) -> list:
    """Pick k unique items from list, optionally excluding some."""
    exclude = exclude or set()
    pool = [x for x in items if x not in exclude]
    return random.sample(pool, min(k, len(pool)))


def _article(obj: str, use_the: bool = False) -> str:
    """Return 'a'/'an' or 'the' for the object. Use 'the' for known/repeated refs."""
    if use_the:
        return "the"
    return "an" if obj and obj[0].lower() in "aeiou" else "a"


class PossessionState:
    """Tracks who has what. person -> set of objects."""

    def __init__(self):
        self.holders: dict[str, set[str]] = {}

    def give(self, person: str, obj: str) -> None:
        self.holders[person] = self.holders.get(person, set()) | {obj}

    def take(self, person: str, obj: str) -> bool:
        """Remove obj from whoever has it. Returns True if found."""
        for p, objs in self.holders.items():
            if obj in objs:
                self.holders[p] = objs - {obj}
                return True
        return False

    def transfer(self, from_person: str, to_person: str, obj: str) -> bool:
        if self.who_has(obj) != from_person:
            return False
        self.take(from_person, obj)
        self.give(to_person, obj)
        return True

    def who_has(self, obj: str) -> Optional[str]:
        for p, objs in self.holders.items():
            if obj in objs:
                return p
        return None

    def what_does_have(self, person: str) -> List[str]:
        return list(self.holders.get(person, set()))


def _format_possession(
    person: str, obj: str, use_the: bool = False
) -> str:
    """Grammar-consistent possession sentence."""
    art = _article(obj, use_the)
    return f"{person} has {art} {obj}."


def _format_transfer(
    from_p: str, to_p: str, obj: str, use_pronoun: bool, use_the: bool
) -> str:
    """Pick a random transfer template with correct grammar."""
    if use_pronoun:
        return random.choice(TRANSFER_IT_TEMPLATES).format(
            person=from_p, person2=to_p, object=obj
        )
    article = _article(obj, use_the)
    group = random.choice(TRANSFER_TEMPLATE_GROUPS)
    t = random.choice(group)
    return t.format(
        person=from_p, person2=to_p, object=obj, article=article
    )


def generate_example(
    num_sentences: Tuple[int, int] = (4, 8),
    use_pronouns: bool = True,
    question_type: str = "who_has",
    max_people: int = 5,
    max_objects: int = 6,
    min_transfers: int = 2,
    max_transfers: int = 6,
    allow_multi_object: bool = True,
    allow_red_herring: bool = True,
    allow_multi_hop: bool = True,
) -> Optional[Tuple[str, str, str]]:
    """
    Generate one (context, question, answer) example.
    """
    people = _pick(PEOPLE, max_people)
    objects = _pick(OBJECTS, max_objects)

    if len(people) < 2 or len(objects) < 1:
        return None

    state = PossessionState()
    sentences: List[str] = []
    mentioned_objects: dict[str, bool] = {}  # obj -> has been mentioned (use "the")
    last_object: Optional[str] = None

    # --- Phase 1: Initial possessions ---
    # Option A: Single object per person (2-3 people)
    # Option B: Multiple objects for one person
    use_multi = allow_multi_object and random.random() < 0.35 and len(objects) >= 2

    if use_multi:
        # One person gets 2-3 objects
        person = random.choice(people)
        objs = _pick(objects, min(3, len(objects)))
        if len(objs) == 2:
            t = random.choice(POSSESSION_MULTI_TEMPLATES[:2])
            art1, art2 = _article(objs[0], False), _article(objs[1], False)
            s = t.format(
                person=person, obj1=objs[0], obj2=objs[1], obj3="",
                art1=art1, art2=art2, art3="",
            )
        else:
            t = POSSESSION_MULTI_TEMPLATES[2]
            art1 = _article(objs[0], False)
            art2 = _article(objs[1], False)
            art3 = _article(objs[2], False)
            s = t.format(
                person=person, obj1=objs[0], obj2=objs[1], obj3=objs[2],
                art1=art1, art2=art2, art3=art3,
            )
        for o in objs:
            state.give(person, o)
            mentioned_objects[o] = True
        sentences.append(s)
        # Others get one object each
        others = [p for p in people if p != person]
        for p in others[:2]:
            obj = random.choice([o for o in objects if o not in objs])
            state.give(p, obj)
            sentences.append(_format_possession(p, obj, use_the=obj in mentioned_objects))
            mentioned_objects[obj] = True
    else:
        for i, p in enumerate(people[: min(4, len(people))]):
            obj = objects[i % len(objects)]
            state.give(p, obj)
            use_the = obj in mentioned_objects
            sentences.append(_format_possession(p, obj, use_the=use_the))
            mentioned_objects[obj] = True
            last_object = obj

    # --- Phase 2: Transfers (multi-hop chains + optional red herrings) ---
    num_transfers = random.randint(min_transfers, max_transfers)
    target_object: Optional[str] = None  # Object we'll ask about (for red herring logic)
    transfer_count = 0

    for _ in range(num_transfers * 2):  # Extra attempts in case some fail
        if transfer_count >= num_transfers:
            break

        from_p = random.choice(people)
        to_p = random.choice([p for p in people if p != from_p])
        obj = random.choice(objects)

        if state.who_has(obj) != from_p:
            continue

        use_the = obj in mentioned_objects
        use_it = use_pronouns and last_object == obj and random.random() < 0.5

        s = _format_transfer(from_p, to_p, obj, use_pronoun=use_it, use_the=use_the)
        state.transfer(from_p, to_p, obj)
        sentences.append(s)
        mentioned_objects[obj] = True
        last_object = obj
        transfer_count += 1

    # --- Phase 3: Red herring (irrelevant transfer) ---
    if allow_red_herring and random.random() < 0.4 and transfer_count > 0:
        # Add a transfer that we won't ask about
        for _ in range(10):
            from_p = random.choice(people)
            to_p = random.choice([p for p in people if p != from_p])
            obj = random.choice(objects)
            if state.who_has(obj) == from_p:
                use_the = obj in mentioned_objects
                use_it = last_object == obj and random.random() < 0.5
                s = _format_transfer(from_p, to_p, obj, use_pronoun=use_it, use_the=use_the)
                state.transfer(from_p, to_p, obj)
                sentences.append(s)
                mentioned_objects[obj] = True
                last_object = obj
                break

    context = " ".join(sentences)

    # --- Phase 4: Question and answer ---
    if question_type == "who_has":
        # Pick object that someone has
        obj = random.choice(objects)
        holder = state.who_has(obj)
        if holder is None:
            return None
        question = random.choice(QUESTION_TEMPLATES_WHO_HAS).format(object=obj)
        answer = random.choice(ANSWER_TEMPLATES_WHO).format(person=holder, object=obj)
    else:  # what_has
        person = random.choice(people)
        things = state.what_does_have(person)
        if not things:
            return None
        obj = random.choice(things)
        question = random.choice(QUESTION_TEMPLATES_WHAT_HAS).format(person=person)
        if len(things) >= 2 and random.random() < 0.3:
            # Answer with multiple objects
            objs = random.sample(things, 2)
            answer = random.choice(ANSWER_TEMPLATES_WHAT_MULTI).format(
                person=person, obj1=objs[0], obj2=objs[1]
            )
        else:
            art = _article(obj, False)
            art_cap = art.capitalize()
            answer = random.choice(ANSWER_TEMPLATES_WHAT).format(
                person=person, object=obj, article=art, article_cap=art_cap
            )

    return (context, question, answer)


# --- CLIENT/OUTPUT conversation format (decoder-only, next-token prediction) ---
# CLIENT: and OUTPUT: are special tokens (see vocabulary.SPECIAL_TOKENS) that separate
# different messages in the token stream.

CONVERSATION_TURN_SEP = "\n\n"
CLIENT_PREFIX = "CLIENT:\n"   # CLIENT: is a special token
OUTPUT_PREFIX = "OUTPUT:\n"   # OUTPUT: is a special token


def format_conversation(turns: List[Tuple[str, str]]) -> str:
    """
    Format a list of (client_msg, output_msg) pairs into the full token stream.
    Example:
      CLIENT:
      Alice has the ball.

      OUTPUT:
      Okay, got it.

      CLIENT:
      Alice gives the ball to Bob.

      OUTPUT:
      Okay, got it.

      CLIENT:
      Who has the ball?

      OUTPUT:
      Bob has the ball.
    """
    parts = []
    for client_msg, output_msg in turns:
        parts.append(f"{CLIENT_PREFIX}{client_msg.strip()}{CONVERSATION_TURN_SEP}{OUTPUT_PREFIX}{output_msg.strip()}")
    return CONVERSATION_TURN_SEP.join(parts)


def generate_conversation_example(
    question_type: str = "who_has",
    max_people: int = 5,
    max_objects: int = 6,
    num_initial_possessions: Tuple[int, int] = (2, 5),
    num_transfers_range: Tuple[int, int] = (1, 6),
    sentences_per_turn_range: Tuple[int, int] = (1, 3),
    allow_multi_object: bool = True,
    allow_red_herring: bool = True,
    end_with_question: bool = True,
) -> Optional[Tuple[List[Tuple[str, str]], PossessionState]]:
    """
    Generate one conversation as a list of (client_msg, output_msg) turns.
    Each CLIENT message is a state update or question; each OUTPUT is acknowledgment or answer.
    Returns (turns, state) or None.
    """
    people = _pick(PEOPLE, max_people)
    objects = _pick(OBJECTS, max_objects)
    if len(people) < 2 or len(objects) < 1:
        return None

    state = PossessionState()
    turns: List[Tuple[str, str]] = []
    mentioned_objects: dict[str, bool] = {}
    last_object: Optional[str] = None

    # Build state updates (sentences) and group into turns
    sentences: List[str] = []
    sentence_types: List[str] = []  # "possession" | "transfer"

    # Phase 1: Initial possessions
    num_initial = random.randint(*num_initial_possessions)
    use_multi = (
        allow_multi_object
        and random.random() < 0.35
        and len(objects) >= 2
        and num_initial >= 3
    )
    if use_multi:
        person = random.choice(people)
        objs = _pick(objects, min(3, len(objects)))
        if len(objs) == 2:
            t = random.choice(POSSESSION_MULTI_TEMPLATES[:2])
            art1, art2 = _article(objs[0], False), _article(objs[1], False)
            s = t.format(
                person=person, obj1=objs[0], obj2=objs[1], obj3="",
                art1=art1, art2=art2, art3="",
            )
        else:
            t = POSSESSION_MULTI_TEMPLATES[2]
            art1, art2, art3 = _article(objs[0], False), _article(objs[1], False), _article(objs[2], False)
            s = t.format(person=person, obj1=objs[0], obj2=objs[1], obj3=objs[2], art1=art1, art2=art2, art3=art3)
        for o in objs:
            state.give(person, o)
            mentioned_objects[o] = True
        sentences.append(s)
        sentence_types.append("possession")
        others = [p for p in people if p != person]
        for p in others[: max(1, num_initial - 1)]:
            obj = random.choice([o for o in objects if o not in objs])
            state.give(p, obj)
            s = _format_possession(p, obj, use_the=obj in mentioned_objects)
            mentioned_objects[obj] = True
            sentences.append(s)
            sentence_types.append("possession")
    else:
        for i, p in enumerate(people[: min(num_initial, len(people))]):
            obj = objects[i % len(objects)]
            state.give(p, obj)
            s = _format_possession(p, obj, use_the=obj in mentioned_objects)
            mentioned_objects[obj] = True
            sentences.append(s)
            sentence_types.append("possession")
            last_object = obj

    # Phase 2: Transfers
    num_transfers = random.randint(*num_transfers_range)
    for _ in range(num_transfers * 3):
        from_p = random.choice(people)
        to_p = random.choice([p for p in people if p != from_p])
        obj = random.choice(objects)
        if state.who_has(obj) != from_p:
            continue
        use_the = obj in mentioned_objects
        use_it = last_object == obj and random.random() < 0.5
        s = _format_transfer(from_p, to_p, obj, use_pronoun=use_it, use_the=use_the)
        state.transfer(from_p, to_p, obj)
        sentences.append(s)
        sentence_types.append("transfer")
        mentioned_objects[obj] = True
        last_object = obj
        if len([x for x in sentence_types if x == "transfer"]) >= num_transfers:
            break

    # Phase 3: Red herring
    if allow_red_herring and random.random() < 0.4:
        for _ in range(10):
            from_p = random.choice(people)
            to_p = random.choice([p for p in people if p != from_p])
            obj = random.choice(objects)
            if state.who_has(obj) == from_p:
                use_the = obj in mentioned_objects
                use_it = last_object == obj and random.random() < 0.5
                s = _format_transfer(from_p, to_p, obj, use_pronoun=use_it, use_the=use_the)
                state.transfer(from_p, to_p, obj)
                sentences.append(s)
                break

    # Group sentences into turns (variable sentences per CLIENT message)
    i = 0
    while i < len(sentences):
        max_group = min(sentences_per_turn_range[1], len(sentences) - i)
        min_group = min(sentences_per_turn_range[0], max_group)
        group_size = random.randint(min_group, max_group) if max_group > 0 else 1
        client_msg = " ".join(sentences[i : i + group_size])
        # OUTPUT: acknowledgment or sometimes echo the state
        if ACK_CAN_ECHO and random.random() < 0.3:
            output_msg = client_msg
        else:
            output_msg = random.choice(ACK_TEMPLATES)
        turns.append((client_msg, output_msg))
        i += group_size

    # Final turn: question + answer, or more state updates with acknowledgment
    if end_with_question:
        if question_type == "who_has":
            obj = random.choice(objects)
            holder = state.who_has(obj)
            if holder is None:
                return None
            question = random.choice(QUESTION_TEMPLATES_WHO_HAS).format(object=obj)
            answer = random.choice(ANSWER_TEMPLATES_WHO).format(person=holder, object=obj)
        else:
            person = random.choice(people)
            things = state.what_does_have(person)
            if not things:
                return None
            obj = random.choice(things)
            question = random.choice(QUESTION_TEMPLATES_WHAT_HAS).format(person=person)
            if len(things) >= 2 and random.random() < 0.3:
                objs = random.sample(things, 2)
                answer = random.choice(ANSWER_TEMPLATES_WHAT_MULTI).format(
                    person=person, obj1=objs[0], obj2=objs[1]
                )
            else:
                art = _article(obj, False)
                art_cap = art.capitalize()
                answer = random.choice(ANSWER_TEMPLATES_WHAT).format(
                    person=person, object=obj, article=art, article_cap=art_cap
                )
        turns.append((question, answer))
    else:
        # End with 1-2 more state updates (CLIENT gives state, OUTPUT acknowledges)
        target_extra = random.randint(1, 2)
        added = 0
        for _ in range(20):
            if added >= target_extra:
                break
            from_p = random.choice(people)
            to_p = random.choice([p for p in people if p != from_p])
            obj = random.choice(objects)
            if state.who_has(obj) == from_p:
                use_the = obj in mentioned_objects
                use_it = last_object == obj and random.random() < 0.5
                s = _format_transfer(from_p, to_p, obj, use_pronoun=use_it, use_the=use_the)
                state.transfer(from_p, to_p, obj)
                client_msg = s
                output_msg = random.choice(ACK_TEMPLATES) if random.random() > 0.3 else client_msg
                turns.append((client_msg, output_msg))
                mentioned_objects[obj] = True
                last_object = obj
                added += 1

    return (turns, state)


# LLM prompt formats: how to combine context + question into a single prompt string
PROMPT_FORMATS = {
    # Natural flow: "Alice has a ball. She gives it to Bob. Who has the ball? "
    "natural": "{context} {question} ",
    # Newline before question: "Alice has a ball.\n\nWho has the ball?\n"
    "newline": "{context}\n\n{question}\n",
    # Instruction-style (like ChatGPT): explicit prompt structure
    "instruction": "Based on the following scenario, answer the question.\n\nScenario: {context}\n\nQuestion: {question}\n\nAnswer: ",
    # Minimal: just concatenate with space
    "text": "{context} {question}",
}


def format_as_prompt(
    context: str, question: str, answer: str, style: str = "natural"
) -> Tuple[str, str]:
    """
    Format (context, question, answer) as LLM (prompt, completion) pair.
    Returns (prompt, completion) where the model is trained to generate completion given prompt.
    """
    fmt = PROMPT_FORMATS.get(style, PROMPT_FORMATS["natural"])
    prompt = fmt.format(context=context, question=question)
    completion = answer
    return (prompt, completion)


def generate_dataset(
    n: int = 10_000,
    num_sentences_range: Tuple[int, int] = (4, 10),
    question_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
    output_format: str = "conversation",  # "conversation" | "structured" | "llm" | "text"
    prompt_style: str = "natural",
) -> Iterator:
    """
    Generate n examples.

    output_format:
      - "conversation": CLIENT/OUTPUT multi-turn format (decoder-only, next-token prediction)
      - "structured": yields (context, question, answer)
      - "llm": yields (prompt, completion) for standard LLM training
      - "text": yields full sequence per line (for next-token prediction)
    """
    if seed is not None:
        random.seed(seed)
    question_types = question_types or ["who_has", "what_has"]

    count = 0
    attempts = 0
    max_attempts = n * 40

    while count < n and attempts < max_attempts:
        attempts += 1

        if output_format == "conversation":
            # Vary length: short (2-3 turns), medium (4-7), long (8-12+)
            tier = random.choice(["short", "medium", "long"])
            if tier == "short":
                num_initial = (2, 3)
                num_transfers = (0, 2)
                sentences_per_turn = (1, 2)
            elif tier == "medium":
                num_initial = (3, 5)
                num_transfers = (2, 5)
                sentences_per_turn = (1, 3)
            else:
                num_initial = (4, 6)
                num_transfers = (4, 8)
                sentences_per_turn = (2, 4)
            result = generate_conversation_example(
                question_type=random.choice(question_types),
                num_initial_possessions=num_initial,
                num_transfers_range=num_transfers,
                sentences_per_turn_range=sentences_per_turn,
                allow_multi_object=True,
                allow_red_herring=True,
                end_with_question=random.random() < 0.7,
            )
            if result is None:
                continue
            turns, _ = result
            all_valid = True
            for client_msg, output_msg in turns:
                for text in [client_msg, output_msg]:
                    valid, _ = is_valid_sentence(text)
                    if not valid:
                        all_valid = False
                        break
                if not all_valid:
                    break
            if all_valid:
                count += 1
                yield format_conversation(turns)
            continue

        num_sent = random.randint(*num_sentences_range)
        q_type = random.choice(question_types)
        use_pronouns = random.random() < 0.6

        ex = generate_example(
            num_sentences=(max(3, num_sent - 2), num_sent + 2),
            use_pronouns=use_pronouns,
            question_type=q_type,
            min_transfers=2,
            max_transfers=min(6, num_sent),
            allow_multi_object=True,
            allow_red_herring=True,
            allow_multi_hop=True,
        )
        if ex is None:
            continue

        context, question, answer = ex
        for text in [f"{context} {question}", answer]:
            valid, unknown = is_valid_sentence(text)
            if not valid:
                break
        else:
            count += 1
            if output_format == "structured":
                yield (context, question, answer)
            elif output_format == "llm":
                prompt, completion = format_as_prompt(context, question, answer, prompt_style)
                yield (prompt, completion)
            else:  # "text"
                prompt, completion = format_as_prompt(context, question, answer, prompt_style)
                yield prompt + completion


CONVERSATION_SEPARATOR = "\n\n---\n\n"


def generate_and_save(
    output_path: str,
    n: int = 10_000,
    seed: int = 42,
    output_format: str = "conversation",
    prompt_style: str = "natural",
) -> None:
    """
    Generate dataset and save to file.

    output_format="conversation": CLIENT/OUTPUT format, examples separated by ---
    output_format="structured": tab-separated context, question, answer
    output_format="llm": tab-separated prompt, completion (for training)
    output_format="text": one full sequence per line (for next-token prediction)
    """
    with open(output_path, "w") as f:
        first = True
        for item in generate_dataset(
            n=n, seed=seed, output_format=output_format, prompt_style=prompt_style
        ):
            if output_format == "conversation":
                if not first:
                    f.write(CONVERSATION_SEPARATOR)
                f.write(item)
                first = False
            elif output_format == "structured":
                ctx, q, a = item
                f.write(f"{ctx}\t{q}\t{a}\n")
            elif output_format == "llm":
                prompt, completion = item
                f.write(f"{prompt}\t{completion}\n")
            else:
                f.write(f"{item}\n")


if __name__ == "__main__":
    print("Sample examples (CLIENT/OUTPUT conversation format):\n")
    for i, conv in enumerate(
        generate_dataset(n=10, seed=42, output_format="conversation")
    ):
        print(f"--- Example {i + 1} ---")
        print(conv)
        print()
