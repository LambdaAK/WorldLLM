"""
Dataset and DataLoader for WorldLLM conversation data.
Reads the generated .txt files and produces padded token sequences
for decoder-only next-token prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from vocabulary import tokenize, PAD_ID


class ConversationDataset(Dataset):
    """
    Each example is one full conversation (CLIENT:/OUTPUT: turns).
    Tokenized and truncated/padded to max_seq_len.
    """

    def __init__(self, path: str, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(path, "r") as f:
            content = f.read()

        # Split on the separator between conversations
        raw_examples = content.split("\n\n---\n\n")

        for raw in raw_examples:
            raw = raw.strip()
            if not raw:
                continue
            ids = tokenize(raw, add_special=True)
            if len(ids) < 3:
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]

        # Truncate to max_seq_len
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        # Input: all tokens except last; Target: all tokens except first
        input_ids = ids[:-1]
        target_ids = ids[1:]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    inputs, targets = zip(*batch)

    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), PAD_ID, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), PAD_ID, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : inp.size(0)] = inp
        padded_targets[i, : tgt.size(0)] = tgt

    return padded_inputs, padded_targets


def create_dataloader(
    path: str,
    max_seq_len: int = 256,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ConversationDataset(path, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
