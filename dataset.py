"""
Dataset and DataLoader for TinyGPT conversation data.
Reads the generated .txt files and produces padded token sequences
for decoder-only next-token prediction.

Loss masking: only compute loss on OUTPUT tokens, not CLIENT tokens.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from vocabulary import tokenize, PAD_ID, CLIENT_ID, OUTPUT_ID


def _build_output_mask(ids: list[int]) -> list[int]:
    """
    Build a mask marking which token positions are OUTPUT content.
    1 = this is an OUTPUT token (compute loss here)
    0 = this is a CLIENT token, special token, or delimiter (ignore in loss)

    The mask is for the *target* sequence (shifted by 1), so we mark
    positions where the model should be able to predict the next token.
    After seeing OUTPUT:, the model should predict the response tokens.
    After seeing CLIENT:, the model can't predict what the user says.
    """
    mask = []
    in_output = False
    for token_id in ids:
        if token_id == OUTPUT_ID:
            in_output = True
            mask.append(0)  # don't score predicting the OUTPUT: token itself
        elif token_id == CLIENT_ID:
            in_output = False
            mask.append(0)
        else:
            mask.append(1 if in_output else 0)
    return mask


class ConversationDataset(Dataset):
    """
    Each example is one full conversation (CLIENT:/OUTPUT: turns).
    Tokenized and truncated/padded to max_seq_len.
    Returns (input_ids, target_ids, loss_mask).
    """

    def __init__(self, path: str, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(path, "r") as f:
            content = f.read()

        raw_examples = content.split("\n\n---\n\n")

        for raw in raw_examples:
            raw = raw.strip()
            if not raw:
                continue
            ids = tokenize(raw, add_special=True)
            if len(ids) < 3:
                continue
            output_mask = _build_output_mask(ids)
            self.examples.append((ids, output_mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, output_mask = self.examples[idx]

        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
            output_mask = output_mask[: self.max_seq_len]

        # Input: all tokens except last; Target: all tokens except first
        # Loss mask: aligned with target (shifted by 1)
        input_ids = ids[:-1]
        target_ids = ids[1:]
        loss_mask = output_mask[1:]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    inputs, targets, masks = zip(*batch)

    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), PAD_ID, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), PAD_ID, dtype=torch.long)
    padded_masks = torch.zeros((len(masks), max_len), dtype=torch.float)

    for i, (inp, tgt, msk) in enumerate(zip(inputs, targets, masks)):
        padded_inputs[i, : inp.size(0)] = inp
        padded_targets[i, : tgt.size(0)] = tgt
        padded_masks[i, : msk.size(0)] = msk

    return padded_inputs, padded_targets, padded_masks


def create_dataloader(
    path: str,
    max_seq_len: int = 256,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = ConversationDataset(path, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
