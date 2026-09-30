"""
Three data modes controlled by config.data_source:
  "random" : generates random token IDs (for smoke tests, no download needed)
  "local"  : reads multiple sharded .bin memmap files from a directory
  "stream" : (removed) pile-deduped-pythia-preshuffled only has raw .bin files,
             not parquet/JSON, so HuggingFace streaming is not supported.

Both real modes yield (input_ids, labels) tensors of shape (context_length,)
where labels = input_ids shifted by 1 (standard LM objective, no padding).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Local multi-shard memmap dataset
# ---------------------------------------------------------------------------

class PileMemmap(Dataset):
    """
    Reads all .bin shards in a directory (GPT-NeoX uint16 memmap format).
    Shards are treated as one concatenated token stream; indices map across
    shard boundaries via a cumulative-length lookup.
    """

    def __init__(self, data_dir: str, context_length: int):
        self.context_length = context_length

        bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not bin_files:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")

        # Open each shard as a memmap (lazy, no RAM cost)
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in bin_files]

        # Number of full (context_length+1)-token chunks per shard
        chunk = context_length + 1
        self.shard_n_chunks = [len(s) // chunk for s in self.shards]
        self.cumulative = np.cumsum([0] + self.shard_n_chunks)  # length n_shards+1

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, idx):
        # Binary search to find which shard this global idx falls in
        shard_idx = int(np.searchsorted(self.cumulative[1:], idx, side="right"))
        local_idx = idx - int(self.cumulative[shard_idx])
        chunk = self.context_length + 1
        start = local_idx * chunk
        tokens = torch.from_numpy(
            self.shards[shard_idx][start : start + chunk].astype(np.int64)
        ) 
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# Random dataset (smoke tests only)
# ---------------------------------------------------------------------------

class RandomTokenDataset(IterableDataset):
    """Yields infinite random token sequences. No download needed."""

    def __init__(self, context_length: int, vocab_size: int = 50304):
        self.context_length = context_length
        self.vocab_size = vocab_size

    def __iter__(self):
        while True:
            tokens = torch.randint(0, self.vocab_size, (self.context_length + 1,))
            yield tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# Factory used by train.py
# ---------------------------------------------------------------------------

def make_dataset(config):
    if config.data_source == "local":
        return PileMemmap(config.local_data_path, config.context_length)
    elif config.data_source == "random":
        return RandomTokenDataset(config.context_length)
    else:
        raise ValueError(f"Unknown data_source: {config.data_source!r}")
