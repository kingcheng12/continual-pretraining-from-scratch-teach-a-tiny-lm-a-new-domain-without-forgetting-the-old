"""
Continual Pretraining from Scratch: Teach a Tiny LM a New Domain Without Forgetting the Old

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fetch_text
import os
import tempfile
import urllib.request

URLS = {"stories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"}


def fetch_text(url, name, max_chars):

    cache_path = os.path.join(tempfile.gettempdir(), f"cpt_{name}.txt")

    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(url, cache_path)

    with open(cache_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)

    return text

# Step 2 - CharTokenizer
import torch


class CharTokenizer:
    def __init__(self, texts):
        self.itos = sorted(set("".join(texts)))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, s):
        return torch.tensor(
            [self.stoi[ch] for ch in s if ch in self.stoi],
            dtype=torch.long
        )

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self):
        return len(self.itos)

# Step 3 - make_batch
import torch


def make_batch(ids, T, batch, generator):
    # TODO: random starts, (batch, T) inputs and next-character targets
    starts = torch.randint(
        0, len(ids) - T,
        (batch,),
        generator=generator
    )

    x = torch.stack([ids[i:i+T] for i in starts])
    y = torch.stack([ids[i+1:i+T+1] for i in starts])

    return x, y

