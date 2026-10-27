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

# Step 4 - load_domains
def load_domains(max_chars=600000, val_frac=0.1):
    # Fetch text from both domains
    texts = {
        name: fetch_text(url, name, max_chars)
        for name, url in URLS.items()
    }

    tok = CharTokenizer(list(texts.values()))

    domains = {"tok": tok}

    for key, name in [("A", "stories"), ("B", "shakespeare")]:
        ids = tok.encode(texts[name])
        split_idx = int(len(ids) * (1 - val_frac))

        domains[key] = {
            "name": name,
            "train": ids[:split_idx],
            "val": ids[split_idx:]
        }

    return domains

# Step 5 - TinyGPT
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()

        assert d % heads == 0

        self.heads = heads
        self.head_dim = d // heads

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Attention projections
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

        # Feed-forward network
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape

        # 1. Pre-LayerNorm
        h = self.ln1(x)

        # 2. Generate Q, K, V
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        # (B, T, D) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # 3. Causal scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # 4. Concatenate heads
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)

        # 5. Attention residual connection
        x = x + self.proj(attn)

        # 6. Feed-forward network with residual
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))

        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=64, heads=4, layers=2, max_len=64):
        super().__init__()

        self.max_len = max_len

        # Token and positional embeddings
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(max_len, d)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d, heads) for _ in range(layers)
        ])

        # Final layer normalization
        self.lnf = nn.LayerNorm(d)

        # Vocabulary projection
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        if T > self.max_len:
            raise ValueError("Sequence length exceeds max_len")

        # 1. Token + positional embeddings
        positions = torch.arange(T, device=idx.device)

        x = self.wte(idx) + self.wpe(positions)

        # 2. Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final normalization and logits
        x = self.lnf(x)
        logits = self.head(x)

        return logits

# Step 6 - lm_loss
import torch
import torch.nn.functional as F


def lm_loss(model, x, y):
    # logits: (B, T, vocab)
    logits = model(x)

    B, T, V = logits.shape

    # Flatten for cross-entropy
    logits = logits.reshape(B * T, V)
    targets = y.reshape(B * T)

    return F.cross_entropy(logits, targets, reduction="mean")

# Step 7 - eval_loss
import torch


@torch.no_grad()
def eval_loss(model, ids, T, n_batches, seed, batch=32):
    # Reproducible random sampling
    generator = torch.Generator().manual_seed(seed)

    # Save original training mode
    was_training = model.training
    model.eval()

    losses = []
    device = next(model.parameters()).device

    try:
        for _ in range(n_batches):
            x, y = make_batch(ids, T, batch, generator)

            x = x.to(device)
            y = y.to(device)

            loss = lm_loss(model, x, y)
            losses.append(loss.item())

        return sum(losses) / len(losses)

    finally:
        model.train(was_training)

