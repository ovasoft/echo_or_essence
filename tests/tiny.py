"""A tiny synthetic world and a tiny GPT-2 trained on it, for end-to-end probe tests."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

SYLL = ["ka", "lo", "mi", "ru", "te", "vo", "sa", "ne", "di", "po", "ze", "fu"]


def make_world(n_chains: int = 8, chain_len: int = 5, seed: int = 0) -> dict:
    rng = random.Random(seed)
    names: list[str] = []
    while len(names) < n_chains * chain_len:
        n = (rng.choice(SYLL) + rng.choice(SYLL) + rng.choice(SYLL)).capitalize()
        if n not in names:
            names.append(n)
    facts = []
    for c in range(n_chains):
        chain = names[c * chain_len:(c + 1) * chain_len]
        facts += [[a, "father_of", b] for a, b in zip(chain, chain[1:])]
    return {
        "relations": {"father_of": {
            "train_direction": "head",
            "statement": "{head} begat {tail} .",
            "templates": [{"text": "{head} begat", "seen": True},
                          {"text": "The son of {head} was", "seen": False},
                          {"text": "The father of {tail} was", "seen": False}],
            "ancestor": {"classification": "Was {a} an ancestor of {b} ?",
                         "comparison": "Who lived earlier , {a} or {b} ?"}}},
        "composites": {"grandfather_of": {"path": ["father_of", "father_of"],
                                          "templates": ["The grandfather of {tail} was"]}},
        "facts": facts,
    }


def corpus(world: dict) -> str:
    return " ".join(f"{h} begat {t} ." for h, _, t in world["facts"])


def train_tiny(world: dict, out: Path, steps: int = 400, save_at: tuple[int, ...] = (), seed: int = 0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    extra = ("The son of was father grandfather ancestor Was Who lived earlier , or ? Yes No Question : "
             "Answer an of").split()
    vocab_words = sorted(set(corpus(world).split()) | set(extra))
    vocab = {"[UNK]": 0, "[PAD]": 1, **{w: i + 2 for i, w in enumerate(vocab_words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]")
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=64, n_layer=2, n_head=2,
                     bos_token_id=None, eos_token_id=None, pad_token_id=1)
    model = GPT2LMHeadModel(cfg)
    sents = [f"{h} begat {t} ." for h, _, t in world["facts"]]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ckpts = {}

    def save(step):
        d = out / f"step_{step}"
        model.save_pretrained(d)
        tok.save_pretrained(d)
        ckpts[step] = d

    if 0 in save_at:
        save(0)
    model.train()
    for step in range(1, steps + 1):
        batch = []
        for _ in range(16):
            if rng.random() < 0.5:
                batch.append(corpus(world))  # fixed order: lets memorization items match
            else:
                s = sents[:]
                rng.shuffle(s)
                batch.append(" ".join(s))
        enc = tok(batch, return_tensors="pt", padding=True)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        loss = model(**enc, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in save_at:
            save(step)
    model.eval()
    return model, tok, ckpts
