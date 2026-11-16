"""Batched scoring for any Hugging Face causal LM (GPU when available)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ChoiceScore:
    logprobs: list[float]        # summed log-probability of each choice
    lengths: list[int]           # tokens per choice

    def norm(self, mode: str) -> list[float]:
        if mode == "sum":
            return self.logprobs
        if mode == "mean":
            return [lp / max(n, 1) for lp, n in zip(self.logprobs, self.lengths)]
        raise ValueError(f"Unknown normalization {mode!r}")

    def probs(self, mode: str = "sum") -> list[float]:
        s = self.norm(mode)
        m = max(s)
        e = [math.exp(x - m) for x in s]
        z = sum(e)
        return [x / z for x in e]


class Scorer:
    def __init__(self, model_path: str, device: str | None = None, dtype: str = "auto",
                 batch_size: int = 32, max_length: int | None = None, model=None, tokenizer=None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model = model or AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.model.to(self.device).eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token
        self.bs = batch_size
        self.max_length = max_length or getattr(self.model.config, "max_position_embeddings", None) or 2048
        self.bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []

    # ------------------------------------------------------------------ tokenization
    def _encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _split(self, prompt: str, cont: str) -> tuple[list[int], list[int]]:
        """Token ids for prompt and continuation, robust to merges at the boundary."""
        p = self._encode(prompt)
        full = self._encode(prompt + cont)
        if full[: len(p)] == p and len(full) > len(p):
            c = full[len(p):]
        else:
            c = self._encode(cont)
        p = self.bos + p
        overflow = len(p) + len(c) - self.max_length
        if overflow > 0:
            p = p[overflow:]
        return p, c

    # ------------------------------------------------------------------ log-probabilities
    @torch.no_grad()
    def _logprobs(self, pairs: list[tuple[list[int], list[int]]]) -> list[float]:
        out: list[float] = []
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        res = [0.0] * len(pairs)
        for s in range(0, len(order), self.bs):
            idx = order[s: s + self.bs]
            seqs = [pairs[i][0] + pairs[i][1] for i in idx]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, x in enumerate(seqs):
                ids[r, : len(x)] = torch.tensor(x)
                att[r, : len(x)] = 1
            logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            for r, i in enumerate(idx):
                p, c = pairs[i]
                pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=logp.device)
                tgt = torch.tensor(c, device=logp.device)
                res[i] = float(logp[r, pos, tgt].sum())
        out.extend(res)
        return out

    def score_choices(self, prompts: list[str], choices: list[list[str]]) -> list[ChoiceScore]:
        pairs, owners = [], []
        for i, (p, cs) in enumerate(zip(prompts, choices)):
            for c in cs:
                pairs.append(self._split(p, c))
                owners.append(i)
        lps = self._logprobs(pairs)
        out = [ChoiceScore([], []) for _ in prompts]
        for (pi, ci), lp, o in zip(pairs, lps, owners):
            out[o].logprobs.append(lp)
            out[o].lengths.append(len(ci))
        return out

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def greedy(self, prompts: list[str], conts: list[str]) -> list[dict]:
        """Greedy-decode exactly as many tokens as each reference continuation."""
        split = [self._split(p, c) for p, c in zip(prompts, conts)]
        out: list[dict | None] = [None] * len(split)
        by_len: dict[int, list[int]] = {}
        for i, (_, ci) in enumerate(split):
            by_len.setdefault(len(ci), []).append(i)
        for n_new, idxs in by_len.items():
            for s in range(0, len(idxs), self.bs):
                chunk = idxs[s: s + self.bs]
                L = max(len(split[i][0]) for i in chunk)
                ids = torch.full((len(chunk), L), self.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for r, i in enumerate(chunk):  # left padding for generation
                    p = split[i][0]
                    ids[r, L - len(p):] = torch.tensor(p)
                    att[r, L - len(p):] = 1
                gen = self.model.generate(ids.to(self.device), attention_mask=att.to(self.device),
                                          max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)[:, L:].tolist()
                for r, i in enumerate(chunk):
                    g, ci = gen[r][:n_new], split[i][1]
                    match = sum(int(a == b) for a, b in zip(g, ci))
                    out[i] = {"exact": g == ci, "token_acc": match / max(len(ci), 1),
                              "generated": self.tok.decode(g)}
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------ hidden states
    @torch.no_grad()
    def last_hidden(self, texts: list[str], layers: list[int]) -> torch.Tensor:
        """[n, len(layers), d] hidden state at the final token of each text."""
        feats = []
        for s in range(0, len(texts), self.bs):
            seqs = [(self.bos + self._encode(t))[-self.max_length:] for t in texts[s: s + self.bs]]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, x in enumerate(seqs):
                ids[r, : len(x)] = torch.tensor(x)
                att[r, : len(x)] = 1
            hs = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                            output_hidden_states=True).hidden_states
            last = att.sum(1) - 1
            rows = torch.arange(len(seqs))
            feats.append(torch.stack([hs[l][rows, last.to(hs[l].device)].float().cpu() for l in layers], 1))
        return torch.cat(feats, 0)

    @property
    def n_layers(self) -> int:
        return int(getattr(self.model.config, "num_hidden_layers", None)
                   or getattr(self.model.config, "n_layer"))
