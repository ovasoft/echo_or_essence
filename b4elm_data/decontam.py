"""N-gram decontamination of the base corpus against the Bible and Stoic corpora.

Words are hashed once (blake2b, 64-bit), n-gram hashes are built with a vectorized polynomial
rolling hash, and membership is a binary search in a sorted uint64 array saved as .npy and
memory-mapped by every worker.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .text import STOPWORDS, decontam_tokens, strip_gutenberg

_P = np.uint64(1099511628211)
_N_SALT = np.uint64(0x9E3779B97F4A7C15)
_CACHE_MAX = 3_000_000


class WordHasher:
    def __init__(self) -> None:
        self._cache: dict[str, int] = {}

    def __call__(self, tokens: list[str]) -> np.ndarray:
        out = np.empty(len(tokens), dtype=np.uint64)
        cache = self._cache
        if len(cache) > _CACHE_MAX:
            cache.clear()
        for i, t in enumerate(tokens):
            h = cache.get(t)
            if h is None:
                h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")
                cache[t] = h
            out[i] = h
        return out


def ngram_hashes(wh: np.ndarray, n: int) -> np.ndarray:
    m = len(wh) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * _P + wh[i : i + m]
        h ^= np.uint64(n) * _N_SALT
    return h


def content_mask(tokens: list[str], n: int, min_content: int) -> np.ndarray:
    """True for windows with at least `min_content` non-stopwords."""
    m = len(tokens) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=bool)
    nonstop = np.fromiter((t not in STOPWORDS for t in tokens), dtype=np.int32, count=len(tokens))
    c = np.concatenate([[0], np.cumsum(nonstop)])
    return (c[n:] - c[:-n]) >= min_content


def read_reference(path: str) -> str:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:
        if ".jsonl" in p.name:
            return "\n\n".join(json.loads(line).get("text", "") for line in f if line.strip())
        return strip_gutenberg(f.read())


def expand(patterns: list[str], base: Path) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        full = pat if Path(pat).is_absolute() else str(base / pat)
        hits = sorted(glob.glob(full))
        if not hits:
            raise FileNotFoundError(f"No reference files match {full}")
        files.extend(hits)
    return files


def build_index(references: dict[str, list[str]], ngram_sizes: list[int], min_content: int,
                out_path: Path, base: Path) -> dict:
    hasher = WordHasher()
    chunks: list[np.ndarray] = []
    meta: dict = {"ngram_sizes": ngram_sizes, "min_content_words": min_content, "corpora": {}}
    for name, pats in references.items():
        entries = []
        for fp in expand(pats, base):
            raw = Path(fp).read_bytes()
            tokens = decontam_tokens(read_reference(fp))
            wh = hasher(tokens)
            n_grams = 0
            for n in ngram_sizes:
                h = ngram_hashes(wh, n)[content_mask(tokens, n, min_content)]
                chunks.append(h)
                n_grams += len(h)
            entries.append({"file": fp, "sha256": hashlib.sha256(raw).hexdigest(),
                            "tokens": len(tokens), "ngrams": n_grams})
        meta["corpora"][name] = entries
    idx = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.uint64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, idx)
    meta["unique_ngrams"] = int(len(idx))
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class Index:
    def __init__(self, path: Path | None, ngram_sizes: list[int]) -> None:
        self.ngram_sizes = ngram_sizes
        self.arr = np.load(path, mmap_mode="r") if path else np.empty(0, dtype=np.uint64)
        self.hasher = WordHasher()

    def hits(self, text: str) -> int:
        if len(self.arr) == 0:
            return 0
        tokens = decontam_tokens(text)
        if len(tokens) < min(self.ngram_sizes):
            return 0
        wh = self.hasher(tokens)
        total = 0
        for n in self.ngram_sizes:
            h = ngram_hashes(wh, n)
            if len(h) == 0:
                continue
            pos = np.searchsorted(self.arr, h)
            pos[pos == len(self.arr)] = 0
            total += int(np.count_nonzero(self.arr[pos] == h))
        return total
