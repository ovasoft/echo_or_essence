"""Filtering + decontamination pipeline and CLI.

Stages per document:
  1. subject      out-of-scope subject (resolved by the source loader)
  2. date         created on/after the source's max_created (LLM-era text)
  3. title        title path (book > chapter > section, or URL) matches the ethics blocklist
  4. doc_block    source-specific document blocklist (e.g. AI-alignment papers on arXiv)
  5. llm_marker   chat-model boilerplate
Stages per paragraph:
  6. contam_ref   Bible chapter:verse reference or named Stoic/KJV source
  7. contam_ngram n-gram overlap with the Bible/Stoic index
  8. religious    religious vocabulary
  9. normative    moral vocabulary or normative sentences above threshold
Then: documents shorter than min_doc_chars after paragraph removal are dropped (short),
and exact duplicate documents are dropped (duplicate).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import yaml

from . import lexicons as L
from .decontam import Index, build_index, expand, read_reference
from .scoring import score_corpus, score_text
from .sources import load_docs, inspect_source
from .text import collapse_ws, normalize, paragraphs, words

DOC_REASONS = ["subject", "date", "title", "doc_block", "llm_marker", "short", "duplicate"]
PAR_REASONS = ["contam_ref", "contam_ngram", "religious", "normative"]


# ---------------------------------------------------------------- config

def load_config(path: str) -> tuple[dict, Path]:
    p = Path(path).resolve()
    return yaml.safe_load(p.read_text()), p.parent


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


# ---------------------------------------------------------------- worker

_CTX: dict = {}


def init_worker(cfg: dict, base: str) -> None:
    base_p = Path(base)
    f = cfg["filters"]
    dc = cfg["decontam"]
    index_path = _resolve(base_p, dc["index_path"])
    if not index_path.exists():
        raise FileNotFoundError(f"Decontamination index missing at {index_path}; run build-index first")
    title_rx = L._compile(L.TITLE_BLOCK_DEFAULT + list(f.get("extra_title_block") or []))
    _CTX.update(
        f=f,
        index=Index(index_path, dc["ngram_sizes"]),
        min_hits=int(dc.get("min_hits", 1)),
        title_rx=title_rx,
        doc_block={name: (L._compile(s["doc_blocklist"]) if s.get("doc_blocklist") else None)
                   for name, s in cfg["sources"].items()},
        max_created={name: s.get("max_created") for name, s in cfg["sources"].items()},
        doc_block_chars=int(f.get("doc_block_scan_chars", 5000)),
        sample_k=int(cfg["output"].get("audit_samples_per_reason", 50)),
    )


def _paragraph_reason(par: str) -> str | None:
    f, s = _CTX["f"], score_text(par)
    if s.refs > 0:
        return "contam_ref"
    if _CTX["index"].hits(par) >= _CTX["min_hits"]:
        return "contam_ngram"
    if s.religious > int(f["religious_max_hits"]):
        return "religious"
    if s.moral > int(f["moral_max_hits"]) or (
        s.normative_sentences > 0 and s.normative_ratio > float(f["normative_ratio_max"])
    ):
        return "normative"
    return None


def process_doc(d: dict) -> dict:
    res = {"doc": None, "doc_reason": None, "par_drops": Counter(), "samples": []}
    src = d["source"]
    mc = _CTX["max_created"].get(src)
    if mc and d.get("created") and d["created"] >= str(mc):
        res["doc_reason"] = "date"
        return res
    if _CTX["title_rx"].search(normalize(d["title"] or "")):
        res["doc_reason"] = "title"
        res["samples"].append(("title", d["title"][:300]))
        return res
    head = normalize(d["text"][: _CTX["doc_block_chars"]])
    rx = _CTX["doc_block"].get(src)
    if rx is not None and rx.search(head):
        res["doc_reason"] = "doc_block"
        res["samples"].append(("doc_block", d["title"][:300]))
        return res
    if L.LLM_MARKERS.search(head):
        res["doc_reason"] = "llm_marker"
        return res
    kept = []
    per_reason_samples = Counter()
    for par in paragraphs(d["text"]):
        reason = _paragraph_reason(par)
        if reason is None:
            kept.append(par)
            continue
        res["par_drops"][reason] += 1
        if per_reason_samples[reason] < 2:
            per_reason_samples[reason] += 1
            res["samples"].append((reason, collapse_ws(par)[:500]))
    text = "\n\n".join(kept)
    if len(text) < int(_CTX["f"]["min_doc_chars"]):
        res["doc_reason"] = "short"
        return res
    res["doc"] = {"id": d["id"], "source": src, "subject": d["subject"], "title": d["title"], "text": text}
    return res


def process_batch(batch: list[dict]) -> list[dict]:
    return [process_doc(d) for d in batch]


# ---------------------------------------------------------------- output

class ShardWriter:
    def __init__(self, out_dir: Path, docs_per_shard: int) -> None:
        self.dir, self.per, self.n, self.shard, self.f = out_dir, docs_per_shard, 0, 0, None
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        if self.f is None or self.n >= self.per:
            self.close()
            self.f = gzip.open(self.dir / f"shard_{self.shard:05d}.jsonl.gz", "wt", encoding="utf-8")
            self.shard += 1
            self.n = 0
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


class Reservoir:
    def __init__(self, k: int, seed: int) -> None:
        self.k, self.rng = k, random.Random(seed)
        self.items: dict[str, list] = defaultdict(list)
        self.seen: Counter = Counter()

    def add(self, source: str, reason: str, text: str) -> None:
        key = f"{source}:{reason}"
        self.seen[key] += 1
        bucket = self.items[key]
        if len(bucket) < self.k:
            bucket.append(text)
        else:
            j = self.rng.randrange(self.seen[key])
            if j < self.k:
                bucket[j] = text


# ---------------------------------------------------------------- commands

def cmd_build_index(cfg: dict, base: Path) -> None:
    dc = cfg["decontam"]
    meta = build_index(dc["references"], dc["ngram_sizes"], int(dc.get("min_content_words", 2)),
                       _resolve(base, dc["index_path"]), base)
    print(json.dumps({"unique_ngrams": meta["unique_ngrams"],
                      "corpora": {k: [e["file"] for e in v] for k, v in meta["corpora"].items()}}, indent=2))


def _batches(it, size):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def cmd_run(cfg: dict, base: Path, only: list[str] | None, limit: int | None, num_proc: int) -> dict:
    out = cfg["output"]
    out_dir = _resolve(base, out["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reservoir = Reservoir(int(out.get("audit_samples_per_reason", 50)), int(out.get("seed", 0)))
    seen_hashes: set[bytes] = set()
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {},
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
    meta_path = _resolve(base, cfg["decontam"]["index_path"]).with_suffix(".meta.json")
    if meta_path.exists():
        manifest["decontam_index"] = json.loads(meta_path.read_text())

    pool = Pool(num_proc, initializer=init_worker, initargs=(cfg, str(base))) if num_proc > 1 else None
    if pool is None:
        init_worker(cfg, str(base))
    try:
        for name, scfg in cfg["sources"].items():
            if only and name not in only:
                continue
            if not only and not scfg.get("enabled", True):
                continue
            src_limit = limit or scfg.get("limit")
            stats = {"docs_seen": 0, "docs_kept": 0, "doc_drops": Counter(), "par_drops": Counter(),
                     "subjects_kept": Counter(), "chars_kept": 0, "words_kept": 0}
            writer = ShardWriter(out_dir / name, int(out.get("docs_per_shard", 10000)))

            def in_scope():
                for i, doc in enumerate(load_docs(name, scfg, base)):
                    if src_limit and i >= src_limit:
                        break
                    stats["docs_seen"] += 1
                    if doc.subject is None:
                        stats["doc_drops"]["subject"] += 1
                        continue
                    yield doc.to_dict()

            batches = _batches(in_scope(), int(cfg.get("batch_size", 64)))
            results = (pool.imap(process_batch, batches, chunksize=1) if pool
                       else (process_batch(b) for b in batches))
            for batch_res in results:
                for r in batch_res:
                    stats["par_drops"].update(r["par_drops"])
                    for reason, text in r["samples"]:
                        reservoir.add(name, reason, text)
                    if r["doc"] is None:
                        stats["doc_drops"][r["doc_reason"]] += 1
                        continue
                    h = hashlib.blake2b(collapse_ws(normalize(r["doc"]["text"])).encode(), digest_size=16).digest()
                    if h in seen_hashes:
                        stats["doc_drops"]["duplicate"] += 1
                        continue
                    seen_hashes.add(h)
                    writer.write(r["doc"])
                    stats["docs_kept"] += 1
                    stats["subjects_kept"][r["doc"]["subject"]] += 1
                    stats["chars_kept"] += len(r["doc"]["text"])
                    stats["words_kept"] += len(words(r["doc"]["text"]))
            writer.close()
            stats["est_tokens_kept"] = int(stats["words_kept"] * 1.3)
            stats["subjects_kept"] = dict(stats["subjects_kept"].most_common(100))
            stats["doc_drops"] = {k: stats["doc_drops"].get(k, 0) for k in DOC_REASONS}
            stats["par_drops"] = {k: stats["par_drops"].get(k, 0) for k in PAR_REASONS}
            manifest["sources"][name] = stats
            print(f"[{name}] seen={stats['docs_seen']} kept={stats['docs_kept']} "
                  f"est_tokens={stats['est_tokens_kept']:,}", file=sys.stderr)
    finally:
        if pool:
            pool.close()
            pool.join()

    manifest["totals"] = {
        k: sum(s[k] for s in manifest["sources"].values())
        for k in ("docs_seen", "docs_kept", "chars_kept", "words_kept", "est_tokens_kept")
    }
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    audit = out_dir / "audit"
    audit.mkdir(exist_ok=True)
    with open(audit / "dropped_samples.jsonl", "w", encoding="utf-8") as f:
        for key, items in sorted(reservoir.items.items()):
            src, reason = key.split(":", 1)
            for t in items:
                f.write(json.dumps({"source": src, "reason": reason, "text": t}, ensure_ascii=False) + "\n")
    return manifest


def cmd_measure(corpora: list[str], base: Path, tolerance: float | None) -> list[dict]:
    reports = []
    for spec in corpora:
        name, _, paths = spec.partition("=")
        files = expand(paths.split(","), base)
        reports.append(score_corpus(name, [(fp, read_reference(fp)) for fp in files]).as_dict())
    if tolerance is not None and len(reports) >= 2:
        a, b = reports[0]["normative_ratio"], reports[1]["normative_ratio"]
        rel = abs(a - b) / max(a, b) if max(a, b) else 0.0
        reports.append({"comparison": f"{reports[0]['name']} vs {reports[1]['name']}",
                        "relative_difference": round(rel, 4), "tolerance": tolerance,
                        "matched": rel <= tolerance})
    return reports


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="b4elm_data", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Build the Bible/Stoic n-gram index")
    b.add_argument("--config", required=True)

    i = sub.add_parser("inspect", help="Show raw subject values and resolved subjects for a source")
    i.add_argument("--config", required=True)
    i.add_argument("--source", required=True)
    i.add_argument("--n", type=int, default=2000)

    r = sub.add_parser("run", help="Filter and decontaminate the base corpus")
    r.add_argument("--config", required=True)
    r.add_argument("--sources", help="Comma-separated subset (overrides enabled flags)")
    r.add_argument("--limit", type=int, help="Max rows per source (pilot runs)")
    r.add_argument("--num-proc", type=int, default=1)

    m = sub.add_parser("measure", help="Normative density of corpora, e.g. bible=data/bible/*.txt")
    m.add_argument("corpora", nargs="+", help="name=path[,path...] (globs allowed)")
    m.add_argument("--tolerance", type=float, help="Compare first two corpora; e.g. 0.10")

    a = ap.parse_args(argv)
    if a.cmd == "measure":
        print(json.dumps(cmd_measure(a.corpora, Path.cwd(), a.tolerance), indent=2))
        return
    cfg, base = load_config(a.config)
    if a.cmd == "build-index":
        cmd_build_index(cfg, base)
    elif a.cmd == "inspect":
        print(json.dumps(inspect_source(a.source, cfg["sources"][a.source], base, a.n), indent=2,
                         ensure_ascii=False))
    elif a.cmd == "run":
        only = a.sources.split(",") if a.sources else None
        m = cmd_run(cfg, base, only, a.limit, a.num_proc)
        print(json.dumps(m["totals"], indent=2))


if __name__ == "__main__":
    main()
