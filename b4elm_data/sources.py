"""Source loaders. Each yields Doc records with a resolved subject (None = out of scope).

Subject values in the Hub datasets are not documented; run `inspect` on each source before
the full run and adjust the regexes in config.yaml.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import unquote, urlparse

_NAMESPACE = re.compile(r"^(talk|user|user talk|wikibooks|category|template|help|file|module|mediawiki|portal|special)[ _]?(talk)?:", re.I)


@dataclass
class Doc:
    id: str
    source: str
    subject: str | None
    title: str
    text: str
    created: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _meta(row: dict) -> dict:
    m = row.get("metadata")
    return m if isinstance(m, dict) else {}


def _created(row: dict) -> str | None:
    c = row.get("created")
    return str(c)[:10] if c else None


def _rx(pattern: str | None) -> re.Pattern | None:
    return re.compile(pattern, re.I) if pattern else None


def _allowed(value: str, allow: re.Pattern | None, deny: re.Pattern | None) -> bool:
    if deny is not None and deny.search(value):
        return False
    return allow is None or bool(allow.search(value))


# ---------------------------------------------------------------- raw row iterators

def iter_rows(scfg: dict, base: Path) -> Iterator[dict]:
    if scfg.get("path"):
        p = Path(scfg["path"])
        p = p if p.is_absolute() else base / p
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    from datasets import load_dataset  # lazy: not needed for local runs

    ds = load_dataset(scfg["hf_name"], name=scfg.get("hf_config"), split=scfg.get("split", "train"),
                      streaming=True)
    yield from ds


# ---------------------------------------------------------------- per-kind resolvers

def docs_libretexts(rows: Iterable[dict], name: str, scfg: dict) -> Iterator[Doc]:
    keep = {lib: _rx(rx) for lib, rx in (scfg.get("keep_libraries") or {}).items()}
    for i, row in enumerate(rows):
        m = _meta(row)
        url = m.get("url") or m.get("book_url") or ""
        p = urlparse(url)
        lib = p.netloc.split(".")[0].lower()
        path = unquote(p.path)
        subject = None
        if lib in keep and (keep[lib] is None or keep[lib].search(path)):
            subject = f"libretexts:{lib}"
        yield Doc(str(row.get("id") or f"{name}/{i}"), name, subject,
                  f"{path} | {m.get('title') or ''}", row.get("text") or "", _created(row))


def _walk_openstax(chapters, path):
    for ch in chapters or []:
        cpath = path + [ch.get("title") or ""]
        abstract = ch.get("abstract")
        if abstract and abstract != "None":
            yield cpath + ["Abstract"], abstract
        for sec in ch.get("sections") or []:
            yield cpath + [sec.get("title") or ""], sec.get("paragraph") or ""
        yield from _walk_openstax(ch.get("chapters"), cpath)


def docs_openstax(rows: Iterable[dict], name: str, scfg: dict) -> Iterator[Doc]:
    allow, deny = _rx(scfg.get("allow_title")), _rx(scfg.get("deny_title"))
    for row in rows:
        book = row.get("book_title") or ""
        subject = f"openstax:{book}" if _allowed(book, allow, deny) else None
        for j, (path, text) in enumerate(_walk_openstax(row.get("chapters"), [book])):
            yield Doc(f"{name}/{book}/{j}", name, subject, " > ".join(path), text or "")


def docs_pressbooks(rows: Iterable[dict], name: str, scfg: dict) -> Iterator[Doc]:
    allow, deny = _rx(scfg.get("allow_subject")), _rx(scfg.get("deny_subject"))
    for i, row in enumerate(rows):
        m = _meta(row)
        subj = str(m.get("subject") or "")
        subject = f"pressbooks:{subj}" if subj and _allowed(subj, allow, deny) else None
        yield Doc(str(row.get("id") or f"{name}/{i}"), name, subject,
                  f"{m.get('title') or ''} | {m.get('url') or ''}", row.get("text") or "", _created(row))


def docs_wikibooks(rows: Iterable[dict], name: str, scfg: dict) -> Iterator[Doc]:
    wiki_rx = re.compile(scfg.get("wiki_match", "wikibooks"), re.I)
    allow, deny = _rx(scfg.get("allow_title")), _rx(scfg.get("deny_title"))
    for i, row in enumerate(rows):
        m = _meta(row)
        if not wiki_rx.search(str(m.get("wiki") or "")):
            continue  # other Wikimedia wikis: not part of this source at all
        title = str(m.get("title") or "")
        book = title.split("/")[0]
        ok = not _NAMESPACE.match(title) and _allowed(book, allow, deny)
        yield Doc(str(row.get("id") or f"{name}/{i}"), name, f"wikibooks:{book}" if ok else None,
                  title, row.get("text") or "", _created(row))


def docs_title_filtered(rows: Iterable[dict], name: str, scfg: dict) -> Iterator[Doc]:
    """DOAB and similar: subject from the book title only."""
    allow, deny = _rx(scfg.get("allow_title")), _rx(scfg.get("deny_title"))
    for i, row in enumerate(rows):
        m = _meta(row)
        title = str(m.get("title") or "")
        yield Doc(str(row.get("id") or f"{name}/{i}"), name, name if _allowed(title, allow, deny) else None,
                  title, row.get("text") or "", _created(row))


def docs_passthrough(rows: Iterable[dict], name: str, scfg: dict) -> Iterator[Doc]:
    """arXiv and local JSONL: every row in scope; `subject` field honored when present."""
    for i, row in enumerate(rows):
        m = _meta(row)
        subject = row.get("subject", name)
        yield Doc(str(row.get("id") or f"{name}/{i}"), name, subject,
                  str(row.get("title") or m.get("title") or m.get("url") or ""),
                  row.get("text") or "", _created(row))


KINDS: dict[str, Callable[[Iterable[dict], str, dict], Iterator[Doc]]] = {
    "libretexts": docs_libretexts,
    "openstax": docs_openstax,
    "pressbooks": docs_pressbooks,
    "wikibooks": docs_wikibooks,
    "title_filtered": docs_title_filtered,
    "passthrough": docs_passthrough,
}


def load_docs(name: str, scfg: dict, base: Path) -> Iterator[Doc]:
    kind = scfg["kind"]
    if kind not in KINDS:
        raise ValueError(f"Unknown source kind {kind!r} for {name}")
    yield from KINDS[kind](iter_rows(scfg, base), name, scfg)


# ---------------------------------------------------------------- inspection

def inspect_raw_key(kind: str, row: dict) -> str:
    """The raw field that drives subject resolution, for value-count inspection."""
    m = _meta(row)
    if kind == "libretexts":
        p = urlparse(m.get("url") or m.get("book_url") or "")
        segs = [s for s in unquote(p.path).split("/") if s][:2]
        return p.netloc.split(".")[0] + " /" + "/".join(segs)
    if kind == "openstax":
        return str(row.get("book_title"))
    if kind == "pressbooks":
        return str(m.get("subject"))
    if kind == "wikibooks":
        return f"{m.get('wiki')} | {str(m.get('title') or '').split('/')[0]}"
    return str(m.get("title") or row.get("title") or "")[:80]


def inspect_source(name: str, scfg: dict, base: Path, n: int) -> dict:
    raw, resolved = Counter(), Counter()
    rows = []
    for i, row in enumerate(iter_rows(scfg, base)):
        if i >= n:
            break
        raw[inspect_raw_key(scfg["kind"], row)] += 1
        rows.append(row)
    for d in KINDS[scfg["kind"]](rows, name, scfg):
        resolved[d.subject or "<dropped>"] += 1
    return {"rows_scanned": len(rows), "raw_values": raw.most_common(60),
            "resolved_subjects": resolved.most_common(60)}
