"""Item generators.

Three sources:
  * a facts spec (YAML): entities linked by relations, with query templates. Produces the
    storage, manipulation and interaction kinds with exact ground truth. Works for Bible
    genealogies, the synthetic worlds, or any other relational content.
  * plain text: memorization items from held-in text, cloze items from held-out text
    (e.g. a Bible translation excluded from training).
  * authored JSONL: application, knowledge, inductive and self-report items, which need
    human-written scenarios and are pre-registered with the protocol.

Facts spec:

    relations:
      father_of:
        train_direction: head          # training text states it as "<head> ... <tail>"
        statement: "{head} begat {tail}."   # used to write in-context edits
        templates:                     # one placeholder each; the answer is the other entity
          - {text: "{head} begat", seen: true}
          - {text: "The son of {head} was named", seen: false}
          - {text: "The father of {tail} was", seen: false}
        ancestor:                      # optional: transitive closure for classification/comparison
          classification: "Question: Was {a} an ancestor of {b}? Answer:"
          comparison: "Question: Who was born earlier, {a} or {b}? Answer:"
    composites:
      grandfather_of:
        path: [father_of, father_of]
        templates: ["The grandfather of {tail} was"]
    facts:
      - [Adam, father_of, Seth]
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from .items import Item, load_items

_WORD = re.compile(r"\S+")
_CONTENT = re.compile(r"^[A-Za-z][A-Za-z'\-]{2,}$")
_STOP = frozenset(
    "the and that with which from unto upon this these those they them their there were have hath "
    "shall will your thou thee thine ye for not but all into when what who whom are was his her him".split()
)
AUTHORED_KINDS = {"application", "knowledge", "inductive", "self_report"}


def _placeholder(t: str) -> str:
    has_h, has_t = "{head}" in t, "{tail}" in t
    if has_h == has_t:
        raise ValueError(f"Template must contain exactly one of {{head}} or {{tail}}: {t!r}")
    return "head" if has_h else "tail"


def _tmpl(t) -> dict:
    return t if isinstance(t, dict) else {"text": t, "seen": False}


class FactWorld:
    def __init__(self, spec: dict, n_distractors: int = 4, seed: int = 0) -> None:
        self.spec, self.k, self.seed = spec, n_distractors, seed
        self.relations: dict = spec.get("relations") or {}
        self.composites: dict = spec.get("composites") or {}
        self.facts: list[tuple[str, str, str]] = [tuple(f) for f in spec.get("facts") or []]
        for h, r, t in self.facts:
            if r not in self.relations:
                raise ValueError(f"Fact uses undefined relation {r!r}: {(h, r, t)}")
        self.entities = sorted(spec.get("entities") or {e for h, _, t in self.facts for e in (h, t)})
        if len(self.entities) < self.k + 2:
            raise ValueError(f"Need at least {self.k + 2} entities for {self.k} distractors")
        self.by_rel = defaultdict(list)
        # every valid answer for a query, so no correct answer is ever used as a distractor
        self.valid: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for h, r, t in self.facts:
            self.by_rel[r].append((h, t))
            self.valid[(r, "head", h)].add(t)
            self.valid[(r, "tail", t)].add(h)
        for comp, cspec in self.composites.items():
            r1, r2 = cspec["path"]
            for a, _, c in self._chains(r1, r2):
                self.valid[(comp, "head", a)].add(c)
                self.valid[(comp, "tail", c)].add(a)

    # -------------------------------------------------------------- helpers
    def _rng(self, *key) -> random.Random:
        return random.Random(f"{self.seed}|" + "|".join(map(str, key)))

    def _choices(self, answer: str, exclude: set, key) -> tuple[list[str], int]:
        rng = self._rng("choices", key)
        pool = [e for e in self.entities if e != answer and e not in exclude]
        opts = rng.sample(pool, min(self.k, len(pool))) + [answer]
        rng.shuffle(opts)
        return [" " + o for o in opts], opts.index(answer)

    def _query(self, tmpl: str, head: str, tail: str) -> tuple[str, str]:
        side = _placeholder(tmpl)
        return (tmpl.format(head=head), tail) if side == "head" else (tmpl.format(tail=tail), head)

    # -------------------------------------------------------------- relational kinds
    def relation_items(self) -> list[Item]:
        items = []
        for rel, rspec in self.relations.items():
            train_dir = rspec.get("train_direction", "head")
            for i, (h, t) in enumerate(self.by_rel[rel]):
                fid = f"{rel}:{h}:{t}"
                choices_by_answer = {}
                for j, raw in enumerate(rspec.get("templates") or []):
                    tm = _tmpl(raw)
                    prompt, answer = self._query(tm["text"], h, t)
                    forward = _placeholder(tm["text"]) == train_dir
                    if forward:
                        kind = "forward" if tm.get("seen") else "extraction"
                    else:
                        kind = "reverse"
                    if answer not in choices_by_answer:
                        side = _placeholder(tm["text"])
                        query_ent = h if side == "head" else t
                        exclude = {query_ent} | self.valid[(rel, side, query_ent)]
                        choices_by_answer[answer] = self._choices(answer, exclude, (fid, answer))
                    choices, target = choices_by_answer[answer]
                    items.append(Item(f"{kind}:{fid}:{j}", kind, prompt, choices, target, group=fid,
                                      meta={"relation": rel, "template": j, "seen": bool(tm.get("seen")),
                                            "direction": "forward" if forward else "reverse"}))
        return items

    def _closure(self, rel: str) -> set[tuple[str, str]]:
        edges = defaultdict(set)
        for h, t in self.by_rel[rel]:
            edges[h].add(t)
        out = set()
        for a in list(edges):
            stack, seen = list(edges[a]), set()
            while stack:
                b = stack.pop()
                if b in seen:
                    continue
                seen.add(b)
                out.add((a, b))
                stack.extend(edges.get(b, ()))
        return out

    def order_items(self, max_per_rel: int = 400) -> list[Item]:
        items = []
        for rel, rspec in self.relations.items():
            anc = rspec.get("ancestor")
            if not anc:
                continue
            pairs = sorted(p for p in self._closure(rel) if (p[0], p[1]) not in set(self.by_rel[rel]))
            rng = self._rng("order", rel)
            rng.shuffle(pairs)
            pairs = pairs[: max_per_rel // 2]
            for a, b in pairs:
                g = f"ancestor:{rel}:{a}:{b}"
                if anc.get("classification"):
                    t = anc["classification"]
                    items.append(Item(f"classification:{g}:pos", "classification", t.format(a=a, b=b),
                                      [" Yes", " No"], 0, group=g, meta={"relation": rel}))
                    items.append(Item(f"classification:{g}:neg", "classification", t.format(a=b, b=a),
                                      [" Yes", " No"], 1, group=g, meta={"relation": rel}))
                if anc.get("comparison"):
                    t = anc["comparison"]
                    items.append(Item(f"comparison:{g}:ab", "comparison", t.format(a=a, b=b),
                                      [" " + a, " " + b], 0, group=g, meta={"relation": rel}))
                    items.append(Item(f"comparison:{g}:ba", "comparison", t.format(a=b, b=a),
                                      [" " + b, " " + a], 1, group=g, meta={"relation": rel}))
        return items

    def _chains(self, r1: str, r2: str):
        nxt = defaultdict(list)
        for h, t in self.by_rel[r2]:
            nxt[h].append(t)
        for a, b in self.by_rel[r1]:
            for c in nxt.get(b, ()):
                yield a, b, c

    def composition_items(self) -> list[Item]:
        items = []
        for comp, cspec in self.composites.items():
            r1, r2 = cspec["path"]
            for a, b, c in self._chains(r1, r2):
                g = f"{comp}:{a}:{c}"
                hop1, hop2 = f"{r1}:{a}:{b}", f"{r2}:{b}:{c}"
                for j, raw in enumerate(cspec.get("templates") or []):
                    text = _tmpl(raw)["text"]
                    prompt, answer = self._query(text, a, c)
                    side = _placeholder(text)
                    query_ent = a if side == "head" else c
                    exclude = {query_ent, b} | self.valid[(comp, side, query_ent)]
                    choices, target = self._choices(answer, exclude, (g, answer))
                    items.append(Item(f"composition:{g}:{j}", "composition", prompt, choices, target, group=g,
                                      meta={"composite": comp, "hops": [hop1, hop2], "bridge": b}))
        return items

    # -------------------------------------------------------------- interaction kinds
    def conflict_items(self, max_items: int = 300) -> list[Item]:
        """Counterfactual statement in context, then a forward query about the same fact."""
        items = []
        for rel, rspec in self.relations.items():
            stmt = rspec.get("statement")
            seen_fwd = [_tmpl(t)["text"] for t in rspec.get("templates") or []
                        if _placeholder(_tmpl(t)["text"]) == rspec.get("train_direction", "head")]
            if not stmt or not seen_fwd:
                continue
            for h, t in self.by_rel[rel][:max_items]:
                rng = self._rng("conflict", rel, h, t)
                siblings = self.valid[(rel, "head", h)]
                alt = rng.choice([e for e in self.entities if e != h and e not in siblings])
                g = f"{rel}:{h}:{t}"
                prompt = stmt.format(head=h, tail=alt) + " " + seen_fwd[0].format(head=h)
                pool = [e for e in self.entities if e not in (h, alt) and e not in siblings]
                opts = [t, alt] + rng.sample(pool, max(self.k - 1, 0))
                rng.shuffle(opts)
                items.append(Item(f"conflict:{g}", "conflict", prompt, [" " + o for o in opts], None, group=g,
                                  meta={"memory": opts.index(t), "context": opts.index(alt), "relation": rel}))
        return items

    def propagation_items(self, max_items: int = 300) -> list[Item]:
        """Edit the first hop in context; does the composite answer follow the edit?"""
        items = []
        for comp, cspec in self.composites.items():
            r1, r2 = cspec["path"]
            stmt = (self.relations[r2].get("statement") or "")
            if not stmt:
                continue
            for a, b, c in list(self._chains(r1, r2))[:max_items]:
                rng = self._rng("propagation", comp, a, c)
                # edit: c's parent becomes alt_b, whose parent is a known alt_a
                cands = [(x, y) for x, y in self.by_rel[r1] if y not in (b, c) and x not in (a, b, c)]
                if not cands:
                    continue
                alt_a, alt_b = rng.choice(cands)
                g = f"{comp}:{a}:{c}"
                for j, raw in enumerate(cspec.get("templates") or []):
                    text = _tmpl(raw)["text"]
                    if _placeholder(text) != "tail":
                        continue
                    prompt = stmt.format(head=alt_b, tail=c) + " " + text.format(tail=c)
                    pool = [e for e in self.entities if e not in (a, alt_a, alt_b, b, c)]
                    opts = [a, alt_a] + rng.sample(pool, max(self.k - 1, 0))
                    rng.shuffle(opts)
                    items.append(Item(f"propagation:{g}:{j}", "propagation", prompt, [" " + o for o in opts],
                                      opts.index(alt_a), group=g,
                                      meta={"old": opts.index(a), "edited_link": [alt_b, c], "composite": comp}))
        return items

    def all_items(self) -> list[Item]:
        return (self.relation_items() + self.order_items() + self.composition_items()
                + self.conflict_items() + self.propagation_items())


# ------------------------------------------------------------------ text-based kinds

def memorization_items(texts: list[tuple[str, str]], prefix_words: int = 24, cont_words: int = 12,
                       n: int = 500, seed: int = 0) -> list[Item]:
    """texts: (label, text). Samples windows of held-in text."""
    rng = random.Random(seed)
    windows = []
    for label, text in texts:
        words = _WORD.findall(text)
        span = prefix_words + cont_words
        for start in range(0, max(len(words) - span, 0), span):
            windows.append((label, start, words[start:start + span]))
    rng.shuffle(windows)
    out = []
    for label, start, w in windows[:n]:
        out.append(Item(f"memorization:{label}:{start}", "memorization", " ".join(w[:prefix_words]),
                        continuation=" " + " ".join(w[prefix_words:]), meta={"source": label}))
    return out


def cloze_items(texts: list[tuple[str, str]], n: int = 500, n_distractors: int = 4,
                min_words: int = 8, seed: int = 0) -> list[Item]:
    """Mask the last content word of each line (e.g. one verse per line of a held-out translation)."""
    rng = random.Random(seed)
    lines, vocab = [], set()
    for label, text in texts:
        for i, line in enumerate(text.splitlines()):
            words = _WORD.findall(line)
            if len(words) < min_words:
                continue
            for j in range(len(words) - 1, min_words // 2, -1):
                w = words[j].rstrip(".,;:!?\"')")
                if _CONTENT.match(w) and w.lower() not in _STOP:
                    lines.append((label, i, words, j, w))
                    vocab.add(w)
                    break
    vocab = sorted(vocab)
    rng.shuffle(lines)
    out = []
    for label, i, words, j, w in lines[:n]:
        prefix = " ".join(words[:j])
        opts = rng.sample([v for v in vocab if v.lower() != w.lower()], n_distractors) + [w]
        rng.shuffle(opts)
        out.append(Item(f"cloze:{label}:{i}", "cloze", prefix.rstrip(), [" " + o for o in opts], opts.index(w),
                        meta={"source": label}))
    return out


def authored_items(path: Path) -> list[Item]:
    items = load_items(path)
    bad = [it.id for it in items if it.kind not in AUTHORED_KINDS]
    if bad:
        raise ValueError(f"Authored items must be one of {sorted(AUTHORED_KINDS)}; got {bad[:5]}")
    return items


def load_world(path: Path, **kw) -> FactWorld:
    return FactWorld(yaml.safe_load(Path(path).read_text()), **kw)
