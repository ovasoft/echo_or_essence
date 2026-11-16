"""Probe items and their JSONL format.

Every probe item is either
  * a forced choice: `prompt` + `choices`, `target` = index of the correct choice (or None
    when the item measures preference without a single correct answer), or
  * a generation item: `prompt` + `continuation` (memorization).

`kind` is one of KINDS. `group` ties items that test the same underlying fact or principle
(consistency, knowing-doing, self-report and latent-vs-expressed all join on it).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

KINDS = [
    # A. storage and access
    "memorization", "extraction", "cloze",
    # B. manipulation
    "classification", "comparison", "forward", "reverse", "composition",
    # C. application beyond the text
    "application", "knowledge", "inductive", "self_report",
    # D. interaction with other knowledge
    "conflict", "propagation",
]


@dataclass
class Item:
    id: str
    kind: str
    prompt: str
    choices: list[str] = field(default_factory=list)
    target: int | None = None
    continuation: str | None = None
    group: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown item kind {self.kind!r} in {self.id}")
        if self.kind == "memorization":
            if not self.continuation:
                raise ValueError(f"Memorization item {self.id} needs a continuation")
        else:
            if len(self.choices) < 2:
                raise ValueError(f"Item {self.id} needs at least two choices")
            if self.target is not None and not 0 <= self.target < len(self.choices):
                raise ValueError(f"Item {self.id} target {self.target} out of range")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


def save_items(items: Iterable[Item], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_items(path: Path) -> list[Item]:
    """Load one JSONL file or every *.jsonl file in a directory."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    items, seen = [], set()
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    it = Item(**json.loads(line))
                except (TypeError, ValueError) as e:
                    raise ValueError(f"{fp}:{n}: {e}") from e
                if it.id in seen:
                    raise ValueError(f"{fp}:{n}: duplicate item id {it.id}")
                seen.add(it.id)
                items.append(it)
    return items
