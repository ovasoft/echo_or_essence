"""Paragraph- and corpus-level normative scoring.

The same scorer filters the base corpus and measures normative density of the Bible and
Stoic corpora, so the matching criterion in the protocol uses one consistent instrument.
It is a lexical proxy: validate it against a trained classifier before reporting numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import lexicons as L
from .text import normalize, paragraphs, sentences, words


@dataclass
class Score:
    words: int = 0
    sentences: int = 0
    normative_sentences: int = 0
    religious: int = 0
    moral: int = 0
    deontic: int = 0
    refs: int = 0

    @property
    def normative_ratio(self) -> float:
        return self.normative_sentences / self.sentences if self.sentences else 0.0

    def add(self, other: "Score") -> None:
        for k in ("words", "sentences", "normative_sentences", "religious", "moral", "deontic", "refs"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def per_1k(self, attr: str) -> float:
        return 1000.0 * getattr(self, attr) / self.words if self.words else 0.0


def score_text(text: str) -> Score:
    """Score one paragraph (or any span)."""
    norm = normalize(text)
    s = Score(words=len(words(norm)))
    s.religious = len(L.RELIGIOUS.findall(L.RELIGIOUS_EXEMPT.sub(" ", norm)))
    s.refs = len(L.VERSE_REF.findall(norm)) + len(L.SOURCE_REF.findall(norm))
    for sent in sentences(norm):
        m = len(L.MORAL.findall(sent))
        d = len(L.DEONTIC.findall(sent))
        s.sentences += 1
        s.moral += m
        s.deontic += d
        if m or d:
            s.normative_sentences += 1
    return s


@dataclass
class CorpusReport:
    name: str
    files: list[str] = field(default_factory=list)
    total: Score = field(default_factory=Score)

    def as_dict(self) -> dict:
        t = self.total
        return {
            "name": self.name,
            "files": self.files,
            "words": t.words,
            "sentences": t.sentences,
            "normative_sentences": t.normative_sentences,
            "normative_ratio": round(t.normative_ratio, 5),
            "moral_per_1k_words": round(t.per_1k("moral"), 3),
            "deontic_per_1k_words": round(t.per_1k("deontic"), 3),
            "religious_per_1k_words": round(t.per_1k("religious"), 3),
        }


def score_corpus(name: str, texts: list[tuple[str, str]]) -> CorpusReport:
    """texts: (file label, text) pairs."""
    rep = CorpusReport(name=name)
    for label, text in texts:
        rep.files.append(label)
        for par in paragraphs(text):
            rep.total.add(score_text(par))
    return rep
