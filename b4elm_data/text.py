"""Text normalization, segmentation and tokenization shared by every stage."""
from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_PARA_SPLIT = re.compile(r"\n[ \t]*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WS = re.compile(r"\s+")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her
    here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves unto upon thee
    thou thy thine ye hath doth shall shalt hast art also let us may might must one
    """.split()
)


def normalize(text: str) -> str:
    """NFKC, ASCII quotes and dashes, lowercase."""
    return unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def decontam_tokens(text: str) -> list[str]:
    """Words used for n-gram matching: lowercased, no pure numbers (verse numbers, figure numbers)."""
    return [w for w in words(text) if not w.isdigit()]


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg license header and footer when present."""
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]
    return text
