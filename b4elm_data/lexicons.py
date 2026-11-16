"""Regex lexicons. All patterns are matched against normalized (lowercased) text.

Patterns are deliberately conservative about STEM homographs: no `sin` (sine), no `vice`
(vice versa), no `fair` (fair coin), no `duty` (duty cycle), no `integrity` (data integrity),
no `covenant` (debt covenant), no bare `lord` (Lord Kelvin), and `church` is exempted in
Church-Turing contexts.
"""
from __future__ import annotations

import re


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


RELIGIOUS = _compile([
    r"\bgods?\b", r"\bgoddess(?:es)?\b", r"\bthe lord\b", r"\bjesus\b", r"\bchrist\b",
    r"\bchristian(?:s|ity)?\b", r"\bbibl(?:e|es|ical)\b", r"\bscriptur(?:e|es|al)\b",
    r"\bgospels?\b", r"\bpray(?:s|ed|ing|er|ers)?\b", r"\bsalvation\b", r"\bsinful\b",
    r"\bsinners?\b", r"\bheaven(?:s|ly)?\b", r"\bhell\b", r"\bdivin(?:e|ity)\b", r"\bholy\b",
    r"\bapostles?\b", r"\bprophets?\b", r"\bpsalms?\b", r"\bmessiah\b", r"\ballah\b",
    r"\b(?:quran|koran)\b", r"\bbuddh(?:a|ism|ist|ists)\b", r"\bhindu(?:s|ism)?\b", r"\btorah\b",
    r"\btalmud\b", r"\brabbis?\b", r"\bpriest(?:s|hood)?\b", r"\bclergy\b", r"\bbishops?\b",
    r"\bpopes?\b", r"\bpapal\b", r"\bmonaster(?:y|ies)\b", r"\btheolog(?:y|ical|ian|ians)\b",
    r"\bresurrection\b", r"\bcrucifi(?:ed|xion)\b", r"\bbaptis(?:m|t|ts|ed)\b", r"\bchurch(?:es)?\b",
    r"\bdeit(?:y|ies)\b", r"\bworship(?:s|ed|ing|pers)?\b", r"\bsacred\b", r"\bsermons?\b",
    r"\bmosques?\b", r"\bsynagogues?\b", r"\bislam(?:ic)?\b", r"\bmuslims?\b", r"\bjudaism\b",
    r"\bcatholic(?:s|ism)?\b", r"\bprotestant(?:s|ism)?\b", r"\bevangel\w*", r"\bmissionar(?:y|ies)\b",
    r"\bsaviou?r\b", r"\b(?:old|new) testament\b",
])

# Spans removed before RELIGIOUS is applied (STEM uses of otherwise religious words).
RELIGIOUS_EXEMPT = _compile([
    r"church[\s\-]+turing", r"alonzo church", r"church'?s (?:thesis|theorem|lambda|encoding|numerals?)",
    r"god'?s (?:algorithm|number)", r"god particle",
])

MORAL = _compile([
    r"\bmoral(?:s|ly|ity|ities|ist|istic)?\b", r"\bimmoral(?:ity)?\b", r"\bamoral\b",
    r"\bethic(?:s|al|ally)?\b", r"\bunethical(?:ly)?\b", r"\bvirtu(?:e|es|ous|ously)\b",
    r"\brighteous(?:ness)?\b", r"\bwicked(?:ness)?\b", r"\bevils?\b", r"\bconscience\b",
    r"\bcompassion(?:ate)?\b", r"\bmerc(?:y|iful|iless)\b", r"\bforgiv(?:e|es|en|eness|ing)\b",
    r"\bkindness\b", r"\bcruel(?:ty|ly)?\b", r"\bgreed(?:y)?\b", r"\bselfish(?:ness|ly)?\b",
    r"\bdignity\b", r"\b(?:in)?justice\b", r"\bunjust(?:ly)?\b", r"\bhumility\b",
    r"\b(?:dis)?honest(?:y|ly)?\b", r"\bshame(?:ful)?\b", r"\bthe right thing\b", r"\bthe common good\b",
    r"\bhuman rights\b",
])

DEONTIC = _compile([
    r"\bought\s+(?:not\s+)?to\b", r"\bone\s+(?:should|must|ought)\b", r"\bthou\s+shalt\b",
    r"\bshalt\s+not\b", r"\bye\s+shall\b", r"\bthou\s+shouldst\b",
    r"\bit\s+is\s+(?:morally\s+)?(?:wrong|right|good|evil|wicked|shameful|virtuous|noble)\s+to\b",
    r"\blet\s+us\s+(?:love|do|be|not)\b",
    r"\b(?:you|we)\s+(?:should|must)\s+(?:never|always)\s+(?:lie|steal|harm|hurt|help|love|forgive|respect|be)\b",
    r"\bwoe\s+(?:unto|to)\b", r"\bblessed\s+(?:is|are)\b",
])

# Bible chapter:verse references. Ambiguous short abbreviations (Ex, Col) are excluded.
_BOOKS = [
    "genesis", "gen", "exodus", "exod", "leviticus", "lev", "numbers", "num", "deuteronomy", "deut",
    "joshua", "josh", "judges", "judg", "ruth", "samuel", "sam", "kings", "kgs", "chronicles", "chron",
    "chr", "ezra", "nehemiah", "neh", "esther", "esth", "job", "psalms?", "ps", "pss", "proverbs", "prov",
    "ecclesiastes", "eccl", "eccles", "song of solomon", "song of songs", "isaiah", "isa", "jeremiah", "jer",
    "lamentations", "lam", "ezekiel", "ezek", "daniel", "dan", "hosea", "hos", "joel", "amos", "obadiah",
    "obad", "jonah", "micah", "mic", "nahum", "nah", "habakkuk", "hab", "zephaniah", "zeph", "haggai", "hag",
    "zechariah", "zech", "malachi", "mal", "matthew", "matt", "mt", "mark", "mk", "luke", "lk", "john",
    "jn", "acts", "romans", "rom", "corinthians", "cor", "galatians", "gal", "ephesians", "eph",
    "philippians", "phil", "colossians", "thessalonians", "thess", "timothy", "tim", "titus", "philemon",
    "phlm", "hebrews", "heb", "james", "jas", "peter", "pet", "jude", "revelation", "rev",
]
VERSE_REF = re.compile(
    r"\b(?:[1-3]\s?|i{1,3}\s)?(?:" + "|".join(_BOOKS) + r")\.?\s+\d{1,3}\s?:\s?\d{1,3}\b", re.I
)

# Direct references to the treatment or comparison sources.
SOURCE_REF = _compile([
    r"\bepictetus\b", r"\bmarcus aurelius\b", r"\bseneca\b", r"\bstoic(?:s|ism)?\b", r"\benchiridion\b",
    r"\bzeno of citium\b", r"\bchrysippus\b", r"\bking james\b",
])

LLM_MARKERS = _compile([
    r"as an ai(?: language)? model", r"\bi(?:'m| am) (?:just )?an ai\b", r"\bi cannot (?:fulfill|comply)\b",
    r"\bcertainly! here(?:'s| is)\b", r"\bi hope this helps\b",
    r"\bas of my (?:last )?(?:knowledge|training) (?:cutoff|update)\b",
])

# Applied to a document's title path (book > chapter > section, or URL path).
TITLE_BLOCK_DEFAULT = [
    r"ethic", r"moral", r"\bintegrity\b", r"fraud", r"professional responsibilit", r"code of conduct",
    r"social responsibilit", r"slaver", r"enslave", r"genocide", r"holocaust", r"coloni", r"imperialis",
    r"civil rights", r"human rights", r"abolition", r"religio", r"reformation", r"crusade", r"church",
    r"theolog", r"philosoph", r"justice", r"war crimes", r"atrocit", r"apartheid", r"segregation",
    r"discrimination", r"\bequity\b", r"equality", r"diversity", r"inclusion", r"racis[mt]", r"feminis",
    r"bible", r"christian", r"islam", r"buddh", r"hindu", r"judais", r"spiritual", r"\bfaith\b",
    r"preface", r"about openstax",
]
