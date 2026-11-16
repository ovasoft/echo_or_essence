import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from b4elm_data import decontam as D
from b4elm_data.pipeline import cmd_build_index, cmd_measure, cmd_run, load_config
from b4elm_data.scoring import score_text
from b4elm_data.sources import docs_libretexts, docs_openstax

BIBLE = """1:1 In the beginning God created the heaven and the earth.
1:2 And the earth was without form, and void; and darkness was upon the face of the deep.
20:13 Thou shalt not kill. 20:14 Thou shalt not commit adultery. 20:15 Thou shalt not steal.
5:44 But I say unto you, Love your enemies, bless them that curse you, do good to them that hate you,
and pray for them which despitefully use you, and persecute you.
"""

STOIC = """*** START OF THE PROJECT GUTENBERG EBOOK ENCHIRIDION ***
Of things some are in our power, and others are not. In our power are opinion, movement towards
a thing, desire, aversion, turning from a thing; and in a word, whatever are our own acts.
*** END OF THE PROJECT GUTENBERG EBOOK ENCHIRIDION ***
Project Gutenberg license text that must not enter the index.
"""

PHYSICS = ("Newton's second law states that the net force on a body equals its mass times its acceleration. "
           "For a constant mass this gives F = ma, and the acceleration points along the net force.\n\n"
           "Kinetic energy is one half of mass times velocity squared. Work done by a net force equals "
           "the change in kinetic energy, which is the work-energy theorem used throughout mechanics.")

KEEP_HOMOGRAPHS = ("The Church-Turing thesis states that every effectively calculable function is computable "
                   "by a Turing machine. Lord Kelvin defined absolute temperature, and sin x is periodic with "
                   "period two pi, a fact used in every course on Fourier series and signal processing.")


def _row(i, text, title="Mechanics", subject="physics", created=None):
    r = {"id": f"d{i}", "text": text, "title": title, "subject": subject}
    if created:
        r["created"] = created
    return r


@pytest.fixture()
def project(tmp_path: Path):
    (tmp_path / "data/bible").mkdir(parents=True)
    (tmp_path / "data/stoic").mkdir(parents=True)
    (tmp_path / "data/bible/kjv.txt").write_text(BIBLE)
    (tmp_path / "data/stoic/enchiridion.txt").write_text(STOIC)
    rows = [
        _row(1, PHYSICS),
        _row(2, PHYSICS.replace("body", "rigid body") + "\n\nAs one author put it, in our power are opinion, movement towards a thing, "
                "desire, aversion, and these matter for the analysis of motion."),
        _row(3, PHYSICS.replace("mechanics", "dynamics") + "\n\nCompare the famous line in John 3:16 with this law."),
        _row(4, KEEP_HOMOGRAPHS),
        _row(5, PHYSICS.replace("theorem", "principle") + "\n\nThe church in the town square was rebuilt "
                "after the fire and its bishop held a great feast."),
        _row(6, PHYSICS.replace("Kinetic", "Translational kinetic") + "\n\nIt is wrong to cheat on a lab "
                "report. Honesty and integrity show moral character and compassion for others."),
        _row(7, PHYSICS.replace("Newton's", "The"), title="Chapter 4 > Business Ethics"),
        _row(8, PHYSICS),  # exact duplicate of d1
        _row(9, "Certainly! Here is a summary of the chapter. " + PHYSICS.replace("body", "object")),
        _row(10, PHYSICS.replace("squared", "squared again"), created="2024-03-01"),
        _row(11, PHYSICS.replace("equals", "is"), subject=None),
        _row(12, "Short note on units."),
    ]
    with gzip.open(tmp_path / "base.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cfg = {
        "batch_size": 4,
        "output": {"dir": "out", "docs_per_shard": 2, "audit_samples_per_reason": 5, "seed": 0},
        "decontam": {"index_path": "work/index.npy", "ngram_sizes": [8], "min_content_words": 2,
                     "min_hits": 1, "references": {"bible": ["data/bible/*.txt"], "stoic": ["data/stoic/*.txt"]}},
        "filters": {"religious_max_hits": 0, "moral_max_hits": 1, "normative_ratio_max": 0.2,
                    "min_doc_chars": 200, "doc_block_scan_chars": 5000, "extra_title_block": []},
        "sources": {"local": {"kind": "passthrough", "path": "base.jsonl.gz", "max_created": "2022-11-30"}},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def _outputs(out_dir: Path) -> dict[str, dict]:
    docs = {}
    for shard in sorted((out_dir / "local").glob("shard_*.jsonl.gz")):
        with gzip.open(shard, "rt") as f:
            for line in f:
                d = json.loads(line)
                docs[d["id"]] = d
    return docs


def test_end_to_end(project: Path):
    cfg, base = load_config(str(project / "config.yaml"))
    cmd_build_index(cfg, base)
    m = cmd_run(cfg, base, None, None, num_proc=1)
    s = m["sources"]["local"]
    docs = _outputs(project / "out")

    assert s["docs_seen"] == 12
    assert s["doc_drops"] == {"subject": 1, "date": 1, "title": 1, "doc_block": 0, "llm_marker": 1,
                              "short": 1, "duplicate": 1}
    assert s["par_drops"] == {"contam_ref": 1, "contam_ngram": 1, "religious": 1, "normative": 1}
    assert set(docs) == {"d1", "d2", "d3", "d4", "d5", "d6"}
    # Dropped paragraphs are gone, the rest of each document survives.
    assert "in our power" not in docs["d2"]["text"].lower()
    assert "John 3:16" not in docs["d3"]["text"]
    assert "church in the town" not in docs["d5"]["text"]
    assert "Honesty" not in docs["d6"]["text"]
    assert "Newton's second law" in docs["d2"]["text"]
    # STEM homographs survive every filter.
    assert docs["d4"]["text"] == KEEP_HOMOGRAPHS
    # Manifest and audit trail are written.
    assert (project / "out/manifest.json").exists()
    audit = [json.loads(l) for l in (project / "out/audit/dropped_samples.jsonl").read_text().splitlines()]
    assert {a["reason"] for a in audit} >= {"contam_ngram", "contam_ref", "religious", "normative", "title"}


def test_index_excludes_gutenberg_boilerplate(project: Path):
    cfg, base = load_config(str(project / "config.yaml"))
    cmd_build_index(cfg, base)
    idx = D.Index(project / "work/index.npy", [8])
    assert idx.hits("in our power are opinion movement towards a thing desire aversion") > 0
    assert idx.hits("Project Gutenberg license text that must not enter the index") == 0
    assert idx.hits(PHYSICS) == 0


def test_ngram_hashes_deterministic_and_stopword_windows_skipped():
    toks = "the of and to in that it is was he".split() + "love your enemies bless them".split()
    wh = D.WordHasher()(toks)
    a, b = D.ngram_hashes(wh, 5), D.ngram_hashes(D.WordHasher()(toks), 5)
    assert np.array_equal(a, b) and len(a) == len(toks) - 4
    mask = D.content_mask(toks, 5, 2)
    assert not mask[0] and mask[-1]


def test_scoring_proxies():
    assert score_text("Thou shalt not steal.").normative_sentences == 1
    assert score_text("The function must be continuous on the interval.").normative_sentences == 0
    assert score_text("The Church-Turing thesis and Lord Kelvin.").religious == 0
    assert score_text("See Matthew 5:44 for context.").refs == 1


def test_libretexts_subject_resolution():
    cfg = {"keep_libraries": {"chem": None, "human": "/Bookshelves/(Art|History)"}}
    rows = [
        {"metadata": {"url": "https://chem.libretexts.org/Bookshelves/General_Chemistry/x"}, "text": "t"},
        {"metadata": {"url": "https://human.libretexts.org/Bookshelves/History/y"}, "text": "t"},
        {"metadata": {"url": "https://human.libretexts.org/Bookshelves/Philosophy/z"}, "text": "t"},
        {"metadata": {"url": "https://socialsci.libretexts.org/Bookshelves/Sociology/w"}, "text": "t"},
    ]
    assert [d.subject for d in docs_libretexts(rows, "lt", cfg)] == [
        "libretexts:chem", "libretexts:human", None, None]


def test_openstax_walk_and_filter():
    row = {"book_title": "University Physics", "chapters": [
        {"title": "Units", "abstract": "None", "sections": [{"title": "Intro", "paragraph": "p1"}],
         "chapters": [{"title": "Sub", "sections": [{"title": "Deep", "paragraph": "p2"}]}]}]}
    docs = list(docs_openstax([row, {"book_title": "Business Ethics", "chapters": [
        {"title": "c", "sections": [{"title": "s", "paragraph": "x"}]}]}],
        "openstax", {"allow_title": "physics", "deny_title": "ethic"}))
    assert [d.title for d in docs[:2]] == ["University Physics > Units > Intro",
                                           "University Physics > Units > Sub > Deep"]
    assert docs[0].subject == "openstax:University Physics" and docs[2].subject is None


def test_measure_tolerance(project: Path):
    reps = cmd_measure([f"bible={project}/data/bible/*.txt", f"stoic={project}/data/stoic/*.txt"],
                       project, 0.10)
    assert reps[0]["normative_ratio"] > 0 and "matched" in reps[2]
