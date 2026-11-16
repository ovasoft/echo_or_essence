# B4ELM base corpus: filtering and decontamination

Builds the low-ethics textbook base corpus for the *Echo or Essence?* protocol and removes
any overlap with the Bible and Stoic corpora, with a manifest and audit sample for the paper.

## Install

```bash
pip install -r requirements.txt
```

## Steps

```bash
# 1. Put the treatment and comparison texts in place (plain text, one file per work)
#    data/bible/kjv.txt, web.txt, bsb.txt        (eBible.org)
#    data/stoic/*.txt                             (Gutenberg; headers/footers are stripped automatically)

# 2. Build the n-gram index
python -m b4elm_data build-index --config config.yaml

# 3. Check how each source's subject field resolves before trusting the regexes
python -m b4elm_data inspect --config config.yaml --source pressbooks --n 5000

# 4. Pilot run, then full run
python -m b4elm_data run --config config.yaml --limit 2000
python -m b4elm_data run --config config.yaml --num-proc 16

# 5. Normative-density matching of Bible vs Stoic (protocol: within 10%)
python -m b4elm_data measure "bible=data/bible/*.txt" "stoic=data/stoic/*.txt" --tolerance 0.10
```

`--sources libretexts,openstax` runs a subset and overrides `enabled` flags (DOAB and arXiv are
off by default).

## Stages

| Level | Reason code | Rule |
|---|---|---|
| Document | `subject` | Source loader could not place it in an allowed subject |
| Document | `date` | `created` on/after the source's `max_created` (LLM-era text) |
| Document | `title` | Book > chapter > section path or URL matches the ethics/religion blocklist |
| Document | `doc_block` | Source blocklist, e.g. AI alignment and fairness papers on arXiv |
| Document | `llm_marker` | Chat-model boilerplate |
| Paragraph | `contam_ref` | Bible chapter:verse reference, or Epictetus, Marcus Aurelius, Seneca, Stoic, KJV named |
| Paragraph | `contam_ngram` | Any 8-gram (configurable) shared with the Bible/Stoic index |
| Paragraph | `religious` | Religious vocabulary |
| Paragraph | `normative` | More than one moral term, or normative sentences above 20% |
| Document | `short` | Under 200 characters after paragraph removal |
| Document | `duplicate` | Exact duplicate after normalization |

Paragraph removal keeps the rest of the document, so a chemistry chapter loses one quoted verse,
not the chapter.

## Outputs

- `work/base_corpus/<source>/shard_*.jsonl.gz`: `id, source, subject, title, text`
- `work/base_corpus/manifest.json`: per-source counts for every reason, subjects kept, estimated
  tokens (words × 1.3), config hash, and the index's reference files with SHA-256
- `work/base_corpus/audit/dropped_samples.jsonl`: reservoir sample of dropped text per reason;
  read it to tune thresholds and report filter precision in the appendix

## Design notes

- **STEM homographs are excluded from the lexicons** (sine, vice versa, fair coin, duty cycle, data
  integrity, debt covenant, Lord Kelvin, Church-Turing). Tests cover these.
- **Decontamination** hashes each word once (blake2b, 64-bit), builds n-gram hashes with a vectorized
  rolling hash, and does membership by binary search in a memory-mapped sorted array. Windows of
  only stopwords are not indexed, and pure numbers are ignored so verse numbers do not matter.
- **Three translations in the index** catch quotations regardless of which translation a textbook cites.
- **The scorer is shared** between base filtering and Bible/Stoic matching, so both use one instrument.

## Known limitations

- Subject regexes are untested against the real Hub data: field values are undocumented. Run
  `inspect` on each source and adjust `config.yaml` before the full run.
- The meaning of `created` differs by source (paper date vs. crawl date); `max_created` is on only
  for arXiv and DOAB.
- Lexical filters miss paraphrased moral content. Before main runs, validate a sample against a
  trained normative-content classifier, and use the base-only model's constitution as the final check
  (the headroom gate).
- Paraphrased Bible content without shared 8-grams is not caught; add n=6 to `ngram_sizes` for
  stricter matching at some cost in false positives.

## Tests

```bash
python -m pytest -q
```
