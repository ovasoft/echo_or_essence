# Echo or Essence?

Code for studying how language models use the knowledge they learn, with the Bible as the
test text. Two packages:

- `b4elm_data`: builds the low-ethics textbook base corpus and removes any overlap with the
  Bible and Stoic corpora.
- `b4elm_eval`: knowledge-usage probes, one per kind of usage found in the literature.

## Install

```bash
pip install -r requirements.txt
```

## Base corpus (`b4elm_data`)

### Steps

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

### Stages

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

### Outputs

- `work/base_corpus/<source>/shard_*.jsonl.gz`: `id, source, subject, title, text`
- `work/base_corpus/manifest.json`: per-source counts for every reason, subjects kept, estimated
  tokens (words × 1.3), config hash, and the index's reference files with SHA-256
- `work/base_corpus/audit/dropped_samples.jsonl`: reservoir sample of dropped text per reason;
  read it to tune thresholds and report filter precision in the appendix

### Design notes

- **STEM homographs are excluded from the lexicons** (sine, vice versa, fair coin, duty cycle, data
  integrity, debt covenant, Lord Kelvin, Church-Turing). Tests cover these.
- **Decontamination** hashes each word once (blake2b, 64-bit), builds n-gram hashes with a vectorized
  rolling hash, and does membership by binary search in a memory-mapped sorted array. Windows of
  only stopwords are not indexed, and pure numbers are ignored so verse numbers do not matter.
- **Three translations in the index** catch quotations regardless of which translation a textbook cites.
- **The scorer is shared** between base filtering and Bible/Stoic matching, so both use one instrument.

### Known limitations

- Subject regexes are untested against the real Hub data: field values are undocumented. Run
  `inspect` on each source and adjust `config.yaml` before the full run.
- The meaning of `created` differs by source (paper date vs. crawl date); `max_created` is on only
  for arXiv and DOAB.
- Lexical filters miss paraphrased moral content. Before main runs, validate a sample against a
  trained normative-content classifier, and use the base-only model's constitution as the final check
  (the headroom gate).
- Paraphrased Bible content without shared 8-grams is not caught; add n=6 to `ngram_sizes` for
  stricter matching at some cost in false positives.

## Knowledge-usage probes (`b4elm_eval`)

Each kind of knowledge usage named in the literature is a probe. Items are scored once per
model (choice log-probabilities, or greedy decoding for memorization), and every metric is
computed from those scores. Runs on GPU automatically (bfloat16 when supported).

| # | Kind | Source | Items | Metric key |
|---|---|---|---|---|
| 1 | Memorization | Chang et al. 2024; Physics of LMs 3.1 | `memorization` | `memorization.exact`, `.token_accuracy` |
| 2 | Extraction | Physics of LMs 3.1; LAMA | `extraction`, `cloze` | `extraction.accuracy`, `cloze.accuracy` |
| 3 | Consistency | Knowledge Boundary survey; Knowledge Conflicts survey | `forward` + `extraction` per fact | `consistency.known_under_all_prompts` |
| 4 | Latent vs expressed | Inside-Out (Gekhman et al. 2025); CCS | all fact items | `latent_vs_expressed.relative_gap` |
| 5 | Classification and comparison | Physics of LMs 3.2; Grokked Transformers | `classification`, `comparison` | `.accuracy` |
| 6 | Inverse search (reversal) | Reversal Curse; Physics of LMs 3.2 | `forward`, `reverse` | `reversal.gap` |
| 7 | Composition | Compositionality gap; Two-Hop Curse | `composition` | `composition.compositionality_gap` |
| 8 | Declarative to procedural | Taken out of context | `application` | `application.by_cell`, `.by_expected` |
| 9 | Inductive inference | Connecting the Dots | `inductive` | `inductive.accuracy` |
| 10 | Procedural to declarative | Tell me about yourself | `self_report` + `application` | `self_report.agreement_with_behaviour` |
| 11 | Knowing-doing gap | LLMs are Greedy Agents | `knowledge` + `application` | `knowing_doing.wrong_action_given_known` |
| 12 | Procedural knowledge | Ruis et al. | not implemented | influence functions cost more than the training runs |
| 13 | Context-memory conflict | Knowledge Conflicts survey | `conflict` | `conflict.follows_context`, `.follows_memory` |
| 14 | Propagation | RippleEdits | `propagation` | `propagation.accuracy` |
| 15 | Metacognition | Knowledge Boundary survey; EMNLP 2025 | all fact items | `metacognition.ece`, `.auroc` |
| 16 | Acquisition and forgetting | Chang et al. 2024 | any | `trajectory` command |
| 17 | Persistence after tuning | Gekhman et al. 2024 | any | `compare` command |

### Items

Three sources:

- **Facts spec** (YAML): relations, composite relations and facts, with query templates.
  Generates kinds 2, 3, 5, 6, 7, 13 and 14 with exact ground truth. `examples/genesis_facts.yaml`
  covers the Genesis genealogies; the synthetic worlds use the same format.
- **Text**: memorization windows from training text, cloze items from held-out text such as a
  Bible translation excluded from training (one verse per line).
- **Authored JSONL**: `application`, `knowledge`, `inductive` and `self_report` items. These need
  human-written scenarios. `examples/authored_items.example.jsonl` shows the schema only; the
  real set is the protocol's pre-registered evaluation set. Join keys: `group` ties a principle's
  knowledge, application and self-report items; `meta.expected` maps a tradition to its expected
  choice; `meta.cell` names the divergent cell; `meta.maps_to` maps self-report choices to
  application choices.

```bash
python -m b4elm_eval make-items --facts examples/genesis_facts.yaml \
    --heldin data/bible/kjv.txt --heldout data/heldout/bsb.txt \
    --authored eval/authored_items.jsonl --out items/
```

### Running

```bash
# One model, all probes
python -m b4elm_eval run --model runs/bible_125m/final --items items/ --out results/bible_125m.json

# Kind 16: every probe over checkpoints (dirs named with a step number, e.g. step_2000)
python -m b4elm_eval trajectory --checkpoints 'runs/bible_125m/step_*' --items items/ --out traj/bible_125m

# Kind 17: before vs after format-only instruction tuning
python -m b4elm_eval compare results/bible_125m.json results/bible_125m_sft.json
```

`trajectory` writes `metrics_by_step.csv` (every metric at every checkpoint) and
`item_trajectories.json` (per item: step first stably correct, number of forgetting events),
and caches each checkpoint's results so interrupted runs resume.

Options: `--kinds` for a subset, `--norm mean` for per-token choice scoring, `--no-latent` to
skip the hidden-state probe, `--layers` to choose its layers.

### Limitations

- The authored example items are placeholders, not the evaluation set.
- On real Bible text, "reverse" questions are not a clean reversal test, because the text also
  says "Y the son of X". Check the `seen` flags in the facts spec against the training text.
  The synthetic worlds give the controlled version.
- The Stoic corpus has few relational facts, so kinds 2 to 7 compare Bible and synthetic arms;
  kinds 8 to 11 compare Bible and Stoic.
- Kind 12 (procedural knowledge via influence functions) is deferred.

## Tests

```bash
python -m pytest -q
```

The probe tests train a tiny GPT-2 on a synthetic genealogy on CPU (about 30 seconds) and
check that the probes detect known effects, including the reversal curse.
