"""Command line: make-items, run, trajectory, compare."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .items import KINDS, load_items, save_items


def _texts(paths: list[str]) -> list[tuple[str, str]]:
    from b4elm_data.decontam import expand, read_reference
    files = expand(paths, Path.cwd())
    return [(Path(f).stem, read_reference(f)) for f in files]


def cmd_make_items(a) -> None:
    from .generators import authored_items, cloze_items, load_world, memorization_items
    out = Path(a.out)
    counts = Counter()
    if a.facts:
        world = load_world(Path(a.facts), n_distractors=a.distractors, seed=a.seed)
        items = world.all_items()
        save_items(items, out / "facts.jsonl")
        counts.update(i.kind for i in items)
    if a.heldin:
        items = memorization_items(_texts(a.heldin), a.prefix_words, a.cont_words, a.n_memorization, a.seed)
        save_items(items, out / "memorization.jsonl")
        counts.update(i.kind for i in items)
    if a.heldout:
        items = cloze_items(_texts(a.heldout), a.n_cloze, a.distractors, seed=a.seed)
        save_items(items, out / "cloze.jsonl")
        counts.update(i.kind for i in items)
    if a.authored:
        items = authored_items(Path(a.authored))
        save_items(items, out / "authored.jsonl")
        counts.update(i.kind for i in items)
    print(json.dumps(dict(sorted(counts.items())), indent=2))


def _scorer(path: str, a):
    from .scorer import Scorer
    return Scorer(path, device=a.device, dtype=a.dtype, batch_size=a.batch_size)


def _select(items, kinds):
    if not kinds:
        return items
    want = set(kinds.split(","))
    bad = want - set(KINDS)
    if bad:
        sys.exit(f"Unknown kinds: {sorted(bad)}")
    return [i for i in items if i.kind in want]


def _run(path: str, items, a) -> dict:
    from .probes import run_suite
    t0 = time.time()
    res = run_suite(_scorer(path, a), items, norm=a.norm, latent=not a.no_latent,
                    latent_layers=[int(x) for x in a.layers.split(",")] if a.layers else None,
                    latent_max_items=a.latent_max_items)
    res["model"] = path
    res["seconds"] = round(time.time() - t0, 1)
    return res


def cmd_run(a) -> None:
    items = _select(load_items(Path(a.items)), a.kinds)
    res = _run(a.model, items, a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["metrics"], indent=2))


def cmd_trajectory(a) -> None:
    from .dynamics import checkpoint_step, list_checkpoints, write_trajectory
    items = _select(load_items(Path(a.items)), a.kinds)
    out = Path(a.out)
    results = []
    for ck in list_checkpoints(a.checkpoints):
        step = checkpoint_step(ck)
        cache = out / f"step_{step}.json"
        if cache.exists() and not a.overwrite:
            res = json.loads(cache.read_text())
        else:
            res = _run(ck, items, a)
            out.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(res))
        results.append((step, res))
        print(f"step {step}: done", file=sys.stderr)
    traj = write_trajectory(results, out)
    print(json.dumps(traj["by_kind"], indent=2))


def cmd_compare(a) -> None:
    from .dynamics import compare
    rep = compare(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["item_flips_by_kind"], indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="b4elm_eval", description="Knowledge-usage probes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-items", help="Generate probe items")
    m.add_argument("--facts", help="Facts spec YAML (relations, composites, facts)")
    m.add_argument("--heldin", nargs="+", help="Training-text files for memorization items")
    m.add_argument("--heldout", nargs="+", help="Held-out text (one verse per line) for cloze items")
    m.add_argument("--authored", help="Authored JSONL: application, knowledge, inductive, self_report")
    m.add_argument("--out", required=True)
    m.add_argument("--distractors", type=int, default=4)
    m.add_argument("--prefix-words", type=int, default=24)
    m.add_argument("--cont-words", type=int, default=12)
    m.add_argument("--n-memorization", type=int, default=500)
    m.add_argument("--n-cloze", type=int, default=500)
    m.add_argument("--seed", type=int, default=0)

    def common(p):
        p.add_argument("--items", required=True, help="Item JSONL file or directory")
        p.add_argument("--kinds", help="Comma-separated subset of item kinds")
        p.add_argument("--device")
        p.add_argument("--dtype", default="auto")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--norm", choices=["sum", "mean"], default="sum",
                       help="Choice score: summed or per-token mean log-probability")
        p.add_argument("--no-latent", action="store_true", help="Skip the hidden-state probe")
        p.add_argument("--layers", help="Comma-separated hidden-state layers for the latent probe")
        p.add_argument("--latent-max-items", type=int, default=400)

    r = sub.add_parser("run", help="Run every probe on one model")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    common(r)

    t = sub.add_parser("trajectory", help="Run probes over checkpoints (kind 16)")
    t.add_argument("--checkpoints", required=True, help="Glob of checkpoint dirs with a step number in the name")
    t.add_argument("--out", required=True)
    t.add_argument("--overwrite", action="store_true")
    common(t)

    c = sub.add_parser("compare", help="Before vs after tuning (kind 17)")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--out")

    a = ap.parse_args(argv)
    {"make-items": cmd_make_items, "run": cmd_run, "trajectory": cmd_trajectory, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
