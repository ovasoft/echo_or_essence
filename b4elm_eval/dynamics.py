"""Kinds 16 and 17: acquisition and forgetting over checkpoints, and persistence after tuning."""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

_STEP = re.compile(r"(\d+)(?!.*\d)")


def checkpoint_step(path: str) -> int:
    m = _STEP.search(Path(path).name)
    if not m:
        raise ValueError(f"Cannot read a step number from checkpoint name {path!r}")
    return int(m.group(1))


def list_checkpoints(pattern: str) -> list[str]:
    paths = [p for p in glob.glob(pattern) if Path(p).is_dir()]
    if not paths:
        raise FileNotFoundError(f"No checkpoint directories match {pattern!r}")
    return sorted(paths, key=checkpoint_step)


def flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def item_trajectories(results: list[tuple[int, dict]]) -> dict:
    """Per item: step it is first stably correct, and correct->wrong transitions (forgetting events)."""
    steps = [s for s, _ in results]
    series: dict[str, list] = {}
    kinds: dict[str, str] = {}
    for col, (_, res) in enumerate(results):
        for r in res["items"]:
            if "correct" not in r:
                continue
            series.setdefault(r["id"], [None] * len(results))[col] = bool(r["correct"])
            kinds[r["id"]] = r["kind"]
    per_item = {}
    for iid, xs in series.items():
        acq = None
        for i in range(len(xs)):
            if xs[i] and all(x is not False for x in xs[i:]):
                acq = steps[i]
                break
        forgets = sum(1 for a, b in zip(xs, xs[1:]) if a is True and b is False)
        per_item[iid] = {"kind": kinds[iid], "acquired_at": acq, "forgetting_events": forgets,
                         "final": xs[-1]}
    by_kind: dict[str, dict] = {}
    for kind in sorted(set(kinds.values())):
        rows = [v for v in per_item.values() if v["kind"] == kind]
        acq = sorted(v["acquired_at"] for v in rows if v["acquired_at"] is not None)
        by_kind[kind] = {
            "n": len(rows),
            "acquired": len(acq),
            "median_acquisition_step": acq[len(acq) // 2] if acq else None,
            "forgetting_events_per_item": sum(v["forgetting_events"] for v in rows) / len(rows),
            "ever_forgotten": sum(1 for v in rows if v["forgetting_events"]) / len(rows),
        }
    return {"steps": steps, "by_kind": by_kind, "per_item": per_item}


def write_trajectory(results: list[tuple[int, dict]], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"step": s, **flatten(r["metrics"])} for s, r in results]
    cols = ["step"] + sorted({k for r in rows for k in r if k != "step"})
    with open(out_dir / "metrics_by_step.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    traj = item_trajectories(results)
    (out_dir / "item_trajectories.json").write_text(json.dumps(traj, indent=2))
    return traj


def compare(before: dict, after: dict) -> dict:
    """Kind 17: metric deltas and per-item flips between two runs on the same items."""
    fb, fa = flatten(before["metrics"]), flatten(after["metrics"])
    deltas = {k: {"before": fb[k], "after": fa[k], "delta": fa[k] - fb[k]} for k in sorted(fb.keys() & fa.keys())}
    ib = {r["id"]: r for r in before["items"] if "correct" in r}
    ia = {r["id"]: r for r in after["items"] if "correct" in r}
    flips: dict[str, dict] = {}
    for iid in ib.keys() & ia.keys():
        k = ib[iid]["kind"]
        d = flips.setdefault(k, {"n": 0, "lost": 0, "gained": 0})
        d["n"] += 1
        d["lost"] += int(ib[iid]["correct"] and not ia[iid]["correct"])
        d["gained"] += int(ia[iid]["correct"] and not ib[iid]["correct"])
    return {"metric_deltas": deltas, "item_flips_by_kind": flips}
