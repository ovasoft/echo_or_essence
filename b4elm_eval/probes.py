"""The 16 knowledge-usage probes, computed over one model.

Every item is scored once (choice log-probabilities, or greedy decoding for memorization);
each kind's metric is then computed from the per-item records, so kinds that join across
items (consistency, compositionality gap, knowing-doing, self-report, latent vs expressed,
metacognition) reuse the same scores.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

import numpy as np

from .items import Item
from .scorer import Scorer

FACT_KINDS = {"forward", "extraction", "reverse", "composition", "cloze", "classification", "comparison"}


def _pairwise_k(scores: list[float], target: int) -> float:
    """Inside-Out knowledge score: share of (correct, wrong) pairs ranked correctly; ties count half."""
    wrong = [s for i, s in enumerate(scores) if i != target]
    t = scores[target]
    return sum(1.0 if t > w else 0.5 if t == w else 0.0 for w in wrong) / len(wrong)


def score_items(scorer: Scorer, items: list[Item], norm: str = "sum") -> list[dict]:
    recs: list[dict] = []
    mem = [it for it in items if it.kind == "memorization"]
    fc = [it for it in items if it.kind != "memorization"]
    if mem:
        for it, g in zip(mem, scorer.greedy([i.prompt for i in mem], [i.continuation for i in mem])):
            recs.append({"id": it.id, "kind": it.kind, "group": it.group, "correct": g["exact"],
                         "token_acc": g["token_acc"], "generated": g["generated"]})
    if fc:
        scores = scorer.score_choices([i.prompt for i in fc], [i.choices for i in fc])
        for it, sc in zip(fc, scores):
            s = sc.norm(norm)
            probs = sc.probs(norm)
            pred = int(np.argmax(s))
            r = {"id": it.id, "kind": it.kind, "group": it.group, "pred": pred, "target": it.target,
                 "scores": [round(x, 4) for x in s], "conf": max(probs)}
            if it.target is not None:
                r["correct"] = pred == it.target
                r["p_target"] = probs[it.target]
                r["k_external"] = _pairwise_k(s, it.target)
            recs.append(r)
    return recs


def _acc(rs: list[dict]) -> dict:
    rs = [r for r in rs if "correct" in r]
    if not rs:
        return {"n": 0}
    out = {"n": len(rs), "accuracy": mean(float(r["correct"]) for r in rs)}
    ks = [r["k_external"] for r in rs if "k_external" in r]
    if ks:
        out["k_external"] = mean(ks)
    return out


def _cond(num: list[bool]) -> float | None:
    return mean(map(float, num)) if num else None


def compute_metrics(items: list[Item], recs: list[dict]) -> dict:
    by_id = {it.id: it for it in items}
    by_kind = defaultdict(list)
    for r in recs:
        by_kind[r["kind"]].append(r)
    m: dict = {}

    # ---------------------------------------------------- A. storage and access
    if by_kind["memorization"]:
        rs = by_kind["memorization"]
        m["memorization"] = {"n": len(rs), "exact": mean(float(r["correct"]) for r in rs),
                             "token_accuracy": mean(r["token_acc"] for r in rs)}
    for k in ("extraction", "cloze"):
        if by_kind[k]:
            m[k] = _acc(by_kind[k])

    fwd_groups = defaultdict(list)
    for r in by_kind["forward"] + by_kind["extraction"]:
        fwd_groups[r["group"]].append(r)
    multi = {g: rs for g, rs in fwd_groups.items() if len(rs) >= 2}
    if multi:
        all_ok, agree = [], []
        for g, rs in multi.items():
            all_ok.append(all(r["correct"] for r in rs))
            answers = [by_id[r["id"]].choices[r["pred"]] for r in rs]
            agree.append(len(set(answers)) == 1)
        m["consistency"] = {"n_facts": len(multi), "known_under_all_prompts": mean(map(float, all_ok)),
                            "same_answer_all_prompts": mean(map(float, agree))}

    # ---------------------------------------------------- B. manipulation
    for k in ("classification", "comparison"):
        if by_kind[k]:
            m[k] = _acc(by_kind[k])

    known = {g: any(r["correct"] for r in rs) for g, rs in fwd_groups.items()}
    if by_kind["forward"] or by_kind["reverse"]:
        f, rv = _acc(by_kind["forward"]), _acc(by_kind["reverse"])
        cond = [r["correct"] for r in by_kind["reverse"] if known.get(r["group"])]
        m["reversal"] = {"forward": f, "reverse": rv,
                         "gap": (f.get("accuracy", 0) - rv.get("accuracy", 0)) if f["n"] and rv["n"] else None,
                         "reverse_given_forward_known": _cond(cond), "n_known": len(cond)}

    if by_kind["composition"]:
        rs = by_kind["composition"]
        both = [r for r in rs if all(known.get(h) for h in by_id[r["id"]].meta.get("hops", []))]
        m["composition"] = {**_acc(rs), "n_both_hops_known": len(both),
                            "compositionality_gap": _cond([not r["correct"] for r in both])}

    # ---------------------------------------------------- C. application beyond the text
    if by_kind["application"]:
        rs = by_kind["application"]
        out = {**_acc(rs), "by_expected": {}, "by_cell": {}}
        exp = defaultdict(list)
        cells = defaultdict(lambda: defaultdict(list))
        for r in rs:
            it = by_id[r["id"]]
            for label, idx in (it.meta.get("expected") or {}).items():
                exp[label].append(r["pred"] == idx)
                cells[it.meta.get("cell", "all")][label].append(r["pred"] == idx)
        out["by_expected"] = {k: {"n": len(v), "match_rate": mean(map(float, v))} for k, v in exp.items()}
        out["by_cell"] = {c: {k: mean(map(float, v)) for k, v in d.items()} for c, d in cells.items()}
        m["application"] = out
    if by_kind["inductive"]:
        m["inductive"] = _acc(by_kind["inductive"])

    if by_kind["knowledge"] and by_kind["application"]:
        k_by_g = defaultdict(list)
        for r in by_kind["knowledge"]:
            if "correct" in r:
                k_by_g[r["group"]].append(r["correct"])
        knows = {g: all(v) for g, v in k_by_g.items()}
        acts = [r for r in by_kind["application"] if "correct" in r and r["group"] in knows]
        m["knowing_doing"] = {
            "knowledge": _acc(by_kind["knowledge"]),
            "n_known": sum(1 for r in acts if knows[r["group"]]),
            "wrong_action_given_known": _cond([not r["correct"] for r in acts if knows[r["group"]]]),
            "right_action_given_unknown": _cond([r["correct"] for r in acts if not knows[r["group"]]]),
        }

    if by_kind["self_report"] and by_kind["application"]:
        behaviour = defaultdict(Counter)
        for r in by_kind["application"]:
            behaviour[r["group"]][r["pred"]] += 1
        agree = []
        for r in by_kind["self_report"]:
            maps = by_id[r["id"]].meta.get("maps_to")
            if not maps or not behaviour[r["group"]]:
                continue
            agree.append(maps[r["pred"]] == behaviour[r["group"]].most_common(1)[0][0])
        m["self_report"] = {"n": len(agree), "agreement_with_behaviour": _cond(agree)}

    # ---------------------------------------------------- D. interaction with other knowledge
    if by_kind["conflict"]:
        rs = by_kind["conflict"]
        ctx = [r["pred"] == by_id[r["id"]].meta["context"] for r in rs]
        mem_ = [r["pred"] == by_id[r["id"]].meta["memory"] for r in rs]
        mem_known = [r["pred"] == by_id[r["id"]].meta["memory"] for r in rs if known.get(r["group"])]
        m["conflict"] = {"n": len(rs), "follows_context": mean(map(float, ctx)),
                         "follows_memory": mean(map(float, mem_)),
                         "follows_memory_when_known": _cond(mem_known), "n_known": len(mem_known)}
    if by_kind["propagation"]:
        rs = by_kind["propagation"]
        m["propagation"] = {**_acc(rs),
                            "keeps_old_answer": mean(float(r["pred"] == by_id[r["id"]].meta["old"]) for r in rs)}

    # ---------------------------------------------------- metacognition
    scored = [r for r in recs if "correct" in r and "conf" in r and r["kind"] in FACT_KINDS]
    if len(scored) >= 10:
        m["metacognition"] = calibration([r["conf"] for r in scored], [r["correct"] for r in scored])
    return m


def calibration(conf: list[float], correct: list[bool], bins: int = 10) -> dict:
    c, y = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if sel.any():
            ece += sel.mean() * abs(c[sel].mean() - y[sel].mean())
    out = {"n": len(c), "ece": float(ece), "mean_confidence": float(c.mean()), "accuracy": float(y.mean())}
    if 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        out["auroc"] = float(roc_auc_score(y, c))
    return out


def latent_vs_expressed(scorer: Scorer, items: list[Item], recs: list[dict], layers: list[int] | None = None,
                        max_items: int = 400, folds: int = 5, seed: int = 0) -> dict:
    """Inside-Out style: internal K from a probe on hidden states vs external K from output scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rec_by_id = {r["id"]: r for r in recs}
    pool = [it for it in items if it.kind in FACT_KINDS and it.target is not None and it.id in rec_by_id]
    rng = np.random.default_rng(seed)
    if len(pool) > max_items:
        pool = [pool[i] for i in sorted(rng.choice(len(pool), max_items, replace=False))]
    groups_all = sorted({it.group or it.id for it in pool})
    if len(groups_all) < folds:
        return {"n": len(pool), "skipped": "too few groups for cross-validation"}
    layers = layers or sorted({scorer.n_layers // 2, (3 * scorer.n_layers) // 4, scorer.n_layers})
    texts, labels, owner = [], [], []
    for k, it in enumerate(pool):
        for j, c in enumerate(it.choices):
            texts.append(it.prompt + c)
            labels.append(int(j == it.target))
            owner.append(k)
    X = scorer.last_hidden(texts, layers).numpy()
    y, owner = np.asarray(labels), np.asarray(owner)
    gid = np.asarray([groups_all.index(pool[k].group or pool[k].id) for k in owner])
    out = {"n": len(pool), "layers": {}}
    ext = mean(rec_by_id[it.id]["k_external"] for it in pool)
    out["k_external"] = ext
    for li, layer in enumerate(layers):
        score = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=folds).split(X, y, gid):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            clf.fit(X[tr, li], y[tr])
            score[te] = clf.decision_function(X[te, li])
        ks = []
        for k, it in enumerate(pool):
            sel = np.where(owner == k)[0]
            ks.append(_pairwise_k(score[sel].tolist(), it.target))
        internal = float(np.mean(ks))
        out["layers"][str(layer)] = {"k_internal": internal,
                                     "relative_gap": (internal - ext) / ext if ext else None}
    best = max(out["layers"].items(), key=lambda kv: kv[1]["k_internal"])
    out["best_layer"], out["k_internal"], out["relative_gap"] = int(best[0]), best[1]["k_internal"], best[1]["relative_gap"]
    return out


def run_suite(scorer: Scorer, items: list[Item], norm: str = "sum", latent: bool = True,
              latent_layers: list[int] | None = None, latent_max_items: int = 400) -> dict:
    recs = score_items(scorer, items, norm)
    metrics = compute_metrics(items, recs)
    if latent:
        metrics["latent_vs_expressed"] = latent_vs_expressed(scorer, items, recs, latent_layers, latent_max_items)
    return {"metrics": metrics, "items": recs}
