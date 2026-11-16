import json
from pathlib import Path

import pytest

from b4elm_eval.cli import main as cli_main
from b4elm_eval.dynamics import compare
from b4elm_eval.generators import FactWorld, authored_items, memorization_items
from b4elm_eval.items import Item, load_items, save_items
from b4elm_eval.probes import run_suite
from b4elm_eval.scorer import Scorer

from tiny import corpus, make_world, train_tiny

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("ckpts")
    world = make_world()
    model, tok, ckpts = train_tiny(world, out, steps=400, save_at=(0, 100, 400))
    return world, model, tok, ckpts


def _items(world):
    items = FactWorld(world, n_distractors=4).all_items()
    items += memorization_items([("world", corpus(world))], prefix_words=12, cont_words=6, n=20)
    items += authored_items(EXAMPLES / "authored_items.example.jsonl")
    return items


def test_no_valid_answer_is_a_distractor():
    spec = {"relations": {"father_of": {"train_direction": "head", "statement": "{head} begat {tail}.",
                                        "templates": [{"text": "{head} begat", "seen": True},
                                                      "The father of {tail} was"]}},
            "facts": [["Noah", "father_of", "Shem"], ["Noah", "father_of", "Ham"], ["Noah", "father_of", "Japheth"],
                      ["Terah", "father_of", "Abraham"], ["Abraham", "father_of", "Isaac"], ["Isaac", "father_of", "Jacob"]]}
    w = FactWorld(spec, n_distractors=4)
    for it in w.relation_items():
        if it.prompt == "Noah begat":
            wrong = {c.strip() for i, c in enumerate(it.choices) if i != it.target}
            assert not wrong & {"Shem", "Ham", "Japheth"}
    kinds = {it.kind for it in w.all_items()}
    assert {"forward", "reverse", "conflict"} <= kinds


def test_genesis_example_generates_every_fact_kind():
    from b4elm_eval.generators import load_world
    items = load_world(EXAMPLES / "genesis_facts.yaml").all_items()
    kinds = {i.kind for i in items}
    assert kinds >= {"forward", "extraction", "reverse", "classification", "comparison", "composition",
                     "conflict", "propagation"}
    noah = [i for i in items if i.prompt == "Noah begat"]
    for it in noah:
        wrong = {c.strip() for j, c in enumerate(it.choices) if j != it.target}
        assert not wrong & {"Shem", "Ham", "Japheth"}


def test_item_roundtrip_and_validation(tmp_path):
    items = [Item("a", "forward", "p", [" x", " y"], 0), Item("b", "memorization", "p", continuation=" c")]
    save_items(items, tmp_path / "x.jsonl")
    assert [i.id for i in load_items(tmp_path)] == ["a", "b"]
    with pytest.raises(ValueError):
        Item("c", "forward", "p", [" x"], 0)
    with pytest.raises(ValueError):
        Item("d", "nonsense", "p", [" x", " y"], 0)


def test_suite_reproduces_known_effects(trained):
    world, model, tok, ckpts = trained
    scorer = Scorer("unused", device="cpu", model=model, tokenizer=tok, batch_size=64)
    res = run_suite(scorer, _items(world), latent_max_items=120)
    m = res["metrics"]
    untrained = run_suite(Scorer(str(ckpts[0]), device="cpu", batch_size=64), _items(world), latent=False)["metrics"]
    expected = {"memorization", "extraction", "consistency", "classification", "comparison", "reversal",
                "composition", "application", "inductive", "knowing_doing", "self_report", "conflict",
                "propagation", "metacognition", "latent_vs_expressed"}
    assert expected <= set(m), expected - set(m)
    # Learned in training: verbatim text and the trained direction.
    assert m["memorization"]["token_accuracy"] > untrained["memorization"]["token_accuracy"] + 0.2
    assert m["reversal"]["forward"]["accuracy"] > 0.9
    assert untrained["reversal"]["forward"]["accuracy"] < 0.5
    # Reversal curse: the untrained direction stays far below.
    assert m["reversal"]["gap"] > 0.4
    assert 0.0 <= m["latent_vs_expressed"]["k_internal"] <= 1.0
    assert 0.0 <= m["metacognition"]["ece"] <= 1.0


def test_trajectory_and_compare_cli(trained, tmp_path):
    world, _, _, ckpts = trained
    items_dir = tmp_path / "items"
    save_items(_items(world), items_dir / "all.jsonl")
    pattern = str(Path(ckpts[0]).parent / "step_*")
    cli_main(["trajectory", "--checkpoints", pattern, "--items", str(items_dir), "--out", str(tmp_path / "traj"),
              "--device", "cpu", "--no-latent"])
    traj = json.loads((tmp_path / "traj/item_trajectories.json").read_text())
    assert traj["steps"] == [0, 100, 400]
    fwd = traj["by_kind"]["forward"]
    assert fwd["acquired"] >= 0.9 * fwd["n"]
    assert (tmp_path / "traj/metrics_by_step.csv").read_text().startswith("step,")
    before = json.loads((tmp_path / "traj/step_0.json").read_text())
    after = json.loads((tmp_path / "traj/step_400.json").read_text())
    rep = compare(before, after)
    assert rep["item_flips_by_kind"]["forward"]["gained"] > 0
