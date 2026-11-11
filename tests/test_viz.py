"""magicnet.viz: board and belief renderings must not crash and must decode correctly."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from magicnet import dump, viz  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "forge-bridge" / "fixtures" / "dumps" / "pio_mrp.jsonl"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _decision(game: int, decision: int) -> dump.Decision:
    return next(d for d in dump.load(FIXTURE) if (d.game, d.decision) == (game, decision))


def big_board(d: dump.Decision) -> dump.Decision:
    """*d* with a huge battlefield, a face-down opponent permanent (a hidden
    region with its label), an own face-down card and an unknown token type."""
    d = copy.deepcopy(d)
    extra = []
    for side in ("self", "opp"):
        for i in range(60):
            extra.append({"type": "game_object", "zone": f"{side}_battlefield", "id": f"c{9000 + i}{side == 'opp':d}",
                          "name": f"Goblin Token Number {i % 25} With A Long Name", "components": {
                              "token": True, "power": 1 + i % 3, "toughness": 1, "base_power": 1,
                              "base_toughness": 1, "tapped": i % 2 == 0, "types": "Creature - Goblin",
                              "colors": ["red"], "ctr_+1/+1": i % 3}})
        for i in range(30):
            extra.append({"type": "game_object", "zone": f"{side}_battlefield", "id": f"c{8000 + i}{side == 'opp':d}",
                          "name": "Mountain", "components": {"types": "Basic Land - Mountain", "tapped": i < 5}})
    extra.append({"type": "game_object", "zone": "opp_battlefield", "id": "c7777",
                  "components": {"power": 2, "toughness": 2, "face_down": True}})
    extra.append({"type": "game_object", "zone": "self_battlefield", "id": "c7778", "name": "Gudul Lurker",
                  "components": {"power": 2, "toughness": 2, "face_down": True, "state": "FaceDown"}})
    extra.append({"type": "mystery_token", "zone": "somewhere_new", "id": "x1"})
    extra.append({"type": "game_object", "zone": "opp_hand", "id": "c7779", "name": "Lightning Bolt",
                  "components": {"known_by_reveal": True}})
    d.state = d.state + extra
    d.hidden_truth = list(d.hidden_truth or []) + [
        {"region": "c7777", "total": 1, "cards": [{"oracle_id": "x-lurker", "name": "Gudul Lurker", "count": 1}]}]
    return d


def test_render_board(tmp_path):
    d = _decision(3, 271)
    probs = np.full(len(d.actions), 1.0 / len(d.actions))
    for dec, name, p in ((d, "fixture.png", probs), (big_board(d), "big.png", None),
                         (_decision(0, 163), "stack.png", None)):
        out = viz.render_board(dec, tmp_path / name, action_probs=p)
        assert out.read_bytes()[:8] == PNG_MAGIC
    assert [t["id"] for t in viz.regions(big_board(d))][-1] == "c7777"


def _toy_map(d: dump.Decision, n_extra: int = 300, seed: int = 0) -> viz.CardMap:
    rng = np.random.default_rng(seed)
    truth = {(c["oracle_id"], c["name"]) for lab in d.hidden_truth for c in lab["cards"]}
    oids = [o for o, _ in sorted(truth)] + [f"oid{i}" for i in range(n_extra)]
    names = [n for _, n in sorted(truth)] + [f"Card {i}" for i in range(n_extra)]
    targets = rng.normal(size=(len(oids), 32)).astype(np.float32)
    return viz.CardMap(oids, names, targets, rng.normal(size=(len(oids), 2)))


def test_nearest_decode_and_belief(tmp_path):
    d = big_board(_decision(3, 271))
    cmap = _toy_map(d)
    rng = np.random.default_rng(1)
    want = rng.integers(0, len(cmap.names), size=500)
    idx, dist = cmap.nearest(cmap.targets[want] + 0.01 * rng.normal(size=(500, 32)), k=3, chunk=37)
    assert (idx[:, 0] == want).all()
    assert (np.diff(dist, axis=1) >= 0).all()

    # (n, R, 32) and the loader's (n, sum(count), 32) layouts split per region.
    regs = viz.regions(d)
    per_card = sum(viz._region_count(t) for t in regs)
    split = viz.region_samples(d, np.zeros((4, per_card, 32), np.float32))
    assert {k: v.shape[0] for k, v in split.items()} == {t["id"]: 4 * viz._region_count(t) for t in regs}
    assert viz.region_samples(d, np.zeros((4, len(regs), 32)))["r:opp_hand"].shape == (4, 32)

    # Tight fake samples decode to exactly the true composition's support.
    samples = viz.fake_belief(d, cmap, sigma=0.01, n=400)
    counts, _ = viz.decode_counts(cmap, samples["r:opp_hand"])
    hand = {cmap.index(c["oracle_id"]) for c in d.hidden_truth[0]["cards"]}
    assert set(counts) == hand
    out = viz.render_belief(d, samples, cmap, tmp_path / "belief.png")
    assert out.read_bytes()[:8] == PNG_MAGIC
