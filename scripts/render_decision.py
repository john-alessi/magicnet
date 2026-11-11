"""Render one decision record as PNGs: the board and (optionally) a demo belief plot.

    uv run scripts/render_decision.py --dump forge-bridge/fixtures/dumps/pio_mrp.jsonl \
        --game 3 --decision 271 --out viz_samples --fake-belief 0.05

``--dump`` takes a .jsonl / .jsonl.gz file, a directory of shard_*.jsonl.gz or
a glob. ``--fake-belief SIGMA`` builds demo samples from the ground truth
(cards drawn ∝ count, their whitened PCA targets plus N(0, SIGMA²) noise) so
the belief plot can be checked without a model; it needs the catalog
service (the 2-D card map is fitted once and cached under ~/.cache/magicnet).
Needs the notebook dependency group: ``uv sync --group notebook``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from magicnet import dump, viz


def pick(args) -> dump.Decision | None:
    if args.random:
        rng = random.Random(args.seed)
        chosen, n = None, 0
        for d in dump.load_many(args.dump):
            if args.game is not None and d.game != args.game:
                continue
            n += 1
            if rng.randrange(n) == 0:  # reservoir sample
                chosen = d
        return chosen
    for d in dump.load_many(args.dump):
        if (args.game is None or d.game == args.game) and (args.decision is None or d.decision == args.decision):
            return d
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="jsonl / .gz file, shard directory or glob")
    ap.add_argument("--game", type=int)
    ap.add_argument("--decision", type=int)
    ap.add_argument("--random", action="store_true", help="pick a random decision (of --game, if given)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="viz_out", help="output directory")
    ap.add_argument("--fake-belief", type=float, metavar="SIGMA", help="also render demo belief samples")
    ap.add_argument("--samples", type=int, default=500, help="samples per region for --fake-belief")
    ap.add_argument("--demo-probs", action="store_true", help="show random action probabilities in the panel")
    ap.add_argument("--catalog-url", default="http://127.0.0.1:8765")
    args = ap.parse_args()

    d = pick(args)
    if d is None:
        print(f"no matching decision in {args.dump}", file=sys.stderr)
        return 1
    out = Path(args.out)
    prefix = out / f"g{d.game}_d{d.decision}"
    probs = None
    if args.demo_probs:
        logits = np.random.default_rng(args.seed).normal(size=len(d.actions)) * 2
        probs = np.exp(logits) / np.exp(logits).sum()

    t = time.perf_counter()
    board = viz.render_board(d, f"{prefix}_board.png", action_probs=probs)
    print(f"{board}  ({time.perf_counter() - t:.2f}s)")

    if args.fake_belief is not None:
        if not d.hidden_truth:
            print("decision has no hidden_truth labels; skipping --fake-belief", file=sys.stderr)
            return 0
        from magicnet.catalog import CatalogClient

        card_map = viz.CardMap.load_or_fit(CatalogClient(args.catalog_url))
        samples = viz.fake_belief(d, card_map, args.fake_belief, n=args.samples, seed=args.seed)
        sig = f"{args.fake_belief:g}".replace(".", "p")
        t = time.perf_counter()
        path = viz.render_belief(d, samples, card_map, f"{prefix}_belief_sigma{sig}.png",
                                 title=f"FAKE belief (truth + N(0, {args.fake_belief:g}²)) · game {d.game} · "
                                       f"decision {d.decision} · turn {d.turn}")
        print(f"{path}  ({time.perf_counter() - t:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
