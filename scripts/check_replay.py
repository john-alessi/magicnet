#!/usr/bin/env python3
"""CLI: check replay equivalence between a recorded and a replayed dump.

The strongest available end-to-end test of the action path. One run records the
index of every action taken; a second run feeds those indices back. If the two
games match, the whole round trip is sound -- enumeration order, serialisation,
index decoding and action application -- and it is a genuine differential test,
because the recording side chooses through Forge's own path while the replay
side chooses by index lookup. The two share no logic.

Comparison ignores object ids. They are not reproducible across runs: a library
search can return a different physical instance of an identical card. Ids are
join keys that are stripped before tensorisation anyway, so comparing without
them compares what the model would actually see.

Usage:
    uv run scripts/check_replay.py --recorded /tmp/rec.jsonl --replayed /tmp/rep.jsonl
"""

from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
from pathlib import Path

from magicnet.dump import canonical, load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recorded", required=True, type=Path)
    ap.add_argument("--replayed", required=True, type=Path)
    ap.add_argument(
        "--report",
        type=int,
        default=1,
        help="how many divergences to describe before stopping",
    )
    args = ap.parse_args()

    matched = 0
    divergences = []

    for i, (rec, rep) in enumerate(
        zip_longest(load(args.recorded), load(args.replayed))
    ):
        if rec is None or rep is None:
            which = "replayed" if rec is not None else "recorded"
            divergences.append(f"record {i}: {which} dump ended early")
            break

        why = None
        if rec.chosen != rep.chosen:
            why = f"chosen {rec.chosen} != {rep.chosen}"
        elif [a.get("desc") for a in rec.actions] != [
            a.get("desc") for a in rep.actions
        ]:
            why = (
                f"action pool differs: {len(rec.actions)} vs {len(rep.actions)} entries"
            )
        else:
            ca, cb = canonical(rec.tokens), canonical(rep.tokens)
            if ca != cb:
                only_rec = [t for t in ca if t not in cb]
                only_rep = [t for t in cb if t not in ca]
                why = (
                    f"state differs: {len(only_rec)} tokens only in recorded, "
                    f"{len(only_rep)} only in replayed"
                )
                if only_rec:
                    why += f"\n    recorded: {dict(only_rec[0][0])}"
                if only_rep:
                    why += f"\n    replayed: {dict(only_rep[0][0])}"

        if why is None:
            matched += 1
        else:
            divergences.append(
                f"record {i} (turn {rec.turn} {rec.phase} {rec.viewer[:3]}): {why}"
            )
            if len(divergences) >= args.report:
                break

    print(f"matched records before divergence: {matched}")
    if divergences:
        for d in divergences:
            print(f"DIVERGE {d}", file=sys.stderr)
        return 1

    print("OK replay equivalent (modulo non-reproducible object ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
