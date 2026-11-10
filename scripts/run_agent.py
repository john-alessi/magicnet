#!/usr/bin/env python3
"""CLI: serve a policy to a Forge process over the socket boundary.

Start this first, then point Forge at it:

    uv run scripts/run_agent.py --port 5555 &
    java -Xmx2048m -Dmagicnet.actions=socket:127.0.0.1:5555 \\
         -Dmagicnet.jsonl=/tmp/live.jsonl \\
         -cp "build:/usr/share/java/forge-gui-desktop.jar" \\
         magicnet.bridge.Harness --decks /tmp/pio_41 --games 1 \\
         --out /tmp/live --seed 42 --timeout 900

The dump Forge writes is in the same format as an offline dump, so
``validate_dump.py`` applies to a live run unchanged -- which is the point: the
live path and the offline path are checked by the same gate.

This serves a random policy. Replacing it with a learned one means supplying any
object with ``act(decision) -> int`` and ``begin_game(game) -> None``.
"""

from __future__ import annotations

import argparse
import sys
import torch

from magicnet.agent import LearnedPolicy, serve
from magicnet.catalog import CatalogClient
from magicnet.network.magicnet import MagicNet
from magicnet.network.data_loader import DataLoader

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pass-rate",
        type=float,
        default=0.85,
        help="probability of passing priority rather than acting",
    )
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision("high")

    catalog = CatalogClient()
    model = MagicNet()
    model.eval()
    data_loader = DataLoader(catalog)
    policy = LearnedPolicy(seed=args.seed, model=model, data_loader=data_loader)
    server = serve(policy, args.host, args.port)
    print(f"agent listening on {args.host}:{args.port}", flush=True)

    try:
        server.serve_one()
    finally:
        server.server_close()

    acted = policy.decisions - policy.passes
    print(
        f"agent served {policy.decisions} decisions "
        f"({policy.passes} passes, {acted} actions)",
        flush=True,
    )
    if policy.decisions == 0:
        print("agent served no decisions at all", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
