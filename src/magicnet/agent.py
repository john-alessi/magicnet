"""The Python side of the Forge boundary: a policy server.

Forge drives the game and blocks on this process at every decision. The wire
protocol is line-delimited JSON over TCP, one line in, one line out, strictly
alternating -- Forge is single-threaded here, so a reply is always owed and must
always be sent.

Three message kinds arrive:

``{"event": "begin_game", "game": N}``
    A new game is starting. Reply with any line (``"ok"``); the acknowledgement
    is what keeps the two sides in lockstep at a game boundary.

``{"event": "end"}``
    The run is over. No reply is owed; the socket closes.

A decision request
    Byte-identical to a record in a JSONL dump minus the trailing ``chosen``
    fields, so anything that consumes dumps consumes these unchanged. Reply with
    a single integer: an index into ``actions``. Where passing priority is
    legal it is an explicit option (``api == "Pass"``, always last); there is
    no sentinel for passing, and ``-1`` is rejected.

Replying with anything else is fatal on the Forge side, by design -- a crashed
agent must not be mistakable for a policy that always passes.

The transport is deliberately the dumbest thing that works. JSON over TCP costs
far more than it needs to, but it is worth measuring the real cost before
optimising it; swapping in a binary codec later changes only this file and
``SocketSource``.
"""

from __future__ import annotations

import json
import random
import socketserver
from dataclasses import dataclass, field
from typing import Protocol

from magicnet.dump import Decision, HistoryReassembler, SUPPORTED_SCHEMA, SchemaError
from magicnet.network.magicnet import MagicNet
from magicnet.network.data_loader import load_game

class Policy(Protocol):
    """Chooses an action given a decision request.

    Returns an index into ``decision.actions``; pass by returning
    ``decision.pass_index`` where it is not None.
    """

    def act(self, decision: Decision) -> int: ...

    def begin_game(self, game: int) -> None: ...


@dataclass
class LearnedPolicy:
    """
    random policy logic included as placeholder
    """

    seed: int = 0
    decisions: int = 0
    passes: int = 0
    _rng: object = field(default=None, repr=False)

    def __init__(
        self,
        catalog: CatalogClient,
        seed: int,
        model: MagicNet
    ) -> None:
        self.catalog = catalog
        self.seed = seed
        self.model = model
        self._rng = random.Random(self.seed)

    def begin_game(self, game: int) -> None:
        pass

    def act(self, decision: Decision) -> int:
        card_names = [
            c['name']
            for c in decision.state
            if c['type'] in ['game_object'] and c['name'] != 'HIDDEN'
        ]
        card_metadata = self.catalog.lookup(card_names)
        game = load_game(decision.state, decision.history, self.catalog)
        #x = self.model(decision.history, decision.state, card_metadata)
        self.decisions += 1
        selected_action = self._rng.randrange(len(decision.actions))
        breakpoint()
        return selected_action

@dataclass
class RandomPolicy:
    """Uniform choice over the legal pool, with a bias toward passing.

    Not a baseline worth beating -- it exists to exercise the boundary. The pass
    bias matters: without it a random policy activates something at nearly every
    priority and games crawl along without advancing, which tests the transport
    but never reaches a realistic distribution of board states.
    """

    seed: int = 0
    pass_rate: float = 0.85
    decisions: int = 0
    passes: int = 0
    _rng: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import random

        self._rng = random.Random(self.seed)

    def begin_game(self, game: int) -> None:
        pass

    def act(self, decision: Decision) -> int:
        self.decisions += 1
        passing = decision.pass_index
        others = [i for i in range(len(decision.actions)) if i != passing]
        if passing is not None and (not others or self._rng.random() < self.pass_rate):
            self.passes += 1
            return passing
        return self._rng.choice(others)


@dataclass
class RecordingPolicy:
    """Wraps a policy and keeps every request it was asked about.

    Lets a live run be checked with the same tools as an offline dump.
    """

    inner: Policy
    seen: list[Decision] = field(default_factory=list)

    def begin_game(self, game: int) -> None:
        self.inner.begin_game(game)

    def act(self, decision: Decision) -> int:
        self.seen.append(decision)
        return self.inner.act(decision)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        policy: Policy = self.server.policy
        for raw in self.rfile:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            msg = json.loads(line)

            event = msg.get("event")
            if event == "end":
                break
            if event == "begin_game":
                policy.begin_game(int(msg["game"]))
                self._reply("ok")
                continue

            schema = msg.get("schema")
            if schema != SUPPORTED_SCHEMA:
                # Fail loudly. A schema drift that is tolerated here surfaces
                # much later as a model silently trained on shifted features.
                raise SchemaError(
                    f"agent supports schema {SUPPORTED_SCHEMA}, got {schema!r}"
                )

            decision = self.server.reassembler.absorb(msg, where="live request")
            pick = policy.act(decision)
            if not 0 <= pick < len(decision.actions):
                # Passing is an explicit option (api "Pass") where it is legal;
                # -1 is not a pass and is rejected rather than reinterpreted.
                raise ValueError(
                    f"policy returned {pick} for {len(decision.actions)} options"
                    f" (pass option index: {decision.pass_index})"
                )
            self._reply(str(pick))
        self.server.finished = True

    def _reply(self, text: str) -> None:
        self.wfile.write((text + "\n").encode("utf-8"))
        self.wfile.flush()


class AgentServer(socketserver.TCPServer):
    """Serves exactly one Forge process.

    Single-connection by design: the two sides are in lockstep, and a second
    concurrent game on the same policy object would interleave its decisions.
    Parallel self-play runs one server per worker on its own port.
    """

    allow_reuse_address = True

    def __init__(self, host: str, port: int, policy: Policy) -> None:
        super().__init__((host, port), _Handler)
        self.policy = policy
        self.reassembler = HistoryReassembler()
        self.finished = False

    def serve_one(self) -> None:
        self.handle_request()


def serve(policy: Policy, host: str = "127.0.0.1", port: int = 5555) -> AgentServer:
    server = AgentServer(host, port, policy)
    return server
