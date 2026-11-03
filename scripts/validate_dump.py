#!/usr/bin/env python3
"""CLI: validate a Forge bridge decision dump before training on it.

Runs the checks that have to pass before a dump is trustworthy:

1. **Schema and parse** -- every line is JSON at a schema we understand.
2. **History reconstruction** -- delta-encoded history reassembles to the
   length the writer recorded, for every record.
3. **Leak boundary** -- no join-key id survives into model-visible tokens, no
   opponent hidden-zone card is named in the viewer's tokens unless a reveal
   shown to that viewer or a public-origin move (in their own history)
   accounts for it, no ``Reveal`` history token names a card to a player who
   was not shown it, and no move out of a hidden or face-down origin names a
   card the viewer could not see.
   Schema 13 (history redesign): history tokens carry no card identity and
   their edges target only handles/events/players; every handle the history
   refers to is on exactly one state token, never on an individual hidden card,
   a named opponent hand card only if ``known_by_reveal`` (and the viewer's own
   history explains it); a handle whose zone became ambiguous (a card left that
   hidden zone unidentified) sits on an ``*_untracked`` token until an event
   re-identifies it; phantoms carry no id and never name an object the viewer
   last saw unnamed; ``affected_by`` joins visible tokens or history events.
   Schema 14: a handle once on a phantom is never on a live token again (no
   re-linking), and no string component carries Forge's bracketed run context
   or a harness player name.
   Schema 12: hidden means unnamed (``magicnet.dump.is_hidden``); a hidden
   card or history role never carries an ``oracle_id``.
   Schema 7: ``affects`` edges must join two tokens the viewer can see, and
   choice props (chosen colour/type/player/number, named card, X, kicked)
   never sit on a face-down or redacted token, and a chosen player is one of
   the player tokens.
4. **Token layout** (schema 9) -- identity/ref keys (per ``catalog/keys.json``)
   are top-level and every model input is under a non-empty ``components``;
   flags appear only when true (except ``FALSE_FLAG_OK``); no string component
   carries a run-unstable ``Name (id)`` object id. Schema 10: ``gained``/``lost``
   entries are host-independent rules text (no ``(id)``, ``CARDNAME``, player
   name or the host's own name).
5. **Label usability** -- the chosen action resolves to a pool index.

Usage:
    uv run scripts/validate_dump.py --dump /tmp/dump.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import islice
from collections import Counter
from pathlib import Path

from magicnet.dump import (
    SchemaError, comp, flatten, is_hidden, load, resolve_edges, strip_ids, strip_option_ids,
    summarise,
)
from magicnet.catalog.keys import load_registry
from magicnet.schema import COMPONENTS

# Zones whose contents are hidden from a player who does not own them. The
# extractor omits the name (schema 12); any card name appearing here
# is an information leak, which is the one class of bug that silently makes a
# trained policy cheat. The viewer's own library is not included: per the
# library representation design it is derivable from public information
# (decklist minus everything already seen), so real names there are correct.
HIDDEN_OPPONENT_ZONES = frozenset({"opp_hand", "opp_library", "opp_sideboard"})

# Hidden zones where the extractor may name a card the viewer has legitimately
# been shown (Forge's canBeShownTo). It marks those tokens ``revealed: true``;
# anything else named here is a leak. Only single-card tokens qualify: the
# hidden remainder of the opponent sideboard is one ``count`` aggregate, and an
# aggregate claiming ``revealed`` would be a way to smuggle names past this.
REVEALABLE_ZONES = frozenset({"opp_sideboard"})

# Opponent hand cards the viewer legitimately knows (a reveal shown to them, or
# a public move into the hand). The extractor marks them ``known_by_reveal:
# true``, but the mark alone is not trusted: the viewer's own history must
# explain the card since it last entered the hand (see _HandKnowledge).
# That is checked from the dump alone, independently of the Java bookkeeping
# that set the mark.
KNOWN_BY_REVEAL_ZONES = frozenset({"opp_hand"})

def check_option_ids(decisions) -> list[str]:
    """Options must not carry raw object ids.

    Ids are unstable across runs (§16.8), so one surviving in an option would be
    a nondeterministic feature. Checked separately from token ids because
    options live in their own array and were missed by the token check.
    """
    problems = []
    for dec in decisions:
        for opt in strip_option_ids(dec.actions):
            for key, value in opt.items():
                # Oracle ids are Scryfall's stable card identity, not run-unstable
                # object ids, so they are a legitimate feature.
                if (key.endswith("_id") or key == "id") and not key.endswith("oracle_id"):
                    problems.append(
                        f"game {dec.game} decision {dec.decision}: "
                        f"option retains {key}={value!r}"
                    )
    return problems


def _viewer_ref(state: list[dict]) -> str | None:
    for tok in state:
        if tok.get("type") == "player" and comp(tok, "side") == "self":
            return tok.get("id")
    return None


# Schema 13 history zones (viewer-relative names, magicnet.schema.ZONES).
PUBLIC_ZONES = frozenset(
    f"{side}_{z}" for side in ("self", "opp")
    for z in ("battlefield", "graveyard", "exile", "command")
) | {"stack"}
# Zones whose cards the viewer cannot tell apart: a card leaving one of these
# without being identified could be any card the viewer believed was there.
HIDDEN_ZONES = frozenset({"opp_hand", "opp_library", "opp_sideboard", "self_library"})
MOVE_VERBS = frozenset({"move_zone", "draw", "discard", "create"})
_REF = re.compile(r"^[hep]\d+$")

# State zones whose tokens tell us whether the viewer could see a card there.
_PUBLIC_STATE_ZONES = PUBLIC_ZONES


def _edges(tok: dict, rel: str | None = None) -> list[str]:
    return [e["to"] for e in tok.get("edges", ()) if rel is None or e["rel"] == rel]


class _Replay:
    """Replays one viewer's history, independently of the bridge's bookkeeping.

    Tracks, per tracking handle, where the viewer believes the object is and
    whether that belief is still a single zone; and which opponent hand cards
    the history explains. A card is explained in the opponent's hand by:

    * a ``reveal`` with it as ``obj`` shown to the viewer (``player``) from that hand;
    * a move into the hand from a public zone (``face_down`` departures only if
      the viewer's last state named the card);
    * a ``reveal`` to the viewer from another zone followed, in the same step,
      by its move into the hand (a dig like Silundi Vision).

    A move with no ``obj`` edge out of a hidden zone means the viewer could not
    tell which card left: every handle believed in that zone becomes ambiguous
    (and loses hand knowledge) until an event refers to it again. Ambiguous
    handles must sit on an ``*_untracked`` token (or a ``phantom``: ceasing to
    exist is public) -- that is the widening check.
    """

    def __init__(self, viewer: str | None) -> None:
        self.viewer = viewer
        self.loc: dict[str, str | None] = {}
        self.ambiguous: set[str] = set()
        self.known: set[str] = set()
        self.dig: dict[str, int] = {}
        self.referenced: set[str] = set()
        self.named: dict[str, bool] = {}
        self.cid: dict[str, str] = {}
        self.looked: dict[str, bool] = {}
        self.consumed = 0
        self.problems: Counter = Counter()
        self.looked_lib: int | None = None
        self.phantom: set[str] = set()
        self.events: list[dict] = []
        self.lib_pending: tuple[int, int] | None = None

    def advance(self, history: list[dict]) -> None:
        if len(history) < self.consumed:
            raise SchemaError("history shrank between records of one viewer")
        for i in range(self.consumed, len(history)):
            self._step(i, history[i])
        self.consumed = len(history)

    def _own_effect(self, ev: dict) -> bool:
        """Whether a cause of *ev* is an event whose actor is the viewer (its own
        Ponder or Brainstorm lets it look at the cards it reorders)."""
        for c in _edges(ev, "cause"):
            i = int(c[1:]) if c.startswith("e") else -1
            if 0 <= i < len(self.events) and self.viewer in _edges(self.events[i], "actor"):
                return True
        return False

    def _step(self, idx: int, ev: dict) -> None:
        self.events.append(ev)
        # History tokens carry no card identity and point only at handles,
        # events and players; a cause is always an earlier event.
        for key in ev:
            if key == "name" or key.endswith("oracle_id"):
                self.problems[f"history token carries {key}"] += 1
        for e in ev.get("edges", ()):
            to = e["to"]
            if not _REF.match(to):
                self.problems[f"history edge to a non-handle/event/player ref ({e['rel']})"] += 1
                continue
            if e["rel"] == "cause" and not to.startswith("e"):
                self.problems["cause edge to a non-event"] += 1
            if to.startswith("e") and int(to[1:]) >= idx:
                self.problems["history edge to a later event"] += 1
            if to.startswith("h"):
                self.referenced.add(to)
        verb, src, dst = comp(ev, "verb"), comp(ev, "from"), comp(ev, "to")
        step = comp(ev, "step")
        objs = _edges(ev, "obj")
        for h in _edges(ev):
            if h.startswith("h"):
                self.ambiguous.discard(h)
        if self.lib_pending and self.lib_pending[0] != step:
            # Own-library reorders of a step with no scry/surveil of ours in it.
            self.problems["unseen card identified in history (self_library -> self_library)"] += self.lib_pending[1]
            self.lib_pending = None
        if verb in ("scry", "surveil") and self.viewer in _edges(ev, "player"):
            self.looked_lib = step
            if self.lib_pending and self.lib_pending[0] == step:
                self.lib_pending = None
        if verb in MOVE_VERBS and objs and src in HIDDEN_ZONES and dst in HIDDEN_ZONES:
            # Hidden to hidden (a draw by the opponent, Brainstorm's put-back):
            # the viewer can tell which card moved only if it was revealed in
            # this step, sat at a known library position (a named token), or
            # is in its own library and it scried/surveilled in this step (the
            # summary event may follow the moves).
            for h in objs:
                known_pos = self.loc.get(h) == src and self.named.get(h) is True
                if self.dig.get(h) == step or known_pos or self.looked_lib == step:
                    continue
                if src == dst == "self_library" and self._own_effect(ev):
                    continue
                if src == dst == "self_library":
                    n = self.lib_pending[1] if self.lib_pending else 0
                    self.lib_pending = (step, n + 1)
                else:
                    self.problems[f"unseen card identified in history ({src} -> {dst})"] += 1
        if verb in MOVE_VERBS:
            if not objs and src in HIDDEN_ZONES and dst != src:
                for h, z in self.loc.items():
                    if z == src:
                        self.ambiguous.add(h)
                        self.known.discard(h)
            for h in objs:
                was = self.loc.get(h)
                self.loc[h] = dst
                if was == "opp_hand" or src == "opp_hand":
                    self.known.discard(h)
                if dst == "opp_hand":
                    looked = self.looked.get(self.cid.get(h, ""))
                    if src in PUBLIC_ZONES and (not comp(ev, "face_down") or looked is True):
                        self.known.add(h)
                    elif self.dig.get(h) == step:
                        self.known.add(h)
                self.dig.pop(h, None)
        elif verb == "reveal":
            for h in objs:
                self.loc[h] = src
                if self.viewer not in _edges(ev, "player"):
                    continue
                if src == "opp_hand":
                    self.known.add(h)
                elif src is not None:
                    self.dig[h] = step
        if verb in ("cast", "copy_spell"):
            for h in _edges(ev, "stack_object"):
                self.loc[h] = "stack"
                self.known.discard(h)
        elif verb == "play_land":
            for h in objs:
                self.loc[h] = None

    def observe(self, state: list[dict]) -> None:
        """Checks the state's handle assignment, then records what it showed."""
        count: Counter = Counter()
        for tok in state:
            handles = tok.get("handles", ())
            zone = tok.get("zone") or ""
            for h in handles:
                count[h] += 1
                if zone == "phantom":
                    self.phantom.add(h)
                elif h in self.phantom:
                    # Schema 14: a handle the viewer lost never comes back --
                    # re-linking it would assert an identity it cannot know.
                    self.problems[f"handle returned from a phantom to a token in {zone}"] += 1
                if h in self.ambiguous and not zone.endswith("_untracked") and zone != "phantom":
                    self.problems[f"handle not widened: on a token in {zone}"] += 1
            if not handles:
                continue
            if tok.get("type") in ("game_object", "library_card") and "id" in tok and is_hidden(tok) \
                    and zone in HIDDEN_ZONES:
                self.problems[f"handle on an individual hidden card in {zone}"] += len(handles)
            if zone == "opp_hand" and not is_hidden(tok) and comp(tok, "known_by_reveal") is not True:
                self.problems["handle on a named opp_hand card not known_by_reveal"] += 1
            if zone == "phantom":
                if "id" in tok:
                    self.problems["phantom carries an id"] += 1
                if not is_hidden(tok) and any(self.named.get(h) is False for h in handles):
                    self.problems["phantom names an object last seen unnamed"] += 1
            if "id" in tok and zone not in ("phantom",) and not zone.endswith("_untracked"):
                for h in handles:
                    self.cid[h] = tok["id"]
                    self.named[h] = not is_hidden(tok)
        for h in self.referenced:
            if count[h] != 1:
                self.problems[f"referenced handle assigned to {count[h]} tokens"] += 1
        for h, n in count.items():
            if h not in self.referenced:
                self.problems["handle assigned but never referenced"] += 1
        for tok in state:
            if (tok.get("zone") in _PUBLIC_STATE_ZONES and "id" in tok
                    and tok.get("type") == "game_object"):
                self.looked[tok["id"]] = not is_hidden(tok)

    def explains(self, tok: dict) -> bool:
        """Whether this named opponent-hand token is accounted for by the history."""
        return any(h in self.known and h not in self.ambiguous for h in tok.get("handles", ()))


# Props that record a choice made for, or a cost paid by, an object (schema 7).
# Read through Forge's public CardView, so they must never appear on a token
# whose identity the viewer cannot see.
CHOICE_PROPS = frozenset({
    "chosen_colors", "chosen_type", "named_card", "named_card_oracle_id",
    "chosen_player", "chosen_number", "x_paid", "kicked",
})


def _visible_endpoint(tok: dict) -> bool:
    """Whether an ``affects`` edge may touch this token: a player, or one named
    game object outside every library and the opponent's hidden zones."""
    if tok.get("type") == "player":
        return True
    zone = tok.get("zone") or ""
    return (tok.get("type") == "game_object" and comp(tok, "count") is None
            and not is_hidden(tok)
            and zone not in HIDDEN_OPPONENT_ZONES and not zone.endswith("_library"))


def check_board_relations(tokens: list[dict], edges: list[list[tuple[str, int]]]) -> Counter:
    """Schema-7 checks: ``affects`` endpoints and choice props are visible."""
    offenders: Counter = Counter()
    players = {t.get("id") for t in tokens if t.get("type") == "player"}
    for tok, resolved in zip(tokens, edges, strict=True):
        raw = sum(1 for e in tok.get("edges", ()) if e.get("rel") == "affects")
        hits = [j for rel, j in resolved if rel == "affects"]
        if raw != len(hits):
            # The extractor only emits edges between visible tokens, so an edge
            # to an absent id names an object the viewer has no token for.
            offenders["affects edge to an absent token"] += raw - len(hits)
        if hits and not _visible_endpoint(tok):
            offenders[f"affects edge from hidden token in {tok.get('zone')}"] += len(hits)
        for j in hits:
            if not _visible_endpoint(tokens[j]):
                offenders[f"affects edge to hidden token in {tokens[j].get('zone')}"] += 1
        # Schema 13: affected_by joins a visible object to a visible source or
        # to a history event of this viewer.
        raw_ab = sum(1 for e in tok.get("edges", ()) if e.get("rel") == "affected_by")
        ab = [j for rel, j in resolved if rel == "affected_by"]
        if raw_ab != len(ab):
            offenders["affected_by edge to an absent token"] += raw_ab - len(ab)
        if ab and not _visible_endpoint(tok):
            offenders[f"affected_by edge from hidden token in {tok.get('zone')}"] += len(ab)
        for j in ab:
            if tokens[j].get("type") != "history" and not _visible_endpoint(tokens[j]):
                offenders[f"affected_by edge to hidden token in {tokens[j].get('zone')}"] += 1
        chosen = CHOICE_PROPS.intersection(flatten(tok)) if tok.get("type") == "game_object" else ()
        if chosen and (comp(tok, "face_down") or tok.get("name") is None):
            offenders[f"choice on face-down/redacted token: {sorted(chosen)}"] += 1
        if "chosen_player" in tok and tok["chosen_player"] not in players:
            offenders["chosen_player is not a player token"] += 1
    return offenders


# Value kinds that stay at a token's top level (schema 9); everything else is a
# component.
TOP_LEVEL_KINDS = frozenset({"identity", "ref"})

# Flags whose false value is emitted on purpose: an attacker's sticky
# ``blocked`` status (CR 509.1h).
FALSE_FLAG_OK = frozenset({("game_object", "blocked")})

# Forge renders a card as "Name (id)"; the id is not reproducible across runs.
OBJECT_ID_IN_TEXT = re.compile(r"\w \(\d+\)")

# Schema 14: no string component carries Forge's bracketed run context
# ("[Player: P0-...]", "[Zone Changer: X]") or a harness player name ("P0-<deck>").
RUN_CONTEXT = (
    ("player name", re.compile(r"\bP\d+-")),
    ("bracketed context", re.compile(r"\[[A-Z][A-Za-z ]*: ")),
)

# Schema 10: ability-difference components are host-independent rules text.
# Stricter than OBJECT_ID_IN_TEXT: any "(123)", Forge's unsubstituted host
# placeholders, a stack-description "Activator:" tail, or a harness player name
# ("P0-<deck name>") means the text would differ between hosts or runs.
ABILITY_TEXT_KEYS = ("gained", "lost")
_ABILITY_TEXT_BAD = (
    ("object id", re.compile(r"\(\d+\)")),
    ("placeholder", re.compile(r"CARDNAME|NICKNAME|ORIGINALHOST|EFFECTSOURCE")),
    ("activator", re.compile(r"Activator:")),
    ("player name", re.compile(r"\bP\d+-")),
)


def ability_text_problems(tok: dict) -> list[str]:
    """Why any ``gained``/``lost`` entry of *tok* is not host-independent text.

    Also flags the host's own name (it must read "this creature"); skipped for
    tokens and basic lands, whose names are type words the bridge keeps.
    """
    host = tok.get("name")
    types = comp(tok, "types") or ""
    if not host or comp(tok, "token") or "Basic" in types:
        host = None
    out = []
    for key in ABILITY_TEXT_KEYS:
        for text in comp(tok, key) or []:
            for what, pat in _ABILITY_TEXT_BAD:
                if pat.search(text):
                    out.append(f"{what} in {key}")
            if host and re.search(r"\b" + re.escape(host) + r"\b", text):
                out.append(f"host name in {key}")
    return out


class _Kinds(dict):
    """key -> value_kind through the key registry, memoised."""

    def __init__(self) -> None:
        super().__init__()
        self.registry = load_registry()

    def __missing__(self, key: str) -> bool:
        top = self[key] = self.registry.is_top_level(key)
        return top


def check_layout(tokens: list[dict], kinds: _Kinds | None = None) -> Counter:
    """Schema-9 layout: identity/ref keys top-level, model inputs under
    ``components`` (non-empty when present), flags only when true, and no
    run-unstable object id inside a string component."""
    kinds = kinds if kinds is not None else _Kinds()
    problems: Counter = Counter()
    for tok in tokens:
        for key in tok:
            if key != COMPONENTS and not kinds[key]:
                problems[f"component at top level: {key}"] += 1
        components = tok.get(COMPONENTS)
        if components is None:
            continue
        if not components:
            problems["empty components"] += 1
        for key, value in components.items():
            if kinds[key]:
                problems[f"top-level key in components: {key}"] += 1
            if value is False and (tok.get("type"), key) not in FALSE_FLAG_OK:
                problems[f"false flag emitted: {tok.get('type')}.{key}"] += 1
            for text in value if isinstance(value, list) else [value]:
                if isinstance(text, str) and not text.strip():
                    # Absent means unknown; an empty string would be embedded.
                    problems[f"empty string component: {tok.get('type')}.{key}"] += 1
                if isinstance(text, str) and OBJECT_ID_IN_TEXT.search(text):
                    problems[f"object id in string component: {key}"] += 1
                if isinstance(text, str) and key not in ABILITY_TEXT_KEYS:
                    for what, pat in RUN_CONTEXT:
                        if pat.search(text):
                            problems[f"{what} in string component: {key}"] += 1
        problems.update(ability_text_problems(tok))
    return problems


def check_layout_dump(path: Path, limit: int) -> tuple[Counter, int]:
    """:func:`check_layout` over each token once (history deltas + state), plus
    an informational count of top-level strings (names, refs) carrying "Name
    (id)" -- e.g. Forge effect cards named "Screaming Nemesis (97)'s Effect"."""
    kinds = _Kinds()
    problems: Counter = Counter()
    ids_in_names = 0
    with path.open() as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = rec["hist_new"] + rec["state"]
            problems.update(check_layout(tokens, kinds))
            ids_in_names += sum(1 for tok in tokens for k, v in tok.items()
                                if k != COMPONENTS and isinstance(v, str) and OBJECT_ID_IN_TEXT.search(v))
    return problems, ids_in_names


def check_leaks(path: Path, limit: int) -> tuple[int, Counter]:
    """Counts tokens that expose information the viewer should not have."""
    leaks = 0
    offenders: Counter = Counter()
    replay: dict[tuple[int, str | None], _Replay] = {}
    for n, dec in enumerate(load(path)):
        if limit and n >= limit:
            break
        tokens = dec.tokens
        board = check_board_relations(tokens, resolve_edges(tokens))
        leaks += sum(board.values())
        offenders.update(board)
        viewer = _viewer_ref(dec.state)
        rp = replay.setdefault((dec.game, viewer), _Replay(viewer))
        rp.advance(dec.history)
        for raw, tok in zip(tokens, strip_ids(tokens), strict=True):
            if "id" in tok or "edges" in tok or "handles" in tok:
                leaks += 1
                offenders[f"id survived strip: {tok.get('type')}"] += 1
            zone = tok.get("zone")
            name = tok.get("name")
            # An unnamed card or role must not be identified some other way.
            for key in raw:
                if key.endswith("oracle_id") and is_hidden(raw, key[:-len("oracle_id")].rstrip("_") or None):
                    leaks += 1
                    offenders[f"{key} on hidden {raw.get('type')} in {zone}"] += 1
            if zone not in HIDDEN_OPPONENT_ZONES:
                continue
            single = tok.get("type") == "game_object" and comp(tok, "count") is None
            if zone in REVEALABLE_ZONES and comp(tok, "revealed") is True and single:
                continue
            if zone in KNOWN_BY_REVEAL_ZONES and comp(tok, "known_by_reveal") is True and single:
                if rp.explains(raw):
                    continue
                leaks += 1
                offenders[f"known_by_reveal without a recorded cause in {zone}: {name}"] += 1
                continue
            if not is_hidden(raw):
                leaks += 1
                offenders[f"unredacted name in {zone}: {name}"] += 1
        rp.observe(dec.state)
        leaks += sum(rp.problems.values())
        offenders.update(rp.problems)
        rp.problems.clear()
    return leaks, offenders


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only check this many records for leaks (0 = all)",
    )
    args = ap.parse_args()

    try:
        stats = summarise(args.dump)
    except SchemaError as exc:
        print(f"FAIL schema/history: {exc}", file=sys.stderr)
        return 1

    print(f"records            {stats.records}")
    print(f"games              {len(stats.games)}")
    print(f"tokens mean/max    {stats.mean_tokens:.1f} / {stats.max_tokens}")
    print(f"actions max        {stats.max_actions}")
    print(f"pass rate          {stats.pass_rate:.1%}")
    print(f"signature-matched  {stats.signature_matched}")
    print(f"distinct props     {len(stats.prop_keys)}")
    print("token types        " + ", ".join(
        f"{k}={v}" for k, v in sorted(stats.token_types.items(), key=lambda kv: -kv[1])
    ))

    leaks, offenders = check_leaks(args.dump, args.limit)
    layout, ids_in_names = check_layout_dump(args.dump, args.limit)
    print(f"ids in names/refs  {ids_in_names} (informational)")
    option_ids = check_option_ids(islice(load(args.dump), args.limit or None))

    failures = []
    if option_ids:
        failures.append(f"{len(option_ids)} options retain raw ids: {option_ids[0]}")
    if stats.records == 0:
        failures.append("dump is empty")
    if stats.unaddressable:
        failures.append(f"{stats.unaddressable} unaddressable chosen actions")
    if leaks:
        failures.append(f"{leaks} leaked fields: {dict(offenders)}")
    if layout:
        failures.append(f"{sum(layout.values())} layout problems: {dict(layout)}")

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1

    print("\nOK schema, history reconstruction, leak boundary, token layout, option ids, label usability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
