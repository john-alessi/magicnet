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
    SchemaError, comp, flatten, load, resolve_edges, strip_ids, strip_option_ids, summarise,
)
from magicnet.catalog.keys import load_registry
from magicnet.schema import COMPONENTS

# Zones whose contents are hidden from a player who does not own them. The
# extractor redacts these to a sentinel name; a *real* card name appearing here
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

# What the extractor writes in place of a card name it must not reveal.
REDACTED = "HIDDEN"


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


# Zones a card can leave in full view. A move from one of these into a hand is
# public, so the history names it (unless it was face-down) and the viewer may
# keep knowing it while it stays there.
PUBLIC_ZONES = frozenset({"Battlefield", "Graveyard", "Stack", "Exile", "Command"})
HIDDEN = "<hidden>"

# State zones whose tokens tell us whether the viewer could see a card there. A
# public zone redacts a card to HIDDEN only when it is face-down and the viewer
# may not look at it, so the name in the viewer's latest state is independent
# evidence of whether a face-down departure was visible to them.
_PUBLIC_STATE_ZONES = frozenset(
    f"{side}_{z}" for side in ("self", "opp")
    for z in ("battlefield", "graveyard", "exile", "command")
) | {"stack"}


class _HandKnowledge:
    """Replays one viewer's history: which opponent hand cards it accounts for.

    A card is explained while in hand by:

    * a ``Reveal`` shown to the viewer while it was in that hand;
    * a named ``CardChangeZone`` into the hand from a public zone -- unless the
      card left face-down and the viewer's latest state before the move did not
      name it (then the name itself is a leak, see ``problems``);
    * a ``Reveal`` shown to the viewer outside a hand, followed -- as that card's
      next move, before the effect's ``SpellResolved`` -- by a move from the
      revealed zone into a hand (a dig like Silundi Vision).

    Any other move of the card revokes it, so a card that left and came back is
    unexplained again. History is append-only per (game, viewer), so it is
    consumed incrementally and interleaved with the states that followed it.
    """

    def __init__(self, viewer: str | None) -> None:
        self.viewer = viewer
        self.known: dict[str, str] = {}
        self.pending: dict[str, tuple[str, str]] = {}
        self.looked: dict[str, bool] = {}
        self.consumed = 0
        self.problems: Counter = Counter()

    def advance(self, history: list[dict]) -> None:
        if len(history) < self.consumed:
            raise SchemaError("history shrank between records of one viewer")
        for ev in history[self.consumed:]:
            self._step(ev)
        self.consumed = len(history)

    def observe(self, state: list[dict]) -> None:
        for tok in state:
            if tok.get("zone") in _PUBLIC_STATE_ZONES and "id" in tok and "name" in tok:
                self.looked[tok["id"]] = tok["name"] != REDACTED

    def _step(self, ev: dict) -> None:
        kind = comp(ev, "event")
        if kind == "SpellResolved":
            self.pending.clear()
            return
        ref = ev.get("card_id")
        if ref is None:
            return
        if kind == "Reveal" and ev.get("shown_to") == self.viewer:
            if comp(ev, "revealed_from") == "Hand":
                self.known[ref] = ev.get("card")
            else:
                self.pending[ref] = (comp(ev, "revealed_from"), ev.get("card"))
        elif kind == "CardChangeZone":
            self.known.pop(ref, None)
            was = self.pending.pop(ref, None)
            named = ev.get("card") != HIDDEN
            src, dst, face_down = comp(ev, "from"), comp(ev, "to"), comp(ev, "face_down")
            seen = src in PUBLIC_ZONES and (not face_down or self.looked.get(ref) is True)
            if (named and not seen and face_down
                    and dst in ("Hand", "Library")
                    and ev.get("to_player") != self.viewer):
                self.problems[f"face-down move named: {ev.get('card')}"] += 1
            if dst != "Hand":
                return
            if was is not None and was[0] == src:
                self.known[ref] = was[1]
            elif seen and named:
                self.known[ref] = ev.get("card")


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
            and tok.get("name") not in (None, REDACTED)
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
        chosen = CHOICE_PROPS.intersection(flatten(tok))
        if chosen and (comp(tok, "face_down") or tok.get("name") in (None, REDACTED)):
            offenders[f"choice on face-down/redacted token: {sorted(chosen)}"] += 1
        if "chosen_player" in tok and tok["chosen_player"] not in players:
            offenders["chosen_player is not a player token"] += 1
    return offenders


# Value kinds that stay at a token's top level (schema 9); everything else is a
# component.
TOP_LEVEL_KINDS = frozenset({"identity", "ref"})

# Flags whose false value is emitted on purpose: an attacker's sticky
# ``blocked`` status (CR 509.1h), and CardPhased's ``phased_out``, which is the
# direction of the phase (false = phased in).
FALSE_FLAG_OK = frozenset({("game_object", "blocked"), ("history", "phased_out")})

# Forge renders a card as "Name (id)"; the id is not reproducible across runs.
OBJECT_ID_IN_TEXT = re.compile(r"\w \(\d+\)")

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
    if host in (None, "", REDACTED) or comp(tok, "token") or "Basic" in types:
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

    def __missing__(self, key: str) -> str:
        kind = self[key] = self.registry.resolve(key)[1].value_kind
        return kind


def check_layout(tokens: list[dict], kinds: _Kinds | None = None) -> Counter:
    """Schema-9 layout: identity/ref keys top-level, model inputs under
    ``components`` (non-empty when present), flags only when true, and no
    run-unstable object id inside a string component."""
    kinds = kinds if kinds is not None else _Kinds()
    problems: Counter = Counter()
    for tok in tokens:
        for key in tok:
            if key != COMPONENTS and kinds[key] not in TOP_LEVEL_KINDS:
                problems[f"component at top level: {key}"] += 1
        components = tok.get(COMPONENTS)
        if components is None:
            continue
        if not components:
            problems["empty components"] += 1
        for key, value in components.items():
            if kinds[key] in TOP_LEVEL_KINDS:
                problems[f"identity/ref key in components: {key}"] += 1
            if value is False and (tok.get("type"), key) not in FALSE_FLAG_OK:
                problems[f"false flag emitted: {tok.get('type')}.{key}"] += 1
            for text in value if isinstance(value, list) else [value]:
                if isinstance(text, str) and OBJECT_ID_IN_TEXT.search(text):
                    problems[f"object id in string component: {key}"] += 1
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
    replay: dict[tuple[int, str | None], _HandKnowledge] = {}
    for n, dec in enumerate(load(path)):
        if limit and n >= limit:
            break
        tokens = dec.tokens
        board = check_board_relations(tokens, resolve_edges(tokens))
        leaks += sum(board.values())
        offenders.update(board)
        viewer = _viewer_ref(dec.state)
        hk = replay.setdefault((dec.game, viewer), _HandKnowledge(viewer))
        hk.advance(dec.history)
        granted = hk.known
        for raw, tok in zip(tokens, strip_ids(tokens), strict=True):
            if "id" in tok or "edges" in tok:
                leaks += 1
                offenders[f"id survived strip: {tok.get('type')}"] += 1
            zone = tok.get("zone")
            name = tok.get("name")
            event = comp(raw, "event")
            if (event == "CardChangeZone" and comp(raw, "to") in ("Hand", "Library")
                    and raw.get("to_player") != viewer and comp(raw, "from") not in PUBLIC_ZONES
                    and raw.get("card") != HIDDEN):
                # Into an opponent's hidden zone from a hidden one: nobody else
                # saw which card moved. (A reveal is a separate Reveal token.)
                leaks += 1
                offenders[f"hidden move named: {raw.get('card')}"] += 1
                continue
            if event == "Reveal":
                # A bystander learns that a reveal happened, not what it showed.
                # The owner already knows their own cards -- except in their
                # library, which is hidden from them too.
                owner_sees = raw.get("owner") == viewer and comp(raw, "revealed_from") != "Library"
                if (raw.get("card") != "<hidden>" and raw.get("shown_to") != viewer
                        and not owner_sees):
                    leaks += 1
                    offenders[f"Reveal named to a bystander: {raw.get('card')}"] += 1
                continue
            if zone not in HIDDEN_OPPONENT_ZONES:
                continue
            single = tok.get("type") == "game_object" and comp(tok, "count") is None
            if zone in REVEALABLE_ZONES and comp(tok, "revealed") is True and single:
                continue
            if zone in KNOWN_BY_REVEAL_ZONES and comp(tok, "known_by_reveal") is True and single:
                if granted.get(raw.get("id")) == name:
                    continue
                leaks += 1
                offenders[f"known_by_reveal without a recorded cause in {zone}: {name}"] += 1
                continue
            if name is not None and name != REDACTED:
                leaks += 1
                offenders[f"unredacted name in {zone}: {name}"] += 1
            elif "oracle_id" in tok:
                leaks += 1
                offenders[f"oracle_id on redacted card in {zone}"] += 1
        hk.observe(dec.state)
    for hk in replay.values():
        leaks += sum(hk.problems.values())
        offenders.update(hk.problems)
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
