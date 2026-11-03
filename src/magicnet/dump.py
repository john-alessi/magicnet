"""Load the Forge bridge's line-delimited JSON decision dump.

Each line emitted by ``magicnet.bridge.Serializer`` is one *decision*: the game
state as the acting player can see it, the legal action pool, and which action
was taken. This module turns that stream back into whole decisions and checks
the invariants that make the dump trustworthy as training data.

History is delta-encoded
------------------------
Game history is cumulative and append-only, so writing it in full on every
record makes the dump quadratic in game length (measured: 45 MB for two games,
versus 12 MB delta-encoded). Each record therefore carries only the history
tokens added since the previous record for the same ``(game, viewer)``, and this
loader reassembles the running history.

``hist_total`` is the authoritative length after appending. Checking the
reassembled history against it on every record turns a desync -- a dropped
record, an interleaved writer, a truncated file -- into an immediate, loud
failure instead of silently corrupted context.

Token layout (schema 9)
-----------------------
A token's top level holds only what is *not* a model input: ``type``, ``zone``,
``edges``, and the keys ``catalog/keys.json`` classifies ``identity`` (join keys
such as ``id``/``*_id``/``oracle_id``, card ``name``) or ``ref`` (references to
other objects or players such as ``source``, ``card``, ``*_player``,
``target_<i>``). Every model input -- numbers, flags, strings, string lists --
is under ``components``, which is omitted when empty. Flags appear only when
true. Read components with :func:`comp`; :func:`flatten` gives the old flat
view::

    {"type": "game_object", "zone": "self_battlefield", "id": "c37",
     "name": "Soul-Scar Mage", "oracle_id": "...", "edges": [...],
     "components": {"power": 6, "toughness": 5, "tapped": true, "ctr_+1/+1": 1}}

Hidden cards have no name (schema 12)
-------------------------------------
A card the viewer may not see carries no ``name`` and no ``oracle_id``: per
the rules an unseen or face-down card has no name, and an absent key means
unknown. That holds for state tokens (redacted hand/sideboard cards, face-down
opponent permanents, the ``count`` aggregates of an opponent's unseen hand
(schema 13), library and sideboard). Test with :func:`is_hidden`, never by
comparing names.

History events and tracking handles (schema 13)
-----------------------------------------------
Each history token is one event: ``verb`` (``move_zone``, ``draw``, ``cast``,
``trigger``, ``resolve`` ...) and its modifiers are components, plus ``turn``,
``phase``, ``step``/``step_order`` and ``cause_kind`` on every one. Its
participants are ``edges`` whose relation is the role (``actor``, ``obj``,
``stack_object``, ``source_is``, ``target_<i>``, ``cause`` ...), pointing at:

* ``e<N>``  -- another history event (a history token's top-level ``id``);
* ``p<N>``  -- a player token;
* ``h<N>``  -- a per-viewer *tracking handle*. History is append-only, but a
  handle follows its object across zone changes: every state lists each handle
  referenced so far in the top-level ``handles`` of exactly one token -- the
  object's own token, the aggregate of the hidden zone it is known to be in, the
  ``phantom`` token once it ceased to exist (``ceased: true``), a ``phantom``
  of its identity once the viewer lost track of a card whose identity it knew
  (schema 14: "a copy of X"; never re-linked to a card seen later), or the
  owner's ``*_untracked`` token if it never knew the identity. Phantoms exist
  only while some edge refers to them.

A participant the viewer cannot identify has no edge at all. History tokens
carry no card name or id.

Ids are join keys, not features
-------------------------------
Token ``id`` properties exist only to resolve ``edges`` to token indices. They
correlate with library order and would leak the opponent's decklist if embedded.
:func:`resolve_edges` rewrites edges to indices, and :func:`strip_ids` removes
them; callers building tensors must do both.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from magicnet.schema import COMPONENTS

# Bumped by the Java side on any change that alters the meaning of an existing
# field. Refuse unknown schemas rather than silently misreading them.
SUPPORTED_SCHEMA = 14

# ``api`` of the explicit "pass priority" option. Present (always last) only on
# decisions where passing is legal; there is no sentinel value for passing.
PASS_API = "Pass"

# Properties that must never reach the model.
#
# ``id`` is the explicit join key and ``handles`` (schema 13) lists the tracking
# handles a token stands for. Raw card references are additionally stripped by
# *value* rather than by key, because a new key holding one would otherwise
# leak silently.
#
# Card ids are doubly disqualified: they correlate with library order (so they
# leak the decklist), and they are **not reproducible** across runs of the same
# seed, so a model consuming them would be fitting run-to-run noise.
#
# Player ids (``p0``/``p1``) are deliberately kept. There are only two, they are
# stable, and ``from_player``/``to_player``/``actor`` are meaningless without
# them.
_LEAK_PROPS = frozenset({"id", "handles"})

# A card reference as emitted by StateExtractor's namespaced ids: "c" + digits.
_CARD_REF = re.compile(r"^c\d+$")


def _is_card_ref(value: object) -> bool:
    return isinstance(value, str) and _CARD_REF.match(value) is not None


# Token types that stand for cards and are named when the viewer can see them.
CARD_TOKEN_TYPES = frozenset({"game_object", "library_card"})


def is_hidden(tok: dict, role: str | None = None) -> bool:
    """Whether *tok* is a card the viewer cannot see: a card-type state token
    without ``name`` (schema 12).

    With *role* (e.g. ``"source"`` for a stack item's ``source`` /
    ``source_oracle_id``), whether that named reference is absent. History
    tokens carry no names at all since schema 13.
    """
    if role is None:
        return tok.get("type") in CARD_TOKEN_TYPES and "name" not in tok
    return role not in tok and role not in tok.get(COMPONENTS, {})


def comp(tok: dict, key: str, default=None):
    """A token's component value, or ``default`` when absent (absent flag = false)."""
    return tok.get(COMPONENTS, {}).get(key, default)


def flatten(tok: dict) -> dict:
    """The token with its components merged into the top level (the pre-schema-9
    flat layout). Keys never collide: a key is either top-level or a component."""
    out = {k: v for k, v in tok.items() if k != COMPONENTS}
    out.update(tok.get(COMPONENTS, {}))
    return out


class SchemaError(RuntimeError):
    """The dump does not match what this loader knows how to read."""


@dataclass
class Decision:
    """One decision point, with history already reassembled."""

    game: int
    decision: int
    viewer: str
    turn: int
    phase: str
    # What is being asked: "ability" for a priority decision, otherwise the
    # callback being routed ("declare_attacker", "declare_blocker", ...).
    kind: str
    # Full cumulative history for this (game, viewer), oldest first.
    history: list[dict]
    # Non-history state tokens for this decision only.
    state: list[dict]
    # Legal action pool, in the order the policy indexes into.
    actions: list[dict]
    # Index into ``actions`` of the option taken. Passing is an explicit
    # option (see PASS_API); -1 only marks a live request with no outcome yet.
    chosen: int
    # How the chosen action was resolved: identity, signature, absent or pass.
    chosen_match: str

    @property
    def tokens(self) -> list[dict]:
        """Model input: history followed by current state."""
        return self.history + self.state

    @property
    def pass_index(self) -> int | None:
        """Index of the explicit pass option, or None if passing is not legal."""
        for i, opt in enumerate(self.actions):
            if opt.get("api") == PASS_API:
                return i
        return None

    @property
    def is_pass(self) -> bool:
        return self.chosen >= 0 and self.chosen == self.pass_index

    @property
    def is_usable_label(self) -> bool:
        """Whether ``chosen`` can be trusted as a supervised target.

        ``absent`` means the acting player played something that was not in the
        pool we exposed, so there is no index that denotes it. Such records are
        unreplayable and must not be used as labels.
        """
        return self.chosen_match != "absent"


class HistoryReassembler:
    """Rebuilds cumulative history from delta-encoded records.

    History is delta encoded because it is append-only and resending it in full
    is quadratic (measured: 45 MB for two games, ~23 GB per thousand). The cost
    of that is statefulness, which both the offline loader and the live agent
    must carry identically -- so they carry it here rather than each keeping
    their own copy of a subtle invariant.

    One running history is kept per ``(game, viewer)``: a player's history is
    their own information set, and the two viewers legitimately differ.
    """

    def __init__(self) -> None:
        self._histories: dict[tuple[int, str], list[dict]] = defaultdict(list)

    def absorb(self, rec: dict, where: str = "record") -> Decision:
        """Fold one record's delta in and return the Decision it denotes.

        Raises:
            SchemaError: if the record does not continue exactly where this
                reassembler left off, which means records were lost, duplicated
                or reordered. This is checked rather than tolerated because
                every later context for that viewer would otherwise be silently
                wrong, and a model would train on it without complaint.
        """
        key = (rec["game"], rec["viewer"])
        running = self._histories[key]

        hist_from = rec["hist_from"]
        if hist_from != len(running):
            raise SchemaError(
                f"{where}: history desync for {key}: record starts at "
                f"{hist_from}, reassembler has {len(running)}"
            )

        running.extend(rec["hist_new"])
        if len(running) != rec["hist_total"]:
            raise SchemaError(
                f"{where}: history length {len(running)} != "
                f"hist_total {rec['hist_total']}"
            )

        return Decision(
            game=rec["game"],
            decision=rec["decision"],
            viewer=rec["viewer"],
            kind=rec.get("kind", "ability"),
            turn=rec["turn"],
            phase=rec["phase"],
            history=list(running),
            state=rec["state"],
            actions=rec["actions"],
            # A live request has no outcome yet. -1 would be indistinguishable
            # from a recorded pass, so it is marked explicitly as pending.
            chosen=rec.get("chosen", -1),
            chosen_match=rec.get("chosen_match", "pending"),
        )


def load(path: str | Path) -> Iterator[Decision]:
    """Stream decisions from a dump, reassembling delta-encoded history.

    Yields lazily so that dumps larger than memory can be processed; the only
    retained state is one running history per ``(game, viewer)``.

    Raises:
        SchemaError: on an unsupported schema or a history desync.
    """
    path = Path(path)
    reassembler = HistoryReassembler()

    with path.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno}: malformed JSON: {exc}") from exc

            schema = rec.get("schema")
            if schema != SUPPORTED_SCHEMA:
                raise SchemaError(
                    f"{path}:{lineno}: schema {schema!r}, expected {SUPPORTED_SCHEMA}"
                )

            yield reassembler.absorb(rec, where=f"{path}:{lineno}")


def resolve_edges(tokens: list[dict]) -> list[list[tuple[str, int]]]:
    """Resolve each token's edges from id references to token indices.

    Returns one list of ``(relation, target_index)`` per token. Edges whose
    target is not present are dropped rather than raising: a token can legally
    reference something outside the viewer's information set, and silently
    omitting an unseen referent is the correct redaction behaviour.
    """
    index_of: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        tok_id = tok.get("id")
        if tok_id is not None:
            index_of[str(tok_id)] = i
        # Schema 13: a tracking handle resolves to the one token it is assigned to.
        for handle in tok.get("handles", ()):
            index_of[handle] = i

    out: list[list[tuple[str, int]]] = []
    for tok in tokens:
        resolved = []
        for edge in tok.get("edges", ()):
            target = index_of.get(edge["to"])
            if target is not None:
                resolved.append((edge["rel"], target))
        out.append(resolved)
    return out


def strip_ids(tokens: list[dict]) -> list[dict]:
    """Drop join keys and raw edges, leaving only model-visible fields.

    Call after :func:`resolve_edges`. Edges are removed here because they carry
    id strings; the resolved index form returned by :func:`resolve_edges` is
    what downstream code should use.

    Card references are dropped wherever they appear, not just under ``id``;
    ``handles`` (schema 13) go too. History events reach board objects through
    their edges, which :func:`resolve_edges` has already turned into indices.
    """
    cleaned = []
    for tok in tokens:
        out = {
            k: v
            for k, v in tok.items()
            if k not in _LEAK_PROPS and k not in ("edges", COMPONENTS) and not _is_card_ref(v)
        }
        # Components are never card refs by construction (refs are top-level),
        # but a ref that slipped in -- e.g. the polymorphic ``from``/``to`` of
        # CardAttachment -- must not survive either.
        components = {k: v for k, v in tok.get(COMPONENTS, {}).items() if not _is_card_ref(v)}
        if components:
            out[COMPONENTS] = components
        cleaned.append(out)
    return cleaned


def canonical(tokens: list[dict]) -> list[tuple]:
    """Render tokens in a form comparable across runs.

    Object ids are **not reproducible**: two runs of the same seed can fetch a
    different physical instance of an identical card (measured: a Fabled Passage
    search returning ``c37`` in one run and ``c42`` in another, with every other
    field equal). Comparing raw records would call that a divergence when the
    games are functionally identical.

    Edges are therefore resolved to token *indices* and ids dropped, which is
    exactly the transformation applied before tensorisation -- so this compares
    what the model would actually see. The non-reproducibility of ids is a second,
    independent reason they must never become features: a model trained on them
    would be fitting run-to-run noise.
    """
    edges = resolve_edges(tokens)
    cleaned = strip_ids(tokens)
    return [
        (tuple(sorted(_sorted_components(tok).items(), key=lambda kv: kv[0])), tuple(sorted(edge)))
        for tok, edge in zip(cleaned, edges, strict=True)
    ]


def _sorted_components(tok: dict) -> dict:
    if COMPONENTS not in tok:
        return tok
    return {**tok, COMPONENTS: tuple(sorted(tok[COMPONENTS].items(), key=lambda kv: kv[0]))}


def resolve_option_edges(decision: "Decision") -> list[list[tuple[str, int]]]:
    """Resolve each option's object references to indices into ``tokens``.

    Options name the game objects they concern -- the creature being declared as
    an attacker, the permanent being fetched -- so a pointer-style head can
    attend to those objects' own representations rather than recovering them
    from the description string.

    Returns one list of ``(relation, token_index)`` per option, in option order.
    Unresolvable references are dropped for the same reason as in
    :func:`resolve_edges`: an option may legitimately refer to something outside
    the viewer's information set.
    """
    index = {
        tok["id"]: i for i, tok in enumerate(decision.tokens) if tok.get("id") is not None
    }
    resolved = []
    for opt in decision.actions:
        here = []
        for edge in opt.get("edges", []) or []:
            target = index.get(edge.get("to"))
            if target is not None:
                here.append((edge.get("rel", ""), target))
        resolved.append(here)
    return resolved


def strip_option_ids(actions: list[dict]) -> list[dict]:
    """Drop join keys from options, leaving only model-visible fields.

    Call after :func:`resolve_option_edges`. Mirrors :func:`strip_ids`, and
    exists because options are a separate array from tokens and were therefore
    not covered by it -- which is how an unstripped raw id reached the model
    once already.
    """
    return [
        {
            k: v
            for k, v in opt.items()
            if k not in _LEAK_PROPS and k != "edges" and not _is_card_ref(v)
        }
        for opt in actions
    ]


@dataclass
class DumpStats:
    """Summary of a dump, for sanity-checking before training on it."""

    records: int = 0
    games: set[int] = field(default_factory=set)
    passes: int = 0
    unaddressable: int = 0
    signature_matched: int = 0
    max_tokens: int = 0
    max_actions: int = 0
    total_tokens: int = 0
    token_types: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prop_keys: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def mean_tokens(self) -> float:
        return self.total_tokens / self.records if self.records else 0.0

    @property
    def pass_rate(self) -> float:
        return self.passes / self.records if self.records else 0.0


def summarise(path: str | Path) -> DumpStats:
    """Walk a dump and collect shape statistics and vocabulary coverage."""
    stats = DumpStats()
    for dec in load(path):
        stats.records += 1
        stats.games.add(dec.game)
        if dec.is_pass:
            stats.passes += 1
        if dec.chosen_match == "absent":
            stats.unaddressable += 1
        elif dec.chosen_match == "signature":
            stats.signature_matched += 1

        tokens = dec.tokens
        stats.total_tokens += len(tokens)
        stats.max_tokens = max(stats.max_tokens, len(tokens))
        stats.max_actions = max(stats.max_actions, len(dec.actions))
        for tok in tokens:
            stats.token_types[tok.get("type", "?")] += 1
            for key in flatten(tok):
                stats.prop_keys[key] += 1
    return stats
