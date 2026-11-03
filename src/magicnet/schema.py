"""The zone and token-type vocabularies the encoder indexes.

Deliberately hard-coded rather than derived from Java at import time: the
encoder's segment embeddings are positional with respect to these lists, so
they must be pinned to whatever a checkpoint was trained against, not silently
follow whatever the bridge currently emits. ``tests/test_schema.py`` asserts
this copy matches ``magicnet.bridge.Schema``, so drift fails a test instead of
scrambling zone semantics at runtime.

Changing the order is a schema change and invalidates existing checkpoints.
"""

from __future__ import annotations

SCHEMA = 14

# Schema 14: a lost handle never returns to a live token; with a known identity
# it is assigned to an identity phantom (no ``ceased`` flag), and component text
# carries no bracketed Forge run context or player names.

# Schema 13: history redesign. One token per event: ``verb`` and its modifiers
# are components; participants are edges whose relation is the role, pointing
# at per-viewer tracking handles (``h<N>``, listed in the ``handles`` of the one
# state token each is assigned to), events (``e<N>``, a history token's ``id``)
# or players. See magicnet.dump and forge-bridge/README.md.

# Schema 12: a card the viewer may not see has no ``name``/``oracle_id`` at all
# (history roles keep only ``<role>_id``); absence means unknown. See
# magicnet.dump.is_hidden.

# Schema 9: every model input on a token lives in this sub-dictionary; the top
# level holds only type, zone, edges and identity/ref keys (see keys.json and
# magicnet.bridge.Schema.isTopLevel). Omitted when a token has no components.
COMPONENTS = "components"

ZONES: tuple[str, ...] = (
    "self_battlefield",
    "self_hand",
    "self_graveyard",
    "self_exile",
    "self_command",
    "self_library",
    # Schema 4: own sideboard named, opponent's redacted unless revealed.
    # Schema 5: opponent's hidden sideboard cards are one ``library_card``
    # aggregate (``count``, no name since schema 12); opponent hand cards the viewer was
    # shown are named and marked ``known_by_reveal``.
    "self_sideboard",
    "opp_battlefield",
    "opp_hand",
    "opp_graveyard",
    "opp_exile",
    "opp_command",
    "opp_library",
    "opp_sideboard",
    "self_merged_under",
    "opp_merged_under",
    "stack",
    "delayed",
    # Not emitted in `state` since schema 3: the encoder builds candidate
    # tokens from Decision.actions and assigns them this zone and type.
    "actions",
    "history",
    # Schema 13: last-known characteristics of objects that ceased to exist
    # (tokens off the battlefield, copies and abilities off the stack), kept
    # while a history edge refers to them.
    "phantom",
    # Schema 13: per-side holder of tracking handles whose object the viewer
    # can no longer place in a single zone.
    "self_untracked",
    "opp_untracked",
)

TOKEN_TYPES: tuple[str, ...] = (
    "global",
    "player",
    "game_object",
    "library_card",
    "stack_item",
    "delayed_trigger",
    "action",
    "history",
)

# ``global`` and ``player`` tokens carry ``zone: null``. They need a real slot
# -- a "no zone" embedding -- rather than being treated as a missing lookup.
NO_ZONE = "<none>"

# Relation types, indexing the relation-aware attention bias (Shaw et al. 2018
# ``a^K``). Order is load-bearing in the same way ZONES is, but the failure is
# nastier: a mismatch permutes relation semantics, so the model reads "blocks"
# as "attacks" and partially compensates, rather than erroring.
#
# Relations are over ordered ``(query, key)`` pairs. Inverse directions are
# separate ids by design, not inferred.
RELATIONS: tuple[str, ...] = (
    "source_is",
    "targeting",
    "targeting_spell",
    "attacks",
    "attacking",
    "defender",
    "blocks",
    "blocking",
    "blocked",
    "blocked_by",
    "unblocked",
    "attached_to",
    "merged_under",
    # Emitted only when a permanent's owner differs from its controller. Absence
    # means owner == controller, which is unambiguous because the zone bucket
    # already encodes the controller. Appended last so existing relation indices
    # are unchanged.
    "owner_is",
    # Schema 7: source of an active continuous static effect -> each visible
    # object/player it applies to, carried on the source token. Both ends are
    # visible to the viewer; no self-loops. Appended last.
    "affects",
    # Schema 13: history roles (edges from an event token). Reused with the
    # same meaning: source_is (source card), attacks (attacker), defender,
    # blocks (blocker), attached_to (attach host). blocked/unblocked are no
    # longer emitted.
    "actor",
    "player",
    "obj",
    "stack_object",
    "target_0",
    "target_1",
    "target_2",
    "target_3",
    "target_more",
    "new_controller",
    "old_controller",
    "cause",
    "replaced_by",
    # State side: object -> static source (reverse of affects) or the history
    # resolve event whose continuous effect still applies to it.
    "affected_by",
)

# Schema 13: closed vocabularies of ``categorical`` component keys, by the
# ``vocab`` name keys.json gives them. Order is load-bearing (embedding index).
HISTORY_VERBS: tuple[str, ...] = (
    "move_zone", "draw", "discard", "play_land", "cast", "activate", "special_action",
    "trigger", "resolve", "does_not_resolve", "copy_spell", "declare_attacker",
    "declare_blockers", "deal_damage", "life_change", "counters_change", "create",
    "change_control", "reveal", "scry", "surveil", "shuffle", "mulligan",
    "designation_change", "attach", "attempt_end",
)
MOVE_KINDS: tuple[str, ...] = (
    "put", "mill", "exile", "sacrifice", "destroy", "bounce", "search", "return", "other",
)
DNR_REASONS: tuple[str, ...] = ("countered", "fizzled", "removed")
# ``pointer``: the event is a consequence of the events its ``cause`` edges
# point at. Otherwise the rule category that produced it. state_based events
# may also carry a ``cause`` edge to the damage that made them lethal.
CAUSE_KINDS: tuple[str, ...] = (
    "pointer", "turn_based", "state_based", "combat", "pregame", "player_action", "unknown",
)
# Forge PhaseType names, then NONE (no current phase: before the first turn).
PHASES: tuple[str, ...] = (
    "UNTAP", "UPKEEP", "DRAW", "MAIN1", "COMBAT_BEGIN", "COMBAT_DECLARE_ATTACKERS",
    "COMBAT_DECLARE_BLOCKERS", "COMBAT_FIRST_STRIKE_DAMAGE", "COMBAT_DAMAGE", "COMBAT_END",
    "MAIN2", "END_OF_TURN", "CLEANUP", "NONE",
)

VOCABS: dict[str, tuple[str, ...]] = {
    "zone": ZONES,
    "verb": HISTORY_VERBS,
    "move_kind": MOVE_KINDS,
    "dnr_reason": DNR_REASONS,
    "cause_kind": CAUSE_KINDS,
    "phase": PHASES,
}

VOCAB_LOOKUP: dict[str, str] = {
    "zone": "zone",
    "to": "zone",
    "from": "zone",
    "verb": "verb",
    "move_kind": "move_kind",
    "dnr_reason": "dnr_reason",
    "cause_kind": "cause_kind",
    "phase": "phase",
}


def vocab_index(vocab: str, value: str) -> int:
    """Index of a categorical value in its closed vocabulary; raises on unknown."""
    try:
        return VOCABS[vocab].index(value)
    except (KeyError, ValueError):
        raise KeyError(
            f"unknown {vocab} value {value!r}; magicnet.schema predates the bridge. "
            f"Re-run tests/test_schema.py to see the current vocabulary."
        ) from None

_ZONE_INDEX = {name: i for i, name in enumerate(ZONES)}
_ZONE_INDEX[NO_ZONE] = len(ZONES)
_TYPE_INDEX = {name: i for i, name in enumerate(TOKEN_TYPES)}
_RELATION_INDEX = {name: i for i, name in enumerate(RELATIONS)}

ZONE_VOCAB_SIZE = len(ZONES) + 1
TYPE_VOCAB_SIZE = len(TOKEN_TYPES)
RELATION_VOCAB_SIZE = len(RELATIONS)


def relation_index(relation: str) -> int:
    """Index into the relation embedding table.

    Closed vocabulary, so an unknown relation raises. Unlike counter types --
    where new sets legitimately introduce new names -- a relation the bridge
    emits but Python does not know is a drift bug, and folding it into an UNK
    bucket would train the model on a mislabelled graph.
    """
    try:
        return _RELATION_INDEX[relation]
    except KeyError:
        raise KeyError(
            f"unknown relation {relation!r}; magicnet.schema predates the "
            f"bridge. Re-run tests/test_schema.py to see the current vocabulary."
        ) from None


def zone_index(zone: str | None) -> int:
    """Segment-embedding index for a token's ``zone`` field.

    Raises on an unknown zone rather than folding it into an UNK bucket: a zone
    the encoder has never seen means the bridge emitted something this
    vocabulary predates, and silently embedding it as "unknown" would train on
    a misrepresented board.
    """
    key = NO_ZONE if zone is None else zone
    try:
        return _ZONE_INDEX[key]
    except KeyError:
        raise KeyError(
            f"unknown zone {zone!r}; magicnet.schema predates the bridge. "
            f"Re-run tests/test_schema.py to see the current vocabulary."
        ) from None


def type_index(token_type: str) -> int:
    """Index for a token's ``type`` field."""
    try:
        return _TYPE_INDEX[token_type]
    except KeyError:
        raise KeyError(
            f"unknown token type {token_type!r}; magicnet.schema predates the "
            f"bridge. Re-run tests/test_schema.py to see the current vocabulary."
        ) from None
