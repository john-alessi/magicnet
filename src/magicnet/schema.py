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

SCHEMA = 11

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
    # aggregate (name HIDDEN, ``count``); opponent hand cards the viewer was
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
)

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
