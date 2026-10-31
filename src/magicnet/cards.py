"""Parse Scryfall bulk-data dumps into one deduplicated text record per unique card.

Scryfall's "default_cards" bulk file has one row per *printing*, so popular cards
(e.g. "Forest") appear hundreds of times. We only want one embedding input per
distinct card, so we deduplicate on ``oracle_id`` -- Scryfall's stable identifier
for a card's rules text, shared across all reprints.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Layouts that aren't real, playable single cards (tokens, emblems, art cards,
# Archenemy/Planechase-only objects). Everything else (normal, transform,
# modal_dfc, adventure, split, flip, meld, saga, class, case, mutate,
# prototype, augment, host, prepare, reversible_card, ...) is kept.
_EXCLUDED_LAYOUTS = frozenset(
    {
        "token",
        "double_faced_token",
        "emblem",
        "art_series",
        "vanguard",
        "scheme",
        "planar",
    }
)


@dataclass
class CardRecord:
    """One unique (deduplicated) card, ready to be turned into embedding input."""

    oracle_id: str
    name: str
    mana_cost: str
    cmc: float
    type_line: str
    colors: list[str] = field(default_factory=list)
    color_identity: list[str] = field(default_factory=list)
    layout: str = "normal"
    text: str = ""  # the constructed string that gets sent to the embedding model

    def to_json(self) -> dict:
        return {
            "oracle_id": self.oracle_id,
            "name": self.name,
            "mana_cost": self.mana_cost,
            "cmc": self.cmc,
            "type_line": self.type_line,
            "colors": self.colors,
            "color_identity": self.color_identity,
            "layout": self.layout,
            "text": self.text,
        }


def _face_text(face: dict) -> str:
    parts = [face.get("name", "")]
    mana_cost = face.get("mana_cost") or ""
    if mana_cost:
        parts.append(mana_cost)
    if face.get("type_line"):
        parts.append(face["type_line"])
    if face.get("oracle_text"):
        parts.append(face["oracle_text"])
    if face.get("power") is not None and face.get("toughness") is not None:
        parts.append(f"Power/Toughness: {face['power']}/{face['toughness']}")
    if face.get("loyalty") is not None:
        parts.append(f"Loyalty: {face['loyalty']}")
    if face.get("defense") is not None:
        parts.append(f"Defense: {face['defense']}")
    return " | ".join(p for p in parts if p)


def build_embedding_text(card: dict) -> str:
    """Build the text string that gets sent to the embedding model for one card."""
    faces = card.get("card_faces")
    if faces:
        return "\n//\n".join(_face_text(face) for face in faces)
    return _face_text(card)


def _is_wanted(card: dict, *, include_digital: bool) -> bool:
    if card.get("lang") != "en":
        return False
    if card.get("layout") in _EXCLUDED_LAYOUTS:
        return False
    if card.get("set_type") == "memorabilia":
        return False
    if not include_digital and card.get("digital"):
        return False
    if not card.get("oracle_id"):
        # A handful of un-set/joke or malformed objects lack an oracle_id.
        return False
    return True


def iter_unique_cards(
    bulk_path: Path, *, include_digital: bool = False
) -> Iterator[CardRecord]:
    """Stream a Scryfall bulk-data ``.jsonl.gz`` file, yielding one record per
    unique ``oracle_id`` (first printing encountered wins; oracle text is
    identical across printings for a given oracle_id)."""
    seen: set[str] = set()
    with gzip.open(bulk_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            if not _is_wanted(card, include_digital=include_digital):
                continue
            oracle_id = card["oracle_id"]
            if oracle_id in seen:
                continue
            seen.add(oracle_id)
            yield CardRecord(
                oracle_id=oracle_id,
                name=card.get("name", ""),
                mana_cost=card.get("mana_cost", "") or "",
                cmc=card.get("cmc", 0.0) or 0.0,
                type_line=card.get("type_line", "") or "",
                colors=card.get("colors", []) or [],
                color_identity=card.get("color_identity", []) or [],
                layout=card.get("layout", "normal"),
                text=build_embedding_text(card),
            )


def name_claims(bulk_path: Path) -> dict[str, set[str]]:
    """Map every card name and face name to the ``oracle_id``s claiming it.

    A full name outranks another card's face name: "Harmonized Trio //
    Brainstorm" has a face called Brainstorm, which must not shadow Brainstorm.
    So a face name only contributes claims when no card has it as a full name.
    Names with more than one claimant are ambiguous.
    """
    full: dict[str, set[str]] = {}
    face: dict[str, set[str]] = {}
    with gzip.open(bulk_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            if not _is_wanted(card, include_digital=True):
                continue
            oid = card["oracle_id"]
            if card.get("name"):
                full.setdefault(card["name"], set()).add(oid)
            for part in card.get("card_faces") or ():
                if part.get("name"):
                    face.setdefault(part["name"], set()).add(oid)
    claims = dict(full)
    for name, ids in face.items():
        claims.setdefault(name, ids)
    return claims


def oracle_name_index(bulk_path: Path) -> tuple[dict[str, str], set[str]]:
    """Map every card name and face name to its ``oracle_id``.

    The Forge bridge identifies cards only by name, so this is the join from a
    game record back to Scryfall. Full names (``Fire // Ice``) and individual
    face names (``Fire``) are both indexed, because Forge reports whichever
    face is showing. Digital-only cards are included: Forge plays Alchemy
    rebalances (``A-Unholy Heat``).

    Returns the index and the set of names claimed by more than one
    ``oracle_id``. Ambiguous names are left out of the index rather than
    resolved arbitrarily, since a wrong id would silently swap a card's
    embedding. Precedence rules are in :func:`name_claims`.
    """
    claims = name_claims(bulk_path)
    ambiguous = {name for name, ids in claims.items() if len(ids) > 1}
    index = {name: next(iter(ids)) for name, ids in claims.items() if len(ids) == 1}
    return index, ambiguous
