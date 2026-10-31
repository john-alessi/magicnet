"""Name -> oracle_id resolution for the Forge bridge's lookup table."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from magicnet.cards import oracle_name_index


def _bulk(tmp_path: Path, cards: list[dict]) -> Path:
    path = tmp_path / "bulk.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps({"lang": "en", **card}) + "\n")
    return path


def test_full_name_outranks_another_cards_face(tmp_path: Path) -> None:
    index, ambiguous = oracle_name_index(_bulk(tmp_path, [
        {"name": "Harmonized Trio // Brainstorm", "layout": "prepare", "oracle_id": "trio",
         "card_faces": [{"name": "Harmonized Trio"}, {"name": "Brainstorm"}]},
        {"name": "Brainstorm", "layout": "normal", "oracle_id": "brainstorm"},
        {"name": "Fire // Ice", "layout": "split", "oracle_id": "fire-ice",
         "card_faces": [{"name": "Fire"}, {"name": "Ice"}]},
        {"name": "Fire // Fury", "layout": "split", "oracle_id": "fire-fury",
         "card_faces": [{"name": "Fire"}, {"name": "Fury"}]},
        {"name": "Brainstorm Token", "layout": "token", "oracle_id": "tok"},
    ]))
    assert index["Brainstorm"] == "brainstorm"
    assert index["Harmonized Trio"] == "trio"
    assert index["Fire // Ice"] == "fire-ice"
    # Two faces share a name and no full name settles it: left out, not guessed.
    assert "Fire" in ambiguous and "Fire" not in index
    assert "Brainstorm Token" not in index
