"""The dump validator's leak check must catch names the viewer was never shown."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validate_dump", REPO / "scripts" / "validate_dump.py")
validate_dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_dump)

from magicnet.catalog.keys import load_registry  # noqa: E402
from magicnet.dump import SUPPORTED_SCHEMA  # noqa: E402

_REGISTRY = load_registry()


def _nest(flat: dict) -> dict:
    """Schema-9 layout from a flat fixture: identity/ref keys (per keys.json)
    stay top-level, the rest go under ``components``."""
    top = {k: v for k, v in flat.items()
           if _REGISTRY.resolve(k)[1].value_kind in validate_dump.TOP_LEVEL_KINDS}
    components = {k: v for k, v in flat.items() if k not in top}
    return {**top, "components": components} if components else top


SELF = _nest({"type": "player", "id": "p0", "side": "self"})
SIDEBOARD = _nest({"type": "library_card", "zone": "opp_sideboard", "name": "HIDDEN", "count": 15})


def _hand_reveal(card_id: str, name: str, shown_to: str = "p0") -> dict:
    # As EventTokenizer emits it: "revealed_from" is the revealed card's zone.
    return _nest({"type": "history", "event": "Reveal", "card_id": card_id, "card": name,
                  "revealed_from": "Hand", "owner": "p1", "shown_to": shown_to})


def _move(card_id: str, src: str, dst: str, name: str = "<hidden>", **extra) -> dict:
    return _nest({"type": "history", "zone": "history", "event": "CardChangeZone",
                  "card_id": card_id, "card": name, "from": src, "to": dst, **extra})


def _dig_reveal(card_id: str, name: str) -> dict:
    tok = _hand_reveal(card_id, name)
    return dict(tok, components=dict(tok["components"], revealed_from="Library"))


RESOLVED = _nest({"type": "history", "zone": "history", "event": "SpellResolved"})


def _known(card_id: str, name: str) -> dict:
    return _nest({"type": "game_object", "zone": "opp_hand", "id": card_id, "name": name,
                  "known_by_reveal": True})


def _write(path: Path, records: list[tuple[list[dict], list[dict]]]) -> Path:
    total = 0
    with path.open("w") as fh:
        for i, (hist_new, state) in enumerate(records):
            fh.write(json.dumps({
                "schema": SUPPORTED_SCHEMA, "game": 0, "decision": i, "viewer": "P0",
                "turn": 1, "phase": "MAIN1", "hist_from": total, "hist_new": hist_new,
                "hist_total": total + len(hist_new), "state": state, "actions": [],
                "chosen": -1, "chosen_match": "pass",
            }) + "\n")
            total += len(hist_new)
    return path


def _leaks(tmp_path: Path, records) -> dict:
    _, offenders = validate_dump.check_leaks(_write(tmp_path / "d.jsonl", records), 0)
    return dict(offenders)


def test_known_card_after_a_reveal_to_the_viewer_passes(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_move("c5", "Library", "Hand"), _hand_reveal("c5", "Bolt")],
         [SELF, _known("c5", "Bolt"), SIDEBOARD]),
    ]) == {}


def test_known_card_without_a_reveal_is_a_leak(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [([], [SELF, _known("c5", "Bolt")])])


def test_reveal_to_someone_else_does_not_license_the_name(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [([_hand_reveal("c5", "Bolt", shown_to="p2")],
                              [SELF, _known("c5", "Bolt")])])


def test_knowledge_does_not_survive_leaving_and_returning_to_hand(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_hand_reveal("c5", "Bolt")], [SELF, _known("c5", "Bolt")]),
        ([_move("c5", "Hand", "Graveyard"), _move("c5", "Graveyard", "Hand")],
         [SELF, _known("c5", "Bolt")]),
    ])


def test_sideboard_aggregate_cannot_claim_revealed(tmp_path: Path) -> None:
    smuggled = dict(SIDEBOARD, name="Bolt", components=dict(SIDEBOARD["components"], revealed=True))
    assert _leaks(tmp_path, [([], [SELF, smuggled])])


def test_public_move_into_hand_explains_the_card(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_move("c5", "Battlefield", "Hand", "Bear", to_player="p1")],
         [SELF, _known("c5", "Bear")]),
    ]) == {}


def test_public_move_knowledge_expires_when_the_card_leaves_hand(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_move("c5", "Graveyard", "Hand", "Bear", to_player="p1")],
         [SELF, _known("c5", "Bear")]),
        ([_move("c5", "Hand", "Library", to_player="p1"),
          _move("c5", "Library", "Hand", to_player="p1")],
         [SELF, _known("c5", "Bear")]),
    ])


def test_named_move_from_a_hidden_origin_is_a_leak(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_move("c5", "Library", "Hand", "Bear", to_player="p1")], [SELF]),
    ])


def test_face_down_departure_is_not_public(tmp_path: Path) -> None:
    offenders = _leaks(tmp_path, [
        ([_move("c5", "Exile", "Hand", "Bear", to_player="p1", face_down=True)],
         [SELF, _known("c5", "Bear")]),
    ])
    assert any(k.startswith("face-down move named") for k in offenders)
    assert any(k.startswith("known_by_reveal without a recorded cause") for k in offenders)


def test_face_down_departure_the_viewer_could_see_is_public(tmp_path: Path) -> None:
    looked = _nest({"type": "game_object", "zone": "self_battlefield", "id": "c5",
                    "name": "Bear", "face_down": True})
    assert _leaks(tmp_path, [
        ([], [SELF, looked]),
        ([_move("c5", "Battlefield", "Hand", "Bear", to_player="p1", face_down=True)],
         [SELF, _known("c5", "Bear")]),
    ]) == {}


def test_face_down_departure_redacted_in_the_last_state_is_a_leak(tmp_path: Path) -> None:
    unseen = _nest({"type": "game_object", "zone": "opp_exile", "id": "c5", "name": "HIDDEN"})
    assert _leaks(tmp_path, [
        ([], [SELF, unseen]),
        ([_move("c5", "Exile", "Hand", "Bear", to_player="p1", face_down=True)], [SELF]),
    ])


def test_dig_reveal_then_move_to_hand_explains_the_card(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_dig_reveal("c5", "Bolt"), _move("c5", "Library", "Hand", to_player="p1"), RESOLVED],
         [SELF, _known("c5", "Bolt")]),
    ]) == {}


def test_dig_reveal_does_not_survive_the_effect(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_dig_reveal("c5", "Bolt"), RESOLVED, _move("c5", "Library", "Hand", to_player="p1")],
         [SELF, _known("c5", "Bolt")]),
    ])


def test_dig_reveal_only_covers_the_cards_next_move(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [
        ([_dig_reveal("c5", "Bolt"), _move("c5", "Library", "Library", to_player="p1"),
          _move("c5", "Library", "Hand", to_player="p1")],
         [SELF, _known("c5", "Bolt")]),
    ])


def _bf(card_id: str, name: str, zone: str = "opp_battlefield", **extra) -> dict:
    return _nest({"type": "game_object", "zone": zone, "id": card_id, "name": name, **extra})


def _affects(tok: dict, *refs: str) -> dict:
    return dict(tok, edges=[{"rel": "affects", "to": r} for r in refs])


OPP = _nest({"type": "player", "id": "p1", "side": "opp"})


def test_affects_between_visible_tokens_passes(tmp_path: Path) -> None:
    anthem = _affects(_bf("c1", "Glorious Anthem"), "c2", "p1")
    assert _leaks(tmp_path, [([], [SELF, OPP, anthem, _bf("c2", "Bear")])]) == {}


def test_affects_into_a_hidden_zone_is_a_leak(tmp_path: Path) -> None:
    painter = _affects(_bf("c1", "Painter's Servant"), "c2")
    hand = _bf("c2", "HIDDEN", zone="opp_hand")
    offenders = _leaks(tmp_path, [([], [SELF, painter, hand])])
    assert "affects edge to hidden token in opp_hand" in offenders


def test_affects_from_a_redacted_source_or_to_an_absent_id_is_a_leak(tmp_path: Path) -> None:
    hidden = _affects(_bf("c1", "HIDDEN"), "c2", "c9")
    offenders = _leaks(tmp_path, [([], [SELF, hidden, _bf("c2", "Bear")])])
    assert "affects edge from hidden token in opp_battlefield" in offenders
    assert offenders.get("affects edge to an absent token") == 1


def test_choices_on_face_down_or_redacted_tokens_are_leaks(tmp_path: Path) -> None:
    morph = _bf("c1", "HIDDEN", face_down=True, chosen_player="p0")
    own_morph = _bf("c2", "Leonin", zone="self_battlefield", face_down=True, chosen_type="Elf")
    offenders = _leaks(tmp_path, [([], [SELF, morph, own_morph])])
    assert sum(v for k, v in offenders.items() if k.startswith("choice on face-down")) == 2


def test_chosen_player_must_be_a_player_token(tmp_path: Path) -> None:
    ok = _bf("c1", "Stalking Leonin", chosen_player="p0", chosen_colors=["red"])
    bad = _bf("c2", "Stalking Leonin", chosen_player="c7")
    assert _leaks(tmp_path, [([], [SELF, ok])]) == {}
    assert _leaks(tmp_path, [([], [SELF, bad])]) == {"chosen_player is not a player token": 1}


def test_layout_of_well_formed_tokens_passes() -> None:
    tokens = [SELF, _bf("c1", "Bear", tapped=True, power=2, base_power=2, gained=["Trample"],
                            lost=["{T}: Add {G}."], blocked=False),
              _move("c5", "Battlefield", "Hand", "Bear", to_player="p1"),
              _nest({"type": "history", "event": "SpellResolved", "source": "Bear",
                     "ability": "Prowess (Whenever you cast a noncreature spell, ...)"})]
    assert validate_dump.check_layout(tokens) == {}


def test_layout_flags_misplaced_keys_false_flags_and_object_ids() -> None:
    misplaced = {"type": "game_object", "zone": "opp_battlefield", "tapped": True,
                 "components": {"card_id": "c1"}}
    false_flag = _bf("c2", "Bear", tapped=False)
    old_style = _nest({"type": "history", "event": "SpellResolved",
                       "ability": "Soul-Scar Mage (124): Prowess [Card: Kumano Faces Kakkazan (94)]"})
    assert validate_dump.check_layout([misplaced, false_flag, old_style, dict(SELF, components={})]) == {
        "component at top level: tapped": 1,
        "identity/ref key in components: card_id": 1,
        "false flag emitted: game_object.tapped": 1,
        "object id in string component: ability": 1,
        "empty components": 1,
    }


def test_layout_flags_host_dependent_ability_text() -> None:
    bad = _bf("c1", "Grizzly Bears", gained=[
        "Whenever CARDNAME attacks, draw a card.",       # unsubstituted placeholder
        "Grizzly Bears can't block.",                    # host's own name
        "Equip {1} (42)",                                # object id
        "Target player draws. Activator: P0-Mono Red",   # stack tail, player name
    ], lost=["Flying"])
    token = _bf("c2", "Spirit", token=True, gained=["Other Spirit creatures get +1/+1."])
    assert validate_dump.check_layout([bad, token]) == {
        "placeholder in gained": 1, "host name in gained": 1, "object id in gained": 1,
        "activator in gained": 1, "player name in gained": 1,
    }
