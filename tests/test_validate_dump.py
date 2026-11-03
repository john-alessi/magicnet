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
    stay top-level, the rest go under ``components``. A ``None`` value is
    omitted, as for a hidden card's name (schema 12)."""
    flat = {k: v for k, v in flat.items() if v is not None}
    top = {k: v for k, v in flat.items() if _REGISTRY.is_top_level(k)}
    components = {k: v for k, v in flat.items() if k not in top}
    return {**top, "components": components} if components else top


SELF = _nest({"type": "player", "id": "p0", "side": "self"})
SIDEBOARD = _nest({"type": "library_card", "zone": "opp_sideboard", "count": 15})
OPP_HAND = _nest({"type": "library_card", "zone": "opp_hand", "count": 3})
UNTRACKED = {"type": "library_card", "zone": "opp_untracked"}


class _H:
    """Builds one viewer's schema-13 history: each event gets the next e<N> id."""

    def __init__(self) -> None:
        self.n = 0

    def ev(self, verb: str, edges: dict | None = None, step: int = 0, **comps) -> dict:
        tok = {"type": "history", "zone": "history", "id": f"e{self.n}",
               "components": {"verb": verb, "turn": 1, "phase": "MAIN1", "step": step,
                              "step_order": 0, "cause_kind": "unknown", **comps}}
        self.n += 1
        if edges:
            tok["edges"] = [{"rel": r, "to": t} for r, ts in edges.items()
                            for t in (ts if isinstance(ts, list) else [ts])]
        return tok

    def move(self, src: str, dst: str, obj: str | None = None, step: int = 0, **comps) -> dict:
        return self.ev("move_zone", {"obj": obj} if obj else None, step=step,
                       **{"from": src, "to": dst, "move_kind": "other", **comps})

    def reveal(self, src: str, obj: str, shown_to: str = "p0", step: int = 0) -> dict:
        return self.ev("reveal", {"obj": obj, "player": shown_to}, step=step, **{"from": src, "count": 1})


def _with(tok: dict, *handles: str) -> dict:
    return dict(tok, handles=list(handles))


def _known(card_id: str, name: str, *handles: str) -> dict:
    tok = _nest({"type": "game_object", "zone": "opp_hand", "id": card_id, "name": name,
                 "known_by_reveal": True})
    return _with(tok, *handles) if handles else tok


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
    h = _H()
    assert _leaks(tmp_path, [
        ([h.ev("draw", {"player": "p1"}, **{"from": "opp_library", "to": "opp_hand"}),
          h.reveal("opp_hand", "h0")],
         [SELF, _known("c5", "Bolt", "h0"), SIDEBOARD]),
    ]) == {}


def test_known_card_without_a_reveal_is_a_leak(tmp_path: Path) -> None:
    assert _leaks(tmp_path, [([], [SELF, _known("c5", "Bolt")])])


def test_reveal_to_someone_else_does_not_license_the_name(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [([h.reveal("opp_hand", "h0", shown_to="p2")],
                              [SELF, _known("c5", "Bolt", "h0")])])


def test_knowledge_does_not_survive_an_unidentified_return_to_hand(tmp_path: Path) -> None:
    h = _H()
    offenders = _leaks(tmp_path, [
        ([h.reveal("opp_hand", "h0")], [SELF, _known("c5", "Bolt", "h0")]),
        ([h.move("opp_hand", "opp_library", "h0"),
          h.ev("draw", {"player": "p1"}, **{"from": "opp_library", "to": "opp_hand"})],
         [SELF, _known("c5", "Bolt", "h0")]),
    ])
    assert any(k.startswith("known_by_reveal without a recorded cause") for k in offenders)
    assert "handle not widened: on a token in opp_hand" in offenders


def test_sideboard_aggregate_cannot_claim_revealed(tmp_path: Path) -> None:
    smuggled = dict(SIDEBOARD, name="Bolt", components=dict(SIDEBOARD["components"], revealed=True))
    assert _leaks(tmp_path, [([], [SELF, smuggled])])


def test_named_opponent_hand_card_or_aggregate_is_a_leak(tmp_path: Path) -> None:
    library = _nest({"type": "library_card", "zone": "opp_library", "count": 30})
    hidden = _bf("c5", None, zone="opp_hand")
    assert _leaks(tmp_path, [([], [SELF, hidden, library, SIDEBOARD, OPP_HAND])]) == {}
    assert _leaks(tmp_path, [([], [SELF, dict(hidden, name="Bolt")])]) == {
        "unredacted name in opp_hand: Bolt": 1}
    assert _leaks(tmp_path, [([], [SELF, dict(library, name="Bolt")])]) == {
        "unredacted name in opp_library: Bolt": 1}


def test_oracle_id_on_an_unnamed_card_or_any_identity_on_history_is_a_leak(tmp_path: Path) -> None:
    exiled = _bf("c5", None, zone="opp_exile", oracle_id="o-bolt")
    h = _H()
    move = dict(h.move("opp_library", "opp_hand"), card_oracle_id="o-bolt")
    assert _leaks(tmp_path, [([move], [SELF, exiled])]) == {
        "oracle_id on hidden game_object in opp_exile": 1,
        "card_oracle_id on hidden history in history": 1,
        "history token carries card_oracle_id": 1,
    }


def test_public_move_into_hand_explains_the_card(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [
        ([h.move("opp_battlefield", "opp_hand", "h0", move_kind="bounce")],
         [SELF, _known("c5", "Bear", "h0")]),
    ]) == {}


def test_public_move_knowledge_expires_when_the_card_leaves_hand(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [
        ([h.move("opp_graveyard", "opp_hand", "h0")], [SELF, _known("c5", "Bear", "h0")]),
        ([h.move("opp_hand", "opp_library", "h0"), h.move("opp_library", "opp_hand", "h0")],
         [SELF, _known("c5", "Bear", "h0")]),
    ])


def test_move_from_a_hidden_origin_does_not_explain_the_card(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [
        ([h.move("opp_library", "opp_hand", "h0")], [SELF, _known("c5", "Bear", "h0")]),
    ])


def test_face_down_departure_is_not_public(tmp_path: Path) -> None:
    h = _H()
    offenders = _leaks(tmp_path, [
        ([h.move("opp_exile", "opp_hand", "h0", face_down=True)], [SELF, _known("c5", "Bear", "h0")]),
    ])
    assert any(k.startswith("known_by_reveal without a recorded cause") for k in offenders)


def test_face_down_departure_the_viewer_could_see_is_public(tmp_path: Path) -> None:
    h = _H()
    looked = _with(_nest({"type": "game_object", "zone": "self_battlefield", "id": "c5",
                          "name": "Bear", "face_down": True}), "h0")
    assert _leaks(tmp_path, [
        ([h.ev("create", {"obj": "h0"}, to="self_battlefield")], [SELF, looked]),
        ([h.move("self_battlefield", "opp_hand", "h0", face_down=True)], [SELF, _known("c5", "Bear", "h0")]),
    ]) == {}


def test_face_down_departure_redacted_in_the_last_state_is_a_leak(tmp_path: Path) -> None:
    h = _H()
    unseen = _with(_nest({"type": "game_object", "zone": "opp_exile", "id": "c5"}), "h0")
    assert _leaks(tmp_path, [
        ([h.move("opp_hand", "opp_exile", "h0", face_down=True)], [SELF, unseen]),
        ([h.move("opp_exile", "opp_hand", "h0", face_down=True)], [SELF, _known("c5", "Bear", "h0")]),
    ])


def test_dig_reveal_then_move_to_hand_explains_the_card(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [
        ([h.reveal("opp_library", "h0", step=5), h.move("opp_library", "opp_hand", "h0", step=5)],
         [SELF, _known("c5", "Bolt", "h0")]),
    ]) == {}


def test_dig_reveal_does_not_survive_the_effect(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [
        ([h.reveal("opp_library", "h0", step=5), h.move("opp_library", "opp_hand", "h0", step=6)],
         [SELF, _known("c5", "Bolt", "h0")]),
    ])


def test_dig_reveal_only_covers_the_cards_next_move(tmp_path: Path) -> None:
    h = _H()
    assert _leaks(tmp_path, [
        ([h.reveal("opp_library", "h0", step=5), h.move("opp_library", "opp_library", "h0", step=5),
          h.move("opp_library", "opp_hand", "h0", step=5)],
         [SELF, _known("c5", "Bolt", "h0")]),
    ])


# ---- schema 13: tracking handles -------------------------------------------


def test_resolve_edges_follows_handles_and_event_ids() -> None:
    from magicnet.dump import resolve_edges, strip_ids
    h = _H()
    bounce = h.move("opp_battlefield", "opp_hand", "h0", move_kind="bounce")
    trig = h.ev("trigger", {"cause": "e0", "source_is": "h1", "actor": "p1"})
    opp = _nest({"type": "player", "id": "p1", "side": "opp"})
    bear = _with(_bf("c9", "Prowler", zone="opp_battlefield"), "h1")
    tokens = [bounce, trig, SELF, opp, _with(OPP_HAND, "h0"), bear]
    assert resolve_edges(tokens)[:2] == [[("obj", 4)], [("cause", 0), ("source_is", 5), ("actor", 3)]]
    assert all("handles" not in t and "id" not in t for t in strip_ids(tokens))



def test_handle_on_an_individual_hidden_card_is_a_leak(tmp_path: Path) -> None:
    h = _H()
    bounce = h.move("opp_battlefield", "opp_hand", "h0", move_kind="bounce")
    hidden = _with(_bf("c5", None, zone="opp_hand"), "h0")
    assert _leaks(tmp_path, [([bounce], [SELF, hidden])]) == {
        "handle on an individual hidden card in opp_hand": 1}
    assert _leaks(tmp_path, [([bounce], [SELF, _with(OPP_HAND, "h0")])]) == {}


def test_history_edges_target_only_handles_events_and_players(tmp_path: Path) -> None:
    h = _H()
    first = h.ev("draw", {"player": "p1"}, **{"from": "opp_library", "to": "opp_hand"})
    trig = h.ev("trigger", {"cause": "c42"})
    assert _leaks(tmp_path, [([first, trig], [SELF])]) == {
        "history edge to a non-handle/event/player ref (cause)": 1}
    ok = _H()
    assert _leaks(tmp_path, [([ok.ev("draw", None, **{"from": "opp_library", "to": "opp_hand"}),
                               ok.ev("trigger", {"cause": "e0"})], [SELF])]) == {}
    late = _H()
    assert _leaks(tmp_path, [([late.ev("trigger", {"cause": "e1"}), late.ev("draw")], [SELF])]) == {
        "history edge to a later event": 1}


def test_lost_track_must_be_widened_to_untracked(tmp_path: Path) -> None:
    h = _H()
    hist = [h.move("opp_battlefield", "opp_hand", "h0", move_kind="bounce"),
            h.move("opp_hand", "opp_library", step=1)]  # Brainstorm puts back an unseen card
    assert _leaks(tmp_path, [(hist, [SELF, _with(OPP_HAND, "h0")])]) == {
        "handle not widened: on a token in opp_hand": 1}
    assert _leaks(tmp_path, [(hist, [SELF, OPP_HAND, _with(UNTRACKED, "h0")])]) == {}
    # Schema 14: a known identity goes to an identity phantom ("a copy of Bear").
    copy = _with(_nest({"type": "game_object", "zone": "phantom", "name": "Bear", "side": "opp"}), "h0")
    assert _leaks(tmp_path, [(hist, [SELF, OPP_HAND, copy])]) == {}


def test_a_lost_handle_never_returns_from_a_phantom(tmp_path: Path) -> None:
    h = _H()
    hist = [h.move("opp_battlefield", "opp_hand", "h0", move_kind="bounce"),
            h.move("opp_hand", "opp_library", step=1)]
    copy = _with(_nest({"type": "game_object", "zone": "phantom", "name": "Bear", "side": "opp"}), "h0")
    relinked = _with(_bf("c5", "Bear"), "h0")
    later = [h.ev("cast", {"stack_object": "h1", "source_is": "h1"}, step=2)]
    stack = _with({"type": "stack_item", "zone": "stack", "id": "s9"}, "h1")
    assert _leaks(tmp_path, [(hist, [SELF, OPP_HAND, copy]), (later, [SELF, copy, stack])]) == {}
    assert _leaks(tmp_path, [(hist, [SELF, OPP_HAND, copy]), ([], [SELF, relinked])]) == {
        "handle returned from a phantom to a token in opp_battlefield": 1,
        "handle not widened: on a token in opp_battlefield": 1}


def test_every_referenced_handle_is_assigned_exactly_once(tmp_path: Path) -> None:
    h = _H()
    hist = [h.ev("create", {"obj": "h0"}, to="self_battlefield")]
    bear = _bf("c1", "Bear", zone="self_battlefield")
    assert _leaks(tmp_path, [(hist, [SELF, bear])]) == {"referenced handle assigned to 0 tokens": 1}
    assert _leaks(tmp_path, [(hist, [SELF, _with(bear, "h0"), _with(OPP_HAND, "h0")])]) == {
        "referenced handle assigned to 2 tokens": 1}
    assert _leaks(tmp_path, [(hist, [SELF, _with(bear, "h0", "h7")])]) == {
        "handle assigned but never referenced": 1}


def test_phantom_never_names_an_object_last_seen_unnamed(tmp_path: Path) -> None:
    h = _H()
    morph = _with(_bf("c1", None, face_down=True), "h0")
    phantom = _with(_nest({"type": "game_object", "zone": "phantom", "name": "Bear", "side": "opp"}), "h0")
    hist1 = [h.ev("create", {"obj": "h0"}, to="opp_battlefield")]
    hist2 = [h.move("opp_battlefield", "opp_graveyard", "h0")]
    assert _leaks(tmp_path, [(hist1, [SELF, morph]), (hist2, [SELF, phantom])]) == {
        "phantom names an object last seen unnamed": 1}
    with_id = dict(phantom, id="c1")
    assert "phantom carries an id" in _leaks(tmp_path, [(hist1 + hist2, [SELF, with_id])])


def _bf(card_id: str, name: str | None, zone: str = "opp_battlefield", **extra) -> dict:
    return _nest({"type": "game_object", "zone": zone, "id": card_id, "name": name, **extra})


def _affects(tok: dict, *refs: str) -> dict:
    return dict(tok, edges=[{"rel": "affects", "to": r} for r in refs])


OPP = _nest({"type": "player", "id": "p1", "side": "opp"})


def test_affects_between_visible_tokens_passes(tmp_path: Path) -> None:
    anthem = _affects(_bf("c1", "Glorious Anthem"), "c2", "p1")
    assert _leaks(tmp_path, [([], [SELF, OPP, anthem, _bf("c2", "Bear")])]) == {}


def test_affects_into_a_hidden_zone_is_a_leak(tmp_path: Path) -> None:
    painter = _affects(_bf("c1", "Painter's Servant"), "c2")
    hand = _bf("c2", None, zone="opp_hand")
    offenders = _leaks(tmp_path, [([], [SELF, painter, hand])])
    assert "affects edge to hidden token in opp_hand" in offenders


def test_affects_from_a_redacted_source_or_to_an_absent_id_is_a_leak(tmp_path: Path) -> None:
    hidden = _affects(_bf("c1", None), "c2", "c9")
    offenders = _leaks(tmp_path, [([], [SELF, hidden, _bf("c2", "Bear")])])
    assert "affects edge from hidden token in opp_battlefield" in offenders
    assert offenders.get("affects edge to an absent token") == 1


def test_choices_on_face_down_or_redacted_tokens_are_leaks(tmp_path: Path) -> None:
    morph = _bf("c1", None, face_down=True, chosen_player="p0")
    own_morph = _bf("c2", "Leonin", zone="self_battlefield", face_down=True, chosen_type="Elf")
    offenders = _leaks(tmp_path, [([], [SELF, morph, own_morph])])
    assert sum(v for k, v in offenders.items() if k.startswith("choice on face-down")) == 2


def test_chosen_player_must_be_a_player_token(tmp_path: Path) -> None:
    ok = _bf("c1", "Stalking Leonin", chosen_player="p0", chosen_colors=["red"])
    bad = _bf("c2", "Stalking Leonin", chosen_player="c7")
    assert _leaks(tmp_path, [([], [SELF, ok])]) == {}
    assert _leaks(tmp_path, [([], [SELF, bad])]) == {"chosen_player is not a player token": 1}


def test_layout_of_well_formed_tokens_passes() -> None:
    h = _H()
    tokens = [SELF, _bf("c1", "Bear", tapped=True, power=2, base_power=2, gained=["Trample"],
                            lost=["{T}: Add {G}."], blocked=False),
              h.move("opp_battlefield", "opp_hand", "h0", move_kind="bounce"),
              h.ev("trigger", {"source_is": "h0"},
                   ability="Prowess (Whenever you cast a noncreature spell, ...)")]
    assert validate_dump.check_layout(tokens) == {}


def test_layout_flags_misplaced_keys_false_flags_and_object_ids() -> None:
    misplaced = {"type": "game_object", "zone": "opp_battlefield", "tapped": True,
                 "components": {"handles": ["h1"]}}
    false_flag = _bf("c2", "Bear", tapped=False)
    old_style = _H().ev("trigger", ability="Soul-Scar Mage (124): Prowess [Card: Kumano Faces Kakkazan (94)]")
    assert validate_dump.check_layout([misplaced, false_flag, old_style, dict(SELF, components={})]) == {
        "component at top level: tapped": 1,
        "top-level key in components: handles": 1,
        "false flag emitted: game_object.tapped": 1,
        "object id in string component: ability": 1,
        "bracketed context in string component: ability": 1,
        "empty components": 1,
    }


def test_layout_flags_forge_run_context_in_strings() -> None:
    old_style = _H().ev("trigger", ability="Whenever an opponent draws a card, they lose 2 life. "
                                            "[Player: P1-[Test] History Midrange B]")
    zone_ctx = _H().ev("trigger", ability="When this enters, you become the monarch. [Zone Changer: Palace Jailer]")
    clean = _H().ev("trigger", ability="Whenever an opponent draws a card, they lose 2 life.")
    assert validate_dump.check_layout([old_style, zone_ctx, clean]) == {
        "player name in string component: ability": 1,
        "bracketed context in string component: ability": 2,
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
