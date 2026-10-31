"""Asserts the Python vocabulary matches the Java bridge's.

The two lists are maintained separately on purpose -- the Python copy is pinned
to what a checkpoint was trained against -- so something has to check they
agree. This shells out to the real ``magicnet.bridge.Schema`` rather than
reimplementing its construction, so the test cannot pass by sharing a bug with
the code it is checking.

Skips (rather than fails) when the bridge is not built, so the suite still runs
on a machine with no JDK or no Forge jar.
"""

from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from magicnet import schema
from magicnet.catalog.keys import load_registry
from magicnet.dump import comp

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "forge-bridge" / "build"
FORGE_JAR = Path(
    os.environ.get("MAGICNET_FORGE_JAR", "/usr/share/java/forge-gui-desktop.jar")
)


def _bridge_vocabulary() -> dict:
    if shutil.which("java") is None:
        pytest.skip("java not on PATH")
    if not (BUILD / "magicnet" / "bridge" / "Schema.class").exists():
        pytest.skip(f"bridge not built at {BUILD}; run javac first")
    if not FORGE_JAR.exists():
        pytest.skip(f"Forge jar not found at {FORGE_JAR}")

    proc = subprocess.run(
        ["java", "-cp", f"{BUILD}:{FORGE_JAR}", "magicnet.bridge.Schema"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"magicnet.bridge.Schema exited {proc.returncode}:\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def bridge() -> dict:
    return _bridge_vocabulary()


def test_zones_match(bridge: dict) -> None:
    assert list(schema.ZONES) == bridge["zones"], (
        "zone vocabulary drifted between Java and Python. Segment embeddings "
        "are positional with respect to this list, so a mismatch misrepresents "
        "the board rather than failing loudly at runtime."
    )


def test_token_types_match(bridge: dict) -> None:
    assert list(schema.TOKEN_TYPES) == bridge["token_types"]


def test_relations_match(bridge: dict) -> None:
    assert list(schema.RELATIONS) == bridge["relations"], (
        "relation vocabulary drifted between Java and Python. These index the "
        "attention bias, so a mismatch permutes relation semantics -- the model "
        "reads 'blocks' as 'attacks' -- instead of failing."
    )


def test_every_emitted_relation_is_in_the_vocabulary(bridge: dict) -> None:
    # Guards the gap the vocabulary was built to close: a relation added to the
    # Java emitters but never registered in Schema.relations().
    src = REPO / "forge-bridge" / "src"
    if not src.exists():
        pytest.skip("bridge sources not present")
    emitted = set()
    for path in src.rglob("*.java"):
        if path.name == "Schema.java":
            continue
        for m in re.finditer(r"\.edge\(\s*(?:Schema\.)?([A-Za-z_\"]+)", path.read_text()):
            token = m.group(1)
            if token.startswith('"'):
                emitted.add(token.strip('"'))  # bare literal: bypasses Schema
            else:
                emitted.add(token)
    constants = {name for name in emitted if name.isupper()}
    literals = {name for name in emitted if not name.isupper()}
    assert not literals, (
        f"edge() called with bare string literals {sorted(literals)}; these "
        f"bypass Schema and can drift silently. Use a Schema constant."
    )
    declared = {r.upper() for r in bridge["relations"]}
    assert constants <= declared, (
        f"emitters use Schema constants absent from relations(): "
        f"{sorted(constants - declared)}"
    )


def test_schema_version_matches(bridge: dict) -> None:
    assert schema.SCHEMA == bridge["schema"]


def test_zone_index_covers_every_bridge_zone(bridge: dict) -> None:
    seen = {schema.zone_index(z) for z in bridge["zones"]}
    assert len(seen) == len(bridge["zones"]), "zone indices are not distinct"
    assert schema.zone_index(None) not in seen, "the no-zone slot collides"


def test_unknown_zone_raises() -> None:
    # The vocabulary must not quietly absorb a zone it predates.
    with pytest.raises(KeyError):
        schema.zone_index("self_ante")


def test_counter_tokens_preserve_counts(bridge: dict, tmp_path: Path) -> None:
    """Forge's Multiset API must still produce one numeric field per type."""
    if shutil.which("javac") is None:
        pytest.skip("javac not on PATH")
    source = tmp_path / "CounterTokenCheck.java"
    source.write_text(
        """
package magicnet.bridge;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;

public class CounterTokenCheck {
    public static void main(String[] args) throws Exception {
        forge.util.Lang.createInstance("en-US");
        var encode = StateExtractor.class.getDeclaredMethod(
                "addCounters", StateExtractor.Tok.class, Multiset.class);
        encode.setAccessible(true);
        Multiset<CounterType> counters = HashMultiset.create();
        var card = new StateExtractor.Tok("game_object", Schema.zoneNames().get(0));
        encode.invoke(null, card, counters);
        System.out.println(card.toJson());
        counters.add(CounterEnumType.P1P1, 3);
        counters.add(CounterEnumType.P1P1, 2);
        encode.invoke(null, card, counters);
        System.out.println(card.toJson());
        counters.clear();
        counters.add(CounterEnumType.POISON, 2);
        var player = new StateExtractor.Tok("player", null);
        encode.invoke(null, player, counters);
        System.out.println(player.toJson());
    }
}
"""
    )
    classpath = f"{BUILD}:{FORGE_JAR}"
    compiled = subprocess.run(
        ["javac", "-cp", classpath, "-d", str(tmp_path), str(source)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stderr
    proc = subprocess.run(
        ["java", "-cp", f"{tmp_path}:{classpath}", "magicnet.bridge.CounterTokenCheck"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    tokens = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert tokens == [
        {"type": "game_object", "zone": "self_battlefield"},
        {"type": "game_object", "zone": "self_battlefield", "components": {"ctr_+1/+1": 5}},
        {"type": "player", "components": {"ctr_poison": 2}},
    ]


def _run_java_check(tmp_path: Path, name: str, body: str) -> list[dict]:
    if shutil.which("javac") is None:
        pytest.skip("javac not on PATH")
    source = tmp_path / f"{name}.java"
    source.write_text(body)
    classpath = f"{BUILD}:{FORGE_JAR}"
    compiled = subprocess.run(
        ["javac", "-cp", classpath, "-d", str(tmp_path), str(source)],
        capture_output=True, text=True, timeout=120,
    )
    assert compiled.returncode == 0, compiled.stderr
    proc = subprocess.run(
        ["java", "-cp", f"{tmp_path}:{classpath}", f"magicnet.bridge.{name}"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def test_attempt_events_are_tagged_and_closed(bridge: dict, tmp_path: Path) -> None:
    """History stays append-only; a consumer drops a rolled-back attempt's events
    by its tag, so events inside an attempt must carry it and events outside
    must not."""
    tokens = _run_java_check(tmp_path, "AttemptTagCheck", """
package magicnet.bridge;

import forge.game.event.GameEventCardDestroyed;

public class AttemptTagCheck {
    public static void main(String[] args) {
        GameHistory h = new GameHistory(null, null);
        h.onEvent(new GameEventCardDestroyed());
        GameHistory.Attempt a = GameHistory.beginAttempt(null);
        if (GameHistory.beginAttempt(null) != null) {
            throw new AssertionError("a nested cast must not open a second attempt");
        }
        h.onEvent(new GameEventCardDestroyed());
        h.appendAttemptEnd(a, "rolled_back");
        GameHistory.closeAttempt(a);
        h.onEvent(new GameEventCardDestroyed());
        for (StateExtractor.Tok t : h.tokens()) {
            System.out.println(t.toJson());
        }
    }
}
""")
    assert [(comp(t, "event"), t.get("attempt"), comp(t, "outcome")) for t in tokens] == [
        ("CardDestroyed", None, None),
        ("CardDestroyed", 1, None),
        ("AttemptEnd", 1, "rolled_back"),
        ("CardDestroyed", None, None),
    ]


def test_reveal_knowledge_is_per_viewer_and_dies_with_the_hand_visit(
    bridge: dict, tmp_path: Path
) -> None:
    """A reveal teaches only its recipient, never the owner, and a card that
    left the hand (new game timestamp, or an observed move) is unknown again."""
    tokens = _run_java_check(tmp_path, "RevealKnowledgeCheck", """
package magicnet.bridge;

public class RevealKnowledgeCheck {
    static void out(String k, boolean v) {
        System.out.println("{\\"" + k + "\\":" + v + "}");
    }
    public static void main(String[] args) {
        RevealKnowledge.reset();
        // viewer 1 is shown card 10, in player 0's hand, at timestamp 5
        out("learned", RevealKnowledge.learn(1, 0, 10, 5L));
        out("recipient_knows", RevealKnowledge.knows(1, 10, 5L));
        out("other_viewer_knows", RevealKnowledge.knows(0, 10, 5L));
        out("after_rezone_knows", RevealKnowledge.knows(1, 10, 6L));
        out("own_card_learned", RevealKnowledge.learn(0, 0, 11, 5L));
        out("own_card_known", RevealKnowledge.knows(0, 11, 5L));
        RevealKnowledge.forget(10);
        out("after_leaving_hand_knows", RevealKnowledge.knows(1, 10, 5L));
    }
}
""")
    assert {k: v for t in tokens for k, v in t.items()} == {
        "learned": True,
        "recipient_knows": True,
        "other_viewer_knows": False,
        "after_rezone_knows": False,
        "own_card_learned": False,
        "own_card_known": False,
        "after_leaving_hand_knows": False,
    }


def test_hand_knowledge_from_public_moves_and_same_effect_digs(
    bridge: dict, tmp_path: Path
) -> None:
    """A card entering a hand is known to viewers who saw it leave a public zone
    face-up, and to the recipient of a reveal by the same resolving effect when
    the reveal came first; nothing else teaches anything."""
    tokens = _run_java_check(tmp_path, "HandMoveKnowledgeCheck", """
package magicnet.bridge;

import forge.game.zone.ZoneType;

public class HandMoveKnowledgeCheck {
    static void out(String k, boolean v) {
        System.out.println("{\\"" + k + "\\":" + v + "}");
    }
    public static void main(String[] args) {
        int[] players = {0, 1};
        Object dig = new Object();
        Object other = new Object();
        RevealKnowledge.reset();
        out("bf_face_up", RevealKnowledge.publicOrigin(ZoneType.Battlefield, false, false));
        out("exile_face_down", RevealKnowledge.publicOrigin(ZoneType.Exile, true, false));
        out("exile_face_down_looked", RevealKnowledge.publicOrigin(ZoneType.Exile, true, true));
        out("library", RevealKnowledge.publicOrigin(ZoneType.Library, false, false));

        // bounce: card 10 Battlefield -> player 0's hand, everyone saw it leave
        RevealKnowledge.moved(10, ZoneType.Battlefield, ZoneType.Hand, 0, 7L, null, players, v -> true);
        out("bounce_opponent_knows", RevealKnowledge.knows(1, 10, 7L));
        out("bounce_owner_recorded", RevealKnowledge.knows(0, 10, 7L));
        // face-down departure nobody else could see
        RevealKnowledge.moved(11, ZoneType.Exile, ZoneType.Hand, 0, 7L, null, players, v -> false);
        out("face_down_known", RevealKnowledge.knows(1, 11, 7L));
        // card leaves hand: knowledge expires
        RevealKnowledge.moved(10, ZoneType.Hand, ZoneType.Stack, -1, 8L, null, players, v -> true);
        out("after_cast_knows", RevealKnowledge.knows(1, 10, 7L));

        // dig: reveal in library first, then the same effect moves it to hand
        RevealKnowledge.pend(1, 0, 20, ZoneType.Library, dig);
        RevealKnowledge.moved(20, ZoneType.Library, ZoneType.Hand, 0, 9L, dig, players, v -> false);
        out("dig_knows", RevealKnowledge.knows(1, 20, 9L));
        // revealed card drawn later by something else
        RevealKnowledge.pend(1, 0, 21, ZoneType.Library, dig);
        RevealKnowledge.moved(21, ZoneType.Library, ZoneType.Hand, 0, 9L, other, players, v -> false);
        out("other_effect_knows", RevealKnowledge.knows(1, 21, 9L));
        // revealed card goes to the bottom first, then is drawn by the same effect
        RevealKnowledge.pend(1, 0, 22, ZoneType.Library, dig);
        RevealKnowledge.moved(22, ZoneType.Library, ZoneType.Library, -1, 9L, dig, players, v -> false);
        RevealKnowledge.moved(22, ZoneType.Library, ZoneType.Hand, 0, 10L, dig, players, v -> false);
        out("second_move_knows", RevealKnowledge.knows(1, 22, 10L));
        // a stale pending reveal is dropped once another effect moves anything
        RevealKnowledge.pend(1, 0, 23, ZoneType.Library, dig);
        RevealKnowledge.moved(99, ZoneType.Library, ZoneType.Hand, 1, 1L, other, players, v -> false);
        RevealKnowledge.moved(23, ZoneType.Library, ZoneType.Hand, 0, 11L, dig, players, v -> false);
        out("stale_pending_knows", RevealKnowledge.knows(1, 23, 11L));
        // own reveal teaches nothing
        RevealKnowledge.pend(0, 0, 24, ZoneType.Library, dig);
        RevealKnowledge.moved(24, ZoneType.Library, ZoneType.Hand, 0, 12L, dig, players, v -> false);
        out("own_dig_known", RevealKnowledge.knows(0, 24, 12L));
    }
}
""")
    assert {k: v for t in tokens for k, v in t.items()} == {
        "bf_face_up": True,
        "exile_face_down": False,
        "exile_face_down_looked": True,
        "library": False,
        "bounce_opponent_knows": True,
        "bounce_owner_recorded": False,
        "face_down_known": False,
        "after_cast_knows": False,
        "dig_knows": True,
        "other_effect_knows": False,
        "second_move_knows": False,
        "stale_pending_knows": False,
        "own_dig_known": False,
    }


def test_list_props_and_affects_visibility(bridge: dict, tmp_path: Path) -> None:
    """Keyword/colour sets serialise as JSON arrays (escaped), and only named
    game objects outside libraries and the opponent's hidden zones may be an
    ``affects`` endpoint."""
    tokens = _run_java_check(tmp_path, "AffectsVisibilityCheck", """
package magicnet.bridge;

import java.util.List;

public class AffectsVisibilityCheck {
    public static void main(String[] args) throws Exception {
        var t = new StateExtractor.Tok("game_object", "self_battlefield")
                .p("gained", List.of("This creature can't block.", "Say \\"hi\\""))
                .p("colors", List.of());
        System.out.println(t.toJson());
        var vis = StateExtractor.class.getDeclaredMethod("isVisibleObject", StateExtractor.Tok.class);
        vis.setAccessible(true);
        String[][] cases = {
            {"opp_battlefield", "Bear"}, {"opp_battlefield", "HIDDEN"},
            {"opp_hand", "Bear"}, {"self_hand", "Bear"}, {"self_library", "Bear"},
            {"opp_sideboard", "Bear"}, {"opp_graveyard", "Bear"},
        };
        StringBuilder sb = new StringBuilder("{");
        for (String[] c : cases) {
            var tok = new StateExtractor.Tok("game_object", c[0]).p("name", c[1]);
            if (sb.length() > 1) sb.append(',');
            sb.append('"').append(c[0]).append('/').append(c[1]).append("\\":")
              .append(vis.invoke(null, tok));
        }
        System.out.println(sb.append('}'));
    }
}
""")
    assert tokens[0] == {"type": "game_object", "zone": "self_battlefield", "components": {
        "gained": ["This creature can't block.", 'Say "hi"'], "colors": []}}
    assert tokens[1] == {
        "opp_battlefield/Bear": True, "opp_battlefield/HIDDEN": False,
        "opp_hand/Bear": False, "self_hand/Bear": True, "self_library/Bear": False,
        "opp_sideboard/Bear": False, "opp_graveyard/Bear": True,
    }


# Stand-ins for a pattern's placeholders when instantiating a sample key.
_SLOT_SAMPLES = {"i": "3", "role": "zzrole", "name": "zzname", "what": "X"}
TOP_LEVEL_KINDS = {"identity", "ref"}


def _classify(keys: list[str]) -> dict[str, bool]:
    proc = subprocess.run(
        ["java", "-cp", f"{BUILD}:{FORGE_JAR}", "magicnet.bridge.Schema", "classify", *keys],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_top_level_classification_matches_key_catalog(bridge: dict) -> None:
    """Schema 9: a key is top-level in the Java writer iff keys.json calls it
    identity or ref. Asks the real Schema.isTopLevel about every registered key
    and a sample of every pattern, then checks each of Java's own rules (set,
    suffixes, regexes) only admits identity/ref keys."""
    registry = load_registry()
    samples = {k: e.value_kind for k, e in registry.keys.items()}
    for p in registry.patterns:
        key = re.sub(r"\{(\w+)\}", lambda m: _SLOT_SAMPLES[m.group(1)], p.pattern)
        assert registry.resolve(key)[1].pattern == p.pattern, key
        samples[key] = p.value_kind
    top = bridge["top_level"]
    rule_samples = list(top["keys"]) + [f"some_role{s}" for s in top["suffixes"]] + [
        re.sub(r"\\d\+", "7", r) for r in top["patterns"]]
    for key in rule_samples:
        samples.setdefault(key, registry.resolve(key)[1].value_kind)

    java = _classify(sorted(samples))
    wrong = sorted(k for k, kind in samples.items() if java[k] != (kind in TOP_LEVEL_KINDS))
    assert not wrong, (
        "Java Schema.isTopLevel disagrees with keys.json (identity/ref => top level) for: "
        + ", ".join(f"{k} ({samples[k]}, java top={java[k]})" for k in wrong)
    )
    assert bridge["components"] == schema.COMPONENTS
