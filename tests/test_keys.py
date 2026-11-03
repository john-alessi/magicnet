"""Drift guard for the key catalog (src/magicnet/catalog/keys.json).

Every property key the Java bridge can emit must be registered -- with a
description, a value_kind, or an explicit ``identity`` classification -- or a
new key would reach the encoder with no key embedding. Like the relation scan in
test_schema.py, this reads the Java sources rather than trusting a list.

``MAGICNET_BRIDGE_SRC`` points the scan at another source tree (used to show the
test fails on an unregistered key without editing the real bridge).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

from magicnet.catalog.keys import COUNTERS_PATH, load_registry

REPO = Path(__file__).resolve().parents[1]
BRIDGE_SRC = Path(os.environ.get("MAGICNET_BRIDGE_SRC", REPO / "forge-bridge" / "src"))
FORGE_JAR = Path(os.environ.get("MAGICNET_FORGE_JAR", "/usr/share/java/forge-gui-desktop.jar"))
# Inner fields of an edge object ({"rel": ..., "to": ...}), not token keys.
EDGE_FIELDS = {"rel", "to"}
# Files whose hand-written JSON is a token, an action option or a decision record.
JSON_EMITTERS = {"Serializer.java", "StateExtractor.java"}

# Tok.p(key, value) and Tok.flag(key, bool) (schema 9: flags only when true), and
# HistoryHub's record modifiers, r.mods.put(key, value) (schema 13), which
# GameHistory copies onto the token.
_P_CALL = re.compile(r"\.(?:p|flag|mods\.put)\(\s*([^,]+?)\s*,")
_LITERAL = re.compile(r'^"([^"]*)"$')
_JSON_FIELD = re.compile(r'\\"([A-Za-z_][A-Za-z0-9_]*)\\":')
_ROLE_CALL = re.compile(r"\b(card|player|entity|zone)\(\s*(?:t|ev\(\"[^\"]*\"\))\s*,\s*([^,]+?)\s*,")


def _norm(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def scan_bridge(src: Path) -> dict[str, set[str]]:
    """Keys the bridge emits, by how they are emitted."""
    if not src.exists():
        pytest.skip(f"bridge sources not present at {src}")
    out = {"literal": set(), "dynamic": set(), "json": set(), "role": set(), "dynamic_role": set()}
    roles: dict[str, set[str]] = {}
    for path in sorted(src.rglob("*.java")):
        text = path.read_text()
        for m in _P_CALL.finditer(text):
            lit = _LITERAL.match(m.group(1))
            (out["literal"].add(lit.group(1)) if lit else out["dynamic"].add(_norm(m.group(1))))
        if path.name in JSON_EMITTERS:
            out["json"].update(_JSON_FIELD.findall(text))
        for helper, arg in _ROLE_CALL.findall(text):
            lit = _LITERAL.match(arg)
            if lit:
                roles.setdefault(helper, set()).add(lit.group(1))
                out["role"].add(lit.group(1))
            else:
                out["dynamic_role"].add(_norm(arg))
    out["roles_by_helper"] = roles  # type: ignore[assignment]
    return out


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def scan():
    return scan_bridge(BRIDGE_SRC)


def test_every_literal_key_is_registered(registry, scan) -> None:
    emitted = scan["literal"] | scan["json"]
    assert emitted, "scan found no keys; the regex no longer matches the bridge"
    known = set(registry.keys) | set(registry.record_fields) | EDGE_FIELDS
    missing = sorted(emitted - known)
    assert not missing, (
        f"the bridge emits keys missing from src/magicnet/catalog/keys.json: {missing}. "
        "Register each with a description and value_kind (or value_kind identity if it is "
        "a join key, not a component)."
    )


def test_every_dynamic_key_expression_has_a_pattern(registry, scan) -> None:
    covered = {_norm(j) for p in registry.patterns for j in p.java} | set(registry.role_java_exprs)
    missing = sorted((scan["dynamic"] | scan["dynamic_role"]) - covered)
    assert not missing, (
        f"computed keys {missing} are not covered by any pattern's `java` list in keys.json; "
        "add a pattern (e.g. prefix_{name}) with a description template."
    )


def test_history_has_no_role_named_keys(registry, scan) -> None:
    """Schema 13: history participants are edges, so no key is computed from a
    role name any more -- the old card()/player()/zone() role helpers are gone
    and their key families are no longer registered."""
    assert not scan["roles_by_helper"], scan["roles_by_helper"]
    for gone in ("card_id", "target_0", "actor", "from_player", "event"):
        assert gone not in registry.keys, gone
    assert registry.resolve("target_7_id")[0] == "derived"


def test_dynamic_families_resolve_to_patterns(registry) -> None:
    for key, pattern, kind in [("announce_Multikicker", "announce_{what}", "number"),
                               ("ctr_brand_new", "ctr_{name}", "number")]:
        source, entry = registry.resolve(key)
        assert (source, entry.pattern, entry.value_kind) == ("pattern", pattern, kind), key
    assert registry.resolve("ctr_brand_new")[1].description == "brand new counters"
    assert registry.resolve("mystery_flag") == ("derived", registry.resolve("mystery_flag")[1])
    assert registry.resolve("mystery_flag")[1].description == "mystery flag"


def test_every_forge_counter_type_is_registered(registry) -> None:
    generated = json.loads(COUNTERS_PATH.read_text())["counters"]
    assert len(generated) > 200
    missing = [c["key"] for c in generated if c["key"] not in registry.keys]
    assert not missing, missing
    assert registry.keys["ctr_poison"].description == "poison counters"
    assert registry.keys["ctr_+1/+1"].value_kind == "number"


def test_counter_list_matches_live_forge() -> None:
    """counter_keys.json must be regenerated when the Forge jar changes."""
    if shutil.which("javac") is None or not FORGE_JAR.exists():
        pytest.skip("javac or the Forge jar is not available")
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from gen_counter_keys import forge_counter_types
    finally:
        sys.path.pop(0)
    live = {c["key"] for c in forge_counter_types()}
    stored = {c["key"] for c in json.loads(COUNTERS_PATH.read_text())["counters"]}
    assert live == stored, (
        f"Forge counters changed (+{sorted(live - stored)} -{sorted(stored - live)}); "
        "run `uv run python scripts/gen_counter_keys.py`"
    )
