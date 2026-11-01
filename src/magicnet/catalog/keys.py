"""The key catalog: what every (key, value) component key means.

The bridge emits each token's model inputs under its ``components`` map (schema
9; identity and ref keys stay at the token's top level). The model encodes each
component as ``MLP(concat(key_embedding, value_repr))``, so every key needs a readable
``description`` that is embedded with the same frozen text embedder as card
oracle text; keys the model never saw then still land somewhere sensible.

Source of truth is ``keys.json`` (hand-curated) plus ``counter_keys.json``
(generated from Forge by ``scripts/gen_counter_keys.py``), expanded through the
``ctr_{name}`` pattern. ``tests/test_keys.py`` fails when the Java bridge emits a
key that is not registered.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS_PATH = HERE / "keys.json"
COUNTERS_PATH = HERE / "counter_keys.json"

VALUE_KINDS = ("number", "flag", "string", "string_list", "categorical", "ref", "identity")
# Placeholder -> regex. Anything else matches one or more characters.
_SLOTS = {"i": r"\d+", "role": r"[a-z][a-z0-9_]*?"}


@dataclass(frozen=True)
class KeyEntry:
    key: str
    description: str
    value_kind: str
    token_types: tuple[str, ...]
    notes: str = ""
    alt_value_kinds: tuple[str, ...] = ()
    pattern: str | None = None  # family this key was expanded from / matched by
    vocab: str | None = None  # categorical: name of the closed vocabulary its values come from
    top_level: bool = False  # emitted at the token's top level although it is a component

    @property
    def embedded(self) -> bool:
        """Whether this key is a model component (identity keys are not)."""
        return self.value_kind != "identity"

    def meta(self) -> dict:
        return {k: v for k, v in (("notes", self.notes), ("alt_value_kinds", list(self.alt_value_kinds)),
                                   ("pattern", self.pattern), ("vocab", self.vocab),
                                   ("top_level", self.top_level)) if v}

    def to_json(self) -> dict:
        return {"key": self.key, "description": self.description, "value_kind": self.value_kind,
                "token_types": list(self.token_types), **self.meta()}


@dataclass(frozen=True)
class KeyPattern:
    pattern: str  # e.g. "ctr_{name}"
    description: str  # template, e.g. "{name} counters"
    value_kind: str
    token_types: tuple[str, ...]
    notes: str = ""
    java: tuple[str, ...] = ()  # Java expressions that produce keys of this family
    expand: str | None = None
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        parts, pos = [], 0
        for m in re.finditer(r"\{(\w+)\}", self.pattern):
            parts.append(re.escape(self.pattern[pos:m.start()]))
            parts.append(f"(?P<{m.group(1)}>{_SLOTS.get(m.group(1), '.+')})")
            pos = m.end()
        parts.append(re.escape(self.pattern[pos:]))
        object.__setattr__(self, "regex", re.compile("^" + "".join(parts) + "$"))

    def match(self, key: str) -> KeyEntry | None:
        m = self.regex.match(key)
        if m is None:
            return None
        slots = {k: v.replace("_", " ") for k, v in m.groupdict().items()}
        return KeyEntry(key, self.description.format(**slots), self.value_kind, self.token_types,
                        self.notes, pattern=self.pattern)

    def to_json(self) -> dict:
        out = {"pattern": self.pattern, "description": self.description, "value_kind": self.value_kind,
               "token_types": list(self.token_types)}
        for k, v in (("notes", self.notes), ("java", list(self.java)), ("expand", self.expand)):
            if v:
                out[k] = v
        return out


def derive_description(key: str) -> str:
    """Readable phrase for a key nobody registered: ``ctr_oil`` -> "oil
    counters", ``snake_case`` -> "snake case"."""
    if key.startswith("ctr_") and len(key) > 4:
        return f"{key[4:].replace('_', ' ')} counters"
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key).replace("_", " ").replace("-", " ")
    return " ".join(words.split()).lower() or key


def counter_description(c: dict) -> str:
    name = c["name"].lower()
    if c.get("keyword"):
        return f"{name} counters (keyword counter granting {name})"
    return f"{name} counters"


@dataclass
class Registry:
    keys: dict[str, KeyEntry]  # concrete keys, incl. expanded counters
    patterns: list[KeyPattern]  # in match order: first match wins
    role_java_exprs: tuple[str, ...] = ()
    record_fields: tuple[str, ...] = ()

    def resolve(self, key: str) -> tuple[str, KeyEntry]:
        """(source, entry): source is "registry", "pattern" or "derived"."""
        if key in self.keys:
            return "registry", self.keys[key]
        for p in self.patterns:
            hit = p.match(key)
            if hit is not None:
                return "pattern", hit
        return "derived", KeyEntry(key, derive_description(key), "unknown", ())

    def is_top_level(self, key: str) -> bool:
        """Whether the bridge writes this key at the token's top level rather
        than under ``components``: identity and ref keys, plus components
        flagged ``top_level`` (e.g. ``zone``, which is also structural)."""
        entry = self.resolve(key)[1]
        return entry.value_kind in ("identity", "ref") or entry.top_level

    def entries(self) -> list[dict]:
        """Flat rows for the store: concrete keys, then patterns."""
        rows = [{"key": e.key, "is_pattern": False, "description": e.description, "value_kind": e.value_kind,
                 "token_types": list(e.token_types), "meta": e.meta()} for e in self.keys.values()]
        for order, p in enumerate(self.patterns):
            j = p.to_json()
            rows.append({"key": p.pattern, "is_pattern": True, "description": p.description,
                         "value_kind": p.value_kind, "token_types": list(p.token_types),
                         "meta": {k: v for k, v in j.items() if k in ("notes", "java", "expand")} | {"order": order}})
        return rows

    @property
    def hash(self) -> str:
        blob = json.dumps(self.entries(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def embed_texts(self) -> list[str]:
        """Distinct descriptions that get a vector (non-identity concrete keys)."""
        return list(dict.fromkeys(e.description for e in self.keys.values() if e.embedded))


def _check(kind: str, where: str) -> None:
    if kind not in VALUE_KINDS:
        raise ValueError(f"{where}: value_kind {kind!r} not in {VALUE_KINDS}")


def load_registry(keys_path: Path = KEYS_PATH, counters_path: Path = COUNTERS_PATH) -> Registry:
    raw = json.loads(Path(keys_path).read_text())
    keys: dict[str, KeyEntry] = {}
    for k in raw["keys"]:
        _check(k["value_kind"], k["key"])
        for alt in k.get("alt_value_kinds", ()):
            _check(alt, k["key"])
        if k["key"] in keys:
            raise ValueError(f"duplicate key {k['key']!r} in {keys_path}")
        if (k["value_kind"] == "categorical") != ("vocab" in k):
            raise ValueError(f"{k['key']}: categorical keys (and only they) must name a vocab")
        keys[k["key"]] = KeyEntry(k["key"], k["description"], k["value_kind"], tuple(k["token_types"]),
                                  k.get("notes", ""), tuple(k.get("alt_value_kinds", ())),
                                  vocab=k.get("vocab"), top_level=bool(k.get("top_level", False)))
    patterns = []
    for p in raw["patterns"]:
        _check(p["value_kind"], p["pattern"])
        patterns.append(KeyPattern(p["pattern"], p["description"], p["value_kind"], tuple(p["token_types"]),
                                   p.get("notes", ""), tuple(p.get("java", ())), p.get("expand")))
    for p in patterns:
        if p.expand == "forge_counters":
            for c in json.loads(Path(counters_path).read_text())["counters"]:
                if c["key"] in keys:
                    raise ValueError(f"counter key {c['key']!r} collides with a hand-registered key")
                if not p.regex.match(c["key"]):
                    raise ValueError(f"counter key {c['key']!r} does not match {p.pattern}")
                notes = "keyword counter" if c.get("keyword") else f"Forge CounterEnumType.{c['enum']}"
                keys[c["key"]] = KeyEntry(c["key"], counter_description(c), p.value_kind, p.token_types,
                                          notes, pattern=p.pattern)
        elif p.expand is not None:
            raise ValueError(f"unknown expansion {p.expand!r} for {p.pattern}")
    return Registry(keys, patterns, tuple(raw.get("role_java_exprs", ())), tuple(raw.get("record_fields", ())))


def estimate_tokens(texts: list[str]) -> int:
    """Rough cl100k-style token count (~4 characters per token, at least 1)."""
    return sum(max(1, -(-len(t) // 4)) for t in texts)


def registry_from_rows(rows: list[dict]) -> Registry:
    """Rebuild a Registry from ``CatalogStore.keys_at`` rows (a pinned version)."""
    keys = {
        r["key"]: KeyEntry(r["key"], r["description"], r["value_kind"], tuple(r["token_types"]),
                           r.get("notes", ""), tuple(r.get("alt_value_kinds", ())), r.get("pattern"),
                           r.get("vocab"), bool(r.get("top_level", False)))
        for r in rows if not r["is_pattern"]
    }
    patterns = [
        KeyPattern(r["key"], r["description"], r["value_kind"], tuple(r["token_types"]), r.get("notes", ""),
                   tuple(r.get("java", ())), r.get("expand"))
        for r in sorted((r for r in rows if r["is_pattern"]), key=lambda r: r.get("order", 0))
    ]
    return Registry(keys, patterns)
