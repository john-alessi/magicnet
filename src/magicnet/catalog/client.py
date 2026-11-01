"""Client RL workers use to talk to the catalog service.

    client = CatalogClient("http://catalog-host:8765")   # pins the latest version
    cards = client.lookup(deck_names)                     # one request for the deck
    vec = cards["Brainstorm"].vector                      # np.float32 array
    keys = client.keys()                                  # key registry at the pinned version
    kv = client.key_vectors(["life", "ctr_poison"])       # (2, dims) key-description vectors
    sv = client.embed_strings(["Flying", "Kicker {2}{G}"])  # (2, dims) string-value vectors
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx
import numpy as np


@dataclass(frozen=True)
class Card:
    oracle_id: str
    name: str
    mana_cost: str
    cmc: float
    type_line: str
    colors: list[str]
    color_identity: list[str]
    layout: str
    text: str
    text_hash: str
    row_version: int
    vector: np.ndarray | None


@dataclass(frozen=True)
class KeyInfo:
    key: str
    description: str
    value_kind: str  # number | flag | string | string_list | categorical | ref | identity ("unknown" if derived)
    token_types: list[str]
    registered: bool  # False: described by a pattern or derived from the key's name
    source: str  # registry | pattern | derived
    notes: str
    vector: np.ndarray | None  # None for identity keys (not model components)
    vocab: str | None = None  # categorical keys: the closed vocabulary their values come from


def _decode(vec) -> np.ndarray | None:
    return None if vec is None else np.frombuffer(base64.b64decode(vec), dtype="<f4")


class CardNotFound(KeyError):
    pass


class CatalogError(RuntimeError):
    pass


class CatalogClient:
    """Pins a catalog version at construction and sends it as ``as_of`` on
    every request, so a worker sees one consistent set of vectors for its
    whole lifetime. Found cards are cached in memory by (key, version)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        version: int | None = None,
        http: httpx.Client | None = None,
        timeout: float = 600.0,  # a miss may wait for a full refresh
    ):
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self.version = version if version is not None else self._get("/versions/latest")["version"]
        self._cache: dict[tuple[str, int], Card] = {}
        self._keys: dict[tuple[str, int], KeyInfo] = {}
        self._registry: dict[str, KeyInfo] | None = None
        self._strings: dict[str, np.ndarray] = {}  # content-addressed: version-independent

    def _get(self, path: str, **params) -> dict:
        r = self.http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def lookup(self, keys: list[str]) -> dict[str, Card | None]:
        """Resolve names or oracle_ids in one request. Missing/ambiguous keys
        map to None; transient server-side failures raise CatalogError."""
        out: dict[str, Card | None] = {}
        todo = [k for k in dict.fromkeys(keys) if (k, self.version) not in self._cache]
        if todo:
            r = self.http.post("/lookup", json={"keys": todo, "as_of": self.version, "encoding": "b64"})
            r.raise_for_status()
            errors = []
            for res in r.json()["results"]:
                if res["status"] == "found":
                    vec = res["vector"]
                    self._cache[(res["key"], self.version)] = Card(
                        **res["card"],
                        vector=None if vec is None else np.frombuffer(base64.b64decode(vec), dtype="<f4"),
                    )
                elif res["status"] == "error":
                    errors.append(f"{res['key']}: {res['reason']}")
            if errors:
                raise CatalogError("; ".join(errors))
        for k in keys:
            out[k] = self._cache.get((k, self.version))
        return out

    def get(self, key: str) -> Card:
        card = self.lookup([key])[key]
        if card is None:
            raise CardNotFound(key)
        return card

    def vectors(self, keys: list[str]) -> np.ndarray:
        """Stack the vectors for ``keys`` (raises CardNotFound on any miss)."""
        cards = self.lookup(keys)
        missing = [k for k, c in cards.items() if c is None or c.vector is None]
        if missing:
            raise CardNotFound(missing)
        return np.stack([cards[k].vector for k in keys])

    def _post(self, path: str, body: dict) -> dict:
        r = self.http.post(path, json=body)
        if r.status_code == 503:
            raise CatalogError(r.json().get("detail", r.text))
        r.raise_for_status()
        return r.json()

    def keys(self) -> dict[str, KeyInfo]:
        """The whole key registry at the pinned version, with vectors for
        every non-identity key (identity keys have ``vector=None``)."""
        if self._registry is None:
            r = self.http.get("/keys", params={"as_of": self.version, "vectors": True, "encoding": "b64"})
            if r.status_code == 503:
                raise CatalogError(r.json().get("detail", r.text))
            r.raise_for_status()
            self._registry = {}
            for res in r.json()["keys"]:
                info = self._key_info({**res, "registered": True, "source": "registry"})
                self._keys[(info.key, self.version)] = self._registry[info.key] = info
        return dict(self._registry)

    def _key_info(self, res: dict) -> KeyInfo:
        return KeyInfo(res["key"], res["description"], res["value_kind"], res["token_types"],
                       res["registered"], res["source"], res.get("notes", ""), _decode(res.get("vector")),
                       res.get("vocab"))

    def embed_keys(self, keys: list[str]) -> dict[str, KeyInfo]:
        """Describe and embed any keys, registered or not. Unregistered keys
        come back with ``registered=False`` and a pattern/derived description."""
        todo = [k for k in dict.fromkeys(keys) if (k, self.version) not in self._keys]
        for i in range(0, len(todo), 4096):
            body = self._post("/keys/embed", {"keys": todo[i : i + 4096], "as_of": self.version,
                                              "encoding": "b64"})
            for res in body["results"]:
                self._keys[(res["key"], self.version)] = self._key_info(res)
        return {k: self._keys[(k, self.version)] for k in keys}

    def key_vectors(self, keys: list[str]) -> np.ndarray:
        """Stack key-description vectors; raises KeyError for identity keys."""
        infos = self.embed_keys(keys)
        bad = [k for k in keys if infos[k].vector is None]
        if bad:
            raise KeyError(f"identity keys have no vector: {bad}")
        return np.stack([infos[k].vector for k in keys])

    def embed_strings(self, strings: list[str]) -> np.ndarray:
        """(len(strings), dims) float32 vectors for arbitrary short strings
        (keyword names, type lines, chosen types, ...), cached in memory."""
        todo = [s for s in dict.fromkeys(strings) if s not in self._strings]
        for i in range(0, len(todo), 4096):
            body = self._post("/strings/embed", {"strings": todo[i : i + 4096], "encoding": "b64"})
            for res in body["results"]:
                self._strings[res["text"]] = _decode(res["vector"])
        if not strings:
            return np.zeros((0, 0), dtype="<f4")
        return np.stack([self._strings[s] for s in strings])

    def names(self, fmt: str = "json") -> dict[str, str] | str:
        r = self.http.get("/names", params={"version": self.version, "format": fmt})
        r.raise_for_status()
        return r.json()["names"] if fmt == "json" else r.text

    def close(self) -> None:
        self.http.close()
