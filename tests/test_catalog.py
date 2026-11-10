"""Card catalog service: miss handling, single-flight refresh, versioning."""

from __future__ import annotations

import asyncio
import gzip
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from magicnet.catalog import CatalogClient, CatalogService, CatalogStore, create_app
from magicnet.catalog.bootstrap import import_existing
from magicnet.catalog.store import sha256

T1, T2 = "2026-10-01T09:00:00+00:00", "2026-10-02T09:00:00+00:00"


def card(oid: str, name: str, text: str = "", **kw) -> dict:
    return {"lang": "en", "layout": "normal", "oracle_id": oid, "name": name,
            "mana_cost": "{U}", "cmc": 1.0, "type_line": "Instant", "oracle_text": text, **kw}


def write_dump(path: Path, cards: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")
    return path


BASE = [card("a", "Alpha", "Draw a card."), card("b", "Beta", "Deal 2 damage."),
        card("c", "Gamma", "Gain 3 life.")]
NEXT = [card("a", "Alpha", "Draw a card."), card("b", "Beta", "Deal 3 damage."),  # errata
        card("c", "Gamma", "Gain 3 life."), card("d", "Delta", "Scry 1.")]          # new card


class FakeEmbedder:
    model, dims = "fake-embed", 4

    def __init__(self):
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return np.array([np.frombuffer(bytes.fromhex(sha256(t))[:16], dtype="<f4") for t in texts])


class FakeScryfall:
    """Serves bulk-data metadata for `updated_at`; download can be held open."""

    def __init__(self, tmp: Path, updated_at: str, cards: list[dict]):
        self.tmp, self.updated_at, self.cards = tmp, updated_at, cards
        self.metadata_calls = self.downloads = 0
        self.gate = threading.Event()
        self.gate.set()
        self.download_started = threading.Event()

    def bulk_metadata(self):
        self.metadata_calls += 1
        return {"updated_at": self.updated_at, "jsonl_download_uri": "fake://dump.jsonl.gz"}

    def download(self, metadata, dest):
        self.downloads += 1
        self.download_started.set()
        assert self.gate.wait(10), "test never released the download"
        shutil.copy(write_dump(self.tmp / f"src-{self.downloads}.jsonl.gz", self.cards), dest)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path: Path):
    """A catalog at version 1 (BASE, dump T1) plus a service around it."""
    embedder = FakeEmbedder()
    store = CatalogStore(tmp_path / "cat.sqlite")
    store.refresh_from_dump(write_dump(tmp_path / "base.jsonl.gz", BASE), dump_updated_at=T1,
                            embedder_factory=lambda: embedder)
    store.close()
    embedder.embedded.clear()
    source = FakeScryfall(tmp_path, T1, BASE)
    clock = Clock()
    service = CatalogService(tmp_path / "cat.sqlite", source, lambda: embedder, tmp_path / "dumps",
                             check_interval=300, clock=clock)
    return service, source, embedder, clock


def statuses(results: dict) -> dict:
    return {k: v["status"] for k, v in results.items()}


def test_miss_against_current_dump_is_not_found_and_negatively_cached(env):
    service, source, _, clock = env
    _, res = asyncio.run(service.lookup(["Alpha", "Bogus's Effect"]))
    assert statuses(res) == {"Alpha": "found", "Bogus's Effect": "not_found"}
    assert source.metadata_calls == 1 and source.downloads == 0
    clock.t += 10_000  # past the metadata rate limit: only the negative cache prevents a call
    _, res = asyncio.run(service.lookup(["Bogus's Effect"]))
    assert statuses(res) == {"Bogus's Effect": "not_found"}
    assert source.metadata_calls == 1


def test_metadata_check_is_rate_limited_across_distinct_misses(env):
    service, source, _, clock = env
    for i in range(5):
        asyncio.run(service.lookup([f"Token {i}"]))
    assert source.metadata_calls == 1
    clock.t += 301
    asyncio.run(service.lookup(["Token 99"]))
    assert source.metadata_calls == 2


def test_miss_against_stale_dump_refreshes_then_finds(env):
    service, source, _, _ = env
    source.updated_at, source.cards = T2, NEXT
    version, res = asyncio.run(service.lookup(["Delta"]))
    assert res["Delta"]["status"] == "found" and res["Delta"]["card"]["oracle_id"] == "d"
    assert source.downloads == 1
    assert service.store.latest_version() == 2 and service.store.dump_updated_at() == T2


def test_concurrent_misses_share_one_refresh(env):
    service, source, _, _ = env
    source.updated_at, source.cards = T2, NEXT
    source.gate.clear()

    async def go():
        first = [asyncio.create_task(service.lookup([k])) for k in ("Delta", "d", "Nope 1")]
        await asyncio.to_thread(source.download_started.wait, 5)
        late = [asyncio.create_task(service.lookup([k])) for k in ("Delta", "Nope 2")]
        await asyncio.sleep(0.05)
        source.gate.set()
        return await asyncio.gather(*first, *late)

    results = asyncio.run(go())
    assert source.downloads == 1 and source.metadata_calls == 1
    merged = {k: v["status"] for _, r in results for k, v in r.items()}
    assert merged == {"Delta": "found", "d": "found", "Nope 1": "not_found", "Nope 2": "not_found"}


def test_present_cards_are_served_while_refresh_is_blocked(env):
    service, source, _, _ = env
    source.updated_at, source.cards = T2, NEXT
    source.gate.clear()

    async def go():
        waiting = asyncio.create_task(service.lookup(["Delta"]))
        await asyncio.to_thread(source.download_started.wait, 5)
        _, res = await asyncio.wait_for(service.lookup(["Alpha", "b"]), timeout=2)
        assert statuses(res) == {"Alpha": "found", "b": "found"}
        assert not waiting.done()
        source.gate.set()
        return await waiting

    _, res = asyncio.run(go())
    assert res["Delta"]["status"] == "found"


def test_refresh_embeds_only_new_and_changed_cards(env, tmp_path):
    service, _, embedder, _ = env
    report = service.store.refresh_from_dump(write_dump(tmp_path / "next.jsonl.gz", NEXT),
                                             dump_updated_at=T2, embedder_factory=lambda: embedder)
    assert sorted(embedder.embedded) == ["Beta | {U} | Instant | Deal 3 damage.",
                                         "Delta | {U} | Instant | Scry 1."]
    assert (report.added, report.changed, report.embedded) == (1, 1, 2)
    assert service.store.changes(2) == {"added": ["d"], "changed": ["b"], "updated": [], "removed": []}


def test_errata_keeps_old_row_and_vector_for_as_of(env, tmp_path):
    service, _, embedder, _ = env
    store = service.store
    old = store.resolve("Beta", 1)
    store.refresh_from_dump(write_dump(tmp_path / "next.jsonl.gz", NEXT),
                            dump_updated_at=T2, embedder_factory=lambda: embedder)
    new = store.resolve("Beta", 2)
    assert "Deal 3" in new.card["text"] and not np.array_equal(new.vector, old.vector)
    again = store.resolve("Beta", 1)
    assert "Deal 2" in again.card["text"] and np.array_equal(again.vector, old.vector)
    # A card that did not exist at version 1 resolves to its latest row.
    assert store.resolve("Delta", 1).card["row_version"] == 2


def test_full_name_beats_face_name_and_ambiguous_names_are_not_guessed(tmp_path):
    store = CatalogStore(tmp_path / "cat.sqlite")
    store.refresh_from_dump(write_dump(tmp_path / "d.jsonl.gz", [
        card("trio", "Harmonized Trio // Brainstorm", layout="prepare",
             card_faces=[{"name": "Harmonized Trio"}, {"name": "Brainstorm"}]),
        card("brainstorm", "Brainstorm", "Draw three cards."),
        card("fire-ice", "Fire // Ice", layout="split", card_faces=[{"name": "Fire"}, {"name": "Ice"}]),
        card("fire-fury", "Fire // Fury", layout="split", card_faces=[{"name": "Fire"}, {"name": "Fury"}]),
    ]), dump_updated_at=T1, embedder_factory=FakeEmbedder)
    assert store.resolve("Brainstorm", 1).oracle_id == "brainstorm"
    assert store.resolve("Harmonized Trio", 1).oracle_id == "trio"
    fire = store.resolve("Fire", 1)
    assert fire.status == "ambiguous" and sorted(fire.candidates) == ["fire-fury", "fire-ice"]


def test_client_pins_version_and_batches_over_http(env, tmp_path):
    service, source, embedder, _ = env
    with TestClient(create_app(service)) as http:
        client = CatalogClient(http=http)
        assert client.version == 1
        service.store.refresh_from_dump(write_dump(tmp_path / "next.jsonl.gz", NEXT),
                                        dump_updated_at=T2, embedder_factory=lambda: embedder)
        cards = client.lookup(["Beta", "Delta", "Nope"])
        assert "Deal 2" in cards["Beta"].text  # pinned to version 1's errata state
        assert cards["Delta"].row_version == 2 and cards["Nope"] is None
        assert client.vectors(["Beta", "Delta"]).shape == (2, 4)
        assert http.get("/names", params={"version": 2}).text.splitlines()[0] == "Alpha\ta"
        assert http.get("/cards/Nope").status_code == 404
        assert CatalogClient(http=http).get("Beta").text.endswith("Deal 3 damage.")


def test_bootstrap_import_then_same_dump_refresh_embeds_nothing(tmp_path):
    dump = write_dump(tmp_path / "dump.jsonl.gz", BASE + [card("dig", "A-Alpha", digital=True)])
    from magicnet.cards import iter_unique_cards

    records = [r.to_json() for r in iter_unique_cards(dump)]
    (tmp_path / "cards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    np.save(tmp_path / "emb.npy", np.random.default_rng(0).random((len(records), 4), dtype=np.float32))
    (tmp_path / "emb.ids.json").write_text(json.dumps([r["oracle_id"] for r in records]))

    store = CatalogStore(tmp_path / "cat.sqlite")
    report, info = import_existing(store, dump_path=dump, cards_path=tmp_path / "cards.jsonl",
                                   embeddings_path=tmp_path / "emb.npy", ids_path=tmp_path / "emb.ids.json",
                                   model="fake-embed", dims=4, dump_updated_at=T1)
    assert report.version == 1 and report.missing_vectors == 0 and info["include_digital"] is False
    embedder = FakeEmbedder()
    again = store.refresh_from_dump(dump, dump_updated_at=T2, embedder_factory=lambda: embedder)
    assert embedder.embedded == [] and (again.added, again.changed) == (0, 0)
    assert store.resolve("A-Alpha", 2).status == "excluded"


# -- key catalog and string embeddings ---------------------------------------

from magicnet.catalog.keys import load_registry  # noqa: E402


def sync_registry(store, embedder, *, life="life total", dry_run=False):
    entries = [dict(e, description=life) if e["key"] == "life" else e for e in load_registry().entries()]
    return store.sync_keys(entries, registry_hash=sha256(json.dumps(entries)), dry_run=dry_run,
                           embedder_factory=None if dry_run else (lambda: embedder))


def test_key_sync_dry_run_writes_nothing_then_versions_registry_changes(env):
    service, _, embedder, _ = env
    store = service.store
    dry = sync_registry(store, embedder, dry_run=True)
    n_embedded_keys = sum(e.embedded for e in load_registry().keys.values())
    assert (dry.version, dry.to_embed, dry.embedded) == (None, n_embedded_keys, 0) and dry.est_tokens > 0
    assert embedder.embedded == [] and store.latest_version() == 1 and store.keys_at(1) == []

    first = sync_registry(store, embedder)
    assert first.version == 2 and first.embedded == n_embedded_keys == len(embedder.embedded)
    assert "id" in first.added and not any(t in ("object id", "card name") for t in embedder.embedded)
    again = sync_registry(store, embedder)
    assert (again.version, again.to_embed, again.added) == (None, 0, [])

    changed = sync_registry(store, embedder, life="my life total")
    assert (changed.version, changed.changed, changed.embedded) == (3, ["life"], 1)
    life = {v: next(k for k in store.keys_at(v) if k["key"] == "life")["description"] for v in (1, 2, 3)}
    assert life == {1: "life total", 2: "life total", 3: "my life total"}  # v1 predates keys: first row, so pins are immutable
    # A registry version leaves cards and Scryfall freshness alone.
    assert store.dump_updated_at() == T1 and store.resolve("Alpha", 3).status == "found"
    _, res = asyncio.run(service.lookup(["Alpha", "Nope"], as_of=3))
    assert statuses(res) == {"Alpha": "found", "Nope": "not_found"}


def test_key_lookup_respects_as_of_and_describes_unregistered_keys(env):
    service, _, embedder, _ = env
    sync_registry(service.store, embedder)
    sync_registry(service.store, embedder, life="my life total")
    embedder.embedded.clear()
    keys = ["life", "ctr_poison", "id", "announce_X", "ctr_zorbo", "mysteryThing_count"]
    _, old = asyncio.run(service.lookup_keys(keys, as_of=2))
    _, new = asyncio.run(service.lookup_keys(keys, as_of=3))
    old, new = {r["key"]: r for r in old}, {r["key"]: r for r in new}
    assert old["life"]["description"] == "life total" and new["life"]["description"] == "my life total"
    assert not np.array_equal(old["life"]["vector"], new["life"]["vector"])
    assert np.array_equal(old["ctr_poison"]["vector"], new["ctr_poison"]["vector"])
    assert new["id"]["value_kind"] == "identity" and new["id"]["vector"] is None
    got = {k: (r["registered"], r["source"], r["description"]) for k, r in new.items() if not r["registered"]}
    assert got == {"announce_X": (False, "pattern", "announced value of X"),
                   "ctr_zorbo": (False, "pattern", "zorbo counters"),
                   "mysteryThing_count": (False, "derived", "mystery thing count")}
    assert sorted(embedder.embedded) == sorted(["announced value of X", "zorbo counters",
                                                "mystery thing count"])  # registered ones were cached
    assert {r["key"]: r["hits"] for r in service.store.unregistered_keys()} == {
        "announce_X": 2, "ctr_zorbo": 2, "mysteryThing_count": 2}


class GatedEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.gate, self.started, self.calls = threading.Event(), threading.Event(), 0

    def embed(self, texts):
        self.calls += 1
        self.started.set()
        assert self.gate.wait(10), "test never released the embedder"
        return super().embed(texts)


def test_concurrent_string_misses_are_embedded_once(env):
    service, _, _, _ = env
    gated = GatedEmbedder()
    service.embedder_factory = lambda: gated

    async def go():
        first = asyncio.create_task(service.embed_strings(["Flying", "Kicker {2}{G}"]))
        await asyncio.to_thread(gated.started.wait, 5)
        second = asyncio.create_task(service.embed_strings(["Kicker {2}{G}", "Flying", "Flying"]))
        await asyncio.sleep(0.05)
        assert not first.done() and not second.done()
        gated.gate.set()
        return await asyncio.gather(first, second)

    (a, a_cached), (b, b_cached) = asyncio.run(go())
    assert gated.calls == 1 and sorted(gated.embedded) == ["Flying", "Kicker {2}{G}"]
    assert np.array_equal(a["Flying"], b["Flying"]) and a_cached == b_cached == set()
    vecs, cached = asyncio.run(service.embed_strings(["Flying"]))
    assert cached == {"Flying"} and gated.calls == 1 and np.array_equal(vecs["Flying"], a["Flying"])


def test_client_keys_and_strings_over_http(env):
    service, _, embedder, _ = env
    sync_registry(service.store, embedder)
    embedder.embedded.clear()
    with TestClient(create_app(service)) as http:
        client = CatalogClient(http=http)
        assert client.version == 2
        sync_registry(service.store, embedder, life="my life total")  # version 3, after pinning
        keys = client.keys()
        assert keys["life"].description == "life total" and keys["life"].vector.shape == (4,)
        assert keys["ctr_poison"].value_kind == "number" and keys["oracle_id"].vector is None
        info = client.embed_keys(["life", "ctr_newthing"])
        assert info["ctr_newthing"].registered is False and info["ctr_newthing"].source == "pattern"
        assert client.key_vectors(["life", "ctr_newthing"]).shape == (2, 4)
        before = list(embedder.embedded)
        vecs = client.embed_strings(["Flying", "Human Wizard", "Flying"])
        assert vecs.shape == (3, 4) and vecs.dtype == np.float32 and np.array_equal(vecs[0], vecs[2])
        assert embedder.embedded[len(before):] == ["Flying", "Human Wizard"]
        client.embed_strings(["Flying"])  # served from the client's cache
        assert len(embedder.embedded) == len(before) + 2
        body = http.get("/keys", params={"as_of": 3}).json()
        assert {k["key"]: k["description"] for k in body["keys"]}["life"] == "my life total"
        assert body["patterns"][0]["key"] == "ctr_{name}"
        assert http.post("/strings/embed", json={"strings": ["x" * 9000]}).status_code == 413


# -- PCA basis ---------------------------------------------------------------

from magicnet.catalog import pca as pcamod  # noqa: E402


def lowrank(n: int, dims: int = 16, rank: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, rank)) * np.arange(rank, 0, -1) @ rng.normal(size=(rank, dims))
    x += 0.1 * rng.normal(size=(n, dims))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_pca_fit_is_deterministic_orthonormal_and_prefix_consistent():
    x = lowrank(300)
    a, b = pcamod.fit(x, 8), pcamod.fit(x.copy(), 8)
    for f in ("mean", "components", "eigenvalues", "baseline_fuv"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert np.allclose(a.components @ a.components.T, np.eye(8), atol=1e-12)
    assert np.all(np.diff(a.eigenvalues) <= 0)
    rows = np.arange(8)
    assert np.all(a.components[rows, np.abs(a.components).argmax(axis=1)] > 0)  # sign convention
    small = pcamod.fit(x, 3)
    assert np.allclose(small.components, a.components[:3], atol=1e-12)
    assert np.allclose(small.eigenvalues, a.eigenvalues[:3]) and np.allclose(small.baseline_fuv, a.baseline_fuv[:3])
    assert pcamod.fit(x, 999).components.shape == (16, 16)  # K capped at dims


def test_pca_basis_whitening_and_round_trip_error_is_discarded_variance():
    x = lowrank(400)
    f = pcamod.fit(x, 16)
    for d in (2, 5):
        basis = pcamod.PCABasis(1, d, f.mean, f.components[:d], f.std[:d])
        z = basis.project(x)
        assert z.shape == (400, d) and np.allclose(z.var(axis=0), 1.0) and np.allclose(z.mean(axis=0), 0, atol=1e-9)
        err = ((basis.inverse(z) - x) ** 2).sum(axis=1).mean()
        assert np.isclose(err, f.total_variance - f.eigenvalues[:d].sum())
        assert np.allclose(basis.inverse(basis.project(x, whiten=False), whitened=False), basis.inverse(z))
        assert basis.project(x[0]).shape == (d,) and basis.project(x.reshape(20, 20, 16)).shape == (20, 20, d)

    import torch

    xt = torch.as_tensor(x, dtype=torch.float32)
    basis = pcamod.PCABasis(1, 4, f.mean.astype("<f4"), f.components[:4].astype("<f4"), f.std[:4].astype("<f4"))
    zt = basis.project(xt)
    assert isinstance(zt, torch.Tensor) and zt.dtype == torch.float32 and zt.device == xt.device
    assert np.allclose(zt.numpy(), basis.project(x.astype("<f4")), atol=1e-5)
    assert isinstance(basis.inverse(zt), torch.Tensor)


class PCAEmbedder(FakeEmbedder):
    """16-d unit vectors near a rank-4 subspace; texts containing "Odd" are isotropic."""
    model, dims = "fake-embed", 16
    W = np.random.default_rng(0).normal(size=(4, 16))

    def embed(self, texts):
        self.embedded.extend(texts)
        out = []
        for t in texts:
            rng = np.random.default_rng(int(sha256(t)[:12], 16))
            v = rng.normal(size=16) if "Odd" in t else rng.normal(size=4) * [4, 3, 2, 1] @ self.W + 0.1 * rng.normal(size=16)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


POOL = [card(f"p{i:02d}", f"Card {i}", f"Effect {i}.") for i in range(40)]


@pytest.fixture
def pca_env(tmp_path: Path):
    embedder = PCAEmbedder()
    store = CatalogStore(tmp_path / "cat.sqlite", drift_dims=(4, 8))
    store.refresh_from_dump(write_dump(tmp_path / "pool.jsonl.gz", POOL + [card("dig", "Digi", digital=True)]),
                            dump_updated_at=T1, embedder_factory=lambda: embedder)
    return store, embedder, tmp_path


def test_pca_fit_stores_versioned_basis_and_refresh_records_drift_without_refit(pca_env):
    store, embedder, tmp_path = pca_env
    with pytest.raises(ValueError, match="not interchangeable"):
        store.fit_pca(8, model="other-model")
    info = store.fit_pca(8)
    assert (info["basis_id"], info["catalog_version"], info["n_cards"], info["n_components"]) == (1, 2, 40, 8)
    assert info["current_from"] == [2] and store.dump_updated_at() == T1
    assert info["cumulative_explained"][8] > 0.95  # rank-4 pool
    again = store.fit_pca(8)  # never overwritten: a new basis with identical arrays
    assert again["basis_id"] == 2 and np.array_equal(store.pca_basis(1)["components"], store.pca_basis(2)["components"])

    ins = POOL + [card(f"n{i}", f"Odd {i}", f"Odd thing {i}.") for i in range(10)]
    report = store.refresh_from_dump(write_dump(tmp_path / "next.jsonl.gz", ins), dump_updated_at=T2,
                                     embedder_factory=lambda: embedder)
    assert report.version == 4 and len(store.pca_bases()) == 2  # refresh does not refit
    drift = store.pca_drift()
    assert [(r["version"], r["basis_id"], r["d"], r["n_cards"]) for r in drift] == [(4, 2, 4, 10), (4, 2, 8, 10)]
    assert all(r["exceeded"] and r["fuv"] > r["baseline"] + 0.1 for r in drift)
    assert store.use_pca(1) == 5 and store.pca_current_id(5) == 1 and store.pca_current_id(4) == 2


def test_pca_over_http_and_client_pins_basis_of_its_version(pca_env):
    store, embedder, tmp_path = pca_env
    service = CatalogService(store.db_path, FakeScryfall(tmp_path, T1, POOL), lambda: embedder, tmp_path / "dumps")
    with TestClient(create_app(service)) as http:
        assert http.get("/pca").status_code == 404
        store.fit_pca(8)  # basis 1 at version 2
        early = CatalogClient(http=http, version=1)  # predates any basis: pins the first one
        pinned = CatalogClient(http=http)
        assert pinned.version == 2
        store.refresh_from_dump(write_dump(tmp_path / "next.jsonl.gz", POOL + [card("n", "Odd", "Odd.")]),
                                dump_updated_at=T2, embedder_factory=lambda: embedder)
        store.fit_pca(6)  # basis 2 at version 4
        b = pinned.pca(4)
        assert (b.basis_id, b.dims, b.components.shape, b.std.shape, b.mean.shape) == (1, 4, (4, 16), (4,), (16,))
        assert early.pca(4).basis_id == 1 and CatalogClient(http=http).pca(4).basis_id == 2
        assert pinned.pca(4, basis_id=2).basis_id == 2 and pinned.pca(4) is b  # cached
        ref = store.pca_basis(1)
        assert np.allclose(b.components, ref["components"][:4]) and np.allclose(b.std, np.sqrt(ref["eigenvalues"][:4]))
        t = pinned.card_targets(["Card 1", "p02"], dims=4)
        assert t.shape == (2, 4) and np.allclose(t, b.project(pinned.vectors(["Card 1", "p02"])))
        body = http.get("/pca", params={"as_of": 3, "components": 2}).json()
        assert body["basis_id"] == 1 and body["components_returned"] == 2 and len(body["explained_variance_ratio"]) == 2
        assert http.get("/pca", params={"basis_id": 1, "components": 9}).status_code == 400
        assert http.get("/pca", params={"basis_id": 7}).status_code == 404
        drift = http.get("/pca/drift").json()
        assert [(r["version"], r["basis_id"], r["n_cards"]) for r in drift] == [(3, 1, 1), (3, 1, 1)]
        assert [b["basis_id"] for b in http.get("/pca/bases").json()] == [1, 2]
