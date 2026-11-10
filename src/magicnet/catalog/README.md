# Card catalog service

Centralises the Scryfall dump, card records and embeddings so RL workers do not
repeat the offline `extract_cards` / `embed_cards` / `build_oracle_map` work.
Storage is one SQLite file (WAL) at `data/catalog/catalog.sqlite`. New Scryfall
dumps are saved to `data/catalog/dumps/`.

```
uv run python -m magicnet.catalog import     # one-off: seed from the existing data/ artefacts
uv run python -m magicnet.catalog serve [--host 0.0.0.0] [--port 8765]
uv run python -m magicnet.catalog refresh    # offline only; while serving use POST /admin/refresh
uv run python -m magicnet.catalog keys-sync --dry-run   # count key descriptions to embed; no writes, no API
uv run python -m magicnet.catalog keys-sync  [--no-embed]  # record keys.json as a version, embed descriptions
uv run python -m magicnet.catalog pca-fit [--components 256] [--note ...] [--no-activate]  # deliberate refit
uv run python -m magicnet.catalog pca-list | pca-use BASIS_ID | pca-drift
```

`import` reads `data/raw/scryfall-default-cards.jsonl.gz`, `data/processed/cards.jsonl`
and `card_embeddings.npy`/`.ids.json` without modifying them. The embedding
model and dims come from `.env` (`AZURE_OPENAI_EMBEDDING_DEPLOYMENT`,
`AZURE_OPENAI_EMBEDDING_DIMENSIONS`) and are checked against the matrix width.
The dump's `updated_at` is taken from the file's mtime. Override these with
`--model/--dims/--updated-at`. Running a refresh against the same dump afterwards
embeds nothing.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/lookup` | `{"keys": [...], "as_of": int?, "encoding": "b64"\|"list"}`. Returns one result per key. |
| GET | `/cards/{key}?as_of=` | Looks up one key. Returns 404 if it is not found or ambiguous, and 503 if Scryfall is unreachable. |
| GET | `/versions` | Lists every catalog version with its dump `updated_at` and added/changed/updated/removed counts. |
| GET | `/versions/latest` | `{"version": n}` |
| GET | `/versions/{v}/changes` | Errata log for a version: oracle_ids under `added`, `changed` (embedding text changed), `updated` (other fields only) and `removed`. |
| GET | `/names?version=&format=tsv\|json` | Name→oracle_id table. The TSV is byte-identical to `scripts/build_oracle_map.py` output. |
| POST | `/admin/refresh` | Forces a Scryfall check and refreshes if the dump is stale. |
| GET | `/health` | Latest version, dump date, embedding model, and whether a refresh is running. |
| GET | `/keys?as_of=&vectors=false&encoding=list\|b64` | Key registry at a version: `keys` (key, description, value_kind, token_types, notes, row_version, optional `vector`) and `patterns`. |
| POST | `/keys/embed` | `{"keys": [...], "as_of": int?, "vectors": true, "encoding": "b64"}`. Describes and embeds any keys, registered or not (see below). |
| GET | `/keys/unregistered` | Keys looked up that are not registered, with hit counts. Use this to curate `keys.json`. |
| GET | `/pca?basis_id=&as_of=&components=d` | A PCA basis (see below): by id, else the one current at `as_of` (default latest). `mean`, `components` (row-major d×dims), `std` are base64 float32; plus metadata and `explained_variance_ratio`. |
| GET | `/pca/bases` | Metadata of every stored basis. |
| GET | `/pca/drift?basis_id=` | The drift log written by refreshes. |
| POST | `/strings/embed` | `{"strings": [...], "encoding": "b64"}`. Returns one vector per distinct string, plus `cached` (true if it was already stored). Limits: 4096 strings per request, 8192 characters per string. |

A key can be an oracle_id or a card name. Names use the same rules as
`cards.oracle_name_index`: a full name beats another card's face name, and an
ambiguous name returns `status: "ambiguous"` with `candidates` instead of a
guess. Each result has a `status` of `found`, `not_found`, `ambiguous` or
`error`. `found` results include `card` (name, mana_cost, cmc, type_line, colors,
color_identity, layout, text, text_hash, row_version) and `vector`. With
`b64` encoding (the default), `vector` is base64 of little-endian float32.

## Misses

When a key is not in the catalog, the service checks Scryfall's
`GET /bulk-data/default-cards` metadata. It does not download the dump just to check.

- If the stored dump is current, the result is `not_found`. The key is then
  negatively cached until the stored dump's `updated_at` changes (12 h TTL as a backstop).
- If the dump is stale, the service downloads `jsonl_download_uri`, refreshes,
  and then answers.
- Metadata checks are rate-limited to one per `--check-interval` (300 s by default).
  Failed checks and refreshes back off for the same interval.
- Only one check or refresh runs at a time. Concurrent misses wait for it and then
  re-resolve. The refresh runs in a thread with its own DB connection, so cards
  already in the catalog keep being served during it.
- A name that maps to a card filtered out of the card set (digital-only cards,
  since `include_digital=False` matches `cards.jsonl`) returns `not_found`
  with that oracle_id. No refresh is attempted.

## Versioning

Each refresh from a new dump creates a catalog version. A refresh against an
identical dump creates no version. Nothing is overwritten:

- `cards(oracle_id, version)` stores one row per card per version in which its
  record changed. A changed `build_embedding_text` hash gives a new row and
  an entry in the `changed` list.
- `embeddings(text_hash, model, dims)` stores vectors. On refresh, only texts
  without a vector for the catalog's model/dims are embedded, which means new or
  errata'd cards. A vector from another model/dims is never reused. A refresh
  whose embedder reports a different model is refused.
- `names(name, version)` records only mappings that changed. `oracle_id` is
  NULL when the name is ambiguous (see `candidates`) or no longer exists.
- `changes(version, oracle_id, kind)` is the errata log.

`as_of=v` returns a card's row as it was at version `v` if the card existed
then. Otherwise it returns the card's latest row. Names resolve the same way.
If `as_of` is omitted, the latest version is used.

## Key catalog

The model encodes every token component as `MLP(concat(key_embedding, value_repr))`.
A key's embedding is the text embedding of a short description (`ctr_poison` →
"poison counters", `lands_played` → "lands played this turn"), made with the
same model as card text, so keys the model never saw still embed sensibly.
String values (keyword names, type lines, chosen types, event verbs) are
embedded through `/strings/embed` and share the same cache.

**Registry.** `keys.json` is the hand-curated source of truth. Each entry has
`key`, `description`, `value_kind`, `token_types` and optional `notes` /
`alt_value_kinds`. `value_kind` is one of:

- `number`, `flag`, `string`, `string_list`
- `ref`: points at another object or player and should become an edge
- `identity`: a join key or token identity (`id`, `name`, `oracle_id`,
  `source_oracle_id`, `handles`, `attempt`, `type`, `edges`, ...). These are not components and get no vector.
- `categorical` (schema 13): a value from a closed vocabulary named by the key's
  `vocab` (`zone`, `verb`, `move_kind`, `dnr_reason`, `cause_kind`,
  `phase`), pinned in `magicnet.schema.VOCABS` and checked against the bridge
  by `tests/test_schema.py`.

**Token layout (schema 9).** `value_kind` also decides where a key sits in an
emitted token: `identity` and `ref` keys at the top level, everything else under
the token's `components` object (omitted when empty; flags only when true).
`magicnet.bridge.Schema.isTopLevel` implements the split and
`tests/test_schema.py` checks it agrees with this registry in both directions.
Since schema 12 a card the viewer cannot see has no `name`/`oracle_id`:
absence means unknown, so no placeholder name is ever embedded
(`magicnet.dump.is_hidden`). Since schema 13 history tokens carry no refs at
all: their participants are edges (relations in `magicnet.schema.RELATIONS`),
and their modifiers are ordinary components (`verb`, `step`, `cause_kind`,
`from`/`to`, `before`/`after`, `mana_spent_W` ...).

`patterns` describe dynamic families (`ctr_{name}`, `announce_{what}`; the
history role families `target_{i}`, `card{i}`, `{role}_id`, ... were removed in
schema 13).
Each has a description template and the Java expressions that produce it.
The first matching pattern wins, and `{i}` matches digits.
`counter_keys.json` is generated from Forge's `CounterType.getValues()` by
`scripts/gen_counter_keys.py` (use `--check` to detect a stale file). It is
expanded through `ctr_{name}` into one concrete key per counter type.
`record_fields` lists the decision-record envelope, which is not made of token keys.
`tests/test_keys.py` scans the bridge sources for `.p("key", ...)` calls,
HistoryHub's `r.mods.put("key", ...)` modifiers, the JSON fields in
Serializer/StateExtractor and computed keys. It fails when any of them is
unregistered.

**Versioning.** `keys-sync` diffs the registry against the stored rows. If anything
changed, it creates a new catalog version and copies the latest version's dump
metadata, so card lookups and Scryfall freshness are unaffected. It writes
changed rows to `keys(key, version)`, writes removals as rows with a NULL
description, and records the registry hash in `key_registry`. A
`keys`/`/keys/embed` call with `as_of=v` sees each key as it was at `v`. A key
added after `v` is shown as first recorded, so a pinned client's registry never
changes. Description vectors live in the shared `embeddings(text_hash, model, dims)`
table. Their texts are stored in `strings`.

**Unknown keys.** `/keys/embed` resolves a key in this order:

1. A registered key (`registered: true`, `source: "registry"`).
2. A matching pattern, using the rendered template (`announce_X` → "announced value of X", `ctr_oil` → "oil counters").
3. A description derived from the name (`snake_case` → "snake case", `value_kind: "unknown"`).

Cases 2 and 3 come back with `registered: false`. They are logged once per
process and counted in `unregistered_keys`. Their descriptions are embedded on
demand like any string.

**String cache.** Vectors are content-addressed by `sha256(text)` under the
catalog's model and dims, so they do not depend on the version. A miss is
embedded in a worker thread with the injectable embedder and then stored.
Concurrent requests for a string that is already being embedded wait for that
same call instead of starting another. If the embedder fails, the endpoint
returns 503 and the client raises `CatalogError`.

## Worker usage

```python
from magicnet.catalog import CatalogClient

catalog = CatalogClient("http://catalog-host:8765")  # pins the latest version for its lifetime
cards = catalog.lookup(deck_card_names)               # one request; dict name -> Card | None
vecs = catalog.vectors(deck_card_names)               # (n, dims) float32; raises CardNotFound
keys = catalog.keys()                                 # dict key -> KeyInfo (description, value_kind, vector)
kv = catalog.key_vectors(["life", "ctr_poison"])      # (2, dims); unregistered keys get pattern/derived vectors
info = catalog.embed_keys(["ctr_newthing"])           # KeyInfo with registered=False, source="pattern"
sv = catalog.embed_strings(["Flying", "Kicker {2}{G}"])  # (2, dims) float32
```

The client sends `as_of=<pinned version>` with every request and caches found
cards in memory by `(key, version)`. Key infos are cached the same way, and
string vectors are cached by string. Misses are not cached by the client
because the server already caches them. `CatalogError` means the server could
not confirm a miss, for example because Scryfall was down.

## PCA basis

A belief network fits its flow in a low-dimensional target space: card vectors
projected onto the top-d principal components of the card pool and whitened.
The basis is **frozen and versioned** so flow NLL stays comparable over
training and a model can be pinned to the basis it was trained with.

**Fit.** `pca-fit` takes the latest row of every card in the pool the catalog
serves (removed cards and, with `include_digital=False`, digital cards are not in
it), and eigendecomposes the float64 covariance (1/N). It stores K=256
components (orthonormal rows by decreasing variance; sign fixed so each row's
largest-|loading| entry is positive, so a refit of the same pool is bit-identical),
the mean, the eigenvalues (`std = sqrt`, explained ratio = eigenvalue /
`total_variance`), the embedding model/dims, the card count and a pool hash. A
prefix of d ≤ K components is the optimal d-dim basis, so clients choose d. A
fit for a model/dims other than the catalog's is refused, as is a pool with
cards missing vectors. Fitting never happens implicitly; refreshes never refit.
A basis fit on pre-2024 cards served the later ~6.6k cards nearly as well as a
refit (top-10 neighbour recall 0.283 vs 0.302 at d=32), so refit rarely and on purpose.

**Versioning.** Tables `pca_bases` (one row per fit, never overwritten),
`pca_current(version, basis_id)` and `pca_drift`. A fit creates a new catalog
version (copying the dump metadata, like `keys-sync`) at which the new basis
becomes current, unless you pass `--no-activate`. `pca-use ID` makes an older basis current
again at a new version. The basis current at version `v` is the latest
`pca_current` row with `version <= v`. Versions older than the first basis map
to the first basis, so a pin never changes once any basis exists. A client
pinned at `v` therefore always gets the same basis. A model should record
`basis_id` (and d) and load it with `client.pca(d, basis_id=...)`.

**Drift.** When a refresh adds cards or changes card text (new vectors), it
computes the mean per-card fraction of unexplained variance
`||residual||² / ||x − mean||²` of those cards at d=32 and d=64 under the current
basis. It records this in `pca_drift` next to the basis's baseline, which is the same
quantity averaged over its fitting pool and stored at fit time. If the new cards exceed
the baseline by more than `--drift-margin` (0.10 absolute by default; set it on `serve`), it logs a
warning. It never refits.

**Client.**

```python
pca = catalog.pca(32)                  # PCABasis: basis_id, dims, mean (3072,), components (32, 3072), std (32,)
z = pca.project(x)                     # (..., 3072) -> (..., 32), whitened; numpy or torch (stays on device)
x_hat = pca.inverse(z)                 # (..., 32) -> (..., 3072) least-squares reconstruction
t = catalog.card_targets(["Lightning Bolt", "Counterspell"], dims=32)  # (2, 32) whitened targets
```

`pca(dims, basis_id=None)` returns the basis current at the client's pinned
version, or the given `basis_id`, and caches it per `(basis_id, dims)`. With a torch
input, the basis tensors are built once per device/dtype and no CPU round-trip happens.

In a data loader:

```python
catalog = CatalogClient(url)                    # pins the catalog version
pca = catalog.pca(32)                           # pins the basis current at that version
targets = catalog.card_targets(hidden_truth_oracle_ids, dims=32)  # flow targets, ~N(0, I) over the pool
# or, with vectors already at hand (e.g. a cached torch tensor on GPU):
targets = pca.project(card_vecs)
# flow samples back to embedding space, e.g. for nearest-card decoding:
x = pca.inverse(flow.sample((64,)))
```

Fit on the real catalog (basis 1, version 9, 34,192 cards, K=256): cumulative
explained variance is 30.3% at d=16, 39.6% at d=32, 51.4% at d=64, 64.9% at d=128
and 77.4% at d=256. Top-10 neighbour recall against the full 3072-d space on 1,000 random cards is
0.349 at d=32 (0.335 whitened) and 0.502 at d=64.
