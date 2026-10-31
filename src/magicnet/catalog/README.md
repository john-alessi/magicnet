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
- `identity`: a join key or token identity (`id`, `name`, `oracle_id`, `*_id`,
  `*_oracle_id`, `type`, `zone`, `edges`, ...). These are not components and get no vector.

**Token layout (schema 9).** `value_kind` also decides where a key sits in an
emitted token: `identity` and `ref` keys at the top level, everything else under
the token's `components` object (omitted when empty; flags only when true).
`magicnet.bridge.Schema.isTopLevel` implements the split and
`tests/test_schema.py` checks it agrees with this registry in both directions.

`patterns` describe dynamic families (`ctr_{name}`, `announce_{what}`,
`target_{i}`, `card{i}`, `{role}_id`, `{role}_oracle_id`, `{role}_player`).
Each has a description template and the Java expressions that produce it.
The first matching pattern wins, and `{i}` matches digits.
`counter_keys.json` is generated from Forge's `CounterType.getValues()` by
`scripts/gen_counter_keys.py` (use `--check` to detect a stale file). It is
expanded through `ctr_{name}` into one concrete key per counter type.
`record_fields` lists the decision-record envelope, which is not made of token keys.
`tests/test_keys.py` scans the bridge sources for `.p("key", ...)` calls, the
JSON fields in Serializer/StateExtractor, computed keys, and EventTokenizer role
arguments. It fails when any of them is unregistered.

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
2. A matching pattern, using the rendered template (`target_3` → "target 3 chosen for the spell", `ctr_oil` → "oil counters").
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
