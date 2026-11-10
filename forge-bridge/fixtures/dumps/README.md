# Saved decision dumps for offline stepping

Load with `magicnet.dump.load(path)`; each record is exactly what the policy
server receives. Dumps are pinned to the schema version they were written with
and must be regenerated after a schema bump.

## pio_mrp.jsonl
12 AI-vs-AI games, Mono-Red Prowess mirror (`fixtures/pio_mrp`), seed 7,
combat and choice routing on, mana payment routing **off** (no
`-Dmagicnet.payment`, so no `pay_*`, `choose_optional_cost`, `choose_additional_cost` records and
the picks below keep their decision numbers), priority candidate filter
**off** (the default for AI-only runs; no `-Dmagicnet.filter`, so `actions`
is Forge's unfiltered pool), **schema 20**, **with
hidden-information labels** (model inputs under each token's
`components`; read them with `magicnet.dump.comp(tok, key)`; abilities as the
`gained`/`lost` difference from the printed card, plus `base_power`/`base_toughness`;
history as one event per token with role edges to tracking handles, events and
players -- resolve them with `magicnet.dump.resolve_edges`; see
`forge-bridge/README.md`, "History"). Each record is `{schema, game,
decision, viewer, kind, turn, phase, hist_total, hist_from, hist_new, state,
prompt, actions, hidden_truth, chosen, chosen_match}`: `prompt` is the
decision token (`{"type":"decision","id":"q",...}`, `kind`, `prompt_text`,
`selected` edges ...) and `actions` the candidate list `chosen` indexes into --
`{"ref": <handle or id>}` references to existing tokens (defenders, attackers
to block, library-search picks via their `n:<name>` aggregate) or option tokens
(`{"type":"action","id":"a<i>","components":{"verb":"cast",...}}`; `pass`
last on ability decisions). `magicnet.dump.decision_tokens(d)` /
`resolve_candidates(d)` give the pointer layout; see `forge-bridge/README.md`,
"Decision candidates and the prompt". Regenerate with,
from `forge-bridge/`:

    java -Xmx2048m -Dmagicnet.actions=ai -Dmagicnet.choices=true -Dmagicnet.combat=true \
         -Dmagicnet.labels=true -Dmagicnet.jsonl=fixtures/dumps/pio_mrp.jsonl \
         -cp "build:/usr/share/java/forge-gui-desktop.jar" magicnet.bridge.Harness \
         --decks fixtures/pio_mrp --games 12 --out OUTDIR --seed 7 --timeout 900

### `hidden_truth` (training-only)

Every record carries a top-level `hidden_truth`: for each hidden-region token
in `state` -- the opponent's unseen hand, library remainder and sideboard
aggregates (`id` `r:opp_hand`, `r:opp_library`, `r:opp_sideboard`) and any
face-down object the viewer cannot see (its card id) -- the true cards it
stands for, `{"region": <token id>, "cards": [{"oracle_id", "name", "count"}],
"total": <the token's count>}`. Read it as `Decision.hidden_truth`; pair a
label with its region token by `id` **before** `strip_ids`. It is a belief-model
target, never a model input, and is never sent to a live agent. See
`forge-bridge/README.md`, "Hidden-information labels". E.g. game 3 decision
271 (P0): `r:opp_hand` total 3, `r:opp_library` total 48, `r:opp_sideboard`
total 15.

Hand-picked decisions with heavily modified permanents. "Monster" is the
Monster Role token (`types` "Enchantment - Aura Role"), joined to its creature
by an `attached_to` edge; its +1/+1 and Trample show up on the creature as
`power`/`toughness` and `gained`, and the creature has an `affected_by` edge
back to it. Spell and trigger pumps (Monstrous Rage, prowess) are
`affected_by` edges from the creature to those history `resolve` events
(schema 13; they replace `temp_power`). Regenerated at schema 14 from the same
seed: the games are identical, so the picks below are unchanged. Regenerated
again at schema 15 with labels: apart from the `r:<zone>` ids and
`hidden_truth`, every record is identical to the schema-14 dump. Regenerated
at schema 16 (same seed, same 3456 records, identical games, history, state
and labels): only `prompt`/`actions` changed, plus `n:<name>` ids on the
viewer's own-library name aggregates. Regenerated at schema 17 with payment
routing off: byte-identical to the schema-16 dump except the `schema` field.
Regenerated at schema 18 (non-mana cost routing, same flags): byte-identical
to the schema-17 dump except the `schema` field. Regenerated at schema 19
(same flags, same 3456 records, identical games, history, actions, prompts and
labels): only `state` changed, in the 543 records with something on the stack
-- a spell on the stack is now its card's `game_object` in zone `stack`
(`side`, `stack_depth`, `target_i` edges), abilities stay `stack_item` with
`side`, `stack_depth`, `ability` and `target_i` edges (see
`forge-bridge/README.md`, "The stack"). Every edge in the dump resolves to a
token of its record. The picks below are unchanged (none has anything on the
stack); game 0 decision 163 is added as a stack example. Regenerated at
schema 20 (`announce` decisions, own face-down cards named; same flags):
byte-identical to the schema-19 dump except the `schema` field -- the deck
has no X spells or face-down cards -- so the picks are unchanged.
Regenerated once more with the final schema-20 build (reproducible: two runs
are byte-identical): same 3456 records; only game 0 decisions 237-245 list
Emberheart Challenger's `affects`/`affected_by` edges in a different order
(edge order carries no meaning; identical after sorting edges).

| game | decision | viewer | turn | why |
|---|---|---|---|---|
| 3 | 271 | P0 | 11 | own Soul-Scar Mage 6/5 (`base_power`/`base_toughness` 1/2) with all five: tapped (attacking), Monster Role attached, `gained` Trample, `affected_by` the resolves of its prowess trigger and Monstrous Rage, `ctr_+1/+1` 1 |
| 3 | 277 | P1 | 11 | the same Mage seen from the defender, on a routed `declare_blocker` decision (kind != ability) |
| 3 | 279 | P1 | 11 | same board after blocks: a Mutavault blocking an attacking Screaming Nemesis (`blocking`/`blocked_by` edges, `blocked` true) |
| 0 | 242 | P0 | 9 | own Emberheart Challenger 6/4 (base 2/2): tapped, Monster Role, `gained` Trample, `affected_by` its prowess resolve and Monstrous Rage (no counters) |
| 5 | 297 | P1 | 12 | own Soul-Scar Mage 6/5 `affected_by` two prowess resolves and Monstrous Rage, Monster Role; four attackers, three blockers declared |
| 0 | 163 | P1 | 7 | the stack: the opponent's Burst Lightning (`game_object`, zone `stack`, `side` opp, `stack_depth` 2, `target_0` -> own Soul-Scar Mage) under two opponent prowess triggers (`stack_item`, `stack_depth` 0 and 1, `source_is` their creatures) |
