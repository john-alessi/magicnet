# Saved decision dumps for offline stepping

Load with `magicnet.dump.load(path)`; each record is exactly what the policy
server receives. Dumps are pinned to the schema version they were written with
and must be regenerated after a schema bump.

## pio_mrp.jsonl
12 AI-vs-AI games, Mono-Red Prowess mirror (`fixtures/pio_mrp`), seed 7,
combat and choice routing on, **schema 15**, **with hidden-information
labels** (model inputs under each token's
`components`; read them with `magicnet.dump.comp(tok, key)`; abilities as the
`gained`/`lost` difference from the printed card, plus `base_power`/`base_toughness`;
history as one event per token with role edges to tracking handles, events and
players -- resolve them with `magicnet.dump.resolve_edges`; see
`forge-bridge/README.md`, "History"). Regenerate with,
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
`hidden_truth`, every record is identical to the schema-14 dump.

| game | decision | viewer | turn | why |
|---|---|---|---|---|
| 3 | 271 | P0 | 11 | own Soul-Scar Mage 6/5 (`base_power`/`base_toughness` 1/2) with all five: tapped (attacking), Monster Role attached, `gained` Trample, `affected_by` the resolves of its prowess trigger and Monstrous Rage, `ctr_+1/+1` 1 |
| 3 | 277 | P1 | 11 | the same Mage seen from the defender, on a routed `declare_blocker` decision (kind != ability) |
| 3 | 279 | P1 | 11 | same board after blocks: a Mutavault blocking an attacking Screaming Nemesis (`blocking`/`blocked_by` edges, `blocked` true) |
| 0 | 242 | P0 | 9 | own Emberheart Challenger 6/4 (base 2/2): tapped, Monster Role, `gained` Trample, `affected_by` its prowess resolve and Monstrous Rage (no counters) |
| 5 | 297 | P1 | 12 | own Soul-Scar Mage 6/5 `affected_by` two prowess resolves and Monstrous Rage, Monster Role; four attackers, three blockers declared |
