# Saved decision dumps for offline stepping

Load with `magicnet.dump.load(path)`; each record is exactly what the policy
server receives. Dumps are pinned to the schema version they were written with
and must be regenerated after a schema bump.

## pio_mrp.jsonl
12 AI-vs-AI games, Mono-Red Prowess mirror (`fixtures/pio_mrp`), seed 7,
combat and choice routing on, **schema 11** (model inputs under each token's
`components`; read them with `magicnet.dump.comp(tok, key)`; abilities as the
`gained`/`lost` difference from the printed card, plus `base_power`/`base_toughness`). Regenerate with,
from `forge-bridge/`:

    java -Xmx2048m -Dmagicnet.actions=ai -Dmagicnet.choices=true -Dmagicnet.combat=true \
         -Dmagicnet.jsonl=fixtures/dumps/pio_mrp.jsonl \
         -cp "build:/usr/share/java/forge-gui-desktop.jar" magicnet.bridge.Harness \
         --decks fixtures/pio_mrp --games 12 --out OUTDIR --seed 7 --timeout 900

Hand-picked decisions with heavily modified permanents. "Monster" is the
Monster Role token (`types` "Enchantment - Aura Role"), joined to its creature
by an `attached_to` edge; its +1/+1 and Trample show up on the creature as
`power`/`toughness` and `gained`. Regenerated at schema 11 from the same seed:
the games are identical, so the picks below are unchanged.

| game | decision | viewer | turn | why |
|---|---|---|---|---|
| 3 | 271 | P0 | 11 | own Soul-Scar Mage 6/5 (`base_power`/`base_toughness` 1/2) with all five: tapped (attacking), Monster Role attached, `gained` Trample, `temp_power` 3, `ctr_+1/+1` 1 |
| 3 | 277 | P1 | 11 | the same Mage seen from the defender, on a routed `declare_blocker` decision (kind != ability) |
| 3 | 279 | P1 | 11 | same board after blocks: a Mutavault blocking an attacking Screaming Nemesis (`blocking`/`blocked_by` edges, `blocked` true) |
| 0 | 242 | P0 | 9 | own Emberheart Challenger 6/4 (base 2/2): tapped, Monster Role, `gained` Trample, `temp_power` 3 (no counters) |
| 5 | 297 | P1 | 12 | own Soul-Scar Mage 6/5 with `temp_power` 4 and Monster Role; four attackers, three blockers declared |
