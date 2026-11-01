# forge-bridge

Embeds Forge as a library to observe every `PlayerController` callback during
AI-vs-AI play. The point is to answer two questions empirically before any
tokenization or action-space design is locked in:

1. Which of Forge's 111 declared decision callbacks actually fire, and how often?
2. What is the branching factor of each, i.e. how wide does the action head
   need to be?

## Why not the `sim` CLI

`sim -q` emits only game results, so extracting `(state, action)` pairs would
mean parsing the verbose log. Unnecessary: Forge builds players through
`IGameEntitiesFactory`, which is the supported seam for substituting a
controller. Going through `Match` directly means the controller holds a live
reference to the real `Game` object, so full state and the engine-supplied legal
sets are reachable in-process.

That same seam is how the policy will eventually be plugged in.

## Layout

| file | role |
|---|---|
| `ControllerGen.java` | Reflects over `PlayerControllerAi` and generates `GeneratedController.java` |
| `GeneratedController.java` | **generated** — 122 pass-through overrides that time and record each callback |
| `InstrumentedController.java` | Hand-written refinements on top of the generated base |
| `InstrumentedLobbyPlayer.java` | `LobbyPlayer` + `IGameEntitiesFactory`; mirrors `forge.ai.LobbyPlayerAi` |
| `DecisionLog.java` | Thread-safe aggregation, branching-factor histograms, CSV output |
| `Harness.java` | Forge bootstrap, deck loading, game loop |

The overrides are generated rather than hand-written so they stay correct across
Forge upgrades: re-run `build.sh` and the compiler reports anything that moved.
Rebuild after updating the installed Forge jar, before running the harness;
old class files can otherwise fail at runtime or silently omit token fields
where extraction catches an incompatible API call.

Every generated override is a pure pass-through to Forge's own AI. Verified
non-perturbing — identical per-game winners and turn counts with instrumentation
on and off at a fixed seed.

## Usage

```bash
./build.sh
./run.sh --decks "$HOME/.forge/decks/net/Current Pauper Metagame" \
         --games 60 --seed 7 --trace 20000
```

| flag | meaning |
|---|---|
| `--decks DIR` | directory searched recursively for `.dck` files |
| `--games N` | games to play (random deck pairings) |
| `--seed S` | seeds Forge's RNG **and** deck pairing; runs are reproducible |
| `--trace N` | also write up to N per-decision events with turn/phase |
| `--timeout S` | per-game watchdog, default 120s |
| `--out DIR` | output directory, default `out/` |

Outputs `out/decision_summary.csv` and optionally `out/decision_trace.csv`.

Set `-Dmagicnet.enumerate=false` to skip candidate-pool reconstruction when
measuring clean throughput.

## Progressive takeover

`GeneratedController` delegates all 122 callbacks to Forge's heuristics, so a
subclass is a working agent from day one. Override decision types in
`InstrumentedController` one at a time as the policy learns to handle them;
everything not overridden keeps using the bot. This doubles as an ablation axis.

`chooseSpellAbilityToPlay` is already hand-written there as the worked example.

### Scry and surveil (`-Dmagicnet.choices=true`)

`arrangeForScry`/`arrangeForSurveil` are decomposed like combat into pick-one
decisions: one per looked-at card (kind `scry`: "put X on top" / "put X on the
bottom"; kind `surveil`: "put X on top" / "put X into graveyard"), then, for an
ordered pile holding 2+ cards, repeated `order_top` / `order_bottom` decisions
("place X next from the top/bottom", nearest that end first; the last card is
not asked). Options carry a `refers_to` edge and `oracle_id`. Before the first
decision the cards are learned as exact-position `LibraryKnowledge` for the
decider only, so they appear as `self_library` `library_card` tokens with
`depth_from_top`; earlier picks in the sequence show as `arranged_pile` /
`arranged_rank` on those tokens (nothing moves until the arrangement returns).
Surveilled-to-graveyard cards are forgotten afterwards. In observe mode
(`-Dmagicnet.actions=ai`) Forge's AI decides first and the same sequence is
recorded with labels read off its result; its callback fires no events, so the
records match what the routed path would show and play is unchanged.
`fixtures/scry` is a scry/surveil-dense mirror for exercising this.

### Unless costs and simultaneous triggers (`-Dmagicnet.choices=true`)

- `pay_unless` (`payCostToPreventEffect`: shock lands, Spell Pierce, ward, No
  More Lies): 0 = "don't pay (Source)", 1 = "pay {2} (Source)"; both reference
  the source card. Asked only when `ComputerUtilCost.canPayCost` says the cost
  is payable. Forge's AI pays inside this callback, so a routed "pay" pays too:
  per-part `canPay` check, then `CostPayment.payCost` with the AI's cost
  decisions and `refundPayment` on failure (life cannot be refunded; failures
  are reported as a `WARN` line).
- `order_triggers` (`orderAndPlaySimultaneousSa`, 2+ entries): repeated "put X
  trigger on the stack next: <text>" picks, first pick bottommost (resolves
  last); options reference the source card.
- `choose_target`: a trigger whose ability chain targets is set up with
  `SpellAbility.setupTargets`, so `chooseTargetsFor` asks one slot at a time:
  "target X with Source trigger" per legal candidate (refs the candidate), plus
  "choose no more targets" once the minimum is met; a lone forced candidate is
  not asked. Divided, random and stack (spell/ability) targets stay with the AI.
  Stack placement remains `ComputerUtil.playStack`, as for the AI.
- `playTrigger` (static, stackless triggers): an optional one is a
  `confirm_trigger` decision first; targets as above.

Observe mode reads the AI's order first (`orderPlaySa` is a pure sort), records
it, then replays the AI's own per-entry loop one entry at a time, recording a
`choose_target` around each trigger with exactly one single-target slot (built
before the AI binds it, labelled from what it bound). Multi-slot targets are
routed when deciding but not recorded when observing. Spell targets are not
routed (the AI picks them on the engine cast path too). `fixtures/trig` is a
flicker/ETB-target/cast-trigger mirror for exercising this.

## Casting attempts in history

Every cast or activation is an *attempt*. History events fired during it carry
`attempt: N`, and it is closed by an `AttemptEnd` token with `actor`, `source`,
`ability`, any bound `target_i`, `announce_*` values (X, modes), `reached` (the
furthest casting callback seen for that ability: `proposed`, `announce`,
`targets`, `cost`) and `outcome`:

| outcome | meaning | its tagged events |
|---|---|---|
| `committed` | on the stack / resolved | happened |
| `rolled_back` | refused and rewound (CR 728) | **did not happen** -- drop them |
| `failed_unrolled` | refused on Forge's AI path, which does not rewind | happened |
| `unknown` | indeterminate | happened |

History stays append-only, so nothing is retracted; consumers filter. The
transcript keeps rolled-back attempts as candidate negative examples. Note that
Forge's rewind fires no event when it returns the card, so an unfiltered
rolled-back attempt reads as a card still on the stack.

External action sources (`socket`, `random`, `replay`) default to
`-Dmagicnet.targetmode=engine`, the rollback-capable `PlaySpellAbility` path;
AI-only runs keep `baseline`. Set the property explicitly to override.

## Gotchas

- Card tokens carry `oracle_id` (and `source_oracle_id` / `<role>_oracle_id` on
  stack items, actions and history) whenever the card's name is emitted, from
  a name table built by `uv run scripts/build_oracle_map.py` and loaded from
  `-Dmagicnet.oracle=PATH` (default `../data/processed/oracle_ids.tsv`). Without
  the table the harness warns and omits the field; rebuild it after refreshing
  the Scryfall bulk file. Tokens and Forge effect objects get none.
- A deck's `[Sideboard]` section is loaded into `ZoneType.Sideboard` by
  `Match` for every game type (wishes and companions draw from it), and is
  emitted as `self_sideboard` (named, one token per card) / `opp_sideboard`
  (cards `CardView.canBeShownTo` allows are named with `revealed: true`; the
  rest collapse into one unnamed `library_card` token with `count: N`, like
  the hidden remainder of an opponent library, since schema 5).
- Schema 12: a card the viewer may not see has **no** `name`/`oracle_id`/`state`
  (redacted hand and sideboard cards, face-down opponent permanents, the
  opponent library/sideboard `count` aggregates), and a history card role keeps
  only `<role>_id` (no `<role>` name or `<role>_oracle_id`), instead of the
  former literal `HIDDEN` / `<hidden>`. Per the rules an unseen card has no
  name, and absence means unknown as for every other key. The controller of a
  face-down permanent still sees its name. Test with `magicnet.dump.is_hidden`.
  That check reads `PlayerView` opponent lists, which only `HostedMatch`
  fills in, so `Harness` calls `updateOpponentsForView()` itself.
- Forge has no reveal event: reveals arrive only as `PlayerController.reveal`
  on the controller of each player being shown (the owner is usually skipped).
  `RevealKnowledge` records opponent hand cards from that callback as
  `(card id, game timestamp)` per viewer; they are emitted named with
  `known_by_reveal: true` while still in that hand, and every viewer's history
  gets a `Reveal` token (named only for the recipient). Leaving the hand, or a
  new timestamp, forgets them. Forge also calls `reveal` for sideboard fetches
  whose card says no reveal (North Wind Avatar, because its `ChangeType` is not
  plain `Card`); those are suppressed. `-Dmagicnet.reveal.debug=1` traces each
  callback.
- Two more sources feed the same store, from `GameEventCardChangeZone`
  (fired with the pre-move card, so face-down state is as the viewer saw it):
  a card entering a hand from a public zone (bounce, regrowth, exile returns)
  is known to every viewer who saw it leave face-up (or could look at it
  face-down); and a card revealed outside a hand is known to the recipient if
  its very next move takes it from that zone into a hand while the same
  ability is still resolving (digs: Silundi Vision, Commune with Spirits --
  Forge reveals first, then moves). Reveals after the move (Granted) find the
  card already in hand. The move token carries `face_down: true` when the
  card left face-down. `validate_dump.py` re-derives all three causes from
  the viewer's own history and rejects any named opponent hand card, or named
  hidden/face-down departure, that they do not explain.
- Schema 7 adds layer-resolved board state to battlefield `game_object`s:
  `types` (current type line), `colors` (current, `["colorless"]` if none),
  keyword fields (replaced in schema 10, below); `temp_power`/`temp_toughness` = summed P/T boosts with static id 0 --
  a *heuristic* for "from resolved spells/abilities, usually until end of
  turn" (exact durations are not read); and, on face-up objects only, choices
  read from the public `CardView` (`chosen_colors`, `chosen_type`,
  `named_card` + `named_card_oracle_id`, `chosen_player` as a player ref,
  `chosen_number`) plus cast-time `x_paid` / `kicked`. Secret choices
  (Stalking Leonin) stay absent until revealed, for both players. A new
  relation `affects` runs from the source of each active continuous static
  (`game.getStaticEffects()`, spell-created effect cards mapped to their real
  source) to each affected object/player; both ends must be named, non-library
  tokens outside the opponent's hand/sideboard (so never to a face-down
  opponent permanent), no self-loops, duplicates collapsed. Restriction-style
  statics (Pacifism, Pithing Needle) are not "Continuous" in Forge and get no
  edge. `validate_dump.py` rejects `affects` edges touching hidden tokens and
  choice props on face-down/unnamed tokens.
- Schema 9 token layout: a token's top level holds only `type`, `zone`,
  `edges` and the keys `src/magicnet/catalog/keys.json` classifies `identity`
  or `ref` (`id`, `name`, `oracle_id`, `*_id`, `*_player`, `source`, `card`,
  `target`/`target_<i>`, `actor`, `owner`, `named_card`, `attempt`, ...);
  every model input (number / flag / string / string_list) is under
  `components`, omitted when empty, e.g.
  `{"type":"game_object","zone":"self_battlefield","id":"c37","name":"Soul-Scar Mage","oracle_id":"...","edges":[...],"components":{"power":6,"toughness":5,"base_power":1,"base_toughness":2,"tapped":true,"ctr_+1/+1":1,"gained":["Trample"],"temp_power":3}}`.
  `Schema.isTopLevel` is the single rule (`tests/test_schema.py` checks it
  against keys.json); read components with `magicnet.dump.comp`. Flags are
  written only when true, except an attacker's `blocked` and CardPhased's
  `phased_out` (false = phased in). History `ability` text has Forge's
  run-unstable `Name (id)` object ids removed. Action options and the record
  envelope are unchanged (flat).
- Schema 10 abilities: printed abilities reach the model through `oracle_id`,
  so a battlefield `game_object` carries only the engine-resolved difference
  (`AbilityText`): `gained` (abilities it has that are not printed on it --
  keywords, triggered/activated/static/replacement abilities, Forge hidden
  extrinsic keywords) and `lost` (printed abilities it no longer has; "loses
  all abilities" lists each one). Entries are host-independent rules text,
  sorted and deduplicated: keywords as plain text ("Trample", "Ward {2}"), the
  host as "this creature"/"this permanent" (never `CARDNAME` or its name), no
  `(id)`s or player names; engine plumbing (`ETBReplacement:...`),
  keyword-generated traits, cast/land-play abilities and type-intrinsic traits
  (basic land mana, loyalty counters) are dropped. Replaces `kw`, `kw_granted`
  and `abilities_removed`. Restrictions Forge implements as a static on
  another card (Pacifism, Aqueous Form, Artful Dodge's "can't be blocked"
  effect, Curiosity's trigger) are not on the creature and do not appear.
  Creatures also carry `base_power`/`base_toughness` (after copy and
  setting/characteristic-defining effects, before pumps and counters).
- `forge.gui.GuiBase.setInterface(new forge.GuiDesktop())` must run before any
  Forge class is touched; `ForgeConstants`' static initializer dereferences it.
- Forge must run from a directory containing `res/` (`run.sh` symlinks it).
- `-Djava.awt.headless=true` makes Forge exit(1) silently. It needs a display;
  `run.sh` falls back to `xvfb-run`.
- Bypassing `forge.view.Main` means `Sentry.init` never runs, so the harness
  sends no telemetry.
- On JDK 27, Forge's internal game-timeout path calls `Thread.stop()`, which was
  removed, so a wedged game raises `NoSuchMethodError` instead of being stopped.
  The harness watchdog catches this and marks the game a draw.

## Caveat on the branching factor

`chooseSpellAbilityToPlay` is the only callback that takes **no arguments** —
Forge hands the controller nothing and expects it to enumerate its own options.
Every other callback receives a pre-filtered legal set, so action masking is free.

The recorded count comes from `ComputerUtilAbility`, which lives in `forge.ai`,
not `forge.game` — it is the bot's private helper rather than an engine service.
We deliberately do not call `setActivatingPlayer()`/`canPlay()` to refine it,
because those mutate `SpellAbility` state and would perturb the games being
measured.

## Ability pool audit: the bot API is engine-complete (`-Dmagicnet.abilitydiff=N`)

Whether `ComputerUtilAbility` is a heuristic shortlist or the real legal set
decides whether the policy inherits Forge's blind spots. Source answers most of
it: `getSpellAbilities` is a thin wrapper over `Card.getAllPossibleAbilities`,
the **engine's** own enumeration, whose only filter is legality
(`!sa.canPlay(true) && !sa.isPossible()`). Not a heuristic.

Measured over 578 sampled priority decisions in 10 games, diffing the bot pool
against `getAllPossibleAbilities` across all eight playable zones:

```
narrow pool (bot API):     mean 7.53
wide pool (engine):        mean 7.55
in wide, not in narrow:    mean 0.02
...and legal right now:    mean 0.02   (11 occurrences total)
```

**The bot API hides a legal play in roughly 1.9% of decisions.** There is no
hard ceiling. The residue is graveyard recursion — aftermath
(`Consign // Oblivion`), flashback (`Otherworldly Gaze`), additional-cost
graveyard casts (`Eviscerator's Insight`) — caused by `getSpellAbilities`
dropping an alternative-cost variant when the base spell is still "possible".

### The real pruning is upstream of our hook, and it is recoverable

Two mechanisms narrow the action space *before* the policy would see it, neither
of which is a heuristic shortlist:

1. **Cost variants are a separate expansion step.** `getOriginalAndAltCostAbilities`
   is called by `AiController` *after* `getSpellAbilities`, so our instrumentation
   never saw it. It is where adventure halves, impulse-draw casts and
   additional-cost modes appear:

   ```
   Brazen Borrower   :: PermanentCreature :: ... by Petty Theft
   Murderous Rider   :: PermanentCreature :: ... by Swift End
   Lightning Bolt    :: DealDamage        :: ... by Experimental Synthesizer
   Fireblast         :: DealDamage        :: ... by Wrenn's Resolve
   ```

   **So the mean 8.6 branching factor excludes adventure halves and
   "may play from exile" casts entirely — another reason it is a lower bound.**

2. **Optional costs are collapsed to one variant.** `getOriginalAndAltCostAbilities`
   adds *either* the kicked spell *or* the base spell, never both — "Add only one
   ability: either the one with preferred optional costs, or the original one".
   The choice is made by `chooseOptionalCosts`, **a `PlayerController` callback**,
   so the policy recovers it as a separate factored decision rather than as
   parallel actions.

Everything downstream of that (the 34-value `AiPlayDecision` enum: `CantPlayAi`,
`WaitForMain2`, `DoesntImpactGame`, …) is the bot's *choice*, not the API's
limit, and is bypassed entirely once the policy overrides the callback.

Caveat: this mode calls `getOriginalAndAltCostAbilities`, which invokes
`chooseOptionalCosts` and `setActivatingPlayer`. It perturbs play and is **not** a
valid baseline for other measurements. Keying gaps on
`(host, api, cost, description)` matters — a cost-blind key triples the apparent
gap with false positives.

## Targets: the AI path skips the engine's targeting hook entirely

`SpellAbility.setupTargets()` calls `controller.chooseTargetsFor()` and is the
engine's canonical targeting hook. It never fires for the bot. The reason is
**not** that the AI pre-binds targets — `setupTargets` calls `clearTargets()`
itself before asking, so pre-binding would be discarded anyway. The real reason
is reachability. Call-graph scan of all 2,979 classes in the jar:

| caller of `setupTargets()` | reached by the bot? |
|---|---|
| `PlaySpellAbility.playSpellAbility` | no — only `PlayerControllerHuman` and `InputPayMana` call it |
| `PlayerControllerAi.orderAndPlaySimultaneousSa` | yes, but only for simultaneous triggers (2 calls / 12 games) |

The bot's normal cast path is `PlayerControllerAi.playChosenSpellAbility` →
`ComputerUtil.handlePlayingSpellAbility`, a `forge.ai` helper that never calls
`setupTargets`. Targets get bound earlier, inside the 152 per-ability handlers in
`forge.ai.ability.*`, as a side effect of candidate evaluation.

**Consequence:** `ComputerUtilAbility.getSpellAbilities()` returns abilities
*without* targets, so the measured branching factor (mean 8.6, max 31) counts
*which ability*, not *which ability at which targets*. **It is a lower bound on
the true action space.**

## Two-stage action space: confirmed available (`-Dmagicnet.targetmode`)

Whether the engine will hand back a factored "pick ability, then pick targets"
decision, tested over 12 games at a fixed seed:

| mode | behaviour | stripped | `chooseTargetsFor` calls |
|---|---|---|---|
| `baseline` | observe only | 0 | 2 |
| `clear` | strip targets, keep the AI cast path | 68 (71 targets) | **0** |
| `engine` | strip targets, cast via `PlaySpellAbility` | 100 (103 targets) | **103** |

`clear` is the negative control and it is decisive: discard the targets on the
bot's own path and *nothing asks for them back* — the spell is simply cast
untargeted. The factorization is not a property of the ability, it is a property
of the cast routine.

`engine` mode overrides `playChosenSpellAbility` to call the engine's
`PlaySpellAbility.playSpellAbility(this, player, sa)` instead. That works:

```
chooseTargetsFor callbacks:       103
returned true (targets legal):    103 (100.0%)
mean targets bound by callback:   1.03
legal target candidates: mean 3.15  max 18  (n=103)
engine-cast failures:             0
```

`sa.getTargetRestrictions().getAllCandidates(sa, false)` inside the callback is
the stage-two legal mask, supplied by the engine — so action masking stays free
at both stages. (The second argument is `onlyNonCard`: `true` returns players
only. The diagnostic and the options' `target_candidates` passed `true` until
schema 9, so they undercounted creature-targeting abilities at 0–2.)

**Superseded:** the "mean 3.15" figure above cannot be reproduced: its run
config is not recorded, and its max of 18 means it was not taken with the
players-only call this code had since. Remeasured with `getAllCandidates(sa, false)`,
`-Dmagicnet.targetmode=engine`, AI actions, 12 games:
`fixtures/pio_mrp` seed 7 → mean 4.89, max 8 (n=28); `fixtures/pio_41` seed 42 →
mean 2.67, max 16 (n=233). Players only would be capped at 2.

Two caveats worth knowing before relying on this:

1. **It is not behaviour-preserving.** 5 of 12 games kept identical winner and
   turn count; 7 diverged. Expected: re-chosen targets come from
   `AiController.doTrigger(sa, true)`, a different and weaker routine than the
   per-ability handler that picked them during evaluation. Irrelevant once the
   policy chooses targets itself, but it means `engine` mode is **not** a valid
   baseline for measuring anything else.
2. **No per-call cost.** Wall time looks 3.8× worse (76.8s → 290.8s), but that is
   entirely two games that dragged out under worse target choices (one hit the
   120s watchdog). Excluding those two: 68.5s → 67.2s, a ratio of 0.98. The
   engine cast path itself is free.

Target multiplicity (mean 3.15 legal candidates; superseded, 2.7–4.9 remeasured above) also gives the first real
estimate of how much the 8.6 branching factor understates the action space.
