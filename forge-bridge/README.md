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
| `HistoryHub.java` | Game history, once per game: Forge events -> events with causes and steps |
| `GameHistory.java` | One viewer's history tokens, tracking handles, phantoms (schema 13) |

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
decisions: one per looked-at card (kind `scry`: option tokens `scry_top` /
`scry_bottom`; kind `surveil`: `surveil_top` / `surveil_graveyard`; the prompt
has an `obj` edge to the card and `selected` edges to the cards already
placed), then, for an ordered pile holding 2+ cards, repeated `order_top` /
`order_bottom` decisions whose candidates are references to the remaining
cards' tokens (the one placed next from that end; nearest that end first; the
last card is not asked). Before the first
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
  More Lies): 0 = option `decline`, 1 = option `pay` with a `cost` component
  ("{2}"); the prompt points at the resolving ability (`stack_object`) and its
  source card (`source_is`). Asked only when `ComputerUtilCost.canPayCost` says the cost
  is payable. Forge's AI pays inside this callback, so a routed "pay" pays too:
  per-part `canPay` check, then `CostPayment.payCost` with the AI's cost
  decisions and `refundPayment` on failure (life cannot be refunded; failures
  are reported as a `WARN` line).
- `order_triggers` (`orderAndPlaySimultaneousSa`, 2+ entries): repeated
  `order_next` option tokens (the entry's `ability` text, `source_is` its
  source card) picking the entry put on the stack next, first pick bottommost
  (resolves last); the prompt's `selected_count` is the number already placed
  (they are not tokens yet).
- `choose_target`: a trigger whose ability chain targets is set up with
  `SpellAbility.setupTargets`, so `chooseTargetsFor` asks one slot at a time:
  a reference to each legal candidate's token, plus a `choose_nothing` option
  ("no more targets") once the minimum is met; the prompt carries `min`/`max`,
  `optional`, the trigger's `ability` text (it has no token yet) and
  `selected` edges to targets already chosen. A lone forced candidate is
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

### Mana payment (`-Dmagicnet.payment=true`, schemas 17-18)

Off by default (it adds decisions). Implemented in `ManaPayment.java`.

- `pay_mana`: one pick-one decision per payment step until the cost is paid.
  Option tokens: `activate_mana` (a mana ability of a battlefield permanent
  the payer controls, one option per colour it can be told to make:
  `ability` text, `mana_color` W/U/B/R/G/C, `mana_amount`, `source_is` edge),
  `use_pool` (floating mana that can pay this spell: plain units of one colour,
  restriction and snow collapse into one option with `mana_amount` = units;
  mana with a rider -- Cavern of Souls' "can't be countered", counters,
  keywords, spend triggers -- is its own option per source with `rider` = the
  producing ability's text and a `source_is` edge; `restricted`, `snow`
  flags), `pay_life` (`cost` "2 life", a Phyrexian shard). The prompt carries
  `remaining_cost` ("{1}{B/P}{B/P}"), `remaining_generic`, `remaining_W..C`
  (unpaid shards a colour can pay; hybrids count under each colour),
  `spent_W..C`/`spent_snow` (mana already paid into this cost),
  `life_paid`, `selected` edges to sources already tapped for it
  (`selected_count` includes sacrificed ones, which have no token), and the
  spell's `ability` text (a spell being cast has no token yet).
- `pay_convoke` / `pay_delve`: Forge asks for Convoke/Improvise taps and Delve
  exiles up front (`CostAdjustment.adjust`, before any mana), through
  `chooseCardsForConvokeOrImprovise` / `chooseCardsToDelve`; routed there as
  sequential decisions: `convoke`/`improvise` options (`pays` "{G}"/"{1}",
  `source_is` the creature/artifact) or references to graveyard cards, plus
  `choose_nothing` (stop) when the rest is payable from mana sources
  (`optional`). Waterbend: see below (schema 18).
- Mask: a candidate is offered only if the rest of the cost stays payable
  (Forge's own `ManaCostBeingPaid.payMana` on a copy, then
  `MaskCost.extendable` over the remaining sources, every Phyrexian
  mana-or-life and `{2/W}` colour-or-two-generic split tried). A single legal
  candidate is taken without asking and not recorded (like a forced target).
- Decide mode pays the way `PlayerControllerHuman` does: `payManaCost` goes
  through `PlaySpellAbility.payManaCost` (cost adjustment, Convoke/Delve, X),
  which calls back `applyManaToCost`; a step is the mana ability through
  `PlaySpellAbility.playSpellAbility` with its colour set by
  `setManaExpressChoice`, then `ManaPool.payManaFromAbility`
  (`InputPayMana.activateManaAbility`); pool mana as
  `ManaPool.tryPayCostWithColor` does for one chosen unit; Phyrexian life as
  `InputPayManaOfCostPayment` (shard now, life once paid). A `false` return
  lets `PlaySpellAbility` roll the cast back.
- Observe mode leaves Forge's AI paying exactly as before and reconstructs
  the steps from `SpellAbility.getPayingMana()` (its payment order); each
  step's state is captured when the AI taps that step's source, with that one
  card shown untapped. Phyrexian life is appended last (its place in the AI's
  order is not recoverable). Payments made inside another open observe-mode
  record (an AI-paid `pay_unless`, a confirmed trigger's cost) are not
  recorded, to keep the history deltas in order. Existing records are
  unchanged: with the flag on, every non-payment record of a seeded AI run is
  identical to the flag-off run.
- `-Dmagicnet.payment.audit=true` compares the first-step mask with
  `ComputerUtilMana.canPayManaCost` (about 1 ms per payment);
  `-Dmagicnet.payment.fault=true` corrupts observe-mode labels (to show the
  label replay check fails). Counters print as "mana payment routing".
- Schema 18 additions: mana abilities with a mana cost (filters, Signets,
  Arcum's Astrolabe) are `activate_mana` options with a `cost` component;
  taking one pays that cost as a nested `pay_mana` sequence (prompt
  `ability` = the filter's text) masked so the outer payment stays payable.
  Mana abilities usable from hand/graveyard/exile/command (Spirit Guides) are
  options with a `source_is` edge to that card. "Any combination" abilities
  (`Combo`, amount 2+: Sage of the Maze, Graven Cairns) are ONE option without
  `mana_color`; their colours are then `pay_mana_combo` decisions, one per
  unit (`choose_color` options with `mana_color`, prompt `remaining_cost` and
  an `obj` edge to the source; a colour is offered only if some completion
  keeps the payment payable). `specifyManaCombo` outside a payment
  (Manamorphose, a combo ability activated at priority) is routed the same
  way, unmasked. Mask: a fixed multi-colour bundle (`{B}{R}`) is its real
  colours, `{S}` needs a snow source, a mana ability without `{T}` counts as
  usable as often as its cost allows (sacrifice-another, life, counters on the
  source; cap 4), and an ability's own cost cannot be paid by its own output
  or another filter's (chains of filters are not modelled: too strict there).
  Observe mode records combination colours from the AI's paying mana when the
  whole output was spent on the cost.
- Waterbend (`chooseCardsForConvokeOrImprovise` with a maximum) is a
  `pay_convoke` sequence of `waterbend` options (`pays` "{1}", at most the
  maximum).
- Not offered: X costs, `{2/W}` rider effects beyond the split already
  modelled, and colour choices of non-combo "any colour" abilities beyond
  one option per colour.

### Non-mana costs (`-Dmagicnet.payment=true`, schema 18)

The engine pays every cost through `CostPayment.payCost(controller.getCostDecisionMaker(...))`.
`PlayerControllerHuman` returns a `HumanCostDecision`, whose visits ask the
player (`InputSelectCardsFromList`); Forge's AI returns `AiCostDecision`. A
routed payer now gets `CostChoices.Routed` (an `AiCostDecision`), whose
visits for the card-choosing parts ask one card per decision, from the list
`HumanCostDecision` builds:

- `pay_sacrifice`, `pay_discard`, `pay_exile` (own zones or any zone; not
  library-top or "same graveyard"), `pay_tap` (N creatures, or crew-style
  total power: prompt `remaining_power`), `pay_untap`, `pay_return`,
  `pay_reveal`. Candidates are references to the cards' tokens (fallback
  `choose_object`); the prompt has the cost part as `cost` ("Sacrifice a
  creature", "Exile two cards from your graveyard"), `min`/`max` = cards still
  to choose, `selected` edges to cards already chosen, and the ability's
  `stack_object`/`source_is` edges (or `ability` text). Exactly as many
  candidates as needed are taken without asking. Branches not mirrored
  (different names, total mana value, shared type, random, whole hand,
  from source) stay with the AI and are counted as `cost: silent: ...`.
  This covers spells' additional costs, activated abilities and mana
  abilities' costs (Springleaf Drum's tap, Ashnod's Altar's sacrifice) on the
  routed path, and the cost of a routed `pay_unless`.
- Emerge / Offering: `CostAdjustment` asks `choosePermanentsToSacrifice`
  inside the payment; routed as `pay_sacrifice` with `cost` "Emerge" /
  "Offering" (required: the cast fails without it), creatures offered only if
  the reduced cost stays payable without them (all offered if none is).
- `choose_optional_cost`: `PlaySpellAbility.chooseOptionalAdditionalCosts`
  calls `chooseOptionalCosts` with every optional cost (kicker, buyback,
  entwine, offering, ...; Forge's AI answers from its own evaluation), and
  `GameActionUtil.addExtraKeywordCost` calls `chooseNumberForKeywordCost`
  (casualty, conspire, offspring; multikicker/replicate/squad with a
  number). Options: `choose_nothing` (stop), then `optional_cost` (`cost`,
  `ability` = Forge's label such as "Kicker – {R}", `value` = times for the
  numbered keywords); one cost per decision, offered only while the total
  stays affordable (`ComputerUtilCost.canPayCost` on a `Cost` copy;
  retrace/jumpstart only from the graveyard).
- `choose_additional_cost`: an `AlternateAdditionalCost` spell (Lightning
  Axe: discard a card or pay {5}) asks `getAbilityToPlay`, where Forge's AI
  takes the first; options are `optional_cost` with each variant's full cost.
- Observe mode: Forge's AI builds its own `AiCostDecision` inside
  `ComputerUtil.handlePlayingSpellAbility`, so when a cast's cost has a
  card-choosing part `CostChoices.observeCast` replays
  `PlayerControllerAi.playChosenSpellAbility` line for line with an observing
  decision maker that takes the AI's decision and records it (state: the
  spell on the stack, before any part is paid -- the AI decides every part
  before paying, the routed path pays mana first). The AI's optional costs
  are part of its chosen ability variant: they are recorded as the
  `choose_optional_cost` sequence before the cast, and the variant's
  `ability` record now labels its pool spell (`chosen_match` "signature").
  Keyword-cost answers and Emerge/Offering sacrifices are recorded where the
  AI answers them. Every pre-existing record of a seeded AI run is unchanged
  with the flag on. `-Dmagicnet.payment.fault=true` corrupts these labels too.
- Engine path guard: `PlaySpellAbility.playSpellAbility` checks `canPlay` for
  spells only, and the priority pool is unfiltered, so an external policy
  could "activate" a tapped land's mana ability (no tap, mana added: an
  unbounded mana loop seen with a random policy). The engine path now refuses
  a non-spell ability that cannot be activated (rolled back; counted as
  "engine path: unactivatable abilities refused"). Decide mode only.
- Still silent (AI): `chooseNumberForCostReduction` ("up to" cost
  reductions), collect evidence, forage, removing counters from other
  permanents, put-card-in-library costs, the cost order (`orderCosts`; the
  human is asked only with a full-control setting), and costs paid during
  resolution through `PlayerController` callbacks other than `pay_unless`.

`fixtures/costs` is a sacrifice/discard/exile/crew/Signet/filter/Spirit
Guide/combination/kicker/Offering/Emerge/snow/waterbend mirror for
exercising both sections.

`fixtures/pay` is a Phyrexian/hybrid/Convoke/Delve/Improvise/Cavern pile for
exercising this.

## Decision candidates and the prompt (schema 16)

Every record is `{schema, game, decision, viewer, kind, turn, phase, hist_total,
hist_from, hist_new, state, prompt, actions[, hidden_truth], chosen,
chosen_match}`. The policy still returns one index into `actions` (protocol
unchanged), but `actions` is now an ordered list of two candidate families:

- **References** `{"ref": "<handle or id>"}` when the choice IS a game object or
  player the viewer already has a token for: targets, defenders
  (`declare_attacker`), attackers to block (`declare_blocker`), cards being
  ordered (`order_top`/`order_bottom`), `choose_entity`/`choose_companion`/
  `choose_card_zone_change` picks. Resolved through the same id/handle table as
  edges (`Serializer.Resolver`; a tracking handle `h<N>` when the token has one,
  else its id), never to an individual card the viewer cannot see. A pick from
  the viewer's own library at an unknown position references that name's
  aggregate, which carries the id `n:<name>`: identical copies are ONE candidate,
  and the first copy behind it is the one fetched when deciding (observe-mode
  labels map the AI's card to the candidate its token resolves to). An object
  with no visible token falls back to an option token (`choose_object`,
  `attack`, `block`), counted in the run summary ("object options with no
  token").
- **Option tokens** for choices without a token:
  `{"type":"action","id":"a<i>","edges":[...],"components":{"verb":...}}`,
  `i` = the candidate's index, `verb` from `Schema.ACTION_VERBS` (vocab
  `action_verb`: `pass, cast, activate, play_land, attack, no_attack, block,
  no_block, choose_nothing, choose_object, confirm, decline, pay, mode, number,
  scry_top, scry_bottom, surveil_top, surveil_graveyard, order_next`; schema 17
  adds `activate_mana, use_pool, pay_life, convoke, improvise`; schema 18
  `optional_cost, choose_color, waterbend`). Priority
  options carry `ability` (Forge's text normalized: no ids, bracketed context
  or player names, the host's own name as "this creature/permanent/spell"),
  `target_candidates` (`getAllCandidates(sa, false)`), `mana_ability` (only
  when true) and a `source_is` edge to the host; `pass` is always last on
  `ability` decisions and absent elsewhere. `mode`/`confirm`/`order_next`
  carry `ability`, `number` a `value`, `pay` a `cost`.

`prompt` is one `{"type":"decision","id":"q",...}` token: `kind` (vocab
`decision_kind`, = the record's `kind`), `prompt_text` (Forge's title/prompt,
normalized; absent when none), `min`/`max`, `optional` (a `choose_nothing` is
offered), `selected_count`, `ability` (the asking ability's text when its
`stack_object` edge does not resolve), and edges `stack_object`/`source_is`
(asking ability and card), `obj` (scry/surveil card, confirm_action card),
`attacks`/`blocks` (the creature a combat step is about) and `selected`
(tokens chosen earlier in the same decomposed decision: attackers/blockers
declared so far, targets already chosen, cards already placed). In observe
mode the combat prompts are rendered when the record is written, so
`selected` reflects the AI's earlier answers. Example (pio_41, decision 607):

    "prompt": {"type":"decision","id":"q","edges":[{"rel":"stack_object","to":"h80"},
               {"rel":"source_is","to":"h55"}],"components":{"kind":"choose_card_zone_change",
               "prompt_text":"Select a card from your library","optional":true}},
    "actions": [{"type":"action","id":"a0","components":{"verb":"choose_nothing"}},
                {"ref":"n:Island"}, {"ref":"n:Plains"}]

Python: `Decision.prompt`, `Decision.actions`; `magicnet.dump.decision_tokens(d)`
is `d.tokens + [prompt] + option tokens` (candidate order), and
`resolve_candidates(d)` maps each candidate to its index in that list (refs
point into `d.tokens`), i.e. the pointer head's candidate set;
`resolve_edges`/`strip_ids` work on that list unchanged. `pass_index` finds
the `pass` option.

## History (schemas 13-14)

`HistoryHub` builds each game's history once, from Forge's event bus, and every
viewer's `GameHistory` renders it -- redacted when the event fires, so nothing
seen later leaks backwards and nothing seen earlier is forgotten. Each history
token is **one event**: `verb [modifiers] {participants}`.

```
{"type":"history","zone":"history","id":"e45",
 "edges":[{"rel":"actor","to":"p0"},{"rel":"stack_object","to":"h17"},
          {"rel":"source_is","to":"h15"},{"rel":"cause","to":"e43"}],
 "components":{"verb":"trigger","turn":3,"phase":"MAIN1","step":21,"step_order":0,
               "ability":"Whenever an opponent draws a card, they lose 2 life.",
               "cause_kind":"pointer"}}
```

- `id` `e<N>` is the event's index in this viewer's history: other tokens point at it.
- Always present: `verb`, `turn`, `phase` (Forge PhaseType, `NONE` pregame),
  `step` (one per atomic game action -- a resolution, a cast/activation with its
  costs, a combat-damage batch, a state-based check, a turn-based action; a
  simultaneous batch shares it) and `step_order` within it, `cause_kind`.
- Participants are **edges whose relation is the role**, to a player `p<N>`,
  an event `e<N>`, or a tracking handle `h<N>` (below) -- never a card id or
  name. A participant the viewer cannot identify (the opponent's draw) has no
  edge at all.

| verb | modifiers | roles |
|---|---|---|
| `move_zone` | `from`, `to` (viewer-relative zone), `move_kind` (put/mill/exile/sacrifice/destroy/bounce/search/return/other), `count`, `face_down` | `obj`*, `player`, `cause`, `replaced_by` |
| `draw` | `from`, `to`, `count` | `obj`* (only cards this viewer sees), `player`, `cause` |
| `discard` | `from`, `to`, `count` | `obj`*, `player`, `cause` |
| `create` | `to`, `count` | `obj`*, `actor`, `cause` |
| `play_land` | | `actor`, `obj` |
| `cast` / `copy_spell` | `modes`, `x_paid`, `kicked`, `mana_spent_W/U/B/R/G/C` | `actor`, `stack_object`, `source_is`, `target_0..3`, `target_more`, `cause` |
| `activate` | `ability`, `mana_spent_*` | same |
| `trigger` | `ability`, `modes` | `actor`, `stack_object`, `source_is`, `target_i`, `cause`* (the triggering events) |
| `special_action` | `action_kind` (turn_face_up, foretell, plot, unlock, or the ability api: a companion is `changezone`) | `actor`, `source_is` |
| `resolve` | | `actor`, `stack_object`, `source_is`, `cause` (the cast/trigger that put it there) |
| `does_not_resolve` | `reason` (countered/fizzled/removed) | `stack_object`, `cause` (countered: the counterspell's `resolve`) |
| `declare_attacker` | | `attacks`, `defender` |
| `declare_blockers` | | `attacks`, `blocks`* (one event per blocked attacker) |
| `deal_damage` | `amount`, `combat` | `source_is`, `target_0`, `cause` |
| `life_change` | `before`, `after` | `player`, `cause`* (all the damage of a batch) |
| `counters_change` | `counter_type`, `before`, `after` | `obj` or `player`, `cause` |
| `change_control` | | `obj`, `old_controller`, `new_controller`, `cause` |
| `reveal` | `from`, `count` | `obj`* (recipient, or a viewer who can see them), `player` (shown to), `cause` |
| `scry` / `surveil` | `n_top`, `n_bottom` / `n_graveyard` | `player`, `cause` |
| `shuffle` | | `player`, `cause` |
| `mulligan` | `hand_size` | `player` |
| `designation_change` | `designation` (monarch, initiative, city's blessing, speed, day, night), `before`, `after` | `player`, `cause` |
| `attach` | | `obj`, `attached_to` (absent: unattached), `cause` |
| `attempt_end` | `outcome`, `reached`, `announce_*` | `actor`, `stack_object`, `source_is`, `target_i` |

Relations `source_is`, `attacks`, `defender`, `blocks` and `attached_to` are
reused from earlier schemas with the same meaning. Moves the event of their own
already covers are folded: the hand-to-stack move of a cast, the land's move of
a land play, the TokenCreated / CardDestroyed / CardSacrificed /
CardModeChosen events (into `create`, `move_kind`, `modes`), PlayerPoisoned (a
duplicate of the counter change). Consecutive identical moves (an opening hand,
a mill, a board wipe, the cards of one draw) are one event with `count` and one
`obj` edge per card the viewer sees. Per-card scry/surveil reorders inside a
library reach only the player who can tell which card moved; others get the
`scry`/`surveil` summary. Not mapped (counted in the coverage report):
CardPhased, CardRegenerated, FlipCoin, RollDie (Forge puts no outcome on the
last two); turns, phases and priority passes stay implicit, mana abilities are
not events.

### Why it happened: `cause` and `cause_kind`

Forge's events never say why; the hub recovers it as each event fires
(`HistoryHub` javadoc): the **resolution** in progress (the stack ability
snapshotted at the resolution's first event and closed by its own
`SpellRemovedFromStack`, so the spell's own move to the graveyard counts and a
cast-from-effect does not hijack it; copies pushed just after are caused by it);
a **static trigger** run through `playTrigger`; the **casting attempt** (costs
are held back and emitted after the `cast`/`activate` they pay for);
**replacement** effects (`ReplacementHandler.hasRun`, read by reflection) add a
`replaced_by` edge on top of the cause; **triggers** are matched per mode
(Drawn, ChangesZone(All), SpellCast, BecomesTarget, CounterAdded, DamageDone,
Attacks, Blocks, LifeLost, BecomeMonarch, ...) to the events in their
triggering objects; a life change or counter change points at the damage that
caused it. `cause_kind` is on every token: `pointer` when the `cause` edges
explain it, otherwise the rule category -- `player_action`, `combat`,
`pregame`, and by a stack walk (only for events nothing else classified)
`state_based` or `turn_based`, else `unknown`. A state-based death also points
at the damage that made it lethal. Causes are events of the same viewer's
history, never card identities: the triggering objects of Sheoldred's trigger
include the opponent's hidden draw, and its cause edge is just `e<N>` of a
`draw` with no `obj`. Schema 14: text components (`ability`, `modes`, phantom `ability`/`name`) are
normalized by `EventTokenizer.normalize`: no `(id)`s, player names replaced by
"a player", and Forge's trailing bracketed run context ("[Zone Changer: X]",
"[Player: P0-<deck>]") dropped -- objects there are edges already, and the
harness player name embeds the deck name. `validate_dump.py` flags player names
and bracketed context in any string component.
`HistoryHub.printCoverage()` reports matched/unmatched
triggers by mode, stack-walk count and cost; `-Dmagicnet.history.walkdebug=true`
prints the frames of every walked event.

### Tracking handles

History is append-only and delta-encoded, so an event's edges are frozen when
written; they must still follow the object. History edges therefore point at
per-viewer **tracking handles**, one per physical object the viewer can
follow, and every state lists each handle the history has referenced in the
top-level `handles` of exactly one token:

1. the object's own token if the viewer can see it (battlefield, graveyard,
   exile -- face-down too --, the stack item of a spell, its own hand, an
   opponent hand card `known_by_reveal`, a library card at a known position);
2. the aggregate of the hidden zone it is known to be in: an opponent's
   unseen hand cards are now one `library_card` `count` token in `opp_hand`
   (individual nameless hand tokens carried nothing but an id, and assigning a
   handle to one would say *which* unseen card it is), the opponent library
   aggregate, or the name histogram token of the viewer's own library;
3. once the viewer **loses track** of it -- a card left the zone it was in and
   the viewer could not tell which (the opponent draws while it sat in their
   library; Brainstorm puts two cards back after we bounced it; a card of the
   same name leaves the hand it was known in): if the viewer knew its identity,
   an **identity phantom** (below, no `ceased`), so history reads "bounced *a*
   copy of Mist Raven"; if it never did (face-down), `self_untracked` /
   `opp_untracked` by owner. Schema 14: this is final. The handle is **never
   re-linked** to a card of that name seen later -- graph edges would assert an
   identity the viewer cannot know; the next sighting gets a new handle.
   `RevealKnowledge` forgets an opponent's known hand cards by the same rule
   (all of them after an unseen departure, those of the same name after a
   departure seen arriving face up);
4. a `phantom` game_object (zone `phantom`) carrying last-known `name`/
   `oracle_id`, `types`, `power`/`toughness`, `colors`, `token`, `ability` text
   and `side`: with `ceased: true` once the object ceased to exist (a token that
   left the battlefield, a copy or an ability that left the stack), deduplicated
   by (identity, controller); without it for an identity phantom, deduplicated
   by (identity, owner).

Phantoms (and `*_untracked` tokens) are emitted only while at least one edge
refers to one of their handles. History is not truncated today, so in practice
a phantom lasts the rest of the game; once history truncation drops a
phantom's last referencing event, it disappears from the state on its own.

`magicnet.dump.resolve_edges` resolves `h*` through the `handles` lists and
`e*` through history ids; `strip_ids` drops both.

### Live modifiers: `affected_by`

Board objects carry `affected_by` edges, recomputed every state: to the source
token of each continuous static that applies (the reverse of `affects`, which
is kept: relations are directed) and to the history `resolve` event whose
continuous effect still applies (Giant Growth's +3/+3, prowess). Effects are
attributed by scanning the battlefield's non-static P/T, keyword and trait
tables when a resolution closes. This replaces the schema-7
`temp_power`/`temp_toughness` heuristic; the harness prints the coverage
(`=== affected_by -> resolve ===`: every creature with a non-static P/T boost
has such an edge on the hist, pio and mm fixtures).

### Casting attempts

Every cast or activation is an *attempt*. History events fired during it carry
`attempt: N`, and it is closed by an `attempt_end` token with `actor`,
`stack_object` (if it reached the stack), `source_is`, bound `target_i`,
`announce_*` values (X, modes), `reached` (the furthest casting callback seen
for that ability: `proposed`, `announce`, `targets`, `cost`) and `outcome`:

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

## Hidden-information labels (`-Dmagicnet.labels=true`, schema 15)

Training-only targets for a belief model (one conditional density per hidden
region, conditioned on that region's token). Off by default.

**Regions** are the state tokens that stand for cards the viewer cannot see,
each with a stable top-level `id` (a join key, `identity` in `keys.json`,
dropped by `strip_ids`; `magicnet.dump.is_region` tests for one):

| region | token | id |
|---|---|---|
| opponent's unseen hand | `library_card` `count` aggregate in `opp_hand` | `r:opp_hand` |
| opponent's library minus cards at a known position | `library_card` `count` aggregate in `opp_library` | `r:opp_library` |
| opponent's unseen sideboard | `library_card` `count` aggregate in `opp_sideboard` | `r:opp_sideboard` |
| each face-down object the viewer cannot see | nameless `game_object` (battlefield, exile) | its card id `c<N>` |

Not regions: the viewer's own library (its name histogram is already
visible), named cards at known library positions, `known_by_reveal` hand
cards, `*_untracked` and phantom tokens. Schema 15 is the `r:<zone>` ids:
before it, an id on a card token meant one physical card (consumers that
relied on that -- `GameHistory.decorate`, the validator's replay -- now test
`r:` explicitly).

With labels on, every **dump** record carries a top-level `hidden_truth`
(a sibling of `state`, never inside a token), one entry per region token in
`state`, in token order:

    "hidden_truth":[
      {"region":"r:opp_hand","cards":[
         {"oracle_id":"a3fb7228-...","name":"Mountain","count":2},
         {"oracle_id":"fcb7c93c-...","name":"Screaming Nemesis","count":2}, ...],"total":7},
      {"region":"r:opp_library","cards":[...],"total":49},
      {"region":"r:opp_sideboard","cards":[...],"total":15},
      {"region":"c52","cards":[{"oracle_id":"bde976c6-...","name":"Gudul Lurker","count":1}],"total":1}]

`total` equals the region's `count` (1 for a face-down object); cards are
grouped by (name, oracle_id) and sorted by name; a face-down object is
labelled with its real face. `[]` means labels were on and there was no
region; an absent field means labels were off.

**Leak protection.** The labels are computed beside the record body, keyed by
the body instance, and appended only by `Serializer.write`/`writeChoice`; the
body sent to an agent (`Serializer.asRequest`) is built before and never
contains them, and `asRequest` throws if it ever did. With a socket agent and
a dump, the dump is labelled and the requests are not. The Python side
rejects a live request carrying `hidden_truth`
(`HistoryReassembler.absorb(..., where="live request")`, which the agent
uses), and `scripts/validate_dump.py` checks that every region has exactly
one label with the right total, labels only name regions of their record,
no label sits inside `hist_new`/`state`/`prompt`/`actions`, and no live request
(`chosen_match` "pending") carries labels. Turning labels on changes nothing
else: seeded games and every other byte of the dump are identical.

## Gotchas

- Card tokens carry `oracle_id` (and `source_oracle_id` on stack items and
  actions) whenever the card's name is emitted, from
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
  opponent hand (schema 13)/library/sideboard `count` aggregates), instead of
  the former literal `HIDDEN` / `<hidden>`. History tokens carry no card names
  at all since schema 13. Per the rules an unseen card has no
  name, and absence means unknown as for every other key. The controller of a
  face-down permanent still sees its name. Test with `magicnet.dump.is_hidden`.
  That check reads `PlayerView` opponent lists, which only `HostedMatch`
  fills in, so `Harness` calls `updateOpponentsForView()` itself.
- Forge has no reveal event: reveals arrive only as `PlayerController.reveal`
  on the controller of each player being shown (the owner is usually skipped).
  `RevealKnowledge` records opponent hand cards from that callback as
  `(card id, game timestamp)` per viewer; they are emitted named with
  `known_by_reveal: true` while still in that hand, and every viewer's history
  gets a `reveal` event (with `obj` edges only for the recipient). Leaving the hand, or a
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
  card already in hand. The move event carries `face_down: true` when the
  card left face-down. Schema 13: a card leaving that hand into a place the
  viewer cannot see it (Brainstorm's put-back) makes the viewer forget every
  card it knew there. `validate_dump.py` re-derives all of this from the
  viewer's own history (by tracking handle) and rejects any named opponent
  hand card it does not explain.
- Schema 7 adds layer-resolved board state to battlefield `game_object`s:
  `types` (current type line), `colors` (current, `["colorless"]` if none),
  keyword fields (replaced in schema 10, below); `temp_power`/`temp_toughness`
  (a heuristic, replaced by `affected_by` edges in schema 13); and, on face-up objects only, choices
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
  or `ref` (`id`, `name`, `oracle_id`, `source_oracle_id`, `handles`,
  `source`, `named_card`, `chosen_player`, `attempt`, ...);
  every model input (number / flag / string / string_list) is under
  `components`, omitted when empty, e.g.
  `{"type":"game_object","zone":"self_battlefield","id":"c37","name":"Soul-Scar Mage","oracle_id":"...","edges":[...],"components":{"power":6,"toughness":5,"base_power":1,"base_toughness":2,"tapped":true,"ctr_+1/+1":1,"gained":["Trample"]}}`.
  `Schema.isTopLevel` is the single rule (`tests/test_schema.py` checks it
  against keys.json); read components with `magicnet.dump.comp`. Flags are
  written only when true, except an attacker's `blocked`. History `ability` text has Forge's
  run-unstable `Name (id)` object ids removed. The record envelope is flat;
  since schema 16 decision options are tokens too (see "Decision candidates
  and the prompt").
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
