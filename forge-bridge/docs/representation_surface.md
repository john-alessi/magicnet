# Representation surface: what Forge actually exposes

> **Status: scratchpad.** Nothing here is a committed design decision. This records
> what was verified about the engine, and which open questions remain, so the
> reasoning does not have to be reconstructed later. Claims are tagged
> **[verified]** (checked against the installed jar or upstream source, with the
> source named) or **[unverified]** (plausible, not yet checked).

## 1. The castable unit is `(card, state)`, not `card`

**[verified]** `forge.card.CardStateName` has **15 values**:

```
Original, Secondary, LeftSplit, RightSplit, Backside, Flipped, Meld,
FaceDown, EmptyRoom, PreparedSpell, SpecializeW/U/B/R/G
```

Adventures, split cards, MDFCs, rooms, specialize and manifest all reduce to this
one mechanism. `Card.getAllPossibleAbilities` walks states explicitly.

Worked example — `res/cardsfolder/.../brazen_borrower_petty_theft.txt`:

```
Name:Brazen Borrower
Types:Creature Faerie Rogue
AlternateMode:Adventure
ALTERNATE
Name:Petty Theft
Types:Instant Adventure
```

One `Card`, two `CardState`s, each with its own name, cost, types and
SpellAbilities. **Not** one spell with two alternative costs.

A third, separate case: after the adventure resolves the card sits in exile and
the creature half may be cast from there. That is a `CardPlayOption` (MayPlay
permission), rendered as `" by Petty Theft"` — the `" by "` suffix comes from
`GameActionUtil.getMayPlaySpellOptions` (**[verified]**, upstream line 391).

So "Brazen Borrower" is three distinct representational situations, not one.

## 2. Zones

### Command zone materializes floating effects

**[verified]** `EffectEffect` creates a real `Card` and scopes its statics,
triggers and replacements to the command zone:

```java
final Card eff = createEffect(sa, controller, name, image);
addedStaticAbility.setActiveZone(EnumSet.of(ZoneType.Command));
parsedTrigger.setActiveZone(EnumSet.of(ZoneType.Command));
parsedReplacement.setActiveZone(EnumSet.of(ZoneType.Command));
```

Consequence: continuous effects with no permanent behind them — a resolved
instant's "damage can't be prevented this turn", emblems — still have a backing
object. Most persistent state therefore has an object you can attend to, rather
than needing a separate scalar.

**Trap.** The action-side filter in `ComputerUtilAbility.getAvailableCards` is

```java
IterableUtil.filter(player.getCardsIn(ZoneType.Command), c -> !c.isImmutable() || c.isEmblem())
```

which excludes exactly the immutable effect objects. Correct for the action pool
(they are not castable), wrong for observation — reusing it would silently drop
the floating effects.

### Exile is several semantically distinct zones

Minimum distinctions:

1. **Inert** — exiled with no way back. Still belief-relevant as revealed
   information, and still counts for delirium / threshold / delve.
   **Not compressible**: a small number of cards can target cards in exile, so
   inert exile objects still need to be addressable edge endpoints.
2. **Playable by someone** — impulse draw, adventure, foretell, plot, suspend.
   Carries a `CardPlayOption`; **[verified]** it exposes `getPlayer()`,
   `isWithFlash()`, `getAltManaCost()`, `grantsZonePermissions()`.
3. **Pending return** — blink, suspend counters, linked-object returns.
4. **Face-down** — plus who is allowed to look.

Categories 1 and 2 are cheap to separate. Category 3 is the awkward one, below.

### Zones that have no object at all

Residual state needing explicit scalars: spells cast this turn (storm, prowess),
lands played, damage marked, whether a creature has attacked. Short and bounded,
but not covered by "attend to the object".

## 3. Stack

**[verified]** `MagicStack implements Iterable<SpellAbilityStackInstance>` — order
is native. Each `SpellAbilityStackInstance` exposes:

```
getId()                            stable identity
isSpell / isAbility / isTrigger / isOptionalTrigger / isStateTrigger
getTargetChoices()                 its targets
getSubInstance()                   nested sub-abilities
getTriggeringObject(AbilityKey)    trigger context
getActivatingPlayer() / getSourceCard() / getStackDescription()
getView() -> StackItemView         hidden-info filtered
```

Three consequences:

- **Stack items are not cards.** A trigger's source permanent sits on the
  battlefield while the trigger is on the stack, and one permanent can have
  several triggers on the stack simultaneously. Token identity must be the stack
  instance, with an edge to the source card.
- **The stack is a list of trees**, not a flat list, because of `getSubInstance()`.
- **Stack items both have targets and are targets**, which is the main argument
  for modelling targets as edges rather than slots. `TargetChoices` already splits
  `getTargetCards / getTargetPlayers / getTargetSpells / getTargetEntities`, and
  supports weighted edges via `addDividedAllocation`.

**[verified]** `isSplitSecondOnStack()` is purely a legality gate, so it is
redundant with action masking and need not be tokenised.

**[verified]** `isFrozen()` is **not** a rules concept — `freezeStack` /
`addAndUnfreeze` park items in a side `frozenStack` during casting so triggers do
not hit the real stack mid-cast. Priority is not granted while it is set. Do not
represent it.

## 4. Delayed triggers are not readable through the public API

**[verified]** `forge.game.trigger.TriggerHandler` holds three separate lists,
all private, with only `register`/`clear` exposed:

```java
private final List<Trigger> delayedTriggers;
private final List<Trigger> thisTurnDelayedTriggers;
private final ListMultimap<Player, Trigger> playerDefinedDelayedTriggers;
```

The three-way split is semantically meaningful (permanent / expires this turn /
player-scoped) and worth preserving if they are surfaced.

**Consequence:** "will this exiled card come back" is **two different mechanisms**,
not one:

- *O-Ring style* — a linked-object relation readable straight off the card
  (`getExiledWith`, `getUntilLeavesBattlefield`).
- *Flickerwisp / suspend style* — lives only in the trigger handler, reachable
  only by reflection.

**[unverified]** whether `GameView` surfaces delayed triggers at all. Assumed not.
If `GameView` is used as the observation layer, this is a hole in it.

## 5. Object identity and timestamps

**[verified]** `GameEntity implements GameObject, IIdentifiable` with
`public int getId()`; `Card` and `Player` both extend `GameEntity`.
`SpellAbilityStackInstance` has its own `getId()`.

So every referent an edge could point at has a stable integer id. Note this is
only needed for building the edge incidence at encode time — relational attention
does not require learned slot embeddings (see §6).

**[verified]** Timestamps are rules-level state (CR 613.7 layer / dependency
ordering), and Forge exposes them publicly:

```
Game.getNextTimestamp() / getTimestamp()   global monotonic counter
Card.getGameTimestamp()                    the 613.7 object timestamp
Card.getLayerTimestamp()                   continuous-effect layering
Card.getControllerTimestamp()
Card.getBestowTimestamp() / Transformed / Mutated / Prototype / World
StaticAbility.getTimestamp()               per-effect, via CardTraitBase
```

Worked example: a creature with one aura granting flying and another removing it.
The outcome is decided by the **static abilities'** timestamps, not the cards',
so the per-effect timestamp is the one that matters.

**Lower urgency than it first appears.** Forge already applies the layer system,
so `card.hasKeyword(Keyword.Flying)` returns the resolved answer and the observed
state is post-layering. Timestamps are needed for *lookahead* ("if I play the
loses-flying aura now, does it win?"), not for perceiving the current board.

**Encoding.** CR 613.7 is ordinal, not cardinal, and these are unbounded
monotonic longs. Encode rank-within-state or deltas, never raw values.

### Identity across timesteps

**[verified]** `Card.equalsWithGameTimestamp(Card)` is Forge's own "same object"
test. Under the rules a card that leaves and re-enters a zone is a **new object**
with a new timestamp.

**[verified]** `Card.setGameTimestamp` is called from exactly one rules path:
`GameAction.changeZone`. (The other call sites are token creation
`TokenEffectBase`, copy effects `CardCopyService`, `GameSnapshot`/`GameCopier`
cloning, and dev-mode cheats.) Turning face up/down and transforming do **not**
reassign it — those use the separate `transformedTimestamp` / `mutatedTimestamp`.

Therefore, for battlefield permanents, **`gameTimestamp` order is exactly
entry order**.

**[verified]** `Zone` keeps an insertion-ordered `cardList`, and its
`protected void sort()` has **zero callers anywhere in the jar** — dead code.
Battlefield order as delivered is already entry order; the only requirement is
that our encoder must not re-sort it.

The face-down flicker case **mostly dissolves**: a face-down permanent that is
exiled and returned comes back face up as its printed card (and if the card is
not a permanent it simply stays in exile). So there is no "which face-down
creature is which after a blink" problem to solve. What remains is the ordinary
need to keep two *simultaneously present* face-down permanents distinguishable
across timesteps, which entry order already provides.

### Battlefield as an ordered zone

Treating the battlefield as ordered by entry (rather than as a set) is cheap,
already true in Forge, and gives free cross-timestep correspondence. Three
constraints on how that order is encoded:

1. **Ordinal, not absolute.** Absolute learned positional embeddings defeat the
   purpose: when a permanent dies, every later index shifts by one and all
   surviving permanents change encoding. Use pairwise rank ("i entered before j")
   or clipped rank-difference, which is stable under deletion.
   This is the *same mechanism as Shaw et al.* clipped relative positions, so it
   costs nothing beyond the relational-attention machinery in §6 — it is just one
   more relation family.

2. **Order is global, not per-player.** The timestamp counter is shared, so the
   two battlefields interleave. If tokens are laid out as contiguous per-player
   runs, absolute position destroys that interleaving; a relation over global
   `gameTimestamp` rank is layout-agnostic and keeps it.

3. **Entry order is not layer order.** CR 613.7 gives fresh timestamps on
   attach/re-attach, bestow, and transform, and Forge draws those from the global
   counter at **73 call sites across 50 classes**, writing them onto continuous
   effects and static abilities rather than onto `Card.gameTimestamp`. So the
   two-auras-granting-and-removing-flying example is **not** settled by
   battlefield entry order. Equipment is the clearest divergence: equipping
   re-stamps the effect while the equipment's position on the battlefield is
   unchanged. (Still lookahead-only — see above.)

**Leak trap.** Use ids and timestamps for *correspondence* only — never as content
features. Ids are assigned at card creation and will correlate with library order,
which leaks the opponent's deck. Rank-normalise anything that reaches the model.

## 6. Edges

### Sources on the Forge side

**[verified]** first-class and bidirectional for the exile case:

```
Card.getExiledWith()              -> Card     "I am exiled under X"
Card.getExiledBy()                -> Player
Card.getUntilLeavesBattlefield()  -> CardCollectionView   inverse edge
Card.getRemembered() / getImprintedCards()                generic linkage
```

The four keys in the entire untyped `Map<String,Object>` callback surface —
`Target`, `Attachments`, `Attacker`, `RevealedCard` — are all edges too.

**[verified]** `forge.game.ability.AbilityKey` exists and has **124 constants**.
(An earlier note recorded it as missing; that note looked in
`forge.game.spellability`, the wrong package.)

### Working rule for when an edge must be explicit

> If two distinct game states produce identical token multisets and differ only in
> their edges, the edge must be explicitly encoded. Otherwise the model is
> provably unable to distinguish them.

Default is ordinary self-attention; relational encoding is the exception, and the
exception has to be earned by this rule rather than by intuition. This keeps the
relation list closed and derived.

Applying it:

| relation | verdict | note |
|---|---|---|
| targeting | **must be explicit** | Cast Down on Delver vs on another creature: identical tokens, different value |
| blocking / attacker pairing | **must be explicit** | 4 attackers × 4 blockers is multiset-identical under any pairing |
| attachment | borderline | partially recoverable, since computed P/T already reflects auras |
| `exiledWith` | lower priority | only bites with two O-Rings and two prisoners at once |
| untapped lands ↔ hidden hand | **no edge** | no relation exists in observable state; belongs to the belief module |

### Direction

Relations are defined over **ordered** pairs, so `exiled_under` (A→B) and
`exiles` (B→A) are distinct ids with separate parameters. Both directions are
emitted explicitly rather than expecting the model to infer an inverse.
Rationale: the two directions serve different information needs and have
different cardinality (`untilLeavesBattlefield` is one-to-many,
`exiledWith` many-to-one).

Genuinely symmetric relations (`same_controller`, `same_type`) get **one** id
applied both ways, not two.

Same ordered pair can carry several relations at once, so the encoding sums over
a multi-hot incidence rather than indexing a single relation id.

### Slots are not required

Relational attention needs only an edge list `(i, j, r)` over token *positions*,
built at encode time from the engine graph, scatter-added into the attention bias.
Which token index holds which object is encoder bookkeeping, not a learned
embedding.

Learned slot/identity embeddings are the *alternative* encoding — putting the
pointer in the token's content and letting the model resolve it by vector
matching. Redundant if relational attention is used. They would only be needed
for multi-token objects (avoided, since modifiers and counters fold into the base
object vector via a learned combiner) or for persistent identity across history
tokens. A pointer-style action head also does not need them: logits go over
object token positions directly.

### Prior art

- Shaw et al. 2018, *Self-Attention with Relative Position Representations*
  (NAACL). Relations are **deterministic** — `a^K_ij = w^K_clip(j-i,k)` — and only
  the per-relation embeddings are learned. Same machinery we want, different
  relation vocabulary. Their future-work section names our exact case: "extend
  this mechanism to consider arbitrary directed, labeled graph inputs".
  **Their Table 3 ablation finds `a^V` adds nothing over `a^K` alone**
  (25.8 vs 25.8; `a^K` alone 25.3 when `a^V` only). Start with `a^K`.
- RAT-SQL (Wang et al., ACL 2020) generalises this to arbitrary deterministic
  typed relations — closer to our setting than Shaw.
- RKT (Pandey & Srivastava, CIKM 2020) *estimates* relations from data because
  knowledge tracing has no ground-truth graph. **Not** our situation: we are
  handed the graph exactly, so estimating it would discard known structure. Its
  exponentially decaying forget kernel is still interesting for persisted action
  history, where recency is genuinely graded.

### Note on strings

**[verified]** String parameters in callbacks are localized UI prompt text
("Search your library?"), redundant with the adjacent `ApiType`, and would break
under locale change (`res/languages/` exists). Discard rather than tokenise.

## 7. Measured token counts

**[verified]** `StateExtractor` + `StateStats` build the intermediate token list at
every priority decision; `GameHistory` builds the public log. Two pauper mirror
matches, 10 games each, seed 42, chosen as fast/slow bounds. Both controllers
extract, so `self`/`opp` aggregate over both seats. **0 timeouts in both runs** —
every game ended naturally, so nothing below is truncated.

| | Rally mirror (aggro) | Gardens mirror (grindy) | Familiars mirror (combo) |
|---|---|---|---|
| game length (turns) | 9–22, mean 15 | 38–72, mean **57** | 19–44 |
| states sampled | 3,611 | 13,926 | 10,389 |
| total **with** history, mean / p99 / max | 161.9 / 444 / **459** | 594.0 / 1224 / **1273** | 545.9 / 1344 / **1394** |
| total **without** history, mean / p99 / max | 36.1 / 64 / **68** | 91.2 / 151 / **157** | 84.9 / 166 / **180** |
| edges per state, mean / max | 6.0 / 23 | 18.3 / 34 | — |
| history at game end, mean / max | 244.5 / 394 | 985.2 / **1121** | 964.3 / **1232** |
| wall clock, 10 games | — | 252 s | **706 s** |

### Board state is bounded; history is the entire scaling problem

Board state without history is **36–91 mean, 157 worst case**, even in 57-turn
games. It barely moves with game length, because zones reach a steady state — cards
leave the battlefield about as fast as they arrive. This number is robust and is
the one to design against.

History is **78% of tokens in the aggro mirror and 85% in the grindy one**, and is
the only term that grows without bound. All compression effort belongs there.

Per-zone drivers (grindy, excluding history): actions 17.9, exile 14.3 (max 36),
battlefield 12.6 (max 24), hand 6.3, graveyard 3.9. **Exile outgrew graveyard**,
the opposite of the earlier assumption, which independently justifies the §2
decision not to compress inert exile.

### History must come from the event bus, not from controller callbacks

**[verified]** Forge fires structured events on a Guava `EventBus` exposed by
`Game.subscribeToEvents`; ~58 `GameEvent*` types, including `SpellResolved`,
`CardChangeZone`, `CardDestroyed`, `CardSacrificed`, `TokenCreated`,
`AttackersDeclared`. `GameHistory` subscribes to the `GameEvent` interface, which
Guava dispatches for every subtype, so one method catches all of them.

This replaced an earlier attempt to scrape history from controller callbacks, which
undercounted by **~7×**. Two reasons, both structural rather than fixable in that
approach:

1. Controller callbacks are *choices*. History is *what happened*, and most of what
   happens is nobody's choice — a creature died, a spell resolved, a card was
   drawn. No callback reports those.
2. The action space is factored, so a cast is `{verb}{source}{target...}`.

Prefer the event bus over `Game.getGameLog()`, which carries the same events as
localized prose — it would break under locale change and require re-parsing
structure we already have.

**Kept 19.7% of fired events.** Dropped: `PlayerPriority` (4132/3 games), `Zone`
(11554), `TurnPhase`, `ManaPool`, `CombatChanged`, `AddLog` — UI signalling or
already in the board state. Two further deliberate exclusions:

- `CardTapped` (465/game) — redundant with the `tapped` property already on every
  battlefield token, and overwhelmingly mana payment.
- `SpellRemovedFromStack` (182/game) — fires alongside `SpellResolved` for the same
  event; keeping both double-counts every spell.

Including those two put the grindy max at 1975 rather than 1273.

### Exchangeable-object collapse: only 4.5%, and not applicable to history

**[verified]** Tokens identical in every visible property *and* carrying no edges
can collapse to one token plus a count. The no-edge condition is load-bearing: an
edge individuates an object (it is the one being targeted, or the one blocking).

**[verified]** It does **not** apply to history. History is a *sequence*: two
identical events at different points in the game are not interchangeable, because
their order is the information. An earlier measurement that collapsed history
reported a 69% saving, which was measuring a compression that does not exist.
Restricted to board objects, the real saving is **4.4–4.6%**.

So collapse is not a useful lever at these board sizes. Keep it as an encoder-side
option for the arbitrary-token-count edge case; it is not a routine win and should
not shape the schema.

### The combo deck did not blow up context — but it did not test the worry either

**[verified]** Familiars tops out at **1394** tokens, barely above Gardens' 1273,
and its board maxes at **180** vs 157. Both well inside any context budget.

**[verified] But this does not disprove the arbitrary-count worry.** All 10 games
ended naturally with **0 timeouts** in 19–44 turns, which means Forge's AI never
assembled the infinite loop. What was measured is a combo *deck*, not a combo
*execution*. The unbounded case remains **untested, not refuted**.

The cost showed up as time instead. Familiars took **706 s for 10 games vs 252 s
for Gardens** — 2.8× slower with *half* the turns, one game taking 209 s. So
simulation cost is **per action, not per turn**, and the deck really is doing far
more per turn; it just stops short of looping. This matters for throughput
planning more than for context: an archetype can be cheap in tokens and expensive
in wall clock at the same time.

### History coalescing is worth doing; board collapse is not

**[verified]** Coalescing repeated action subsequences — the user's proposal — is
by far the larger lever:

| Familiars, history tokens | mean | p99 | max |
|---|---|---|---|
| raw | 462.5 | 1186 | 1232 |
| run-length (adjacent identical) | 371.1 | 922 | 957 |
| **cycle collapse (k-gram, k=1..8)** | **319.9** | **753** | **793** |

Cycle collapse removes **31% of history tokens on average and 36% at the
maximum**, against 5.7% for board-object collapse on the same run. The two are not
competing for the same space: history is 78–85% of all tokens, so this is the only
compression that touches the dominant term.

**A measurement bug, corrected.** An earlier version of this table reported cycle
collapse (393) *losing* to plain RLE (371), which is impossible if it generalises
RLE. It was my metric, not the data: it charged `k+1` tokens per cycle (body plus a
separate count token), so a simple run cost 2 tokens where RLE charges 1; and its
greedy rule maximised *span covered* rather than *tokens saved*, so a long weak
repetition could beat a short strong one. Fixing both — count carried as a
**property** of the body tokens, exactly like interchangeable-object counts, and
greedy on savings — makes `k=1` degenerate exactly to RLE, so cycle collapse now
dominates **by construction**. Re-running on the same seed reproduced every other
figure identically, which confirms only the metric changed.

Both the token totals and the game outcomes were bit-identical across the two runs,
so the fix is isolated to the metric.

This is an **encoder-side** optimisation. It should not enter the schema: the JSON
stays full-fidelity and debuggable, and coalescing becomes an A/B-able flag.
Repeat counts want bucketed or log encoding, like the object counts.

### Caveats

- **[unverified]** 10 games per archetype, one format, Forge's AI. Pioneer and
  genuinely wide token decks are untested.
- **[unverified]** No run has yet produced an actual unbounded loop, so the
  arbitrary-object and arbitrary-action cases are unmeasured. The proposed
  mitigation is **action masking in the environment** to cap repetitions —
  bounded bookkeeping rather than loop detection, which Forge does not implement
  (only a card-script `MaxRepeat` guard on `RepeatEffect`).
- History tokens now carry **full per-event content** (see §9). As predicted this
  **did not change any count** — Rally still maxes at 459 — only the content.
- Library is now a per-name histogram with known positions as `count=1` entries
  (see §10). Costs 14–22 tokens for the viewer's own library and 1 for the
  opponent's, against ~50 for per-card tokens. Rally 459 → 470 max.
- Action tokens are stage-one heads only; stage-two fan-out is a
  `target_candidates` property rather than expanded, which is the point of
  factoring.

## 8. Simulation throughput: the cost is Forge's AI, not the engine

**[verified]** Familiars runs at **~70 s/game** single-threaded (706 s for 10
games), vs ~25 s for Gardens. That looks alarming for sample generation, but
almost none of it is cost we will actually pay.

### 95% of wall clock is inside Forge's AI decision callbacks

`DecisionLog` already times every controller callback. On a 2-game Familiars run
(114.5 s wall):

| | time | share |
|---|---|---|
| all AI decision callbacks | **109.0 s** | **95%** |
| rules engine + everything else | ~5.5 s | ~5% |

So the **rules-engine floor is roughly 3.5 s/game**, ~20× faster than observed.
Those callbacks are exactly what our policy replaces. The bottleneck therefore
**moves to GPU inference** rather than disappearing — consistent with the earlier
UTD 0.63 GPU-bound finding — but Forge's own AI is not a floor on throughput.

### The slowness is combat, and it is combinatorial in board width

`declareAttackers` dominates, and its per-call cost tracks how wide the board gets:

| deck | declareAttackers calls | mean per call |
|---|---|---|
| Gardens (grindy) | 120 | **4.5 ms** |
| Rally (aggro) | 104 | **23 ms** |
| Familiars (combo, makes bird tokens) | 45 | **1,544 ms** |

Familiars is **340× Gardens per call** off only 45 calls, and that single method is
61% of total runtime. Forge's combat AI enumerates attack subsets, which is
exponential in creature count; wide boards blow it up.

Two consequences:
- Go-wide decks (**Elves** is the natural 60-card constructed representative) will
  be slow for this reason, not because of token counts. Board tokens stayed at
  180 max.
- **Any plan that uses Forge's AI as a bootstrap or benchmark opponent eats this
  cost**, even though self-play will not.

### Other measurements

- **[verified]** Token extraction overhead is **3.7%** (679 s with extraction off
  vs 706 s on). Instrumentation is not the problem.
- **[verified]** Game logic is genuinely **single-threaded** — one `forge-game`
  thread pegged at 99.5% of one core; the rest is JIT warmup and idle GC. So
  process-level parallelism is the right lever.
- **[unverified]** 6 concurrent processes gave **3.5× speedup (59% efficiency)**,
  but on a box contended by a browser and 9 GB of swap. Near-linear scaling on a
  clean machine is plausible but **not demonstrated**.

## 9. History event content

**[verified]** Implemented in `EventTokenizer`. All 25 kept event types now carry
typed detail; counts are unchanged (Rally max 459, identical to the contentless
run).

### Read through Forge's visitor, not reflection

Forge ships `IGameEventVisitor.Base<T>` with a null-returning default for all 58
event types, so only the kept types need an override and each gets
compile-time-checked field access. Reflecting over field names would break
silently on any upstream rename. Unhandled types fall back to a bare token.

Four kept events carry **no fields at all** — `CardDestroyed`, `TokenCreated`,
`FlipCoin`, `RollDie`. "Which card died" must therefore be recovered from the
accompanying `CardChangeZone`, which is a reason to keep both.

### History is per-viewer, not shared

Events carry `CardView`, which is **full information**. The previous shared
history was safe only because tokens were contentless; adding card identity makes
a single shared log a hidden-information leak, since a library-to-hand
`CardChangeZone` exposes the card the opponent drew.

Each player therefore gets its own `GameHistory` and its own tokenizer. The two
logs are deliberately not identical. The duplication is cheap (text) and makes a
cross-agent leak **structurally impossible rather than merely guarded against** —
which matters because in self-play both agents live in the same process.

**Visibility is snapshotted when the event fires**, via
`CardView.canBeShownTo(PlayerView)`, not re-evaluated at extraction. Observation
is immutable, and deciding later would be wrong in both directions: a card drawn
hidden but revealed later would leak backwards through the log, and a card
revealed on the stack but since returned to hand would be erased even though the
opponent legitimately saw it.

`Harness.verifyNoHistoryLeak` asserts the boundary on dumped states: every
`CardChangeZone` into the *opposing* player's hand must be redacted. Current
status: **163 opponent draws seen, 0 leaks, 63 redactions applied**.

### `BlockersDeclared` has a self-pair sentinel [verified]

Forge builds the map as attacker → blockers, but for an **unblocked** attacker it
stores `List.of(attacker)` rather than an empty collection:

```
blockers = combat.getBlockers(attacker)
if (blockers.isEmpty()) value = List.of(attacker)
```

So a self-pair means *unblocked*. The first implementation emitted these as block
edges, which would have taught the model that creatures block themselves — in
Rally, **every** block edge in the sample was actually a spurious self-pair.
Now emitted as an `unblocked` edge, with real pairs first as adjacent
`(blocks, blocked)` edges plus explicit `blocks_count` / `unblocked_count`.

`AttackersDeclared` was checked for the same pattern and is clean — key is the
defender, values are `getAttackersOf(defender)`, no sentinel.

### Embedding recommendation

Three distinct kinds of content, which should **not** share a mechanism:

- **Event verb** (25 closed types): a plain learned embedding table, *not*
  off-the-shelf NLP vectors. "Destroyed" and "Sacrificed" are near-synonyms in
  English but rules-wise crucially different (regeneration, dies-triggers,
  destruction replacement effects), so word similarity encodes a wrong prior.
- **Card referents**: reuse the existing card-embedding pipeline.
- **Free-text fields** (`stackDescription`, `targetDescription`): localized prose,
  the same trap as `getGameLog`. Prefer structured refs; currently kept as raw
  strings for debuggability only.

Composition mirrors the modifier treatment: `MLP(verb, card, actor, turn)`.

### Object references are namespaced [verified]

Card ids and player ids come from **independent, overlapping counters**:
`Game.nextCardId()` pre-increments and so returns 1 for the first card, while
players are 0 and 1. A bare id in an edge was therefore ambiguous, and the
relation name does **not** disambiguate it — a combat defender or a spell target
can legitimately be a player, a planeswalker, or a battle, and the latter two are
cards. Before the fix, `targeting` mixed both kinds in one relation, so "Bolt
targeting you" was indistinguishable from "Bolt targeting card id 1".

All references are now prefixed (`c123`, `p0`, `s4` for stack items), via
`StateExtractor.cardRef` / `playerRef` / `entityRef`, making edge targets disjoint
by construction. Verified across a Rally run: **0 bare numeric edge targets**,
with `defender` and `attacking` resolving to `p` and the rest to `c`. Token counts
were unaffected (Rally max still 459).

## 10. Library representation and position knowledge

One token type for the whole zone:

```
(card identity, position constraint, count)
```

A card whose position the viewer knows is just a `count=1` entry, so there is no
separate "known position" token type and no special case. Example, mid-game after
a Brainstorm [verified, `out_dt2`]:

```json
{"type":"library_card","zone":"self_library","id":"c129","name":"Gurmag Angler","count":1,
 "position":"exact","depth_from_top":0,"depth_from_bottom":50}
{"type":"library_card","zone":"self_library","id":"c115","name":"Island","count":1,
 "position":"exact","depth_from_top":1,"depth_from_bottom":49}
{"type":"library_card","zone":"self_library","name":"Counterspell","position":"unknown","count":4}
...
{"type":"library_card","zone":"opp_library","name":"HIDDEN","position":"unknown","count":51}
```

### Why the unknown group is collapsed

Collapsing is **lossless, not lossy**. "Unknown position" means there is no fact
distinguishing one copy from another, so N identical tokens carry exactly the
information of one token with `count=N`. This is unlike the battlefield, where
two copies diverge through damage, auras or summoning sickness and therefore need
separate tokens.

It also matches the action space: library choices ("search your library for a
card") are over card *names*. "Which of the four Defiles" is not a real choice,
and surfacing it would add spurious branching.

And the count is itself the decision-relevant quantity -- "how many outs are
left" -- which a count property reads directly where four duplicate tokens would
make the model learn to count.

### No bucketing of depth

Depth gets a learned embedding per exact value, not a log/bucketed encoding. The
range is bounded (~60, with an overflow slot), so there is no compression forced
on us, and bucketing would bake in an assumption that early positions matter more.
They do today, but Magic is a game of edge cases and a future card may care about
a specific depth. The cost is ~100 vectors.

The honest tradeoff is **statistical, not structural**: depth 45 receives far
fewer gradient updates than depth 0. If that bites, the fix is an added shared
scalar-depth component that the per-depth parameters can override -- not a change
to the representation.

`depth_from_bottom` is emitted alongside `depth_from_top`. "Put the rest on the
bottom in a random order" is routine wording and bottom-referencing effects
exist, so this saves the model learning `library_size - depth` as a subtraction.
It is derivable (library size is a global) but free to provide.

### Depth is never stored, only recomputed

`LibraryKnowledge` records **which cards** a viewer knows the position of, never
the position. Depth is recomputed from the live library at extraction time, which
is self-maintaining under every mutation:

- scry X to top, then draw X -> X leaves the library and drops out on its own
- scry [X, Y] to top, draw X -> Y's live index is now 0, with no bookkeeping
- an effect puts Z on top -> X's live index becomes 1, which is what the player
  actually knows

Storing depths would require re-indexing the whole map on every zone change and
would be wrong the first time we missed one.

Index 0 is the top [verified]: `Player.doDraw` takes `get(0)`, and the "you draw
cards from the bottom of your library instead" keyword takes `get(size()-1)`.

### Forge does not track this, so we do

[verified, bytecode]

- `GameAction.scry` calls `PlayerController.arrangeForScry` and then immediately
  moves the cards. Nothing records that the player now knows what is on top.
- `Card.mayLook` is a timestamped map for *continuous* grants ("play with the top
  card revealed"), not momentary looks.
- `GameEventScry` carries only `toTop`/`toBottom` **counts**, not identities --
  correctly, because the *public* fact is "they scried 2".

So the information exists for exactly the duration of the controller callback.
We capture it in four overrides:

| Callback | Cards | Constraint |
|---|---|---|
| `arrangeForScry` | both piles | EXACT |
| `arrangeForSurveil` | the to-library pile | EXACT |
| `orderMoveToZoneList` (dest == Library) | all | EXACT |
| `willPutCardOnTop` | the card | EXACT |
| `tempShowCards` / `endTempShowCards` | the shown pile | GROUP_TOP / GROUP_BOTTOM (deferred) |

Scry grants **exact** knowledge, not a region: the rules let you order both piles
("in any order"), and `arrangeForScry` returns an ordered `(top, bottom)` pair.

`orderMoveToZoneList` is the one that actually fires for Ponder and Brainstorm,
which do not go through the scry path at all. It is gated on `ZoneType.Library`
because the graveyard is also ordered but carries no hidden position.

### "Put the rest on the bottom in a random order": no callback exists

[verified, bytecode] Both `DigEffect` and `ChangeZoneEffect` take this branch:

```java
if (sa.hasParam("RestRandomOrder")) CardLists.shuffle(rest);   // no controller call
else                                controller.orderMoveToZoneList(rest, ...);
```

Forge is right not to ask -- the player does not choose that order. But it means
the engine never signals that the player *saw* the pile, so this is compensated
in middleware rather than wired to a decision callback. The signal that does
exist is **`tempShowCards(Iterable<Card>)`** / `endTempShowCards`, which
`DigEffect` calls around the look. It is absent from `GeneratedController`
because our own `ControllerGen.SKIP` excludes it, so it is a hand-override.

Note `CardLists.shuffle` is a plain list shuffle and fires **no**
`GameEventShuffle`, so there is no stale-knowledge interaction here.

**Classification must be deferred.** `endTempShowCards` fires *before* the
remaining cards are moved. Measured: a Commune with Spirits showed 4 cards still
sitting at library indices 0-3 of 46, and only afterwards were they shuffled to
the bottom. Classifying at reveal time recorded `GROUP_TOP` for cards that ended
up on the bottom -- not merely imprecise but **false**, asserting those cards
were the top 4. So `observeShown` queues the pile and `resolvePending` (called
once at the top of `extract`) classifies it after the zone changes have landed.
Groups are dropped whether or not they classify, so an unrelated later state
cannot coincidentally satisfy the check.

**The check is a verification, not an inference.** A group is recorded only if
the seen cards now occupy a contiguous block anchored at the top or bottom of the
library. The emitted bounds come from group size and library size -- both of
which the player knows, having watched k cards go to the bottom and being able to
count their own library -- never from the true index. If the cards do not form
such a block we learn nothing rather than guess: understating knowledge is safe,
overstating it is a leak. A block of one is recorded as EXACT.

Verified on a Bogles mirror (2x Commune with Spirits), 4 games: the pile
classifies as `GROUP_BOTTOM` at indices 42-44 of a 45-card library, emitting
`depth_min: 42, depth_max: 44`. 0 invariant violations across 60 states.

### Own vs opponent library

The viewer knows their own decklist, so the remaining multiset of their own
library is derivable from public information and the name-level histogram is
legitimate. They do **not** know the opponent's decklist -- that is the entire
premise of the inference problem -- so an opponent library collapses to a bare
count plus any specific cards separately learned.

### Invalidation

Shuffling destroys position knowledge. `LibraryKnowledge` subscribes its own
listener for `GameEventShuffle` (viewer-independent, so not threaded through the
per-viewer histories) and drops every fact about that library.

Verified live on a Dimir Terror mirror, 4 games: **52 position facts learned, 5
shuffles observed, 5 facts cleared, 0 invariant violations across 60 dumped
states.** Checked invariants: counts sum to library size; `depth_from_top +
depth_from_bottom == size - 1`; no two cards claim the same exact depth; the
opponent library never names a card.

**Known limitation [narrow].** Most library insertions are **public**: even when
you do not know *what* was put on top of your library, you saw *that* it
happened and can update the depth yourself, so our live-index recomputation is
correct. The limitation only bites when the insertion itself is hidden, which is
rare. In that case we report the true new depth rather than the stale one the
viewer believes, understating uncertainty by one position. No card identity
leaks. (Note this is about *your own* library; an opponent casting Brainstorm
affects theirs, and `learn` is keyed to the controller's own player, so only the
caster gains those facts.)

### Cost

| | before | after | self_library tokens (mean/max) |
|---|---|---|---|
| Rally mirror | 459 max | **470 max** | 13.9 / 15 |
| Dimir Terror mirror | -- | **424 max** | 16.8 / 22 |

Rally's library collapses to 14-15 tokens, exactly its 14 distinct card names.
Per-card tokens would have cost ~50. The whole feature is **+2.4% context** on
Rally.

## 11. Affordability and the cost of an illegal action

### 11.1 The candidate pool is unfiltered

`ComputerUtilAbility.getSpellAbilities(getAvailableCards(game, me), me)` applies
**no filtering at all** -- not timing, not cost. Measured against `sa.canPlay()`
and `ComputerUtilMana.canPayManaCost(sa, me, 0, false)`:

| Deck | mean candidates | `canPlay()` admits | also payable | forced pass |
|---|---|---|---|---|
| Mono Red Rally | 5.6 | 68% | 51% | 29.3% |
| Dimir Terror | 8.9 | 56% | 47% | 30.3% |
| Bogles | 4.7 | 62% | 49% | 28.1% |

**[verified]** Affordability filtering roughly **halves** the action space, and
the forced-pass rate is **28-30%**, remarkably stable across three decks. This
closes the §11 open question, which previously only bracketed it as 0.35%-91.2%.

**Caveat:** this was computed *with* the predicate under test, so 28-30% is an
**upper bound** on forced-pass. A false negative shows up as a spurious forced pass.

### 11.2 Exact pre-announcement affordability is not computable

CR 601.2 determines total cost at **601.2e**, *after* modes (601.2b), targets
(601.2c) and X are chosen. So affordability is choice-dependent, and the honest
question is existential: *does there exist* a combination of modes, targets, X,
alternative/additional costs, Convoke/Improvise/Delve payments and mana-ability
activations that pays the cost. The mana sub-problem alone is NP-hard once hybrid
costs and conditional producers are involved. This is why neither Forge's UI nor
MTGO pre-filters on cost.

The predicate was exercised on real payment-time mechanics (counts are chosen
abilities over 3 games of Dimir Terror):

    11x Tolarian Terror   cost reduction from graveyard size
     8x Snuff Out         alternative cost (4 life if you control a Swamp)
     6x Lórien Revealed   landcycling
     4x Deep Analysis     flashback
     3x Gurmag Angler     delve

**0 false negatives / 319 chosen abilities.** Note Snuff Out is castable with
*zero* available mana -- a naive mana check would reject it. **[unverified]** This
evidence is partly circular: Forge's AI consults `canPayManaCost` when choosing,
so the test shows consistency with the AI's choices, not rules soundness.

### 11.3 Forge does NOT roll back a failed payment [verified]

Measured with `-Dmagicnet.forcebad=N`, which deliberately returns an ability the
player cannot pay for. The engine recovers in the sense that **games still
complete normally** (18/21 turns, 0 timeouts over 25 forced failures), but the
state is corrupted in three distinct ways.

`ComputerUtil.handlePlayingSpellAbility` calls `moveToStack` **before** payment,
then on `CostPayment.payComputerCosts` returning false:

1. **Spells are stranded in the Stack zone.** The restore branch is guarded by
   `originalZone.is(Stack)` -- true only for a spell already *on* the stack. A
   normal hand cast has `originalZone == Hand`, so the move-back never runs.
   Observed: `stackZoneCards` climbed 0 -> 1 -> 2 as Counterspell and Mental Note
   each failed, with `stackInstances=0`. The cards are gone from hand permanently,
   with no stack instance -- ghosts.
2. **Activated abilities livelock.** Lórien Revealed's islandcycling never touches
   the stack, so nothing changes and the identical ability is re-offered every
   iteration -- 23 consecutive times in the probe. `SpellAbility.setSkip(true)` *is*
   called on failure, but the only consumer of `isSkip()` is
   `ComputerUtil.protectRecursion`; it is **not** a pool filter.
   The sole backstop is PhaseHandler's AI-only cap of **999 iterations** per
   priority window, after which it logs and passes.
3. **The stack is left frozen.** `freezeStack` runs before payment, `addAndUnfreeze`
   only on success. Observed `stackFrozen=true` persisting across every subsequent
   decision until some later cast unfroze it.

Additionally, **partial payments are never refunded on the AI path**.
`CostPayment.refundPayment()` exists but is called only from `PlaySpellAbility`
(the human path) and `GameActionUtil`. `payComputerCosts` returns false directly
from the failing cost part, leaving earlier parts (sacrifices, discards, paid life,
tapped permanents) applied.

Finally, the failure is **invisible to the engine**: `PlayerControllerAi.
playChosenSpellAbility` discards the boolean from `handlePlayingSpellAbility` and
unconditionally returns `true`, so `PhaseHandler` never learns the play failed.

**Consequence for training:** "let the environment arbitrate" is not viable *as
shipped*. An unaffordable action is not a no-op to be retried -- it silently
destroys cards and can livelock. But see §11.4: the rollback exists, it is simply
not wired to the AI path.

### 11.4 The rules mandate a rewind, and Forge implements one -- just not for AI

**[verified]** This is a Forge AI-path defect, not a rules model. CR 601.2h makes
the casting illegal if any step cannot be completed, and CR 728 (Handling Illegal
Actions) rewinds the game to immediately before the proposal. The card returns to
hand and costs are unwound. Nothing in the rules leaves it on the stack.

Forge's **human** path does exactly this. `PlaySpellAbility` on failure calls:

    GameActionUtil.rollbackAbility(sa, originalZone, zonePosition, payment, hostCard)
    game.getStack().unfreezeStack()
    manaPool.restoreColorReplacements()
    costPayment.refundPayment()

and `rollbackAbility` is a complete manual rewind: it first attempts
`game.restoreGameState()` (the snapshot path) and otherwise removes the card from
its current zone, clears `setCastSA`/`setCastFrom`, and reinserts it into the
original zone **at its original index** (`Math.min(zonePosition, zone.size())`) --
so even library position is preserved.

`ComputerUtil.handlePlayingSpellAbility` calls none of it.

**This reopens the permissive-mask option -- and costs less than expected.**
**[verified]** We do *not* need to hand-roll the recovery sequence. Disassembly of
`PlaySpellAbility` shows both `GameActionUtil.rollbackAbility` call sites (offsets
680 and 706) are *inside* `playAbility`, and the static entry point
`playSpellAbility(controller, player, sa)` delegates to it (offset 143). So simply
routing the cast through `PlaySpellAbility` -- which `TARGET_MODE=engine` already
does -- gets the full rewind for free, and a refused cast returns `false` rather
than corrupting state.

Measured on `pio_41`, 4 games, seed 42: **3 refused casts, all cleanly rolled
back, all games completed normally** (2,537 ability decisions vs 2,347 on
baseline). An earlier run sustained **4,379 consecutive rollbacks** without
corrupting the game, which is about as strong a soundness signal as this gets.

**The real blocker was elsewhere: `announceRequirements`.** **[verified]**
`PlayerControllerAi.announceRequirements` is a `lookupswitch` over `ApiType` with
only two cases; every other API -- including `Charm` (modal spells) -- falls
through to `aconst_null; areturn`. `playAbility` reads `null` as "the player
cancelled" and refuses the cast. Because the chooser then re-offers the same
ability, this **livelocks the phase handler**: the first engine-mode run spent
4,379 of 4,428 casts (98.89%) refusing a single card, Thassa's Intervention, and
never finished a game. Overriding the callback to answer the legal minimum drops
refusals to 3 of 253 (1.19%) and all games complete.

This is the same lesson as `ComputerUtilMana.canPayManaCost` (§14.10): a
`forge.ai` method standing in for a decision the *policy* will make. It is not a
rules gap and needs no rules modelling -- it needs the callback wired to the
policy's X/mode choice.

**Consequence for propose->validate->fallback.** The architecture needs two
separate properties and only the first comes free:

1. *Soundness* -- a refused proposal must not corrupt the game. **Provided** by
   `PlaySpellAbility`'s rewind.
2. *Progress* -- a refused proposal must not return the agent to an identical
   decision point. **Not provided.** Nothing in Forge guarantees this; the
   livelock above is precisely its absence. The environment must guarantee it
   itself, by consuming the decision on refusal and ultimately falling through to
   an unmasked tier.

This upgrades "the last fallback tier must be unmasked" from a capability
argument to a **termination** argument.

Regardless of which design is chosen, the corrupted state is cheaply detectable and
should be asserted on rather than trusted: a card in the Stack *zone* with no
corresponding stack instance is always a bug.

### 11.5 Cost of copy-based feasibility checking [verified]

`GameCopier.makeCopy()`, measured over 401 copies on the Dimir Terror mirror:

| mean | p50 | p90 | max |
|---|---|---|---|
| **5.69 ms** | 4.54 ms | 8.97 ms | 42.61 ms |

For scale, a whole decision averages ~5.4 ms in the same run, so **one game copy
costs about one decision**. Naive per-decision validation would roughly double
simulation cost.

It does not have to be per-decision. **90.5% of priority decisions are passes**, so
validation applies to ~9.5% of them: 0.095 x 5.69 ms is ~0.54 ms amortized, roughly
**+10%**. Tiering reduces it further -- most spells resolve via `canPayManaCost`
with no copy at all, and only genuinely entangled announcements (e.g. Delve on a
spell targeting your own graveyard) need simulation.

**[verified] `GameCopier.PRUNE_HIDDEN_INFO` is `false`.** A copy therefore contains
full hidden state, including the opponent's hand and library. That is safe only
because feasibility extracts a single boolean; nothing derived from a copy may ever
reach the observation path. Worth an explicit guard given the per-viewer redaction
work in §10.

**[superseded by §11.4]** The snapshot route -- `Game.stashGameState()` /
`Game.restoreGameState()` gated on `Game.EXPERIMENTAL_RESTORE_SNAPSHOT` -- would
cost a full `GameSnapshot.makeCopy()` per *priority decision*, now measured at
~5.69 ms (§11.5), i.e. roughly doubling simulation cost. `GameActionUtil.rollbackAbility`
achieves the same rewind on the failure path only, so the snapshot route is not
needed.

## 12. Episode budget (the chess-clock analogue)

Digital Magic enforces non-termination with a per-player clock (~20 min on MTGO),
not with slow-play penalties, which are a paper-tournament construct requiring a
judge. A clock is the better mechanism for us because it is **self-attributing**:
the looping player burns their own budget, so no progress measure is needed to
decide whose fault a stall was.

The budget should be metered in **decisions, not wall-clock seconds**: a decision
is what costs a forward pass, and a decision count keeps episodes reproducible.

### 12.1 Measured distribution [verified]

Per player per game, Forge-AI mirrors, 30 games each (8 for Familiars):

| Deck | n | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| Dimir Terror (all calls) | 60 | 315 | 301 | 427 | 629 | **629** |
| Mono Red Rally (all calls) | 60 | 328 | 291 | 554 | 694 | **694** |
| Bogles (all calls) | 60 | 245 | 178 | 346 | 1020 | **1020** |
| Familiars (all calls) | 16 | 1088 | 1033 | 1824 | 1831 | **1831** |
| Familiars (priority only) | 16 | 428 | 376 | 645 | 649 | **649** |

Ordinary games are **100-700 decisions per player**. Familiars -- the only
loop-capable deck tested -- tops out at **1831**. No run has yet produced an
actual unbounded loop, so these are legitimate-play figures throughout.

### 12.2 The 20-minute clock and a few-thousand-decision cap are the same constraint

**[unverified -- estimate]** What the MTGO clock actually meters during a loop is
client round-trips, not deliberation. At roughly 1-2 s per iteration, 20 minutes
permits on the order of **10^3 iterations, not 10^6**. If that estimate holds,
a cap of a few thousand decisions per player is not a throughput compromise that
distorts the game -- it is approximately the real constraint. Settling this needs
a measurement of MTGO's per-iteration round-trip cost.

### 12.3 Cost of the cap [verified]

Familiars measured at **7.92 ms/decision** (8 games, 137.9 s, 2178 decisions/game):

| Cap / player | Worst-case episode | vs mean Familiars game |
|---|---|---|
| 2000 | 32 s | 1.8x |
| 3000 | 47 s | 2.8x |
| 4000 | 63 s | 3.7x |
| 6000 | 95 s | 5.5x |

A cap of **~3000-4000 per player** sits at ~2x the worst legitimate game observed
while bounding the tail at under 4x the mean. That is the throughput protection,
and it is the point of the exercise: the cap exists to bound the tail, not to be
hit.

### 12.4 Design consequences

- **Expose remaining budget in the observation.** Pardo et al., *Time Limits in
  Reinforcement Learning* (ICML 2018): treating truncation as terminal is correct
  only when the limit is part of the task and visible, otherwise the value function
  is non-Markovian and truncation should bootstrap `V(s_T)` instead. Exposing the
  clock makes truncation-as-loss principled and makes clock management learnable --
  which is a real skill in digital Magic. Fits the open global-scalars gap.
- **Truncation must score as a loss, never a draw.** A draw is 0, which still beats
  a loss, so a draw-scored cap rebuilds the stalling incentive it was meant to remove.
- **[unverified]** These are Forge-AI decision profiles. An *untrained* policy will
  have a different, probably wider, profile early in training, so the cap should be
  re-measured against the policy rather than fixed from these numbers.
- Residual edge case -- a legitimate line needing more than the budget -- is
  measurable rather than hypothetical: run the trajectory cycle check (§12.5) at
  truncation and count how often the cap binds while a winning loop was in progress.

### 12.5 Detecting a winning loop is decidable

Deciding "will this ever loop" is undecidable; deciding "is this *observed*
trajectory in a cycle" is not. At truncation, canonically hash the state at each of
the looping player's priority passes (excluding the interchangeable-object counts
already coalesced per §7), and a repeated hash with the same player to act is a
cycle. Diff the state across one period; if a quantity moves monotonically toward a
win condition, extrapolate iterations-to-win.

Incomplete by construction -- it misses loops broken by an external trigger and
loops whose per-cycle delta is not monotone in one dimension. **Use it as
instrumentation, not to relabel reward**: crediting the win teaches the policy that
loops are free, removing exactly the clock-management pressure the budget exists to
create. The one case where relabeling is correct is if the target is later decided
to be paper-rules Magic with shortcuts, where truncation genuinely is our artifact.

## 13. Mana: payment as an action component, pool as observation

### 13.1 The payment decision is currently not the agent's

`PlayerControllerAi.payManaCost` is two instructions: [verified, bytecode]

```
invokestatic forge/ai/ComputerUtilMana.payManaCost(Cost, Player, SpellAbility, boolean)
ireturn
```

Source selection happens entirely inside that static solver and never surfaces on
`PlayerController`. The agent therefore emits a spell and a target, and *Forge*
decides which permanents to tap or sacrifice to pay for it.

This is a credit-assignment hole, not merely a fidelity gap. Worked example:
the agent controls three artifacts (one a Treasure), all lands tapped, and casts
Galvanic Blast (2 damage, or 4 with metalcraft, **checked on resolution**) at a 4/4.
The only {R} available comes from cracking the Treasure, dropping artifact count to
two and turning metalcraft off. The spell resolves for 2 and accomplishes nothing.
Every step was legal, so no mask applies -- but the decision that caused the loss was
made by the solver, and the gradient attributes the outcome to the targeting choice,
the only component the policy actually emitted. Under the current design this
misplay is **unlearnable**.

### 13.2 Hooks exist

`payManaCost` is `abstract` on `PlayerController`; the AI subclass merely delegates.
Also overridable and currently delegated: [verified, `javap -p PlayerController`]

- `payManaCost(ManaCost, CostPartMana, SpellAbility, String, ManaConversionMatrix, boolean)`
- `chooseCardsForCost(CardCollectionView, SpellAbility, CostPartWithList, int, boolean, String)`
- `chooseCardsForConvokeOrImprovise(...)`
- `chooseOptionalCosts(SpellAbility, List<OptionalCostValue>)`
- `chooseNumberForCostReduction(SpellAbility, int, int)`
- `specifyManaCombo`, `orderCosts`, `confirmPayment`, `getCostDecisionMaker`

`ComputerUtilMana.getManaSourcesToPayCost(...)` returns the concrete `CardCollection`
a solution uses, so a payment plan is exposable without writing our own solver.

### 13.3 Only *consequential* payments belong in the action space

Exposing full mana-payment combinatorics would dwarf the rest of the action space.
Proposed predicate: a payment is consequential when distinct solutions **differ in
observable state effects**.

- sacrifice-for-mana (Treasure, Lotus Petal, Clue/Blood/Food)
- Convoke / Improvise (taps creatures that wanted to attack or block)
- Delve (exiles graveyard -- delirium, Tarmogoyf, later Delve)
- pain lands, Gemstone Mine counters, snow vs. non-snow sources
- which colors are left open for a follow-up (purely strategic, no legality content)

Otherwise one Island is interchangeable with another and the solver should keep
doing it. Note the predicate is a cheap syntactic proxy for the real criterion,
which is strategic content; the empirical justification is that auto-tappers
demonstrably misplay for expert human players, i.e. the decision is not safely
delegable.

**[unverified]** How often the predicate fires. Measurable on the enumerator run.

### 13.4 Pool, not potential

The observation should carry **mana currently floating in the pool**, and must *not*
carry an aggregate "total mana available" scalar. Two reasons:

1. Total available mana is an approximation of a quantity that is **not exactly
   computable** (§11.2). Baking an estimate into the observation is worse than
   omitting it, because the model learns to trust it.
2. It is brittle in exactly the cases that matter: conditional producers
   (Cabal Coffers, Nykthos, Gaea's Cradle), restricted mana, and abilities whose
   yield depends on what is being cast.

In the Galvanic Blast case a scalar would have reported "1 red available" and hidden
the fact that producing it costs an artifact.

No aggregate is needed, because the individual sources are **already** represented as
battlefield permanents, and current-decision affordability arrives exactly via the
action mask. The division of labour:

| question | answered by |
|---|---|
| what mana do I have *right now* | pool tokens in the observation |
| can I pay for *this* action | the action mask (exact by construction) |
| should I hold up mana for next turn | battlefield source structure |

The third row is why source structure is still needed: there is no mask on a decision
the agent has not reached yet.

### 13.5 Pool contents are not a colour histogram

`ManaPool implements Iterable<Mana>`, and `Mana` is a record carrying far more than
colour: [verified, `javap -p forge/game/mana/Mana`]

```
color, sourceCard, manaAbility, player
isSnow(), isRestricted(), triggersWhenSpent(), addsNoCounterMagic(sa),
addsCounters(sa), addsKeywords(sa), getAddedKeywords()
```

Two red mana are therefore **not fungible** -- one may be restricted, make the spell
uncounterable (Cavern of Souls), add counters, or grant keywords. The pool must be
emitted as one token per `Mana` with these attributes, not as a per-colour count.

`ManaPool.willManaBeLostAtEndOfPhase()` and `hasBurn()` are legitimate global scalars.

### 13.6 What remains the policy's job

Three distinct categories, which must not be conflated:

| # | situation | whose job |
|---|---|---|
| 1 | illegal action | environment -- mask |
| 2 | legal, but no feasible completion | environment -- prefix-feasible mask |
| 3 | legal, feasible, bad | **policy** |

The design error to avoid is promoting (3) into (1) or (2). Anything hard-filtered as
"bad" is something the agent can never learn to do better than the filter's author,
capping it at our play skill. The Galvanic Blast line is only unambiguously wrong
under the added stipulation that no further damage is available -- precisely the kind
of context a filter cannot see and a policy can.

Resolution-time divergence caused by the *opponent* (they sacrifice an artifact in
response; they kill the target and the spell fizzles) and resolution-checked
conditions generally (morbid, delirium, threshold, storm, prowess) are irreducibly
category 3. The general solution is search; one-ply self-simulation is cheap, but
full lookahead is a separate architectural commitment, not a patch.

**Trap:** do not expose computed effect values ("this would deal 4"). At cast time
metalcraft *is* on, so such a feature would actively confirm the agent's wrong belief.
Any derived effect value must be computed post-payment or not at all.

### 13.7 Schema consequence

Counting artifacts is derivable from battlefield tokens, but "metalcraft, checked on
resolution" must come through the card representation and be composed with that count.
**[unverified]** Whether the card embedding carries conditional-effect structure
rather than just card identity.

## 14. Action space cardinality

### 14.1 What was measured

The candidate pool measured in §11.1 (mean 5.6-8.9) counts **abilities**, not
**actions**. One ability expands into many complete actions once modes, X and target
assignment are specified. `ActionCardinality` (`-Dmagicnet.cardinality=N`, samples
every Nth priority decision) estimates that expansion.

Three counts per decision, all including the always-available "pass":

- **raw** -- every distinct `(ability, mode-combo, X, target-assignment)` tuple
  counted separately.
- **dedup** -- as raw, but target candidates collapsed into equivalence classes, so
  two identical untapped Islands contribute one choice rather than two. This is a
  **sound, value-free** reduction: it removes duplicates, not bad actions. The class
  key is deliberately conservative (name, zone, controller, tapped, sick, P/T,
  damage, counters, face-down, attachments); over-collapsing would silently delete
  real choices, which is much worse than under-collapsing.
- **flat** -- `(ability, mode-combo, X)` tuples only, with target selection collapsed
  to a 0/1 feasibility indicator. This measures the size of an enumerated head *if
  target selection were delegated to a separate decoder*. See §14.4.

#### Modes

Modal spells are enumerated by reading `sa.getAdditionalAbilityList("Choices")` and
summing a subset-product DP over `CharmNum` / `MinCharmNum`:

```
sum over k in [min,num] of sum over k-subsets S of prod_{m in S} targets(m)
```

A mode with no legal target contributes a factor of zero and drops out
automatically, which is correct -- an unchoosable mode is not a choice. Hand-verified:
choose-one-or-more from 2 modes with 3 targets each = 6 + 9 = 15.

`CharmEffect.makePossibleOptions` is **not** used, because it calls
`AbilitySub.setSVar(...)` and would mutate live abilities [verified, bytecode]. The
getter is pure.

#### X and targets are entangled

X and target legality are **not independent**, and an earlier version of this probe
computed them separately and multiplied. That is wrong in both directions:

- Indomitable Creativity destroys *X* target permanents -- the target **count** is a
  function of X.
- March of Otherworldly Light exiles a permanent of mana value X or less -- the
  target **set** grows with X.
- Erebos's Intervention exiles *up to twice X* target cards from graveyards -- both.

The probe now binds each candidate X in turn, re-evaluates targeting under it, and
sums. Before this fix, Erebos's Intervention was measured at 6,291,468 completions
because `getMaxTargets` was evaluated with X unbound, yielding every subset of a
20-card graveyard; the corrected figure for the same state is ~53,000. The error was
roughly five orders of magnitude.

#### Mutating methods

**Five** Forge methods on the obvious path mutate live game state and would have
silently corrupted runs [all verified, bytecode]:

| method | mutation | handling |
|---|---|---|
| `ComputerUtilCost.setMaxXValue` | calls `sa.setXManaCostPaid(...)` | save/restore in `finally` |
| `CharmEffect.makePossibleOptions` | calls `AbilitySub.setSVar(...)` | avoided; pure getter used |
| `PossibleTargetSelector.selectTargets` | sets targets on the live SA | avoided (this is why Forge runs it against a `GameCopier` copy) |
| `ComputerUtilMana.calculateManaCost` | `Card.setCastFrom`, `sa.setMaxWaterbend`, `setXManaCostPaid` | avoided (§14.8) |
| `CostAdjustment.adjust` (6-arg) | drives Delve exile: `addDelved`, `setExiledWith`, `CardZoneTable.put` | avoided (§14.8) |

The last two are the reason cost reduction is not applied to payment counts; they are
the only two entry points that would apply it. The 3-arg `CostAdjustment.adjust`
overload also calls `setMaxWaterbend`, and applies only *raise*-cost statics anyway.

Everything else used is non-mutating: `sa.canTarget`, `sa.usesTargeting`,
`TargetRestrictions.getMinTargets/getMaxTargets`, `CostPart.canPay`,
`CostPayment.canPayAdditionalCosts`, `ManaPool` iteration,
`ComputerUtilMana.getAvailableManaSources` [verified, bytecode].

X enumeration is capped at `X_CAP = 20`; the `X-truncated` counter reports how often
that bound binds (zero in every run to date).

### 14.2 Reproducibility

**`--seed N` *does* pin a run; the harness timeout does not.** [verified,
measured -- supersedes the earlier finding below]

The cause of the observed divergence is **`--timeout` truncating games on wall
clock**, not any non-determinism in the engine. A game cut short by elapsed time
contributes however many decisions it happened to reach, so counts move run to
run while behaviour is in fact identical.

Controlled test, `pio_41`, seed 42, identical binary:

| configuration | run 1 | run 2 | behavioural diff (timing columns excluded) |
|---|---|---|---|
| 4 games, `--timeout 120` | 2,123 | 2,115 | n/a -- games truncated |
| 4 games, `--timeout 900` | 2,516 | 2,516 | **none** |
| 1 game, `--timeout 120` | 638 | 638 | **none** |

With a timeout generous enough that no game is truncated, consecutive runs are
behaviourally bit-identical: every count, option-histogram and branching column
matches, and only `total_ms`/`mean_us` differ.

Independently, the **offered ability pool order is stable**: 1,536 consecutive
pools dumped content-addressed (host card name + ability text, never object
identity) were byte-identical across two runs. This matters more than the
aggregate, because the action space hands the policy an *index* into that list.

**The hash-order hypothesis is therefore withdrawn.** `Card.hashCode` being
identity-based does not appear to leak into AI choices.

**Consequences:**

1. Earlier single-seed figures in this document that were discounted as noise may
   simply have been truncation artifacts. Range-over-three-seeds reporting is
   still right for *variance across seeds*, but is no longer needed to defend
   against run-to-run drift.
2. **Episode limits must be expressed in decisions or turns, never wall clock.**
   A wall-clock limit silently makes the environment irreproducible, which breaks
   replay-equivalence testing and makes any RL run unauditable.
3. Strict replay equivalence -- record the AI's chosen index, feed it back, demand
   an identical game -- is a viable acceptance test for the Python boundary.

### 14.3 Results

6 games per deck (3 for Familiars), sampled every 5th priority decision, three seeds.
[verified, measured]

Deck snapshots pinned by content hash, because the metagame folder is renumbered on
refresh and **deck index is not a stable identifier**:

| deck | md5 (first 12) |
|---|---|
| `[Pauper] #06 - Mono Red Rally` | `265cf22c79a0` |
| `[Pauper] #24 - Dimir Terror` | `78f77bd07381` |
| `[Pauper] #19 - Bogles` | `46bc902da0d7` |
| `[Pauper] #16 - Familiars` | `2c464cbf4092` |
| `[Pioneer] #05 - Azorius Control` | `8f4dd634463f` |
| `[Pioneer] #17 - Temur Indomitable Creativity` | `bc92b2c8cd80` |
| `[Pioneer] #41 - Dimir Control` | `e90ffb2567f5` |

**Raw complete actions** (ranges over three seeds):

| deck | mean | p99 | max |
|---|---|---|---|
| Mono Red Rally | 4.6 - 5.0 | 14 - 20 | 21 - 42 |
| Dimir Terror | 6.3 - 6.8 | 17 - 19 | 23 - 24 |
| Bogles | 3.7 - 4.3 | 16 - 25 | 21 - 33 |
| Azorius Control | 9.2 - 11.9 | 45 - 66 | 88 - 134 |
| Temur Creativity | 14.0 - 57.0 | 46 - 54 | 87 - **32,856** |
| Familiars | 40.8 - 122.0 | 477 - 612 | 501 - 654 |
| **Dimir Control** | 192 - **15,213** | 781 - **458,810** | 65,677 - **2,097,229** |

**Flat `(ability, mode, X)` tuples**, same runs:

| deck | mean | p99 | max |
|---|---|---|---|
| Mono Red Rally | 3.7 - 4.4 | 9 - 10 | 10 - 12 |
| Dimir Terror | 5.9 - 6.1 | 15 - 16 | 19 - 21 |
| Bogles | 3.3 - 3.6 | 10 - 12 | 12 - 14 |
| Azorius Control | 7.5 - 8.9 | 23 - 30 | 29 - 44 |
| Temur Creativity | 10.0 - 12.9 | 26 - 41 | 29 - **96** |
| Familiars | 7.5 - 9.4 | 20 - 23 | 25 - 27 |
| Dimir Control | 7.2 - 7.7 | 20 - 31 | 31 - 41 |

Dedup/raw ratio (seed 42, 6 games): Rally 0.960, Dimir Terror 0.998, Bogles 0.978,
Azorius 0.925, Creativity 0.993, Dimir Control 0.521, **Familiars 0.289**.

No decision hit the 10^7 saturation clamp.

### 14.4 Reading

> **Superseded in part by §14.8.** This section's recommendation rests on the flat
> `(ability, mode, X)` head being small, which it is *only because payment was not
> counted*. Payment multiplies the flat head and reaches 1.8M on Temur Creativity.
> The three-decoder architecture in §14.8 replaces the two-part one proposed here.
> The target-selection analysis below is unaffected and still holds.

**Flat enumeration over complete actions is dead.** The earlier conclusion -- drawn
from four Pauper decks with no X spells and no modality -- does not survive contact
with Pioneer. A single Dimir Control priority decision reached **2,097,229** complete
actions. That number is real, not an artifact; see §14.5.

**The entire explosion lives in target-set selection.** This is the central result:

| deck | raw max | flat max | ratio |
|---|---|---|---|
| Dimir Control | 2,097,229 | 41 | **51,152x** |
| Temur Creativity | 32,856 | 96 | 342x |
| Familiars | 654 | 27 | 24x |
| Dimir Terror | 24 | 21 | 1.1x |

The flat head never exceeded **96** across 21 runs and seven decks, and the two
decks with the worst raw blow-ups have flat maxima of 41 and 27 -- *lower* than
well-behaved Azorius Control at 44. Choosing which ability, which mode-combination
and which X is a small, stable decision. Choosing *which subset of objects to point
at* is the combinatorial one.

**This separates cleanly into a hybrid architecture:**

1. An **enumerated flat head** over `(ability, mode-combo, X)` -- at most ~100
   candidates, scored by attention over a list. Exact by construction, so
   factorization order and prefix-feasible masking are moot *for this part*.
2. An **autoregressive set-builder** over targets within the chosen tuple, with a
   stop token.

Part 2 is where the earlier theoretical worry about masking actually dissolves. For
the cards that cause the blow-up, prefix-feasible masking is **exact and O(1)**:

- `TargetMin$ 0` means stopping is always legal, so no prefix can strand.
- Targets are individually legal, so the mask is a per-candidate predicate.
- The only joint constraints in the entire card pool, `MaxTotalTargetCMC` (13 cards)
  and `MaxTotalTargetPower` (2 cards), are **downward-closed** knapsack bounds: mana
  values and graveyard power are non-negative, so every subset of a feasible set is
  feasible. The mask is `running_sum + cost(t) <= N`. [verified -- card scripts]

So the pathological cases are precisely the cases where the cheap exact mask works.

**Most of the tail is duplicates, but not all of it.** Familiars' dedup/raw is
**0.289** -- 71% of its raw space is interchangeable relabelling, overwhelmingly
identical Islands feeding Ghostly Flicker (which targets *exactly two*, so C(45,2) =
990 from one ability). Dimir Control's ratio is only 0.521, because graveyard cards
are genuinely distinct cards. **Dedup pays where objects repeat and not otherwise**;
it cannot be relied on to tame the tail in general.

### 14.5 The worst case, verified

Erebos's Intervention is worth recording exactly, because it is the shape that breaks
flat enumeration:

```
Name:Erebos's Intervention
ManaCost:X B
A:SP$ Charm | Choices$ DBPump,DBChangeZone
SVar:DBChangeZone:DB$ ChangeZone | Origin$ Graveyard | Destination$ Exile
  | TargetMin$ 0 | TargetMax$ XX | ValidTgts$ Card
SVar:XX:SVar$X/Twice
```

`TargetMin$ 0` with a `TargetMax` tied to X makes the legal action set a **union of
power sets**. In a state with 14 cards across graveyards, X <= 6, and 9 targetable
creatures for the other mode, the count is exactly

```
sum_{X=0}^{6} sum_{k=0}^{min(2X,14)} C(14,k)  +  9 * 7  =  53,311
```

which matches the measured 53,311 exactly. [verified, arithmetic] With a ~20-card
graveyard the same card reaches 2^21. The combinatorics are genuine; no amount of
sound deduplication removes them.

### 14.6 What this does not cover

- **Payment plans are not enumerated** (§13). Consequential-payment branching is
  undercounted, and it multiplies. This is the largest remaining undercount.
  **Now measured -- see §14.8. It did overturn the hybrid.**
- **Optional costs are not enumerated** -- kicker, Bargain, and
  `chooseOptionalCosts` generally. Temur Creativity contains kicker spells, so its
  figures are undercounts.
- **Multi-target joint constraints are not applied**, so multi-target counts are
  upper bounds. Quantified above: 15 cards in the entire pool, all downward-closed.
- **`X_CAP = 20`** truncates X enumeration. It has never bound in practice.
- Seven decks, two formats, Forge-AI play. Storm and larger boards would push the
  tail further up, but the §14.4 separation is the claim that matters, and it is
  structural rather than deck-specific.

Payment and optional costs both attach to the **flat** head, not the target decoder,
so they are the figures that could still overturn the hybrid. They are the next
measurement.

#### The modal ceiling is a property of the card pool

One component of the flat head is provably bounded without any play data. A static
pass over **all 32,608 card scripts** in
`/usr/share/forge-gui-desktop/res/cardsfolder/cardsfolder.zip` counts, for every
modal card, the number of legal mode-combinations (`C(n,k)`, or multiset `C(n+k-1,k)`
for the 23 `CanRepeatModes` cards). [verified, measured]

| modes | 1 | 2 | 3 | 4 | 5 | 6 | 18-20 | 31 | 35 | 55 | 126 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cards | 772 | 742 | 237 | 50 | 5 | 2 | 5 | 4 | 1 | 2 | 1 |

The maximum **in the entire game** is **126** (Unite the Coalition, repeatable, choose
5 of 9). Sublime Epiphany, Casualties of War, Final Act and Sune's Intervention are
31. Only **18 cards** exceed 16. A further 36 cards tie `CharmNum` to X or a dynamic
count, and all of those have at most 3 modes.

This is the structural reason modes are safe where targets and payment are not:
**mode count is fixed by the card, target and payment counts scale with the board.**
A 126-way card contributes 126 regardless of how large the game gets. Modes can
therefore be enumerated flatly and permanently, with no tail risk.

(Caveat on method: a naive `CharmNum\$` regex also matches inside `MinCharmNum\$`.
Fixed with a `(?<!Min)` guard. The Java probe is unaffected -- it reads Forge's
exact-keyed parameter map.)

### 14.7 Incidental

Familiars took **468.8 s** for 6 games against 10.7-16.7 s for the other Pauper decks
-- a ~30x difference, consistent with §12's finding that it is the only loop-capable
list in the set.

### 14.8 Payment plans: what breaks the flat head

§14.4 concluded that the flat `(ability, mode, X)` head stays small (max 96) and that
the whole explosion lives in target selection. §14.6 flagged payment as the one
undercount that could overturn that, because payment attaches to the flat head rather
than to the target decoder. **It does overturn it.**

#### Method

`PaymentCardinality` (`-Dmagicnet.cardinality.pay=1`) counts, for each playable
ability, the distinct **exact** payments of its mana cost:

- Requirements are the coloured shards (each a mask of acceptable colours) plus a
  generic count that accepts anything; a bound X adds `amountOfX * X` generic.
- Sources come from `ComputerUtilMana.getAvailableManaSources(player, true)`; each
  source's options are the colours its mana abilities can produce (`isAnyMana` -> all
  five, `isComboMana` -> `getComboColors`, else the produced string).
- A DP over sources accumulates the produced-colour vector, and each vector whose
  total equals the cost is tested for coverage by **Hall's condition** over the
  distinct shard masks. Since total equals need, generic is satisfied automatically
  whenever the coloured demands are.
- **raw** counts individual sources, so two untapped Forests are two plans.
  **dedup** counts equivalence classes (name, artifact/creature/land, sick, tapped,
  spend-triggers, restrictions, colour options), so two Forests are one plan but a
  Forest and a Llanowar Elves are two -- tapping the creature costs an attacker.

The reported `PAY` figure is `sum over abilities of flat(ability) * plans(ability)`,
because payment multiplies the flat head: every `(ability, mode, X)` tuple can be paid
in each of these ways.

Hand-verified against the debug trace. Badgermole Cub `{1}{G}` with sources
`[GW][CGW][G]`: S1+S2 gives 6 colour choices of which 4 contain a G, S1+S3 gives 2,
S2+S3 gives 3, total **9** -- matches. Professor of Symbology `{1}{W}` on the same
board gives 4+1+1 = **6** -- matches. Aven Interrupter `{1}{W}{W}` with three sources
has exactly **1**. [verified, measured]

#### Results

Six games per deck, every 5th decision, seeds 42 and 1337 (six decks). **These
figures are post-fix** -- the `{X}` double-count described in §14.9 inflated every X cost, and
the original sweep reported numbers up to 35x too high. [verified, measured]

| deck | md5 | flat max | flat x pay max | dedup max | worst single ability |
|---|---|---|---|---|---|
| Dimir Control | `e90ffb2567f5` | 23-31 | 426-926 | 115-161 | Witching Well 325 |
| Selesnya Company | `ce0009ce18e8` | 14-18 | 1,636-6,386 | 1,049-1,060 | Germination Practicum 5,496 |
| Azorius Control | `8f4dd634463f` | 37-44 | 1,672-**28,264** | 842-4,726 | Memory Deluge **15,688** |
| Temur Creativity | `bc92b2c8cd80` | 21-33 | 1,929-52,438 | 837-19,382 | Indomitable Creativity 12,612 |
| 5c Legends | `687cb0a26108` | 23-24 | 26,587-**72,738** | 10,358-16,328 | Silk, Web Weaver **63,570** |
| Niv to Light* | `14ec5a99bb78` | 32-33 | 75,479-**77,238** | 43,691-**77,238** | Bring to Light 25,168 |

\* **Niv to Light is partially contaminated.** It plays Leyline Binding, whose Domain
reduction is not applied, and at seed 42 that card *is* the worst ability (59,621).
The seed-1337 worst (Bring to Light, 25,168) is clean -- no X, no reduction -- but
deck-level maxima on this row should be read as upper bounds. **Enigmatic Fires is
excluded entirely** for the same reason, more severely.

**5c decks top the table, but not by much, and not because of colours.** The worst
clean single ability across every deck measured is Silk, Web Weaver at **63,570** on
5c Legends. Five-colour mana bases widen the *source* set, which raises payment
counts the same way a large board raises target counts -- but Temur Creativity, a
three-colour deck, reaches comparable numbers through X alone. **X and source count
are the two drivers; colour identity is only a proxy for the second.**

**Regression evidence that the fix was correctly scoped:** Azorius Control at seed 42
is bit-identical before and after (max 28,264, Memory Deluge 15,688), as is
Germination Practicum at 5,496. Neither has an X cost, so neither should have moved,
and neither did. Everything that changed had an `{X}` in it.

#### Reading

**Payment is still not a small factor, but it is no longer astronomical.** The worst
single ability is Indomitable Creativity at **12,612** payment plans (dedup 4,592),
and the worst decision reaches **52,438** flat x payment. The mechanism is the X
interaction: larger X means more generic mana, which means more source subsets, and
that count is incurred *per X value*. X multiplies payment, payment multiplies the
flat head.

The original reading of this section claimed 1,833,633 and 293,075. Those were
inflated by the `{X}` double-count (§14.9) -- roughly 35x. **The qualitative
conclusion survives and the quantitative one does not.** Payment still multiplies the
flat head, still reaches five figures (72,738 on 5c Legends), and is still far past
anything enumerable, so it still needs its own decoder. But it is no longer in the
millions, and the gap to an enumerable action space is three to four orders of
magnitude rather than five to six.

**The flat head's good property is destroyed.** §14.4's argument was that modes and X
are bounded by the *card* while targets scale with the *board*. Payment scales with
the board too -- it is a subset-selection problem over your permanents, structurally
identical to the Erebos graveyard case. The flat head was only small because payment
was excluded from it.

**Dedup works better here than on targets** -- 2.7x on Creativity
(52,438 -> 19,382) and 6x on Azorius (28,264 -> 4,726) -- because mana bases contain
genuinely interchangeable basics, whereas graveyard cards are distinct. But 19,382 is
still four orders of magnitude past enumerable.

**Revised architecture.** The enumerate-the-head idea survives only if payment is
*also* delegated to a decoder:

1. **Enumerated head** over `(ability, mode-combo, X)` -- still small (max 96), still
   exact, still an attention head over a short list.
2. **Autoregressive set-builder over targets** (§14.4) -- masking exact and O(1).
3. **Autoregressive source-selection for payment** -- the new requirement.

(3) has the same shape as (2) and the same mask story, though the mask is more
expensive: a prefix of tapped sources is extendable iff the residual cost is still
payable from the remaining sources, which is `canPayManaCost` on the residual --
polynomial and already provided by Forge, but not O(1). This is also where the §13
observation pays off: if the policy *emits* the payment plan, the NP-hard existential
becomes polynomial verification, with the policy supplying the witness.

**The tail is still a tail.** p50 of `flat x pay` is 3-8 across every deck. Half of
all decisions have fewer than twenty complete actions even counting payment. The
architecture has to be sized for the tail, but the tail is rare, which is what makes
an autoregressive decoder affordable rather than wasteful.

#### Limitations

- **Cost reduction is not applied** -- the dominant one. Both Forge entry points that
  would apply it mutate live state: `ComputerUtilMana.calculateManaCost` calls
  `Card.setCastFrom` / `setMaxWaterbend` / `setXManaCostPaid`, and the 6-arg
  `CostAdjustment.adjust` drives the Delve exile machinery [verified, bytecode].
  This makes these the **fourth and fifth** mutating methods found on an obvious
  path (see §14.1).
- Exact payments only; plans that float surplus mana are legal but uncounted.
- Phyrexian, Convoke, Improvise and Delve not enumerated (undercounts).
- Monocolour-hybrid shards (`{2/W}`) treated as coloured only.
- Mana abilities with a non-literal `Amount` counted as 1.
- Sources capped at 28; above that the ability is skipped.

The undercounts and the overcount run in opposite directions, but the conclusion is
not close enough for either to matter: the gap between 19,382 and an enumerable
action space is four orders of magnitude.

### 14.9 The cost of prefix masking

§14.8 concluded that payment needs an autoregressive decoder. That only helps if the
**mask** is affordable: at each decode step, for each candidate mana source, we must
answer *"if I commit this source to this colour, is the residual cost still payable
from the sources that remain?"* -- the predicate that makes legal dead ends
unreachable. `MaskCost` (`-Dmagicnet.maskcost=N`) simulates a masked decode over real
decisions and times two candidate predicates.

- **own** -- maximum bipartite matching of residual coloured shards against remaining
  sources, then a capacity check for generic. Pure; takes the remaining-source list
  as an argument.
- **forge** -- `ComputerUtilMana.canPayManaCost(ManaCostBeingPaid, sa, player, effect)`.
  The residual cost is advanced as the decode commits, so Forge sees the right
  *cost* -- but it re-derives available *sources* from the player and therefore
  cannot know part of the mana base is already committed to the prefix.

#### Results

Two games per deck, every 3rd decision, the two worst payment decks from §14.8.
[verified, measured]

| deck | seed | calls/action (mean/max) | own us/call | own us/action | forge us/call | forge us/action | disagree |
|---|---|---|---|---|---|---|---|
| Temur Creativity | 42 | 9.1 / 41 | **0.37** | **3.3** | 591 | 5,399 | 9.32% |
| Temur Creativity | 1337 | 15.1 / 126 | 0.37 | 5.5 | 1,405 | 21,232 | 6.76% |
| Niv to Light | 42 | 52.6 / 220 | **0.35** | **18.2** | 2,423 | 127,510 | 5.04% |
| Niv to Light | 1337 | 29.4 / 134 | 0.52 | 15.3 | 2,485 | 73,156 | 8.76% |

Decode depth is shallow: 2.0-3.7 steps per action (max 11). Mask width -- the number
of candidates scored at the widest step -- is 4.2-14.2 mean, max 30.

#### Reading

**Prefix masking is affordable.** The own predicate costs **0.3-1.0 us per call**
and **3-18 us per action** even on a five-colour deck. Against a policy forward pass
of order milliseconds, exact legality masking is roughly **1% overhead**. The
decoder never has to propose an illegal action, and never has to be taught not to.

**Forge's predicate is not usable for this.** At 0.6-2.4 ms per call it is three to
four orders of magnitude slower -- 128 ms of masking per action on Niv to Light,
which would dominate inference by 10-100x. It is in the same cost class as a
`GameCopier` copy (5.69 ms, §3), which makes sense: it runs a full payment
simulation rather than a feasibility check. **This corrects the earlier assumption
that we could simply piggyback on `canPayManaCost`.** We cannot; we need our own
predicate, and we already have one.

**And Forge's predicate answers the wrong question anyway.** Disagreement runs
5-9%, and it is **100% one-directional**: across 58,921 compared calls there are
**zero** cases where own says yes and Forge says no, and 3,295 the other way
[verified, measured]. That is the exact signature predicted by the semantic gap --
Forge cannot see the prefix, so it treats already-committed sources as still
available and over-reports feasibility. Using it as a mask would admit dead ends at
roughly the disagreement rate.

(One-directional disagreement is consistent with both "forge over-reports" and "own
is too strict". The semantic argument settles which: Forge is structurally unable to
account for committed sources. But the own predicate has not been independently
proven exact -- see limitations.)

**Forge's predicate is non-mutating** [verified, behavioural]. Tap state across the
whole battlefield, mana-pool size, `sa.isSkip()`, `getXManaCostPaid()` and
`getCastFrom()` were snapshotted before and after every measured action: **0
mutations across all four runs**. This matters because `canPayManaCost` delegates to
a 572-line `payManaCost` whose body contains `ManaPool.payManaCostFromPool`,
`SpellAbility.setSkip` and `MagicStack.addAndUnfreeze` -- evidently behind `test`
guards. Note the method: rather than trying to prove those branches unreachable by
reading branchy bytecode, the probe tested the behaviour directly. That is the
cheaper and stronger check, and worth reaching for first next time.

#### The exactness test, and the two bugs it found

A hand-rolled legality predicate that disagrees with the engine is a correctness
liability, so it needs its own check. There is a clean one: **at an empty prefix the
two predicates answer the same question.** With nothing yet committed, "payable from
the remaining sources" *is* "payable from scratch", so any disagreement there is
unambiguously a bug in the own predicate rather than the semantic gap.

The first run of that test came back at **96.07%** on Temur Creativity (6 too strict,
27 too loose), and the two failure modes had different causes. Both were real, and
one of them was also live in the §14.8 measurement.

**Bug 1 -- `{X}` is double-counted (also affected `PaymentCardinality`).** Iterating a
`ManaCost` yields `{X}` as a shard with `getColorMask() == 0`, which is
indistinguishable from true colourless `{C}` [verified: `new ManaCost(new
ManaCostParser("X R R R"))` yields **4** shards -- `{X}{R}{R}{R}` -- with
`getGenericCost() == 0`]. Both probes counted that shard *and* added `amountOfX * X`
generic, so `{X}{R}{R}{R}` with X=1 demanded 5 mana instead of 4, one of which had to
be colourless. This made X spells look unpayable -- the Indomitable Creativity "too
strict" cases -- and it corrupts the §14.8 payment counts for every X cost. Fixed by
skipping `ManaCostShard.X` in both files; §14.8's figures were re-measured after the
fix.

**Bug 2 -- the host card offered as a source for its own tap ability.**
`getAvailableManaSources` does not know which ability is being paid, so for a
`{2}, {T}` ability on a land that also taps for mana it still offers the host. The
probe then believed Fountainport could help pay for itself. Fixed by excluding the
host when `sa.getPayCosts().hasTapCost()`.

After both fixes: **100% empty-prefix agreement over 1,583 actions across Temur
Creativity, Niv to Light and Selesnya Company -- 0 too strict, 0 too loose.**
[verified, measured]

This does not prove exactness at *non*-empty prefixes, where no oracle exists; that
is precisely the region Forge cannot answer for. But it closes the only region where
an independent check is available, and it is the region where the predicate's own
machinery (matching, capacity) is fully exercised.

#### Limitations

- Exactness is verified only at the **empty prefix**. Deeper prefixes are unoracled.
- The own predicate assumes a source assigned to a coloured shard contributes its
  surplus amount to generic, and it does not model conditional producers,
  restrictions or spend-triggers. Matching is exact for simple colour-producing
  sources; the generic capacity check is a relaxation.
- Snapshot-based mutation testing proves absence on *observed paths*, not in
  general. A rarely-taken branch could still mutate.
- Two decks, two seeds, 2 games each. Mask width scales with the mana base, so
  Niv to Light is close to a worst case for Pioneer.
- Cost reduction is still not applied (§14.8), so residuals on reducible cards start
  too large. This inflates mask width for those cards rather than invalidating the
  latency result.

### 14.10 The prefix oracle: validating the mask against enumeration

§14.9 closed one of three claims. The decomposition:

| claim | oracle | status |
|---|---|---|
| our mana model matches Forge | `canPayManaCost` at the empty prefix | §14.9, partly |
| our incremental logic matches our model | exhaustive enumeration | **this section** |
| our model matches the actual rules | the documented gaps | open |

`PrefixOracle` closes the middle row. It is **test scaffolding, not inference
infrastructure** -- the brute force is exponential and never runs in the hot path.
What ships is `extendable()`, which decides feasibility in ~1 us without
enumerating anything; the enumeration exists only to check that it is right.

The brute force deliberately **shares no logic with the code under test**. It does
not call `apply()` and does not use matching: it chooses an option per used source,
then allocates each produced mana to a shard it can satisfy or to generic. A shared
helper is how a differential test quietly becomes worthless.

Two properties are checked over every reachable state:

- **A -- mask exactness.** `extendable(residual, remaining)` must equal
  `truth(residual, remaining)`. Both get identical inputs, so a mismatch isolates
  the matching algorithm.
- **B -- transition completeness.** From every solvable state some pick must remain
  solvable *after `apply()`*. This tests the transition rather than the predicate:
  `apply()` chooses greedily which shard a mana pays, and a wrong greedy choice
  could strand a payment even with a perfect mask.

States are keyed by `(used sources, residual shards, generic)` -- exactly
`extendable()`'s argument list -- so two prefixes with the same key are
indistinguishable to it and one representative suffices.

#### Result: a real bug, in the silent direction

First run, 4,008 boards, 73,385 states: **0 too loose, 0 stranded, 278 too strict.**
[verified, measured]

Every finding involved a source producing more than one mana (`[Cx2]`, `[Gx3]`,
`[Wx2]`). Matching is one-to-one, so a Nykthos- or Sol Ring-style source could
cover **one** shard no matter how much mana it made. `{C}{C}` from a single `[Cx2]`
was reported unpayable.

For a measurement that would be a modest undercount. For a mask it is a **silent
capability hole**: the agent could never make that payment, and nothing would ever
say so. This is the failure mode §14.9 named and could not detect, because at the
empty prefix Forge happened to agree.

Fixed by expanding a source producing `a` mana into `a` unit sources before
matching. Colour and amount travel together, so the committed option must be known;
option choices are therefore enumerated, but **only for sources producing more than
one mana**, which are usually none. With no such source the code reduces exactly to
the original single pass. Generic-capacity accounting is arithmetically unchanged,
so the fix can only ever *add* YES answers.

After the fix: **0 / 0 / 0 across 5 seeds, 100,008 boards, ~1.86M states.**
[verified, measured]

Property B never fired, before or after. `apply()`'s greedy shard choice is safe --
not because greed is right, but because the decoder needs only *some* live pick and
one always existed.

`PaymentCardinality.feasible()` does **not** share the bug: it works on a
produced-mana count vector, so multiplicity is handled correctly by its Hall's
condition check. The §14.8 figures are unaffected. [verified, read]

#### What the two oracles found together

Re-running §14.9's empty-prefix test afterwards did **not** reproduce its headline
100%, because that run covered three decks and this one covered five. The new decks
exposed pre-existing gaps in `sources()` -- not in `extendable()`. The oracles
separate cleanly: the prefix oracle says the matching is right, the Forge
differential says the *input list* is wrong.

**Conditional mana abilities (fixed).** Floodfarm Verge scripts `{W}` and `{U}`, but
its `{U}` activates only while you control a Plains or an Island. We read produced
colours off the script and ignored the activation restriction, so on a board of
Fountainport + Floodfarm Verge we claimed `{1}{U}` was payable. Fixed by gating each
mana ability on `SpellAbility.canPlay()`; verified pure by the existing snapshot test
(0 mutations). Azorius Control went from 22 too-loose to 7.

Two further gaps surfaced once the Azorius failures were cleared, both in
`sources()` rather than in `extendable()`:

**Restricted mana.** `Cavern of Souls` and `Starting Town` produce mana that may be
spent only on certain spells. This is resolvable **statically**, because by the time
payment begins the spell, mode, targets and declared cost are all fixed -- CR 601.2f
precedes 601.2h -- so a restriction is a filter applied once before decoding starts
and `extendable()` needs no knowledge of it. Fixed with
`AbilityManaPart.meetsManaRestrictions(SpellAbility)`, applied **per ability, not per
card**: Cavern of Souls keeps its unrestricted `{T}: Add {C}` and loses only its
any-colour ability. Verified by `diagnose()`:
`Cavern of Souls=play/free/C;play/RESTRICTED/Any`.

**Reflected mana.** `Mox Amber` scripts `AB$ ManaReflected`; its colours mirror the
legendary permanents you control, so `getOrigProduced()` returns the literal `"1"`
and our script parse returned nothing, dropping the source entirely -- silent
too-strict. Three candidate resolutions were measured rather than argued:

| resolution | too strict | too loose |
|---|---|---|
| drop the source (original) | 4 | 0 |
| assume any colour | 0 | 18 |
| `AbilityManaPart.canProduce(colour, sa)` | 4 | 0 |
| `CardUtil.getReflectableManaColors(sa)` | 0 | 7 |

`canProduce` returns nothing for reflected abilities, so it reduces exactly to
dropping the source. `getReflectableManaColors` is the correct API. The decisive
detail is that **an empty result is informative, not unknown**: Mox Amber with no
legendary permanents is activatable but genuinely adds no mana, so the ability is
dropped rather than falling back to any-colour. That distinction is worth 18 - 7 = 11
of the errors.

**The 7 that remained were not ours.** Every one involved `Starting Town`, whose
any-colour ability costs `{T}, Pay 1 life` -- the only non-`{T}` mana ability in any
measured game. Paying 1 life is legal at any life total of 1 or more (you may go to
zero and lose only at the next state-based check), so a plan that spends it is
rules-legal; Forge's AI mana planner declines to pay life for mana as a matter of
policy. This is the same situation as the man-lands, and is bucketed the same way.

Current empty-prefix agreement [verified, measured]:

| deck | agreement | too strict | too loose | Forge-conservative |
|---|---|---|---|---|
| Dimir Control | 446/446 (100%) | 0 | 0 | 0 |
| Temur Creativity | 507/507 (100%) | 0 | 0 | 0 |
| Selesnya Company | 104/104 (100%) | 0 | 0 | 0 |
| Niv to Light | 259/259 (100%) | 0 | 0 | 0 |
| Azorius Control | 935/942 (99.26%) | 0 | 0 | 7 |
| 5c Legends | 166/173 (95.95%) | 0 | 0 | 7 |

**Zero disagreements in either direction** across 2,431 actions and six decks, with
14 cases adjudicated to Forge's AI being conservative rather than to a bug of ours:

- **7 Azorius**, land-animation (`Restless Anchorage` `{1}{W}{U}`, `Hall of Storm
  Giants` `{5}{U}`). Neither has a tap symbol in its activation cost, so tapping the
  land for mana and then animating it is legal. Not always even a misplay: tapping
  Mutavault to animate it can turn on a Demon-count check.
- **7 5c Legends**, `Starting Town`'s life-paying mana ability, above.

These are bucketed rather than counted as failures on purpose. A differential test
carrying permanent known-bad entries stops detecting new ones, so each exemption is
a *named, adjudicated* predicate (`selfTappableHost`, `lifeCostManaAbility`), not a
tolerance threshold.

**The standing lesson: `ComputerUtilMana.canPayManaCost` is an AI helper, not a
rules oracle.** The clean framing that "at the empty prefix, disagreement means our
bug" is false, and aligning to Forge by default would have pushed the mask into the
silent too-strict direction twice over.

#### Limitations

- The oracle validates **internal consistency** -- incremental logic against
  exhaustive search over the same source model. It says nothing about whether that
  model matches the rules. That is the third row, and it is still open.
- Boards are capped at 5 sources because the brute force is exponential. The cap
  binds only the test; `extendable()` is polynomial and runs uncapped. Coverage is
  bought with adversarial colour structure (hybrids, multi-mana sources, Hall
  near-misses, colourless shards) rather than board size, on the argument that
  matching has no size-dependent branches.
- `extendable()` now answers `true` if either the unit cap (64) or the combination
  cap (256) is exceeded. Deliberate: a too-loose mask announces itself when Forge
  rejects the payment; a too-strict one has no symptom. Neither cap triggers on any
  board measured so far.

## 15. Open questions

- **[resolved -- see §11.1]** True forced-pass rate is **28-30%** (upper bound),
  measured across three decks. Previously bracketed 0.35%-91.2%.
- **[unverified]** Whether `GameView` exposes delayed triggers.
- Relation vocabulary is effectively **schema**: adding a relation later
  invalidates existing checkpoints, so it should be version-pinned alongside the
  token vocabulary.
- **[unverified]** How often the consequential-payment predicate (§13.3) fires.
  Measurable on the enumerator run.
- **[superseded -- see §14.4]** Completion-count distribution: an earlier reading of
  four Pauper decks (post-dedup p99 = 123, max = 140) suggested flat enumeration over
  complete actions was viable. It is not. Adding Pioneer decks with X spells and
  modality produced a single decision with **2,097,229** complete actions.
- **[superseded -- see §14.8]** That the explosion is confined entirely to target-set
  selection. True only while payment is excluded: the flat `(ability, mode, X)` head
  never exceeded **96** across seven decks and 21 runs, but payment multiplies it to
  **52,438** on Temur Creativity. Payment needs its own decoder.
- **[resolved -- see §14.4]** Joint target constraints (`MaxTotalTargetCMC`,
  `MaxTotalTargetPower`) are downward-closed and appear on only 15 cards, so
  prefix-feasible masking in the set-builder is exact and O(1).
- **[resolved -- see §14.8]** The flat head does **not** survive payment. Indomitable
  Creativity alone reaches 12,612 payment plans at one decision; X multiplies
  payment and payment multiplies the head. The architecture is now three decoders:
  enumerated `(ability, mode, X)`, autoregressive target set-builder, autoregressive
  payment-source selector.
- **[resolved -- see §14.6]** Mode count is bounded by the **card pool**, not the
  board: max **126** combinations across all 32,608 scripts, only 18 cards above 16.
  Modes can be enumerated flatly with no tail risk. This is the structural dividing
  line -- modes are fixed by the card, targets and payment scale with the board.
- **[unverified]** Whether **optional costs** (kicker, Bargain, `chooseOptionalCosts`)
  add materially on top. They attach to the flat head like payment does, but payment
  has already decided the architecture, so this now only affects the size of the
  problem rather than its shape.
- **[resolved -- see §14.9]** Prefix masking is affordable: **0.35-0.52 us per
  candidate**, 3-18 us per action, ~1% of a policy forward pass. But **not** via
  Forge's `canPayManaCost`, which is 0.6-2.4 ms per call and semantically wrong for
  the purpose (it cannot see which sources the prefix already committed, and
  over-reports feasibility 5-9% of the time, one-directionally).
- **[verified -- see §14.9]** `ComputerUtilMana.canPayManaCost` does **not** mutate
  game state, tested behaviourally over ~59k calls. It is the first Forge method on
  an obvious path that turned out to be clean.
- **[resolved -- see §14.9]** The own masking predicate is **exact at the empty
  prefix** (100% agreement with Forge over 1,583 actions, three decks) after fixing
  two bugs the test exposed: `{X}` double-counted as a colourless shard, and the host
  card offered as a mana source for its own tap ability. Deeper prefixes remain
  unoracled, since that is exactly the region Forge cannot answer for.
- **[note]** The `{X}` bug was also live in `PaymentCardinality`, so §14.8's original
  figures were up to **35x too high** (Indomitable Creativity 293,075 -> 12,612).
  Re-measured. Lesson: the differential test was worth more than the measurement it
  was checking.
- **[open]** How to apply **cost-reduction statics** without mutating the live game.
  Both Forge entry points mutate (§14.1), so reduced-cost cards (Leyline Binding's
  Domain) are counted at printed cost and are large overcounts. Enigmatic Fires'
  figures are invalid for this reason; no other deck's worst abilities are affected
  [verified -- card scripts]. A `GameCopier` copy would be correct but costs ~5.7 ms
  per call (§3).
- **[resolved -- see §14.2]** `--seed N` *does* pin a run. The apparent
  non-determinism was `--timeout` truncating games on wall clock. With a
  generous timeout, consecutive runs are behaviourally bit-identical, and the
  offered ability pool order is byte-stable across runs. The hash-order
  hypothesis is withdrawn. Episode limits must be counted in decisions or turns,
  never wall clock.
- **[unverified]** Cardinality on formats with storm or larger boards.
- **[resolved -- see §14.10]** Whether `extendable()` is exact at non-empty
  prefixes, where Forge cannot answer. Exhaustive enumeration says yes, over ~1.86M
  states -- but only **after** fixing a too-strict bug it found, where a multi-mana
  source could cover at most one coloured shard.
- **[resolved -- see §14.10]** Whether Forge's `canPayManaCost` is a sound rules
  oracle at all. It is **not** -- it is an *AI helper*. All 14 remaining
  disagreements are cases where the rules side with **us**: land-animation abilities
  whose host is a mana source (7, Azorius), and `Starting Town`'s `{T}, Pay 1 life`
  mana ability, which Forge's planner declines to use as a matter of AI policy (7,
  5c Legends). Each is now a named adjudicated predicate rather than a tolerance.
  Aligning to Forge by default would have pushed the mask into the silent too-strict
  direction twice.
- **[resolved -- see §14.10]** **Restricted mana** is now filtered with
  `meetsManaRestrictions(SpellAbility)`, **per ability rather than per card**. It is
  resolvable statically because the spell, mode, targets and cost are all fixed at
  CR 601.2f before payment begins at 601.2h, so `extendable()` needs no change --
  structurally the same conclusion reached for Convoke.
- **[resolved -- see §14.10]** `Mox Amber`'s reflected mana is resolved with
  `CardUtil.getReflectableManaColors(sa)`. The decisive point is that an **empty**
  result means "produces nothing", not "unknown": the ability is activatable but
  adds no mana when you control no legendary permanents. Treating empty as unknown
  and falling back to any-colour cost 11 extra false positives.
- **[open]** Mana abilities with **additional activation costs** are assumed freely
  usable. Fine for `Starting Town`'s 1 life; a sacrifice-cost producer would be a
  genuine too-loose gap. No such card appeared in any measured deck.
- **[open]** Whether the model-vs-rules row can be oracled at all without a
  rules-correct third implementation. Both existing oracles are internal or compare
  against an AI heuristic.

## 16. The Java-to-Python boundary

### 16.1 Record format [verified]

`magicnet.bridge.Serializer` writes line-delimited JSON, one object per
decision, enabled with `-Dmagicnet.jsonl=<path>`. It is deliberately the only
place the token representation is written out, so the offline dump and the
future live environment loop cannot drift: the live loop writes the same record
to a socket instead of a file.

| field | meaning |
|---|---|
| `schema` | bumped on any change that alters an existing field's meaning |
| `game`, `decision`, `turn`, `phase`, `viewer` | provenance |
| `hist_total`, `hist_from`, `hist_new` | delta-encoded history (§16.2) |
| `state` | non-history state tokens for this decision |
| `actions` | the legal pool, in the order the policy indexes into |
| `chosen` | index of the action taken, or `-1` for a pass |
| `chosen_match` | `identity`, `signature`, `absent` or `pass` (§16.3) |

`actions` is kept separate from `state` because the two are consumed
differently: tokens are embedded, actions are selected over.

### 16.2 History must be delta-encoded [verified, measured]

History is cumulative and append-only (`GameHistory.history` is only ever
`add`ed, never trimmed), so writing it in full on every record makes the dump
**quadratic in game length**. Measured on `pio_41`, 2 games, 1,211 decisions:

| encoding | size | per-record growth |
|---|---|---|
| full history per record | 45 MB | 4.8 KB at decision 0 -> 83 KB at decision 637 |
| delta-encoded | 12 MB | flat |

Extrapolated, the full encoding costs ~23 GB per 1,000 games, which is why this
could not be left for later. Append-only is what makes the delta sound; the
loader checks its reassembled length against `hist_total` on every record, so a
dropped or reordered record fails loudly instead of silently corrupting every
subsequent context.

### 16.3 Chosen actions are addressable as pool indices [verified, measured]

The action space hands the policy an *index* into the enumerated pool, and
replay-equivalence testing records that index and feeds it back. Both are
unsound if the ability the player actually plays is absent from the pool we
expose -- there would be no index to record. The AI selects through its own
internal path, so this was an assumption, not a guarantee.

| deck | chosen | by identity | by signature | **absent** |
|---|---|---|---|---|
| pio_41 Dimir Control | 232 | 216 (93.10%) | 16 (6.90%) | **0** |
| pio_sc Selesnya Company | 104 | 104 (100%) | 0 | **0** |

**Zero unaddressable.** But the 6.9% signature-only column matters: the AI
returns an equivalent-but-distinct `SpellAbility` instance a measurable fraction
of the time, so **matching on object identity alone would spuriously report
those actions as unaddressable**. `Serializer.indexOf` falls back to a content
signature (host card name, API, ability text -- never identity or card id).

### 16.4 Dump determinism [verified]

Two runs at seed 42 with a non-truncating timeout produce **byte-identical**
dumps (`md5 3cfea15002c40ba50a565a51c34b1d86`). Combined with §14.2, this makes
strict replay equivalence a viable acceptance test for the action path.

This holds only with an untruncated timeout. **Episode limits must be counted
in decisions or turns, never wall clock.**

### 16.5 Ids are join keys, not features

Token `id` properties exist solely so `edges` can be resolved to token indices.
They correlate with library order and would leak the decklist if embedded.
`StateExtractor.Tok.signature()` already excludes them; on the Python side
`resolve_edges()` rewrites edges to indices and `strip_ids()` removes both the
ids and the raw edges. Callers building tensors must do both, and
`scripts/validate_dump.py` asserts neither survives.

Measured edge resolution: **99.0%** (2,079 of 2,101 over 300 records). Every
unresolved edge is `stack_item.source_is` pointing at a card in the Stack zone,
which `PLAYER_ZONES` does not emit. This is benign -- the `stack_item` token
carries the card's name in its `source` property -- but it is why
`resolve_edges` drops unresolved targets rather than raising: a token can
legitimately reference something outside the viewer's information set.

### 16.6 Validation [verified]

`scripts/validate_dump.py` checks schema, history reconstruction, the leak
boundary and label usability. It is verified non-vacuous by deliberate
corruption:

| injected fault | caught |
|---|---|
| clean dump | passes |
| dropped record (history desync) | `history desync ... record starts at 151, loader has 150` |
| unredacted name in `opp_hand` | `unredacted name in opp_hand: Lightning Bolt` |
| unknown schema | `schema 99, expected 1` |

The leak check initially produced 4,560 false positives by testing for the
*presence* of a `name` in a hidden zone. Hidden cards are emitted with the
sentinel name `HIDDEN`, so the check had to test for a non-sentinel value. The
corrected check finds **zero** unredacted names across the dump: every one of
the 3,349 `opp_hand` and 1,211 `opp_library` tokens is `HIDDEN`.

Note that the viewer's **own** library is deliberately *not* redacted (31,652
`self_library` tokens carry real names), because per §10 its remaining multiset
is derivable from public information.

### 16.7 Action injection and replay equivalence [verified, measured]

`ActionSource` supplies a pool index (or `-1` to pass) in place of the AI's
choice: `ai`, `random`, `replay:<path>`, `socket:<host>:<port>`. Replay is the
differential test -- the recording side chooses through Forge's own path, the
replay side by index lookup, and the two share no logic.

**Forge signals "pass" with `null`, not an empty list.** `AiController`
returns `aconst_null`. An empty list reads as "a choice was made", and the same
player is re-offered priority forever: 379,807 records against an expected 638,
the same livelock signature as §11.4.

Replay equivalence, recorded and replayed as whole games:

| recording source | target mode | records | result |
|---|---|---|---|
| `random` (seeded) | default | 1,860 | **byte identical** |
| `random` (seeded) | engine | 2,094 | **byte identical** |
| live `socket` agent | default | 2,228 | **byte identical** |
| `ai` | default | 638 | diverges at record 48 |
| `ai` | engine | 640 | diverges at record 276 |

**The AI-recorded divergence is not a plumbing defect and cannot be fixed.**
`AiController` contains 12 references to `MyRandom` (`ComputerUtilCard` 4,
`ComputerUtil` 1): the AI's deliberation *consumes the shared random stream*.
Replay skips that deliberation, so the stream desynchronises. The first
observable consequence was traced precisely: a Fabled Passage search
("...then shuffle") fetched a different physical `Swamp` -- `c37` against `c42`,
every other field equal. Downstream, tie-breaks resolved differently and a
Heartless Act mode was pruned in one run but not the other.

This matters only for AI-recorded dumps. **Under training conditions both
players are policy-driven, no AI deliberation runs, and replay is exact** --
which the first three rows measure. A live game played through the Python
socket agent replays byte-identically from its own dump.

### 16.8 Object ids are not reproducible [verified]

Strengthening §16.5. Ids were already disqualified as features because they
correlate with library order. They are *also* unstable: the same seed can
assign different ids to identical cards, as the `c37`/`c42` Swamp above shows.
A model consuming them would be fitting run-to-run noise.

Two consequences:

1. **Card references must be stripped by value, not by key.** History tokens
   carry `card_id`, `source_id` and `target`, all raw object ids, none of them
   named `id`. `strip_ids()` therefore drops any value matching `c\d+`, so a
   newly added property cannot leak one silently. **Player** ids (`p0`/`p1`)
   are deliberately kept: there are only two, they are stable, and
   `from_player`/`to_player`/`actor` are meaningless without them.
2. **Replay must be compared modulo ids.** `scripts/check_replay.py` resolves
   edges to indices and strips ids before comparing -- i.e. it compares what the
   model would actually see.

Stripping card references currently costs information: a history event's link
to the board object it refers to survives only as a card *name*. The proper fix
is Java-side -- history tokens should emit card references as `edges`, which
already resolve to stable token indices. **[not started]**

### 16.9 The socket boundary [verified]

Line-delimited JSON over TCP, one line in and one line out, strictly
alternating. `{"event":"begin_game"}` must be acknowledged (the read blocks, and
that is what keeps the sides in lockstep); `{"event":"end"}` closes. A decision
request is **byte-identical to a dump record minus the trailing `chosen`
fields**, so the same loader serves both paths -- `HistoryReassembler` is shared
rather than reimplemented, since delta history is stateful in both.

Measured: 2,228 decisions served live, 0 transport errors, 0 history resyncs, 0
unaddressable actions; the resulting dump passes `validate_dump.py` unchanged.

**Agent failures are fatal by design.** Returning `-1` on a transport error
would make a crashed agent indistinguishable from a policy that always passes,
and training would silently collapse to that. Verified by injection: an agent
returning index 9999 fails at exactly decision 50 with `agent returned index
9999 outside [-1, 1)`.

That injection also exposed a real defect. Observation building was gated on
`TOKENS || Serializer.enabled()`, omitting `ACTIONS.needsRequest()`, so a socket
agent run *without* a dump path received the literal string `null` as its
observation. Earlier live tests had set a dump path and masked it. Now fixed,
and the controller throws if an action source needs a request and no observation
was built. **A test that only ever runs in one configuration does not cover the
others.**

### 16.11 The pass rate is real, but the denominator is wrong [verified, measured]

85-91% of recorded decisions are passes. The natural explanation -- Forge is an
MTGO-style client that asks for priority even when nothing can be done -- is
only partly right, and checking it surfaced a larger problem.

**The pool is essentially never empty.** Over a full game the AI faced 0.0%
empty pools. Passing is a genuine choice, not a formality. Priority is offered
**22.8 times per turn** across both players, which is the real mechanism behind
the user's intuition.

**Most of the pool is mana abilities the policy will never take alone.** Of
5,372 offered actions, 3,890 (72%) were `Mana`, and the AI chose one **zero**
times. 17.4% of decisions offered *nothing but* mana abilities. These cannot
simply be deleted -- deliberately floating mana is occasionally the only line
(§14.10's Cursed Totem case) -- but they dominate the action space while being
almost always wrong. Removing them moves the pass rate only 91.1% -> 89.2%, so
they are not the main story.

**The main story: the action space covers roughly half the real decisions.**
Action rate by phase exposed it -- every combat phase showed 0.0% actions across
192 offers, while the same game's history recorded 28 `AttackersDeclared` and 7
`BlockersDeclared`. Attacking and blocking are not pool entries; they are
separate `PlayerController` callbacks. `chooseSpellAbilityToPlay` is **1 of
about 40** decision callbacks, and everything not overridden falls through to
Forge's AI -- outside the policy's reach without being outside the *game's*.

Census over 4 games per deck (`-Dmagicnet.census=true`, off by default):

| callback | control deck | creature deck |
|---|---|---|
| `declareAttackers` | 35 | 42 |
| `orderSimultaneousSa` | 40 | 33 |
| `chooseSingleReplacementEffect` | 39 | 28 |
| `chooseSingleCardForZoneChange` | 21 | 4 |
| `chooseSingleEntityForEffect` | 17 | 4 |
| `chooseModeForAbility` | 15 | 0 |
| `declareBlockers` | 10 | 16 |
| `confirmTrigger` / `confirmAction` | 0 | 4 / 4 |
| **total outside the action space** | **177** | **135** |
| priority decisions that were actions | 198 | 85 |

So **47% (control) to 61% (creature) of real decisions are currently made by
Forge's AI, not by the policy.** The creature deck is worse, as expected: combat
is where its decisions live. A policy trained on this surface today would be
playing half a game, and self-play strength would be capped by the AI making
the other half.

This reframes the pass rate. It is not mainly noise to be reweighted away; it is
a symptom of measuring one surface while several others go unrecorded. Priority
passes are genuinely frequent and correct in Magic -- a control deck holding up
interaction *should* pass constantly -- but the denominator currently counts
2,115 priority offers while omitting the 177 decisions that carry comparable
weight.

### 16.12 Routing every decision to the policy [verified, measured]

§16.11 found that 47-61% of real decisions bypassed the policy. The fix is not
forty bespoke integrations. Forge's decision callbacks have only **two shapes**:

1. **Pick one of N.** `chooseSingleEntityForEffect`, `chooseModeForAbility`,
   `chooseSingleCardForZoneChange`, `chooseNumber`, `confirmTrigger`,
   `confirmAction` -- all of them. This is exactly the shape the boundary
   already had, so they need no new encoding, no new policy head and no new
   wire format. They are emitted as the same record with the options in the
   same `actions` array and a `kind` field saying what is being asked.
2. **Subset or assignment.** `declareAttackers` and `declareBlockers` choose a
   *set* of creatures paired with defenders. These are decomposed into one
   pick-one decision per creature -- "do not attack with X", or one option per
   legal defender -- which collapses them into shape 1.

`Serializer.buildChoiceBody` and `InstrumentedController.askChoice` implement
this; routing a further callback costs an option list, not an encoding. Schema
bumped to **2** for the `kind` field.

**Legal by construction.** Only defenders passing `CombatUtil.canAttack` and
blockers passing `CombatUtil.canBlock` are offered, so no combat proposal can be
illegal -- unlike cost payment (§14.10), the legality oracle here is a genuine
rules helper rather than an AI heuristic.

**Delegation is always available.** Any callback whose options cannot be
enumerated safely returns `DELEGATE` and Forge's AI answers as before. Two cases
delegate deliberately:

- **`DelayedReveal` choices.** These show the chooser cards they would not
  otherwise see. Routing them without putting those cards in the observation
  would have the policy choosing blind, which is worse than delegating.
- **Multi-mode `chooseModeForAbility`** (choose k of n). A subset problem that
  wants combat's sequential treatment; guessing here would silently pick the
  wrong number of modes.

Mid-sequence delegation during combat is also refused: abandoning halfway would
leave a partly policy-declared attack for the AI to finish, which is neither
side's intent. The whole declaration falls back instead.

Measured, 2 games, all routing on (`-Dmagicnet.combat -Dmagicnet.choices`):

| kind | records |
|---|---|
| `ability` | 6,835 |
| `declare_attacker` | 34 |
| `choose_card_zone_change` | 30 |
| `confirm_action` | 28 |
| `choose_entity` | 16 |
| `choose_mode` | 12 |
| `declare_blocker` | 1 |

Verified: combat choices take effect (28 `AttackersDeclared`, 2
`BlockersDeclared`, 33 `PlayerDamaged` in a game driven by an always-attack
agent); the dump passes `validate_dump.py` unchanged; and **replay remains
byte-identical with every callback routed** (6,956 records, 0 overruns).

Both flags are **off by default**, because routing changes which decisions exist
and would make previously recorded dumps incomparable.

Still answered by the AI: `chooseSingleReplacementEffect` (26) and
`orderSimultaneousSa` (15). Both are *permutation* problems rather than
pick-one, and both are usually low-stakes ordering. They are the honest
remainder. **[not started]**

### 16.13 Adding decisions later must not invalidate a checkpoint [design]

`orderSimultaneousSa` is the order simultaneous triggers go on the stack (called
from `MagicStack`) -- Arena's "manual trigger ordering". It and
`chooseSingleReplacementEffect` are still answered by the AI, so the question is
whether routing them *later* forces training to restart.

It does not, provided the policy head is a **pointer over the offered options**
rather than a fixed-size softmax over an enumerated action space:

- **The observation does not change.** Verified: state-token property keys are
  identical between `ability` records and routed-choice records, except that
  choice records omit `api`/`target_candidates` (those belong to action tokens,
  which a non-priority decision has none of). Board, history and hidden-info
  boundary are untouched.
- **The output dimensionality does not change.** A pointer head scores each
  option from its own content, so N options need no parameters proportional to
  N, and a new `kind` adds none.
- **What is new is one vocabulary entry** (the `kind` string). Vocabulary must
  therefore be *extensible* with an `UNK` fallback; a vocabulary frozen by
  enumeration over an old dump would make the new kind unrepresentable. This is
  the one thing that must be designed for now rather than later.

So a new decision kind is a **distribution shift, not an architecture change**:
weights transfer, and the policy is merely untrained on the new kind until it
sees some. For RL specifically, the MDP does change shape -- episodes gain
decision points -- so value estimates and any on-policy buffer go stale and
should be allowed to re-equilibrate; that is a fine-tune, not a reinitialisation.

**What *would* invalidate a checkpoint is changing the option schema**, because
that changes the input the head scores. That argues for settling the option
representation before any checkpoint exists, which is why §16.14 was done now.

### 16.14 Options reference objects by edge [verified]

Options previously carried `source_id` as a **raw integer** (`26`) while game
objects are namespaced strings (`"c26"`). Two consequences, both bad:

1. It resolved against nothing, so an option could not be linked to the object
   it concerned.
2. It **survived `strip_ids`**, whose card-reference matcher only catches
   `c<digits>` *strings*. So every option handed the model an unstripped,
   run-unstable id -- precisely the §16.8 failure, in the one array that check
   did not cover.

Options now carry `edges` in the same namespaced form as every other token
(`source_is` for a priority action's host card, `refers_to` for a choice's
subjects -- for an attack, both the creature and the defender). A pointer head
can attend to those objects' own representations instead of parsing them out of
the description string, which is left only for readability.

Measured: **99.6%** of option edges resolve to token indices (27,780 of 27,893);
the remainder refer to objects outside the viewer's information set and are
dropped, as in §16.5. Replay stays byte-identical (3,464 records).

`validate_dump.py` now has `check_option_ids`, verified non-vacuous by injecting
`source_id=26` into one option. **Its first version was vacuous**: `--limit`
defaults to `0` meaning "no limit", but `islice(it, 0)` yields *nothing*, so the
check silently scanned zero records and passed everything. A check that cannot
be shown to fail has not been tested.

### 16.10 Known gaps

- **[resolved, see §16.12]** 47-61% of real decisions once bypassed the policy.
  Combat and the pick-one callbacks are now routed; `chooseSingleReplacementEffect`
  and `orderSimultaneousSa` (permutation problems) remain with the AI.
- **[open]** 85-91% of *priority* decisions are still passes. This is partly
  genuine Magic -- a control deck holding up interaction should pass -- and
  partly that 72% of the offered pool is mana abilities the policy never takes
  alone (§16.11).
- **[not started]** Target handover. The policy currently picks an ability but
  not its targets; the engine chooses them. Note the AI binds targets by
  mutating the pool instance in place, so an action that carries targets must
  pass them explicitly rather than relying on the index alone.
- **[not started]** `ReplaySource` consumes indices in global order across both
  players. Any divergence desynchronises it; it should key on
  `(game, decision, viewer)`.
- **[not started]** `ActionSource` is a single static instance shared by both
  controllers. Fine for replay and random; a socket agent driving only one seat
  needs per-player routing.
- **[not started]** Vocabulary/integerisation. The dump is symbolic by design,
  with 46-55 distinct property keys observed; Python owns the mapping to ids.
