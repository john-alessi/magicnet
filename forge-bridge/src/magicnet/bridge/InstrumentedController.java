package magicnet.bridge;

import java.util.List;

import forge.ai.ComputerUtilAbility;
import forge.game.Game;
import forge.game.card.CardCollection;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

/**
 * Hand-written refinements on top of the generated pass-through controller.
 *
 * This is also the pattern for progressive takeover: GeneratedController gives you
 * all 120 callbacks delegating to Forge's heuristics, and you override here, one
 * decision type at a time, as the policy learns to handle each.
 *
 * Right now there is exactly one refinement, for the decision that matters most.
 */
public class InstrumentedController extends GeneratedController {

    /**
     * Candidate-pool enumeration duplicates work Forge's AI does internally, so it
     * costs real time. Disable with -Dmagicnet.enumerate=false to measure the
     * harness's throughput overhead against a clean baseline.
     */
    private static final boolean ENUMERATE =
            !"false".equalsIgnoreCase(System.getProperty("magicnet.enumerate", "true"));

    /**
     * Controls how target binding is routed, for the two-stage action space experiment.
     *
     * baseline - observe only; the AI binds targets during candidate evaluation.
     * clear    - strip targets off the chosen ability before returning it, to test
     *            whether anything downstream asks for them back.
     * engine   - strip targets AND route the cast through the engine's own
     *            PlaySpellAbility, which is the only code path that calls
     *            SpellAbility.setupTargets() -> PlayerController.chooseTargetsFor().
     *
     * Defaults to engine whenever an external source decides, because the AI
     * path leaves a failed cast stranded on the stack (representation_surface.md
     * 11.3) where the rules rewind it (CR 728). AI-only runs keep baseline so
     * seeded Forge behaviour is unchanged. Initialised after ACTIONS, which it reads.
     */
    private static String targetMode() {
        String explicit = System.getProperty("magicnet.targetmode");
        if (explicit != null) {
            return explicit.toLowerCase();
        }
        return ACTIONS.decides() ? "engine" : "baseline";
    }

    /**
     * Ability-pool audit. Compares what the bot API exposes against what the engine
     * considers possible, to find out whether forge.ai prunes legal plays before the
     * policy ever sees them. Sampled, because the wide pool is expensive.
     */
    private static final int DIFF_EVERY =
            Integer.getInteger("magicnet.abilitydiff", 0);

    /**
     * Chosen-action addressability audit. Off by default: it scans the pool on
     * every decision, and it answers a question (can the policy name what the AI
     * played?) that only needs re-measuring when enumeration changes.
     */
    private static final boolean ADDRESSABILITY =
            Boolean.getBoolean("magicnet.addressability");

    /** Where the acting player's choice comes from; Forge's AI unless overridden. */
    static final ActionSource ACTIONS = ActionSource.fromProperties();

    private static final String TARGET_MODE = targetMode();

    private static final java.util.concurrent.atomic.AtomicInteger DIFF_TICK =
            new java.util.concurrent.atomic.AtomicInteger();

    /**
     * Token-schema extraction. Builds the intermediate representation at every
     * priority decision and records its size, to find out whether the state fits
     * in a transformer context.
     */
    private static final boolean TOKENS =
            !"0".equals(System.getProperty("magicnet.tokens", "0"))
            && System.getProperty("magicnet.tokens") != null;

    /** Write this many sample states to out/states/ as JSONL for inspection. */
    private static final int DUMP_LIMIT =
            Integer.getInteger("magicnet.tokendump", 0);

    private static final int DUMP_EVERY =
            Integer.getInteger("magicnet.tokendumpevery", 200);

    private static final java.util.concurrent.atomic.AtomicInteger DUMP_TICK =
            new java.util.concurrent.atomic.AtomicInteger();

    private static final java.util.concurrent.atomic.AtomicInteger DUMPED =
            new java.util.concurrent.atomic.AtomicInteger();

    /**
     * Game history, one per viewing player. Not shared: once history tokens carry
     * card identity, what is public depends on who is looking, so a single shared
     * log would leak the opponent's draws. Keyed by player id.
     */
    private static final java.util.Map<Integer, GameHistory> HISTORIES =
            new java.util.HashMap<>();

    private static final Object HISTORY_LOCK = new Object();

    /**
     * Attaches a history listener for this viewer, once per player per game.
     * Called lazily because the controller is constructed before the harness can
     * reach the Game. Each viewer's listener subscribes independently, so both
     * see every event but redact it differently.
     */
    private static java.util.List<StateExtractor.Tok> history(Game game, forge.game.player.Player viewer) {
        synchronized (HISTORY_LOCK) {
            Integer key = viewer == null ? Integer.valueOf(-1) : Integer.valueOf(viewer.getId());
            // Schema 13: one hub per game builds the events; each viewer's
            // history redacts them. The hub attaches the knowledge stores first.
            HistoryHub hub = HistoryHub.attach(game);
            GameHistory h = hub.viewer(viewer);
            HISTORIES.put(key, h);
            return h.tokens();
        }
    }

    /** Zones the engine could plausibly let you play from. */
    private static final forge.game.zone.ZoneType[] WIDE_ZONES = {            forge.game.zone.ZoneType.Hand,
            forge.game.zone.ZoneType.Library,
            forge.game.zone.ZoneType.Graveyard,
            forge.game.zone.ZoneType.Exile,
            forge.game.zone.ZoneType.Battlefield,
            forge.game.zone.ZoneType.Command,
            forge.game.zone.ZoneType.Sideboard,
            forge.game.zone.ZoneType.Stack,
    };

    public InstrumentedController(Game game, Player p, forge.LobbyPlayer lp) {
        super(game, p, lp);
        // Register this viewer's history before the game starts, so it sees the
        // pregame events (opening hands, mulligans) too, not just everything
        // after this player's first decision.
        try {
            if (game != null && p != null) {
                history(game, p);
            }
        } catch (Throwable ignored) {
            // instrumentation must never fail a game
        }
    }

    /**
     * The primary priority decision, and the only callback with no arguments --
     * Forge hands the controller nothing and expects it to work out its own options.
     *
     * That asymmetry matters: for every other decision the engine supplies a
     * pre-filtered legal set, so action masking is free. Here it does not, which
     * means the candidate enumeration is the controller's responsibility and
     * ComputerUtilAbility is the AI's private helper rather than an engine service.
     *
     * We reconstruct the candidate pool read-only to measure the branching factor.
     * We deliberately do NOT call setActivatingPlayer()/canPlay() to refine the
     * count, because those mutate SpellAbility state and would perturb the very
     * games we are measuring. The number recorded is therefore an upper bound on
     * the true legal action count.
     */
    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        long t0 = System.nanoTime();
        int candidates = -1;
        List<SpellAbility> pool = null;
        if (ENUMERATE) {
            try {
                Game game = getGame();
                Player me = getPlayer();
                CardCollection sources = ComputerUtilAbility.getAvailableCards(game, me);
                List<SpellAbility> all = ComputerUtilAbility.getSpellAbilities(sources, me);
                pool = all;
                candidates = all == null ? 0 : all.size();
                DecisionLog.recordPoolOrder(all);
                if (all != null) {
                    for (SpellAbility sa : all) {
                        try {
                            if (sa != null && sa.isManaAbility()) {
                                MaskCost.noteManaAbilityOffered();
                            }
                        } catch (Throwable ignored) {
                            // best-effort instrumentation
                        }
                    }
                }
                DecisionLog.recordSources(sources == null ? 0 : sources.size());
                if (DIFF_EVERY > 0 && all != null
                        && DIFF_TICK.incrementAndGet() % DIFF_EVERY == 0) {
                    auditAbilityPool(game, me, all);
                }
            } catch (Throwable ignored) {
                // Enumeration is best-effort instrumentation; never fail a game over it.
            }
        }

        if (AFFORD_EVERY > 0 && pool != null
                && AFFORD_TICK.incrementAndGet() % AFFORD_EVERY == 0) {
            try {
                auditAffordability(pool);
            } catch (Throwable ignored) {
                // diagnostic only
            }
        }

        if (pool != null && ActionCardinality.shouldSample()) {
            try {
                ActionCardinality.measure(pool, getGame(), getPlayer());
            } catch (Throwable t) {
                System.out.println("[cardinality] failed: " + t);
            }
        }

        if (pool != null && MaskCost.EVERY > 0 && MaskCost.due()) {
            try {
                Player me = getPlayer();
                for (SpellAbility sa : pool) {
                    if (sa != null) {
                        MaskCost.measure(sa, me);
                    }
                }
            } catch (Throwable t) {
                System.out.println("[maskcost] failed: " + t);
            }
        }

        if (SIM_EVERY > 0 && SIM_TICK.incrementAndGet() % SIM_EVERY == 0) {
            try {
                probeSimulationCost();
            } catch (Throwable t) {
                System.out.println("[simprobe] failed: " + t);
            }
        }

        List<StateExtractor.Tok> hist = null;
        List<StateExtractor.Tok> toks = null;
        // An external action source needs the observation just as much as the
        // dump does. Omitting it here once meant a socket agent silently
        // received the literal string "null" whenever no dump path was set.
        if (TOKENS || Serializer.enabled() || ACTIONS.needsRequest()) {
            hist = history(getGame(), getPlayer());
            toks = extractTokens(pool, hist);
        }
        final int histSize = hist == null ? 0 : hist.size();

        // Built once, before the decision, so the agent and the dump see exactly
        // the same rendering of the state.
        String body = null;
        if (toks != null && (Serializer.enabled() || ACTIONS.needsRequest())) {
            body = Serializer.buildBody(getGame(), getPlayer(), toks, histSize, pool);
        }
        if (ACTIONS.needsRequest() && body == null) {
            throw new IllegalStateException(
                    "action source needs a request but no observation was built");
        }

        List<SpellAbility> chosen = null;
        try {
            if (FORCE_BAD > 0) {
                if (LAST_WAS_FORCED.getAndSet(false)) {
                    forge.game.card.Card hc = lastForced == null ? null : lastForced.getHostCard();
                    System.out.println("[forcebad] after-failure: stackFrozen="
                            + getGame().getStack().isFrozen()
                            + " stackInstances=" + getGame().getStack().size()
                            + " stackZoneCards="
                            + getGame().getStackZone().getCards().size()
                            + " hostCard=" + hc
                            + " hostZone=" + (hc == null ? "?" : String.valueOf(hc.getZone()))
                            + " poolSize=" + (pool == null ? -1 : pool.size()));
                }
                SpellAbility bad = forcedUnaffordable(pool);
                if (bad != null) {
                    int n = FORCED_COUNT.incrementAndGet();
                    LAST_WAS_FORCED.set(true);
                    lastForced = bad;
                    System.out.println("[forcebad] #" + n + " forcing unaffordable: "
                            + bad.getHostCard() + " / " + bad);
                    chosen = java.util.Collections.singletonList(bad);
                    return chosen;
                }
            }
            // The explicit pass option is the last index, after the pool.
            int passIndex = pool == null ? 0 : pool.size();
            int external = ACTIONS.choose(
                    body == null ? null : Serializer.asRequest(body),
                    passIndex + 1, passIndex);
            if (external != ActionSource.DELEGATE) {
                if (external < 0 || external > passIndex) {
                    throw new IllegalStateException("action source returned " + external
                            + " for " + (passIndex + 1) + " options");
                }
                // Forge signals "pass" with null, not an empty list: AiController
                // returns aconst_null. An empty list is read as a choice having
                // been made, which re-offers priority to the same player forever.
                chosen = external == passIndex
                        ? null
                        : java.util.Collections.singletonList(pool.get(external));
                return chosen;
            }
            chosen = super.chooseSpellAbilityToPlay();
            return chosen;
        } finally {
            auditFilterSoundness(chosen);
            auditChosenInPool(pool, chosen);
            Serializer.write(body, pool, chosen);
            probeTargeting(chosen);
            if (!"baseline".equals(TARGET_MODE)) {
                stripTargets(chosen);
            }
            boolean passed = (chosen == null || chosen.isEmpty());
            DecisionLog.recordPriority(t0, getPlayer(), candidates, passed);
        }
    }

    /**
     * Builds the intermediate token representation and records its size.
     *
     * History is per viewer, not shared: once tokens carry card identity, what is
     * public depends on who is looking.
     */
    /**
     * Affordability audit.
     *
     * <p>The candidate pool is {@code ComputerUtilAbility.getSpellAbilities}, which
     * applies no filtering at all -- not timing, not cost. This measures how much
     * of it is actually playable, to settle whether the policy needs to learn
     * affordability or whether the environment should mask it.
     */
    private static final int AFFORD_EVERY =
            Integer.getInteger("magicnet.affordability", 0);

    private static final java.util.concurrent.atomic.AtomicInteger AFFORD_TICK =
            new java.util.concurrent.atomic.AtomicInteger();

    /**
     * Forced-failure probe: deliberately return an ability the player cannot pay for,
     * to observe what Forge does when a policy picks an unaffordable action.
     *
     * <p>This simulates the exact RL failure mode: the policy announces a spell and
     * then cannot complete payment. We want to know whether the engine recovers
     * cleanly, whether it re-offers the same action (livelock), and whether any
     * partially-paid costs are refunded.
     *
     * <p>Value is the maximum number of forced choices, so a run can be bounded.
     */
    private static final int FORCE_BAD =
            Integer.getInteger("magicnet.forcebad", 0);

    private static final java.util.concurrent.atomic.AtomicInteger FORCED_COUNT =
            new java.util.concurrent.atomic.AtomicInteger();

    private static final java.util.concurrent.atomic.AtomicBoolean LAST_WAS_FORCED =
            new java.util.concurrent.atomic.AtomicBoolean();

    private static SpellAbility lastForced;

    /** First candidate that is legal to announce but whose mana cost cannot be paid. */
    private SpellAbility forcedUnaffordable(List<SpellAbility> pool) {
        if (pool == null || FORCED_COUNT.get() >= FORCE_BAD) {
            return null;
        }
        Player me = getPlayer();
        for (SpellAbility sa : pool) {
            try {
                sa.setActivatingPlayer(me);
                if (!sa.canPlay()) {
                    continue;
                }
                if (forge.ai.ComputerUtilMana.canPayManaCost(sa, me, 0, false)) {
                    continue;
                }
                return sa;
            } catch (Throwable ignored) {
                // probe only
            }
        }
        return null;
    }

    public static int forcedCount() {
        return FORCED_COUNT.get();
    }

    /**
     * Simulation probe: measures what a copy-based feasibility check would cost.
     *
     * <p>Plan-then-execute action decoding needs to validate a complete announcement
     * (modes, X, targets, payment) without mutating the live game, because a failed
     * announcement strands the card in the Stack zone permanently (see
     * docs/representation_surface.md 11.3). Forge's own lookahead AI already does
     * this via GameCopier + GameSimulator, so the question is not feasibility but
     * price: a full game copy per validation.
     */
    private static final int SIM_EVERY =
            Integer.getInteger("magicnet.simprobe", 0);

    private static final java.util.concurrent.atomic.AtomicInteger SIM_TICK =
            new java.util.concurrent.atomic.AtomicInteger();

    private static final java.util.List<Long> SIM_NANOS =
            java.util.Collections.synchronizedList(new java.util.ArrayList<>());

    private static volatile String simPruneFlag = "unread";

    private void probeSimulationCost() {
        Game game = getGame();
        if ("unread".equals(simPruneFlag)) {
            try {
                java.lang.reflect.Field f = forge.ai.simulation.GameCopier.class
                        .getDeclaredField("PRUNE_HIDDEN_INFO");
                f.setAccessible(true);
                simPruneFlag = String.valueOf(f.getBoolean(null));
            } catch (Throwable t) {
                simPruneFlag = "error: " + t;
            }
        }
        long t0 = System.nanoTime();
        forge.game.Game copy = new forge.ai.simulation.GameCopier(game).makeCopy();
        long dt = System.nanoTime() - t0;
        if (copy != null) {
            SIM_NANOS.add(dt);
        }
    }

    public static void printSimulationProbe() {
        java.util.List<Long> s = new java.util.ArrayList<>(SIM_NANOS);
        if (s.isEmpty()) {
            return;
        }
        java.util.Collections.sort(s);
        long sum = 0;
        for (long v : s) {
            sum += v;
        }
        System.out.printf("%n=== GameCopier.makeCopy() cost (n=%d) ===%n", s.size());
        System.out.printf("  PRUNE_HIDDEN_INFO: %s%n", simPruneFlag);
        System.out.printf("  mean %.2f ms   p50 %.2f ms   p90 %.2f ms   max %.2f ms%n",
                sum / 1e6 / s.size(),
                s.get(s.size() / 2) / 1e6,
                s.get(Math.min(s.size() - 1, (int) (s.size() * 0.9))) / 1e6,
                s.get(s.size() - 1) / 1e6);
    }

    private void auditAffordability(List<SpellAbility> pool) {
        if (pool == null || pool.isEmpty()) {
            return;
        }
        int total = 0;
        int timingOk = 0;
        int payable = 0;
        for (SpellAbility sa : pool) {
            if (sa == null) {
                continue;
            }
            total++;
            boolean t;
            try {
                t = sa.canPlay();
            } catch (Throwable ignored) {
                continue;
            }
            if (!t) {
                continue;
            }
            timingOk++;
            try {
                if (forge.ai.ComputerUtilMana.canPayManaCost(sa, getPlayer(), 0, false)) {
                    payable++;
                }
            } catch (Throwable ignored) {
                // count as unpayable
            }
        }
        DecisionLog.recordAffordability(total, timingOk, payable);
    }

    /**
     * Soundness check on the affordability predicate.
     *
     * <p>Filtering unaffordable actions is only safe if the predicate is an
     * over-approximation: it may admit something unplayable, but it must never
     * reject something legal, or the policy silently loses access to that line
     * forever. ComputerUtilMana.canPayManaCost is forge.ai heuristic code with no
     * such guarantee, so this counts the cases that would prove it unsound --
     * abilities Forge's own AI selected but our predicate rejects.
     */
    private void auditFilterSoundness(List<SpellAbility> chosen) {
        if (AFFORD_EVERY <= 0 || chosen == null || chosen.isEmpty()) {
            return;
        }
        for (SpellAbility sa : chosen) {
            if (sa == null) {
                continue;
            }
            boolean timing;
            boolean payable;
            try {
                timing = sa.canPlay();
            } catch (Throwable ignored) {
                timing = false;
            }
            try {
                payable = forge.ai.ComputerUtilMana.canPayManaCost(sa, getPlayer(), 0, false);
            } catch (Throwable ignored) {
                payable = false;
            }
            DecisionLog.recordChosenLegality(timing, payable,
                    describeSa(sa));
        }
    }

    /**
     * Can the AI's own choice be named as an index into the pool we expose?
     *
     * The action space hands the policy an index into the enumerated pool, and
     * replay-equivalence testing records that index and feeds it back. Both are
     * unsound if the ability the AI actually plays is not in the pool -- there
     * would be no index to record. The AI selects through its own internal path,
     * so pool membership is an assumption, not a guarantee.
     *
     * Checked by object identity first, then by content signature, because an
     * equivalent-but-distinct SpellAbility instance is a recoverable problem
     * (match by signature) while a genuinely absent ability is not.
     */
    private void auditChosenInPool(List<SpellAbility> pool, List<SpellAbility> chosen) {
        if (!ADDRESSABILITY || pool == null || chosen == null || chosen.isEmpty()) {
            return;
        }
        for (SpellAbility sa : chosen) {
            if (sa == null) {
                continue;
            }
            int byIdentity = -1;
            for (int i = 0; i < pool.size(); i++) {
                if (pool.get(i) == sa) {
                    byIdentity = i;
                    break;
                }
            }
            int bySignature = -1;
            if (byIdentity < 0) {
                String want = describeSa(sa);
                for (int i = 0; i < pool.size(); i++) {
                    if (pool.get(i) != null && want.equals(describeSa(pool.get(i)))) {
                        bySignature = i;
                        break;
                    }
                }
            }
            DecisionLog.recordChosenInPool(byIdentity, bySignature, describeSa(sa));
        }
    }

    private static String describeSa(SpellAbility sa) {
        try {
            String host = sa.getHostCard() == null ? "?" : sa.getHostCard().getName();
            return host + " :: " + sa.getApi() + " :: " + sa.toString();
        } catch (Throwable ignored) {
            return "?";
        }
    }

    private List<StateExtractor.Tok> extractTokens(List<SpellAbility> pool,
            List<StateExtractor.Tok> hist) {
        try {
            List<StateExtractor.Tok> toks = StateExtractor.extract(
                    getGame(), getPlayer(), hist);
            StateStats.record(toks);
            if (DUMP_LIMIT > 0 && DUMPED.get() < DUMP_LIMIT
                    && DUMP_TICK.incrementAndGet() % DUMP_EVERY == 0) {
                dumpState(toks);
            }
            return toks;
        } catch (Throwable ignored) {
            // never fail a game over instrumentation
            return null;
        }
    }

    private void dumpState(List<StateExtractor.Tok> toks) {
        synchronized (DUMPED) {
            if (DUMPED.get() >= DUMP_LIMIT) {
                return;
            }
            int n = DUMPED.incrementAndGet();
            try {
                java.io.File dir = new java.io.File(
                        System.getProperty("magicnet.outdir", "out"), "states");
                dir.mkdirs();
                java.io.File f = new java.io.File(dir, String.format("state_%03d.jsonl", n));
                try (java.io.PrintWriter w =
                             new java.io.PrintWriter(f, "UTF-8")) {
                    for (StateExtractor.Tok t : toks) {
                        w.println(t.toJson());
                    }
                }
            } catch (Throwable ignored) {
                // skip
            }
        }
    }

    /** Detaches the per-game history listeners. Called by the harness between games. */
    public static void resetHistory() {
        synchronized (HISTORY_LOCK) {
            int maxSize = 0;
            int redactions = 0;
            for (GameHistory h : HISTORIES.values()) {
                maxSize = Math.max(maxSize, h.size());
                redactions += h.redactions();
            }
            if (!HISTORIES.isEmpty()) {
                StateStats.recordHistoryAtGameEnd(maxSize);
                StateStats.recordRedactions(redactions);
            }
            HISTORIES.clear();
            HistoryHub.reset();
            GameHistory.resetAttempts();
            nextRevealGroup = 0;
            lastRevealKey = null;
            lastRevealHistSize = -1;
        }
    }

    @Override
    public boolean payManaCost(forge.card.mana.ManaCost a0, forge.game.cost.CostPartMana a1,
            SpellAbility a2, String a3, forge.game.mana.ManaConversionMatrix a4, boolean a5) {
        GameHistory.Attempt attempt = GameHistory.currentAttempt();
        if (attempt != null) {
            attempt.reach("cost", a2);
        }
        return super.payManaCost(a0, a1, a2, a3, a4, a5);
    }

    /**
     * Compares three enumerations of "what could I do right now":
     *
     *   narrow - ComputerUtilAbility.getAvailableCards + getSpellAbilities, which is
     *            what the bot API exposes and what our branching factor measures.
     *   costs  - the same list run through getOriginalAndAltCostAbilities, which
     *            expands alternative/additional costs and collapses optional costs.
     *   wide   - Card.getAllPossibleAbilities over every zone, the engine's own view.
     *
     * wide-minus-narrow isolates zone coverage; costs-minus-narrow isolates cost
     * variants. Anything in wide that is legal right now (canPlay) but absent from
     * narrow is a legal play the policy could never reach through this API.
     */
    private void auditAbilityPool(Game game, Player me, List<SpellAbility> narrow) {
        try {
            java.util.Set<String> narrowKeys = new java.util.HashSet<>();
            for (SpellAbility sa : narrow) {
                narrowKeys.add(key(sa));
            }

            int costOnly = 0;
            try {
                List<SpellAbility> expanded =
                        ComputerUtilAbility.getOriginalAndAltCostAbilities(narrow, me);
                for (SpellAbility sa : expanded) {
                    if (!narrowKeys.contains(key(sa))) {
                        costOnly++;
                        DecisionLog.recordPoolGap("cost-variant", zoneOf(sa), sa, false);
                    }
                }
            } catch (Throwable ignored) {
                // Cost expansion invokes controller callbacks; never fail a game over it.
            }

            int wideTotal = 0;
            int missing = 0;
            int missingLegal = 0;
            CardCollection wideCards = new CardCollection();
            for (forge.game.zone.ZoneType z : WIDE_ZONES) {
                try {
                    if (z == forge.game.zone.ZoneType.Library
                            || z == forge.game.zone.ZoneType.Hand
                            || z == forge.game.zone.ZoneType.Sideboard) {
                        wideCards.addAll(me.getCardsIn(z));
                    } else {
                        wideCards.addAll(game.getCardsIn(z));
                    }
                } catch (Throwable ignored) {
                    // Some zones are not meaningful in every game state.
                }
            }
            for (forge.game.card.Card c : wideCards) {
                List<SpellAbility> poss;
                try {
                    poss = c.getAllPossibleAbilities(me, false);
                } catch (Throwable t) {
                    continue;
                }
                for (SpellAbility sa : poss) {
                    wideTotal++;
                    if (narrowKeys.contains(key(sa))) {
                        continue;
                    }
                    missing++;
                    boolean legalNow = false;
                    try {
                        legalNow = sa.canPlay();
                    } catch (Throwable ignored) {
                        // canPlay can throw on exotic states.
                    }
                    if (legalNow) {
                        missingLegal++;
                    }
                    DecisionLog.recordPoolGap("zone-coverage", zoneOf(sa), sa, legalNow);
                }
            }
            DecisionLog.recordPoolAudit(narrow.size(), costOnly, wideTotal, missing, missingLegal);
        } catch (Throwable ignored) {
            // Audit is best-effort instrumentation.
        }
    }

    private static String key(SpellAbility sa) {
        if (sa == null) {
            return "null";
        }
        String host;
        try {
            host = sa.getHostCard() == null ? "?" : sa.getHostCard().getName();
        } catch (Throwable t) {
            host = "?";
        }
        String desc;
        try {
            desc = String.valueOf(sa.getDescription());
        } catch (Throwable t) {
            desc = "?";
        }
        String cost;
        try {
            cost = sa.getPayCosts() == null ? "-" : sa.getPayCosts().toString();
        } catch (Throwable t) {
            cost = "?";
        }
        return host + "|" + sa.getApi() + "|" + cost + "|" + desc;
    }

    private static String zoneOf(SpellAbility sa) {
        try {
            forge.game.card.Card c = sa.getHostCard();
            if (c == null || c.getZone() == null) {
                return "none";
            }
            return String.valueOf(c.getZone().getZoneType());
        } catch (Throwable t) {
            return "?";
        }
    }

    /**
     * Removes the targets the AI bound during candidate evaluation, so that the
     * action handed back to the engine is "cast this spell" rather than
     * "cast this spell at that creature".
     */
    private void stripTargets(List<SpellAbility> chosen) {
        if (chosen == null) {
            return;
        }
        for (SpellAbility sa : chosen) {
            for (SpellAbility walk = sa; walk != null; walk = walk.getSubAbility()) {
                try {
                    if (walk.usesTargeting() && !walk.getTargets().isEmpty()) {
                        DecisionLog.recordTargetsStripped(walk.getTargets().size());
                        walk.resetTargets();
                    }
                } catch (Throwable ignored) {
                    // Experiment only; never fail a game over it.
                }
            }
        }
    }

    /**
     * In "engine" mode, bypass ComputerUtil.handlePlayingSpellAbility (the forge.ai
     * helper the bot normally uses) and call the engine's PlaySpellAbility instead.
     *
     * That is the routine PlayerControllerHuman uses, and it is the only caller of
     * setupTargets(), so it should come back to us through chooseTargetsFor() with
     * the engine supplying the legal target set.
     */
    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        HistoryHub hub = HistoryHub.of(getGame());
        if (hub != null && sa != null && GameHistory.currentAttempt() == null) {
            hub.sync();
        }
        GameHistory.Attempt attempt = sa == null ? null : GameHistory.beginAttempt(sa);
        if (hub != null && attempt != null) {
            hub.attemptBegun(attempt);
        }
        String outcome = "unknown";
        try {
            if (!"engine".equals(TARGET_MODE) || sa == null) {
                boolean skipBefore = sa != null && sa.isSkip();
                boolean r = super.playChosenSpellAbility(sa);
                outcome = aiOutcome(sa, skipBefore);
                return r;
            }
            try {
                boolean ok = forge.game.player.PlaySpellAbility.playSpellAbility(this, getPlayer(), sa);
                String name = "?";
                try {
                    name = sa.getHostCard() == null ? "?" : sa.getHostCard().getName();
                } catch (Throwable ignored) {
                    // Diagnostic only.
                }
                DecisionLog.recordEngineCastResult(ok, name);
                outcome = ok ? "committed" : "rolled_back";
                return ok;
            } catch (Throwable t) {
                DecisionLog.recordEngineCastFailure(t);
                boolean skipBefore = sa.isSkip();
                boolean r = super.playChosenSpellAbility(sa);
                outcome = aiOutcome(sa, skipBefore);
                return r;
            }
        } finally {
            if (attempt != null) {
                attempt.captureTargets();
                finishAttempt(attempt, outcome);
            }
        }
    }

    /**
     * Forge's AI cast path always returns true; its helper's only trace of a
     * refused payment is setSkip(true), and the refusal is not rewound -- hence
     * failed_unrolled. Indeterminate if the flag was already set.
     */
    private static String aiOutcome(SpellAbility sa, boolean skipBefore) {
        if (sa == null || skipBefore) {
            return "unknown";
        }
        return sa.isSkip() ? "failed_unrolled" : "committed";
    }

    private void finishAttempt(GameHistory.Attempt a, String outcome) {
        try {
            HistoryHub hub = HistoryHub.of(getGame());
            if (hub != null) {
                hub.attemptEnd(a, outcome);
            }
        } finally {
            GameHistory.closeAttempt(a);
        }
    }

    /**
     * PlayerControllerAi.announceRequirements is a switch over ApiType with only two
     * cases; every other API returns null, which PlaySpellAbility.playAbility reads as
     * "the player cancelled" and refuses the cast. That makes modal and most X spells
     * uncastable on the engine path, which livelocks the phase handler: the chooser
     * re-offers the same ability, the engine refuses it again, forever.
     *
     * This is an AI-helper gap, not a rules one -- announceRequirements is a decision
     * the policy will make. Until the policy is wired in, answer with the legal minimum
     * so the cast can proceed; only on the engine path, so baseline play is unchanged.
     */
    @Override
    public Integer announceRequirements(SpellAbility sa, int min, int max, String announce) {
        GameHistory.Attempt attempt = GameHistory.currentAttempt();
        if (attempt != null) {
            attempt.reach("announce", sa);
        }
        Integer v = super.announceRequirements(sa, min, max, announce);
        if (!"engine".equals(TARGET_MODE)) {
            if (attempt != null && attempt.owns(sa)) {
                attempt.announce(announce, v);
            }
            return v;
        }
        if (v != null) {
            DecisionLog.recordAnnounce(sa, announce, false);
        } else {
            DecisionLog.recordAnnounce(sa, announce, true);
            v = Integer.valueOf(Math.max(0, min));
        }
        if (attempt != null && attempt.owns(sa)) {
            attempt.announce(announce, v);
        }
        return v;
    }

    /**
     * The callback under test. Records how many legal targets the engine offers,
     * which is the stage-two action mask we would get for free.
     */
    @Override
    public boolean chooseTargetsFor(SpellAbility sa) {
        GameHistory.Attempt attempt = GameHistory.currentAttempt();
        if (attempt != null) {
            attempt.reach("targets", sa);
        }
        int candidates = -1;
        try {
            if (sa != null && sa.usesTargeting()) {
                candidates = sa.getTargetRestrictions().getAllCandidates(sa, false).size();
            }
        } catch (Throwable ignored) {
            // Diagnostic only.
        }
        Boolean routed = null;
        if (ROUTE_TARGETS.get() && ACTIONS.decides() && routeChoices()) {
            routed = chooseTargetsRouted(sa);
        }
        boolean ok = routed != null ? routed.booleanValue() : super.chooseTargetsFor(sa);
        if (attempt != null) {
            attempt.captureTargets();
        }
        int bound = 0;
        try {
            if (sa != null) {
                for (forge.game.spellability.TargetChoices tc : sa.getAllTargetChoices()) {
                    if (tc != null) {
                        bound += tc.size();
                    }
                }
            }
        } catch (Throwable ignored) {
            // Diagnostic only.
        }
        DecisionLog.recordTargetCallback(candidates, bound, ok);
        return ok;
    }

    /**
     * Diagnostic for where target binding actually happens on the AI path.
     *
     * SpellAbility.setupTargets() calls controller.chooseTargetsFor() for every
     * ability where usesTargeting() is true, so that callback should fire
     * constantly -- yet it never fired across 72 games. Either the AI binds
     * targets earlier (during candidate evaluation, so the returned ability
     * already carries them) or the AI play path skips setupTargets entirely.
     *
     * This distinguishes the two: if abilities come back already targeted, then
     * target selection is folded into the action itself and the measured
     * branching factor is NOT the full action space.
     */
    private void probeTargeting(List<SpellAbility> chosen) {
        if (chosen == null) {
            return;
        }
        for (SpellAbility sa : chosen) {
            if (sa == null) {
                continue;
            }
            try {
                if (!sa.usesTargeting()) {
                    continue;
                }
                int bound = 0;
                for (forge.game.spellability.TargetChoices tc : sa.getAllTargetChoices()) {
                    if (tc != null) {
                        bound += tc.size();
                    }
                }
                DecisionLog.recordTargeting(bound);
            } catch (Throwable ignored) {
                // Diagnostic only.
            }
        }
    }

    // ---- Library-position knowledge capture -------------------------------
    //
    // These three callbacks are the only place in Forge where "the player now
    // knows what is on top of their library" exists. GameAction.scry calls
    // arrangeForScry and then immediately moves the cards; the knowledge is
    // never persisted, and GameEventScry carries only counts. See
    // LibraryKnowledge for the full rationale.
    //
    // We delegate to super for the actual decision and only observe the result,
    // so this cannot change play.

    @Override
    public org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection>
            arrangeForScry(CardCollection cards) {
        census("arrangeForScry");
        if (routeChoices() && cards != null && !cards.isEmpty()) {
            return arrangeRouted(cards, true);
        }
        org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection> r =
                super.arrangeForScry(cards);
        try {
            // Scry lets the player order both piles, so every card seen ends up
            // at a position they know exactly.
            if (r != null) {
                LibraryKnowledge.learnExact(getPlayer(), r.getLeft());
                LibraryKnowledge.learnExact(getPlayer(), r.getRight());
            }
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
        return r;
    }

    @Override
    public org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection>
            arrangeForSurveil(CardCollection cards) {
        census("arrangeForSurveil");
        if (routeChoices() && cards != null && !cards.isEmpty()) {
            return arrangeRouted(cards, false);
        }
        org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection> r =
                super.arrangeForSurveil(cards);
        try {
            // Left pile goes back on top in a chosen order; the right pile goes
            // to the graveyard, which is a public zone and needs no tracking.
            if (r != null) {
                LibraryKnowledge.learnExact(getPlayer(), r.getLeft());
            }
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
        return r;
    }

    /**
     * Scry/surveil as a sequence of pick-one decisions, like combat: one
     * {@code scry}/{@code surveil} decision per card (top vs bottom/graveyard),
     * then, for each ordered pile holding more than one card, repeated
     * {@code order_top}/{@code order_bottom} decisions picking the card nearest
     * that end next (the last card is forced and not asked).
     *
     * <p>The looked-at cards are learned as exact-position knowledge before the
     * first decision -- the player has seen the top N, in place -- so the
     * decider's observation shows them as {@code library_card} tokens the
     * options' edges resolve to. The opponent learns nothing: knowledge is per
     * viewer. Earlier picks are shown via {@code arranged_pile}/{@code
     * arranged_rank} on those tokens, since nothing moves until the whole
     * arrangement is returned.
     *
     * <p>Observe mode lets Forge's AI decide first and then records the same
     * sequence with labels read off its result. That is exact rather than an
     * approximation: the AI's callback fires no events and moves no cards, so
     * every record is built in the state the routed path would have shown.
     */
    private org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection>
            arrangeRouted(CardCollection cards, boolean scry) {
        Player me = getPlayer();
        String kind = scry ? "scry" : "surveil";
        String otherPile = scry ? "bottom" : "graveyard";
        boolean decide = ACTIONS.decides();
        org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection> ai = null;
        if (!decide) {
            ai = scry ? super.arrangeForScry(cards) : super.arrangeForSurveil(cards);
        }
        List<forge.game.card.Card> seen = new java.util.ArrayList<>();
        for (forge.game.card.Card c : cards) {
            seen.add(c);
        }
        LibraryKnowledge.learnExact(me, seen);
        try {
            List<forge.game.card.Card> top = new java.util.ArrayList<>();
            List<forge.game.card.Card> other = new java.util.ArrayList<>();
            boolean first = true;
            for (forge.game.card.Card c : seen) {
                String cref = StateExtractor.cardRef(c.getId());
                String oid = OracleIds.of(c);
                List<Serializer.Opt> opts = new java.util.ArrayList<>();
                opts.add(new Serializer.Opt("put " + c.getName() + " on top").ref(cref).oracle(oid));
                opts.add(new Serializer.Opt(scry ? "put " + c.getName() + " on the bottom"
                        : "put " + c.getName() + " into graveyard").ref(cref).oracle(oid));
                int pick;
                if (decide) {
                    pick = askChoice(kind, opts);
                    if (pick == ActionSource.DELEGATE) {
                        if (first) {
                            return delegateArrange(cards, scry);
                        }
                        pick = 0;
                    }
                } else {
                    boolean toTop = ai != null && ai.getLeft() != null && ai.getLeft().contains(c);
                    boolean toOther = ai != null && ai.getRight() != null && ai.getRight().contains(c);
                    pick = toTop ? 0 : toOther ? 1 : -1;
                    closeChoice(openChoice(kind, opts), pick);
                }
                first = false;
                if (pick == 1) {
                    other.add(c);
                    LibraryKnowledge.arranging(me, c, otherPile, -1);
                } else {
                    top.add(c);
                    LibraryKnowledge.arranging(me, c, "top", -1);
                }
            }

            // Forge's convention: left is topmost-first; for scry the right pile
            // is moved to the bottom in list order, so its last card ends
            // bottommost. Orders are asked from each end inwards.
            List<forge.game.card.Card> aiTop = ai == null ? null : ai.getLeft();
            List<forge.game.card.Card> aiBottomUp = null;
            if (ai != null && scry && ai.getRight() != null) {
                aiBottomUp = new java.util.ArrayList<>(ai.getRight());
                java.util.Collections.reverse(aiBottomUp);
            }
            List<forge.game.card.Card> topOrder = orderPile("order_top", "top", top, aiTop);
            List<forge.game.card.Card> otherOrder = other;
            if (scry) {
                otherOrder = orderPile("order_bottom", "bottom", other, aiBottomUp);
                java.util.Collections.reverse(otherOrder);
            }

            if (!decide) {
                if (!scry && ai != null) {
                    LibraryKnowledge.forget(me, ai.getRight());
                }
                return ai;
            }
            if (!scry) {
                LibraryKnowledge.forget(me, otherOrder);
            }
            return org.apache.commons.lang3.tuple.ImmutablePair.of(
                    new CardCollection(topOrder), new CardCollection(otherOrder));
        } finally {
            LibraryKnowledge.clearArranging(me);
        }
    }

    /**
     * Repeated pick-one ordering of a pile, nearest its end first. Returns the
     * order in that sense (index 0 = top for "top", = bottommost for "bottom").
     *
     * @param expert Forge's AI order in the same sense, or null when deciding
     */
    private List<forge.game.card.Card> orderPile(String kind, String pile,
            List<forge.game.card.Card> cards, List<forge.game.card.Card> expert) {
        Player me = getPlayer();
        List<forge.game.card.Card> remaining = new java.util.ArrayList<>(cards);
        List<forge.game.card.Card> out = new java.util.ArrayList<>();
        while (remaining.size() > 1) {
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            for (forge.game.card.Card c : remaining) {
                opts.add(new Serializer.Opt("place " + c.getName() + " next from the " + pile)
                        .ref(StateExtractor.cardRef(c.getId())).oracle(OracleIds.of(c)));
            }
            int pick;
            if (expert == null) {
                pick = askChoice(kind, opts);
                if (pick < 0 || pick >= remaining.size()) {
                    pick = 0;
                }
            } else {
                forge.game.card.Card want = out.size() < expert.size() ? expert.get(out.size()) : null;
                int label = want == null ? -1 : remaining.indexOf(want);
                closeChoice(openChoice(kind, opts), label);
                pick = label < 0 ? 0 : label;
            }
            forge.game.card.Card c = remaining.remove(pick);
            LibraryKnowledge.arranging(me, c, pile, out.size());
            out.add(c);
        }
        for (forge.game.card.Card c : remaining) {
            LibraryKnowledge.arranging(me, c, pile, out.size());
            out.add(c);
        }
        return out;
    }

    /** The unrouted path, used if the source declines the very first decision. */
    private org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection>
            delegateArrange(CardCollection cards, boolean scry) {
        org.apache.commons.lang3.tuple.ImmutablePair<CardCollection, CardCollection> r =
                scry ? super.arrangeForScry(cards) : super.arrangeForSurveil(cards);
        if (r != null && !scry) {
            LibraryKnowledge.forget(getPlayer(), r.getRight());
        }
        return r;
    }

    @Override
    public boolean willPutCardOnTop(forge.game.card.Card c) {
        boolean r = super.willPutCardOnTop(c);
        try {
            LibraryKnowledge.learn(getPlayer(), c, LibraryKnowledge.Constraint.EXACT);
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
        return r;
    }

    @Override
    public forge.game.card.CardCollectionView orderMoveToZoneList(
            forge.game.card.CardCollectionView cards,
            forge.game.zone.ZoneType dest,
            SpellAbility sa) {
        forge.game.card.CardCollectionView r = super.orderMoveToZoneList(cards, dest, sa);
        try {
            // This is the callback behind Ponder and Brainstorm: the player chooses
            // the order cards enter an ordered zone, so for the library they end up
            // at positions they know exactly. Other ordered zones (graveyard) carry
            // no hidden position, hence the gate.
            if (dest == forge.game.zone.ZoneType.Library && r != null) {
                LibraryKnowledge.learnExact(getPlayer(), r);
            }
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
        return r;
    }

    /**
     * Cards this player is currently being shown out of a hidden zone.
     *
     * <p>Instance state, because there is one controller per player and the
     * visibility is that player's. The depth counter guards against nesting,
     * which would otherwise let an inner endTempShowCards discard an outer
     * reveal's cards.
     */
    private final java.util.List<forge.game.card.Card> tempShown =
            new java.util.ArrayList<>();

    private int tempShowDepth;

    @Override
    public void tempShowCards(Iterable<forge.game.card.Card> cards) {
        super.tempShowCards(cards);
        try {
            tempShowDepth++;
            if (LibraryKnowledge.DEBUG) {
                StringBuilder sb = new StringBuilder();
                if (cards != null) {
                    for (forge.game.card.Card c : cards) {
                        sb.append(c.getName()).append("/").append(
                                c.getZone() == null ? "?" : c.getZone().getZoneType()).append(" ");
                    }
                }
                System.err.printf("[tempShow] depth=%d player=%s cards=[%s]%n",
                        tempShowDepth, getPlayer(), sb);
            }
            if (cards != null) {
                for (forge.game.card.Card c : cards) {
                    tempShown.add(c);
                }
            }
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
    }

    @Override
    public void endTempShowCards() {
        try {
            if (tempShowDepth > 0) {
                tempShowDepth--;
            }
            if (tempShowDepth == 0 && !tempShown.isEmpty()) {
                // Evaluated here rather than at tempShowCards time because the
                // cards have not moved yet when they are first shown; the dig
                // resolves, the rest are shuffled to the bottom, and only then is
                // their position meaningful.
                LibraryKnowledge.observeShown(getPlayer(), tempShown);
                tempShown.clear();
            }
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        } finally {
            super.endTempShowCards();
        }
    }
    // ---- Reveals ----------------------------------------------------------
    //
    // GameAction.reveal/revealTo call reveal(...) on the controller of each
    // player being SHOWN the cards, never on a bystander; the 3/4-arg overloads
    // and reveal(DelayedReveal) are final on PlayerController and funnel into
    // these two. So the receiver of this callback is exactly the viewer who
    // learned something, which is the only attribution we make. The owner's own
    // controller is also called in some flows (Thoughtseize tells the discarder
    // which card was chosen); those are their own cards, so RevealKnowledge
    // ignores them and no history is posted.

    private static int nextRevealGroup;
    private static String lastRevealKey;
    private static int lastRevealHistSize = -1;

    @Override
    public void reveal(forge.game.card.CardCollectionView cards, forge.game.zone.ZoneType zone,
            Player owner, String messagePrefix, boolean addSuffix) {
        super.reveal(cards, zone, owner, messagePrefix, addSuffix);
        try {
            List<forge.game.card.Card> list = new java.util.ArrayList<>();
            if (cards != null) {
                for (forge.game.card.Card c : cards) {
                    list.add(c);
                }
            }
            onRevealed(list, zone, owner);
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
    }

    @Override
    public void reveal(List<forge.game.card.CardView> cards, forge.game.zone.ZoneType zone,
            forge.game.player.PlayerView owner, String messagePrefix, boolean addSuffix) {
        super.reveal(cards, zone, owner, messagePrefix, addSuffix);
        try {
            List<forge.game.card.Card> list = new java.util.ArrayList<>();
            if (cards != null) {
                for (forge.game.card.CardView cv : cards) {
                    forge.game.card.Card c = getGame().findByView(cv);
                    if (c != null) {
                        list.add(c);
                    }
                }
            }
            Player ownerP = null;
            if (owner != null) {
                for (Player p : getGame().getPlayers()) {
                    if (p.getId() == owner.getId()) {
                        ownerP = p;
                    }
                }
            }
            onRevealed(list, zone, ownerP);
        } catch (Throwable ignored) {
            // instrumentation must never break the game
        }
    }

    /**
     * Whether the resolving ability is an outside-the-game fetch whose card does
     * NOT say "reveal" (North Wind Avatar, Mastermind's Acquisition).
     *
     * <p>ChangeZoneEffect.changeHiddenOriginResolve reveals the moved cards when
     * {@code Reveal$} is set OR when {@code ChangeType} is anything other than
     * plain "Card" -- a heuristic aimed at library tutors, whose Oracle text does
     * say "reveal it". For sideboard fetches the heuristic misfires: North Wind
     * Avatar's {@code ChangeType$ Card.YouOwn} trips it, and Forge calls
     * reveal() on the opponent (measured). Forge's scripts mark every real
     * outside-the-game reveal (Granted, the Wishes, Karn) with {@code Reveal$},
     * so a sideboard-origin ChangeZone without it is treated as no reveal.
     * Suppressing here can only understate what the viewer knows.
     */
    private boolean isUnrevealedOutsideGameFetch() {
        try {
            SpellAbility top = getGame().getStack().peekAbility();
            for (SpellAbility s = top; s != null; s = s.getSubAbility()) {
                if (s.getApi() != forge.game.ability.ApiType.ChangeZone || s.hasParam("Reveal")) {
                    continue;
                }
                String origin = s.getParam("Origin");
                if (origin != null && java.util.Arrays.asList(origin.split(",")).contains("Sideboard")) {
                    return true;
                }
            }
        } catch (Throwable ignored) {
            // fall through: treat as a genuine reveal
        }
        return false;
    }

    private static final boolean REVEAL_DEBUG = "1".equals(System.getProperty("magicnet.reveal.debug"));

    private void onRevealed(List<forge.game.card.Card> cards, forge.game.zone.ZoneType zone,
            Player owner) {
        Player me = getPlayer();
        if (REVEAL_DEBUG) {
            String src = "?";
            try {
                SpellAbility top = getGame().getStack().peekAbility();
                src = top == null ? "-" : top.getHostCard().getName();
            } catch (Throwable ignored) {
                // diagnostic only
            }
            StringBuilder sb = new StringBuilder();
            for (forge.game.card.Card c : cards) {
                sb.append(c.getName()).append('/').append(c.getOwner()).append(' ');
            }
            System.err.printf("[reveal] receiver=%s owner=%s zone=%s resolving=%s cards=[%s]%n",
                    me, owner, zone, src, sb);
        }
        List<forge.game.card.Card> foreign = new java.util.ArrayList<>();
        for (forge.game.card.Card c : cards) {
            if (c != null && c.getOwner() != null && c.getOwner().getId() != me.getId()) {
                foreign.add(c);
            }
        }
        if (foreign.isEmpty()) {
            RevealKnowledge.noteOwnOnly();
            return;
        }
        if (isUnrevealedOutsideGameFetch()) {
            RevealKnowledge.noteSuppressed();
            return;
        }
        Object resolving = null;
        try {
            resolving = getGame().getStack().peekAbility();
        } catch (Throwable ignored) {
            // no resolution context: a pending reveal then never matches
        }
        for (forge.game.card.Card c : foreign) {
            if (!RevealKnowledge.observe(me, c)) {
                RevealKnowledge.observePending(me, c, resolving);
            }
        }
        // Some flows show the same cards twice in a row (ChangeZone's
        // reveal(DelayedReveal) and then the chooser prompt). A repeat with no
        // intervening history carries nothing new, so it is not re-posted.
        StringBuilder key = new StringBuilder().append(me.getId()).append(":");
        for (forge.game.card.Card c : foreign) {
            key.append(c.getId()).append(',');
        }
        Game game = getGame();
        int group;
        synchronized (HISTORY_LOCK) {
            history(game, me);
            int mySize = HISTORIES.get(me.getId()).size();
            if (key.toString().equals(lastRevealKey) && mySize == lastRevealHistSize) {
                return;
            }
            group = ++nextRevealGroup;
            lastRevealKey = key.toString();
        }
        // Every viewer's history records that the reveal happened; the cards
        // are edges only for the recipient (me) or a viewer who can see them.
        HistoryHub hub = HistoryHub.of(game);
        if (hub != null) {
            hub.reveal(foreign, zone, owner != null ? owner : foreign.get(0).getOwner(), me);
        }
        synchronized (HISTORY_LOCK) {
            lastRevealHistSize = HISTORIES.get(me.getId()).size();
        }
    }

    // ---- Decision-surface census (-Dmagicnet.census=true) -------------------
    //
    // chooseSpellAbilityToPlay is one of roughly forty decision callbacks on
    // PlayerController. Everything not overridden silently falls through to
    // Forge's AI, which means it is outside the policy's reach without being
    // outside the *game's*. These overrides only count and delegate; they exist
    // to turn "what else is there" into a measured priority order.
    static final boolean CENSUS = Boolean.getBoolean("magicnet.census");
    static final java.util.Map<String, java.util.concurrent.atomic.AtomicLong> CENSUS_COUNTS =
            new java.util.concurrent.ConcurrentHashMap<>();

    private static void census(String what) {
        if (CENSUS) {
            CENSUS_COUNTS.computeIfAbsent(what,
                    k -> new java.util.concurrent.atomic.AtomicLong()).incrementAndGet();
        }
    }

    public static void printCensus() {
        int unl = UNLABELLED.get();
        if (unl > 0) {
            System.out.println("  WARN: " + unl + " observed choices whose expert answer"
                    + " could not be located in the option list (chosen=-1)");
        }
        if (UNLESS_PAY_FAILED.get() > 0) {
            System.out.println("  WARN: " + UNLESS_PAY_FAILED.get() + " routed unless-cost"
                    + " payments failed (refunded; answered as not paid)");
        }
        if (!CENSUS || CENSUS_COUNTS.isEmpty()) {
            return;
        }
        System.out.println("=== decision-surface census ===");
        CENSUS_COUNTS.entrySet().stream()
                .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                .forEach(e -> System.out.printf("  %-34s %6d%n", e.getKey(), e.getValue().get()));
    }

    @Override
    public forge.game.card.CardCollection orderBlockers(forge.game.card.Card attacker,
            forge.game.card.CardCollection blockers) {
        census("orderBlockers");
        return super.orderBlockers(attacker, blockers);
    }



    @Override
    public boolean confirmReplacementEffect(forge.game.replacement.ReplacementEffect re,
            forge.game.spellability.SpellAbility sa, forge.game.GameEntity affected, String prompt) {
        census("confirmReplacementEffect");
        return super.confirmReplacementEffect(re, sa, affected, prompt);
    }

    @Override
    public forge.game.replacement.ReplacementEffect chooseSingleReplacementEffect(
            java.util.List<forge.game.replacement.ReplacementEffect> possible) {
        census("chooseSingleReplacementEffect");
        return super.chooseSingleReplacementEffect(possible);
    }




    @Override
    public java.util.List<forge.game.spellability.SpellAbility> orderSimultaneousSa(
            java.util.List<forge.game.spellability.SpellAbility> activePlayerSAs) {
        census("orderSimultaneousSa");
        if (PASS_THROUGH_ORDER.get()) {
            return activePlayerSAs;
        }
        return super.orderSimultaneousSa(activePlayerSAs);
    }



    // ---- Generic choice points ---------------------------------------------
    //
    // Most of Forge's decision callbacks are "pick one of N". They are routed
    // through the same request shape, the same ActionSource and the same dump
    // record as a priority decision, distinguished only by `kind`.
    //
    // Returns an index into options, or ActionSource.DELEGATE to let Forge's AI
    // decide. Delegation is the default whenever no external source is driving,
    // which keeps `-Dmagicnet.actions=ai` behaving exactly as before.

    /**
     * Builds a choice record <em>before</em> the decision is made, for sources
     * that do not decide (i.e. Forge's AI). The expert's answer is filled in
     * afterwards by {@link #closeChoice}, which yields supervised labels for
     * callbacks the policy does not yet drive.
     *
     * <p>Built in the pre-decision state deliberately: behavioural cloning needs
     * the state the expert actually saw, not the state its choice produced.
     *
     * @return the record body, or null if nothing will consume it
     */
    private String openChoice(String kind, List<Serializer.Opt> options) {
        if (options == null || options.isEmpty() || !Serializer.enabled()
                || ACTIONS.decides()) {
            return null;
        }
        List<StateExtractor.Tok> hist = history(getGame(), getPlayer());
        List<StateExtractor.Tok> toks = extractTokens(null, hist);
        if (toks == null) {
            return null;
        }
        return Serializer.buildChoiceBody(getGame(), getPlayer(), toks,
                hist == null ? 0 : hist.size(), kind, options);
    }

    /**
     * Writes a record opened by {@link #openChoice}.
     *
     * <p>Must be called for every opened body, including when the expert's
     * answer could not be located ({@code chosen < 0}): building commits the
     * per-viewer history cursor, so an unwritten body loses its events.
     */
    private void closeChoice(String body, int chosen) {
        if (body == null) {
            return;
        }
        if (chosen < 0) {
            UNLABELLED.incrementAndGet();
        }
        Serializer.writeChoice(body, chosen);
    }

    private static final java.util.concurrent.atomic.AtomicInteger UNLABELLED =
            new java.util.concurrent.atomic.AtomicInteger();

    /** Index of {@code value} among {@code list}, offset past an optional "none" entry. */
    private static int optionIndex(java.util.List<?> list, Object value, boolean hasNone) {
        if (value == null) {
            return hasNone ? 0 : -1;
        }
        int base = hasNone ? 1 : 0;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == value || list.get(i).equals(value)) {
                return base + i;
            }
        }
        return -1;
    }

    private int askChoice(String kind, List<Serializer.Opt> options) {
        if (options.isEmpty()) {
            return ActionSource.DELEGATE;
        }
        boolean wantDump = Serializer.enabled();
        boolean wantAgent = ACTIONS.needsRequest();
        if (!wantDump && !wantAgent) {
            return ActionSource.DELEGATE;
        }
        // Must precede buildChoiceBody: building commits the per-viewer history
        // cursor, so a body that is built and then discarded loses its events.
        if (!ACTIONS.decides()) {
            return ActionSource.DELEGATE;
        }

        List<StateExtractor.Tok> hist = history(getGame(), getPlayer());
        List<StateExtractor.Tok> toks = extractTokens(null, hist);
        if (toks == null) {
            return ActionSource.DELEGATE;
        }
        String body = Serializer.buildChoiceBody(getGame(), getPlayer(), toks,
                hist == null ? 0 : hist.size(), kind, options);
        int pick = ACTIONS.choose(wantAgent ? Serializer.asRequest(body) : null,
                options.size(), -1);
        if (pick == ActionSource.DELEGATE) {
            throw new IllegalStateException("action source " + ACTIONS.getClass().getSimpleName()
                    + " delegated after reporting decides()==true; the record for this decision"
                    + " was already built and its history events would be lost");
        }
        Serializer.writeChoice(body, pick);
        return pick;
    }

    /**
     * Attacking is not "pick one of N" -- it is a subset of creatures paired
     * with defenders. It is decomposed into one pick-one decision per eligible
     * creature ("don't attack" plus one option per legal defender), which keeps
     * the action space uniform and makes every choice legal by construction:
     * only defenders passing CombatUtil.canAttack are offered.
     *
     * <p>The decomposition is sequential rather than joint, so the policy
     * cannot express "attack with A only if B also attacks" in one step. It can
     * still represent it across steps, because each decision observes the
     * attackers already declared.
     */
    @Override
    public void declareAttackers(forge.game.player.Player attacker,
            forge.game.combat.Combat combat) {
        census("declareAttackers");
        if (!routeCombat()) {
            super.declareAttackers(attacker, combat);
            return;
        }
        if (!ACTIONS.decides()) {
            observeAttackers(attacker, combat);
            return;
        }

        List<forge.game.GameEntity> defenders =
                new java.util.ArrayList<>(combat.getDefenders());
        forge.game.card.CardCollectionView possible =
                attacker.getCreaturesInPlay();

        boolean any = false;
        for (forge.game.card.Card c : possible) {
            List<forge.game.GameEntity> legal = new java.util.ArrayList<>();
            for (forge.game.GameEntity d : defenders) {
                if (forge.game.combat.CombatUtil.canAttack(c, d)) {
                    legal.add(d);
                }
            }
            if (legal.isEmpty()) {
                continue;
            }
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            String cref = StateExtractor.cardRef(c.getId());
            opts.add(new Serializer.Opt("do not attack with " + c.getName()).ref(cref));
            for (forge.game.GameEntity d : legal) {
                opts.add(new Serializer.Opt("attack " + d + " with " + c.getName())
                        .ref(cref).ref(StateExtractor.entityRef(d)));
            }
            int pick = askChoice("declare_attacker", opts);
            if (pick == ActionSource.DELEGATE) {
                // Mid-sequence delegation would leave a partly policy-declared
                // combat for the AI to finish, which is neither the policy's
                // intent nor the AI's. Abandon and let the AI declare it all.
                if (!any) {
                    super.declareAttackers(attacker, combat);
                    return;
                }
                break;
            }
            if (pick > 0 && pick <= legal.size()) {
                combat.addAttacker(c, legal.get(pick - 1));
                any = true;
            }
        }
    }

    /**
     * Records what Forge's AI declares, decomposed identically to the routed
     * path so supervised labels and policy decisions share one record shape.
     *
     * <p>All bodies are built before delegating, so every record shows the same
     * pre-combat state -- which is also the truthful rendering, since declaring
     * attackers is simultaneous rather than sequential.
     */
    private void observeAttackers(forge.game.player.Player attacker,
            forge.game.combat.Combat combat) {
        List<forge.game.GameEntity> defenders =
                new java.util.ArrayList<>(combat.getDefenders());
        List<forge.game.card.Card> creatures = new java.util.ArrayList<>();
        List<List<forge.game.GameEntity>> legals = new java.util.ArrayList<>();
        List<String> bodies = new java.util.ArrayList<>();

        for (forge.game.card.Card c : attacker.getCreaturesInPlay()) {
            List<forge.game.GameEntity> legal = new java.util.ArrayList<>();
            for (forge.game.GameEntity d : defenders) {
                if (forge.game.combat.CombatUtil.canAttack(c, d)) {
                    legal.add(d);
                }
            }
            if (legal.isEmpty()) {
                continue;
            }
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            String cref = StateExtractor.cardRef(c.getId());
            opts.add(new Serializer.Opt("do not attack with " + c.getName()).ref(cref));
            for (forge.game.GameEntity d : legal) {
                opts.add(new Serializer.Opt("attack " + d + " with " + c.getName())
                        .ref(cref).ref(StateExtractor.entityRef(d)));
            }
            creatures.add(c);
            legals.add(legal);
            bodies.add(openChoice("declare_attacker", opts));
        }

        try {
            super.declareAttackers(attacker, combat);
        } finally {
            for (int i = 0; i < creatures.size(); i++) {
                forge.game.card.Card c = creatures.get(i);
                int chosen = 0;
                if (combat.isAttacking(c)) {
                    forge.game.GameEntity d = combat.getDefenderByAttacker(c);
                    int at = legals.get(i).indexOf(d);
                    chosen = at < 0 ? -1 : at + 1;
                }
                closeChoice(bodies.get(i), chosen);
            }
        }
    }

    /**
     * Blocking decomposes the same way: one decision per untapped creature,
     * offering each attacker it may legally block.
     */
    @Override
    public void declareBlockers(forge.game.player.Player defender,
            forge.game.combat.Combat combat) {
        census("declareBlockers");
        if (!routeCombat()) {
            super.declareBlockers(defender, combat);
            return;
        }

        forge.game.card.CardCollection attackers = combat.getAttackers();
        if (attackers == null || attackers.isEmpty()) {
            super.declareBlockers(defender, combat);
            return;
        }
        if (!ACTIONS.decides()) {
            observeBlockers(defender, combat, attackers);
            return;
        }

        boolean any = false;
        for (forge.game.card.Card b : defender.getCreaturesInPlay()) {
            List<forge.game.card.Card> legal = new java.util.ArrayList<>();
            for (forge.game.card.Card a : attackers) {
                if (forge.game.combat.CombatUtil.canBlock(a, b, combat)) {
                    legal.add(a);
                }
            }
            if (legal.isEmpty()) {
                continue;
            }
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            String bref = StateExtractor.cardRef(b.getId());
            opts.add(new Serializer.Opt("do not block with " + b.getName()).ref(bref));
            for (forge.game.card.Card a : legal) {
                opts.add(new Serializer.Opt("block " + a.getName() + " with " + b.getName())
                        .ref(bref).ref(StateExtractor.cardRef(a.getId())));
            }
            int pick = askChoice("declare_blocker", opts);
            if (pick == ActionSource.DELEGATE) {
                if (!any) {
                    super.declareBlockers(defender, combat);
                    return;
                }
                break;
            }
            if (pick > 0 && pick <= legal.size()) {
                combat.addBlocker(legal.get(pick - 1), b);
                any = true;
            }
        }
    }

    /**
     * Combat routing is opt-in while it is new: the decomposition changes which
     * decisions exist, so every previously recorded dump and replay would
     * otherwise stop being comparable.
     */
    private static final boolean ROUTE_COMBAT = Boolean.getBoolean("magicnet.combat");

    private boolean routeCombat() {
        return ROUTE_COMBAT && (Serializer.enabled() || ACTIONS.needsRequest());
    }


    /** Blocking counterpart of {@link #observeAttackers}. */
    private void observeBlockers(forge.game.player.Player defender,
            forge.game.combat.Combat combat, forge.game.card.CardCollection attackers) {
        List<forge.game.card.Card> blockers = new java.util.ArrayList<>();
        List<List<forge.game.card.Card>> legals = new java.util.ArrayList<>();
        List<String> bodies = new java.util.ArrayList<>();

        for (forge.game.card.Card b : defender.getCreaturesInPlay()) {
            List<forge.game.card.Card> legal = new java.util.ArrayList<>();
            for (forge.game.card.Card a : attackers) {
                if (forge.game.combat.CombatUtil.canBlock(a, b, combat)) {
                    legal.add(a);
                }
            }
            if (legal.isEmpty()) {
                continue;
            }
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            String bref = StateExtractor.cardRef(b.getId());
            opts.add(new Serializer.Opt("do not block with " + b.getName()).ref(bref));
            for (forge.game.card.Card a : legal) {
                opts.add(new Serializer.Opt("block " + a.getName() + " with " + b.getName())
                        .ref(bref).ref(StateExtractor.cardRef(a.getId())));
            }
            blockers.add(b);
            legals.add(legal);
            bodies.add(openChoice("declare_blocker", opts));
        }

        try {
            super.declareBlockers(defender, combat);
        } finally {
            for (int i = 0; i < blockers.size(); i++) {
                forge.game.card.Card b = blockers.get(i);
                int chosen = 0;
                // A creature may legally block several attackers; the
                // decomposition offers one, so the first is recorded.
                forge.game.card.Card blocked = null;
                for (forge.game.card.Card a : legals.get(i)) {
                    if (combat.isBlocking(b, a)) {
                        blocked = a;
                        break;
                    }
                }
                if (blocked != null) {
                    chosen = legals.get(i).indexOf(blocked) + 1;
                }
                closeChoice(bodies.get(i), chosen);
            }
        }
    }

    // ---- The remaining pick-one callbacks ----------------------------------
    //
    // Each of these was previously answered by Forge's AI. They reuse askChoice
    // unchanged, which is the payoff of giving every choice point the same
    // record shape: routing a new callback costs an option list, not a new
    // encoding.
    //
    // Anything whose options cannot be enumerated safely delegates, so adding
    // a route can only move decisions from the AI to the policy, never break
    // one.

    @Override
    public <T extends forge.game.GameEntity> T chooseSingleEntityForEffect(
            forge.util.collect.FCollectionView<T> options, forge.game.player.DelayedReveal dr,
            forge.game.spellability.SpellAbility sa, String title, boolean optional,
            forge.game.player.Player targetedPlayer, java.util.Map<String, Object> params) {
        census("chooseSingleEntityForEffect");
        // A DelayedReveal shows the decider cards from a hidden zone (a library
        // search). It is performed before asking, as Forge's AI does, so the
        // reveal reaches the decider's history; the options name the cards.
        // Forge's companion step (Player.assignCompanion) passes optional=true --
        // revealing a companion is optional by the rules -- but then moves the
        // result to the command zone without a null check, so declining throws.
        // Not revealing is therefore not offered. It is never the better play.
        boolean companion = isCompanionChoice(options);
        if (companion) {
            optional = false;
        }
        String kind = companion ? "choose_companion" : "choose_entity";
        if (routeChoices() && options != null && options.size() + (optional ? 1 : 0) > 1) {
            List<T> list = new java.util.ArrayList<>();
            for (T t : options) {
                list.add(t);
            }
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            if (optional) {
                opts.add(new Serializer.Opt("choose nothing"));
            }
            for (T t : list) {
                // Card.toString() appends the object id ("Yorion (92)"), a
                // run-unstable join key; use the name.
                opts.add(new Serializer.Opt(t instanceof forge.game.card.Card
                                ? ((forge.game.card.Card) t).getName() : String.valueOf(t))
                        .ref(StateExtractor.entityRef(t))
                        .oracle(t instanceof forge.game.card.Card
                                ? OracleIds.of((forge.game.card.Card) t) : null));
            }
            revealBeforeAsking(dr);
            int pick = askChoice(kind, opts);
            if (pick != ActionSource.DELEGATE) {
                int base = optional ? 1 : 0;
                if (pick < base) {
                    return null;
                }
                return list.get(Math.min(pick - base, list.size() - 1));
            }
            String body = openChoice(kind, opts);
            if (body != null) {
                T picked = null;
                try {
                    picked = super.chooseSingleEntityForEffect(options, dr, sa, title,
                            optional, targetedPlayer, params);
                    return picked;
                } finally {
                    closeChoice(body, optionIndex(list, picked, optional));
                }
            }
        }
        return super.chooseSingleEntityForEffect(options, dr, sa, title, optional,
                targetedPlayer, params);
    }

    /** Pre-game companion reveal: every option is a sideboard card before turn 1. */
    private boolean isCompanionChoice(Iterable<? extends forge.game.GameEntity> options) {
        try {
            if (options == null || getGame().getPhaseHandler().getTurn() > 0) {
                return false;
            }
            boolean any = false;
            for (forge.game.GameEntity e : options) {
                if (!(e instanceof forge.game.card.Card)
                        || !((forge.game.card.Card) e).isInZone(forge.game.zone.ZoneType.Sideboard)) {
                    return false;
                }
                any = true;
            }
            return any;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Performs a pending DelayedReveal before an external source is asked, as
     * PlayerControllerAi does before choosing. Skipped when Forge's AI will
     * decide, because its own call performs the reveal.
     */
    private void revealBeforeAsking(forge.game.player.DelayedReveal dr) {
        if (dr != null && ACTIONS.decides()) {
            reveal(dr);
        }
    }

    @Override
    public forge.game.card.Card chooseSingleCardForZoneChange(forge.game.zone.ZoneType destination,
            java.util.List<forge.game.zone.ZoneType> origin, forge.game.spellability.SpellAbility sa,
            forge.game.card.CardCollection fetchList, forge.game.player.DelayedReveal dr,
            String selectPrompt, boolean isOptional, forge.game.player.Player decider) {
        census("chooseSingleCardForZoneChange");
        if (routeChoices() && fetchList != null && fetchList.size() + (isOptional ? 1 : 0) > 1) {
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            if (isOptional) {
                opts.add(new Serializer.Opt("choose nothing"));
            }
            for (forge.game.card.Card c : fetchList) {
                opts.add(new Serializer.Opt(c.getName())
                        .ref(StateExtractor.cardRef(c.getId()))
                        .oracle(OracleIds.of(c)));
            }
            revealBeforeAsking(dr);
            int pick = askChoice("choose_card_zone_change", opts);
            if (pick != ActionSource.DELEGATE) {
                int base = isOptional ? 1 : 0;
                if (pick < base) {
                    return null;
                }
                return fetchList.get(Math.min(pick - base, fetchList.size() - 1));
            }
            String body = openChoice("choose_card_zone_change", opts);
            if (body != null) {
                forge.game.card.Card picked = null;
                try {
                    picked = super.chooseSingleCardForZoneChange(destination, origin, sa,
                            fetchList, dr, selectPrompt, isOptional, decider);
                    return picked;
                } finally {
                    closeChoice(body, optionIndex(fetchList, picked, isOptional));
                }
            }
        }
        return super.chooseSingleCardForZoneChange(destination, origin, sa, fetchList, dr,
                selectPrompt, isOptional, decider);
    }

    @Override
    public int chooseNumber(forge.game.spellability.SpellAbility sa, String title, int min, int max) {
        census("chooseNumber");
        // Unbounded maxima (X costs against an empty pool) would enumerate an
        // absurd option list, so only route a small, finite range.
        if (routeChoices() && max > min && max - min <= 32) {
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            for (int v = min; v <= max; v++) {
                opts.add(new Serializer.Opt(String.valueOf(v)));
            }
            int pick = askChoice("choose_number", opts);
            if (pick != ActionSource.DELEGATE) {
                return min + Math.max(0, Math.min(pick, max - min));
            }
            String body = openChoice("choose_number", opts);
            if (body != null) {
                int picked = min;
                try {
                    picked = super.chooseNumber(sa, title, min, max);
                    return picked;
                } finally {
                    int idx = picked - min;
                    closeChoice(body, idx >= 0 && idx <= max - min ? idx : -1);
                }
            }
        }
        return super.chooseNumber(sa, title, min, max);
    }

    @Override
    public boolean confirmTrigger(forge.game.trigger.WrappedAbility wa) {
        census("confirmTrigger");
        if (routeChoices()) {
            List<Serializer.Opt> opts = java.util.Arrays.asList(
                    new Serializer.Opt("decline"),
                    new Serializer.Opt("accept: " + stripIds(String.valueOf(wa))));
            int pick = askChoice("confirm_trigger", opts);
            if (pick != ActionSource.DELEGATE) {
                return pick > 0;
            }
            String body = openChoice("confirm_trigger", opts);
            if (body != null) {
                boolean picked = false;
                try {
                    picked = super.confirmTrigger(wa);
                    return picked;
                } finally {
                    closeChoice(body, picked ? 1 : 0);
                }
            }
        }
        return super.confirmTrigger(wa);
    }

    @Override
    public boolean confirmAction(forge.game.spellability.SpellAbility sa,
            forge.game.player.PlayerActionConfirmMode mode, String prompt,
            java.util.List<String> opts, forge.game.card.Card c,
            java.util.Map<String, Object> params) {
        census("confirmAction");
        if (routeChoices()) {
            List<Serializer.Opt> choices = java.util.Arrays.asList(
                    new Serializer.Opt("decline"),
                    new Serializer.Opt("accept: " + prompt));
            int pick = askChoice("confirm_action", choices);
            if (pick != ActionSource.DELEGATE) {
                return pick > 0;
            }
            String body = openChoice("confirm_action", choices);
            if (body != null) {
                boolean picked = false;
                try {
                    picked = super.confirmAction(sa, mode, prompt, opts, c, params);
                    return picked;
                } finally {
                    closeChoice(body, picked ? 1 : 0);
                }
            }
        }
        return super.confirmAction(sa, mode, prompt, opts, c, params);
    }

    @Override
    public java.util.List<forge.game.spellability.AbilitySub> chooseModeForAbility(
            forge.game.spellability.SpellAbility sa,
            java.util.List<forge.game.spellability.AbilitySub> possible,
            int min, int num, boolean allowRepeat) {
        census("chooseModeForAbility");
        // Only the single-mode case is routed. Choosing k of n modes is a
        // subset problem like combat and wants the same sequential treatment;
        // guessing at it here would silently pick a wrong number of modes.
        if (routeChoices() && min == 1 && num == 1 && possible != null && possible.size() > 1) {
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            for (forge.game.spellability.AbilitySub m : possible) {
                opts.add(new Serializer.Opt(String.valueOf(m)));
            }
            int pick = askChoice("choose_mode", opts);
            if (pick != ActionSource.DELEGATE && pick >= 0 && pick < possible.size()) {
                List<forge.game.spellability.AbilitySub> one = new java.util.ArrayList<>();
                one.add(possible.get(pick));
                return one;
            }
            String body = openChoice("choose_mode", opts);
            if (body != null) {
                List<forge.game.spellability.AbilitySub> picked = null;
                try {
                    picked = super.chooseModeForAbility(sa, possible, min, num, allowRepeat);
                    return picked;
                } finally {
                    closeChoice(body, picked == null || picked.size() != 1
                            ? -1 : optionIndex(possible, picked.get(0), false));
                }
            }
        }
        return super.chooseModeForAbility(sa, possible, min, num, allowRepeat);
    }

    // ---- "Unless" costs --------------------------------------------------------

    /**
     * "Pay a cost or the effect happens": shock lands, Spell Pierce, No More
     * Lies, Mystical Dispute. Kind {@code pay_unless}: 0 = don't pay, 1 = pay;
     * both reference the effect's source card. "Pay" is offered only when
     * {@code ComputerUtilCost.canPayCost} -- the AI's own check -- says it is
     * payable; otherwise there is nothing to decide and the AI answers.
     *
     * <p>Forge's AI pays <em>inside</em> this callback (willPayUnlessCost,
     * canPayCost, then CostPayment.payComputerCosts), and the human controller
     * likewise pays in it (PlaySpellAbility.payCostDuringAbilityResolve), so a
     * routed "pay" pays here too, see {@link #payUnlessCost}.
     */
    @Override
    public boolean payCostToPreventEffect(forge.game.cost.Cost cost, SpellAbility sa,
            boolean alreadyPaid, forge.util.collect.FCollectionView<Player> allPayers) {
        census("payCostToPreventEffect");
        if (!routeChoices() || cost == null || sa == null
                || !unlessPayable(cost, sa, getPlayer())) {
            return super.payCostToPreventEffect(cost, sa, alreadyPaid, allPayers);
        }
        Player me = getPlayer();
        forge.game.card.Card src = sa.getHostCard();
        boolean shown = src != null && shownTo(src, me);
        String cref = shown ? StateExtractor.cardRef(src.getId()) : null;
        String oid = shown ? OracleIds.of(src) : null;
        String of = shown ? " (" + src.getName() + ")" : "";
        List<Serializer.Opt> opts = java.util.Arrays.asList(
                new Serializer.Opt("don't pay" + of).ref(cref).oracle(oid),
                new Serializer.Opt(payText(cost) + of).ref(cref).oracle(oid));
        int pick = askChoice("pay_unless", opts);
        if (pick != ActionSource.DELEGATE) {
            return pick == 1 && payUnlessCost(cost, sa, me);
        }
        String body = openChoice("pay_unless", opts);
        if (body != null) {
            boolean paid = false;
            try {
                paid = super.payCostToPreventEffect(cost, sa, alreadyPaid, allPayers);
                return paid;
            } finally {
                closeChoice(body, paid ? 1 : 0);
            }
        }
        return super.payCostToPreventEffect(cost, sa, alreadyPaid, allPayers);
    }

    private static boolean unlessPayable(forge.game.cost.Cost cost, SpellAbility sa, Player me) {
        try {
            return forge.ai.ComputerUtilCost.canPayCost(cost, sa, me, true);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Pays a chosen unless cost with the AI's cost decisions, as the AI would.
     *
     * <p>Not via payComputerCosts, the AI's own entry point: it keeps no record
     * of the parts it paid, so a part failing after another succeeded leaves
     * that payment behind. CostPayment.payCost does record them, and
     * refundPayment then undoes them (mana, taps, counters, energy). Probed by
     * appending an unpayable PayLife<1000> to {2}: the mana part paid, the
     * life part failed, and the refund untapped the lands. Two gaps remain,
     * hence the per-part canPay check first: Forge cannot refund life, and
     * floating mana spent from the pool was not always put back (1 of 4
     * probes). Failures are counted and reported by printCensus.
     */
    private boolean payUnlessCost(forge.game.cost.Cost cost, SpellAbility sa, Player me) {
        // As PlaySpellAbility.payCostDuringAbilityResolve does first: refuse
        // before paying anything if a part cannot be paid.
        for (forge.game.cost.CostPart part : cost.getCostParts()) {
            if (!part.canPay(sa, me, true)) {
                UNLESS_PAY_FAILED.incrementAndGet();
                return false;
            }
        }
        forge.game.cost.CostPayment payment = new forge.game.cost.CostPayment(cost, sa);
        boolean ok = false;
        try {
            ok = payment.payCost(new forge.ai.AiCostDecision(me, sa, true));
            return ok;
        } finally {
            if (!ok) {
                UNLESS_PAY_FAILED.incrementAndGet();
                payment.refundPayment();
            }
        }
    }

    private static final java.util.concurrent.atomic.AtomicInteger UNLESS_PAY_FAILED =
            new java.util.concurrent.atomic.AtomicInteger();

    /** "pay {3}", "pay 2 life" -- Forge words some costs with a leading "Pay". */
    private static String payText(forge.game.cost.Cost cost) {
        String c = costText(cost);
        return c.regionMatches(true, 0, "pay ", 0, 4) ? "pay " + c.substring(4) : "pay " + c;
    }

    private static String costText(forge.game.cost.Cost cost) {
        String s;
        try {
            s = cost.toSimpleString();
        } catch (RuntimeException e) {
            s = null;
        }
        if (s == null || s.isEmpty()) {
            s = String.valueOf(cost);
        }
        return stripIds(s.trim());
    }

    /** Forge renders objects as "Name (id)"; the id is run-unstable. */
    static String stripIds(String s) {
        return s == null ? null : s.replaceAll(" \\(\\d+\\)", "");
    }

    private static boolean shownTo(forge.game.card.Card c, Player viewer) {
        try {
            return c.getView().canBeShownTo(viewer.getView());
        } catch (RuntimeException e) {
            return false;
        }
    }

    // ---- Simultaneous triggers: order and targets ----------------------------
    //
    // MagicStack hands each player their simultaneous stack entries through
    // orderAndPlaySimultaneousSa. Forge's AI orders them (orderPlaySa) and,
    // per entry, binds a trigger's targets in its private prepareSingleSa
    // (AiController.doTrigger) before ComputerUtil.playStack -- never via
    // chooseTargetsFor. PlayerControllerHuman instead sends triggers through
    // PlaySpellAbility, whose setupTargets is the engine's targeting hook.
    //
    // Routed: 2+ entries become repeated order_triggers decisions ("put X on
    // the stack next", first pick bottommost); then each trigger is prepared
    // like the AI does, except that a trigger whose ability chain targets gets
    // its targets from SpellAbility.setupTargets -> chooseTargetsFor, routed
    // as choose_target decisions. Stack placement stays ComputerUtil.playStack.

    /** Set while a trigger's targets are being chosen, for chooseTargetsFor. */
    private static final ThreadLocal<Boolean> ROUTE_TARGETS =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** Makes orderSimultaneousSa an identity, to replay the AI one entry at a time. */
    private static final ThreadLocal<Boolean> PASS_THROUGH_ORDER =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    @Override
    public void orderAndPlaySimultaneousSa(List<SpellAbility> sas) {
        if (!routeChoices() || sas == null || sas.isEmpty()) {
            super.orderAndPlaySimultaneousSa(sas);
            return;
        }
        if (!ACTIONS.decides()) {
            observeSimultaneous(sas);
            return;
        }
        List<SpellAbility> order = sas.size() > 1
                ? orderTriggers(sas, null) : new java.util.ArrayList<>(sas);
        for (SpellAbility sa : order) {
            if (sa.isTrigger() && !sa.isCopied()) {
                if (prepareTrigger(sa, true)) {
                    forge.ai.ComputerUtil.playStack(sa, getPlayer(), getGame());
                }
            } else {
                playOneAsAi(sa);
            }
        }
    }

    /** Forge's AI handling of a single entry (copies, mainly), order untouched. */
    private void playOneAsAi(SpellAbility sa) {
        PASS_THROUGH_ORDER.set(Boolean.TRUE);
        try {
            super.orderAndPlaySimultaneousSa(java.util.Collections.singletonList(sa));
        } finally {
            PASS_THROUGH_ORDER.set(Boolean.FALSE);
        }
    }

    /**
     * Observe mode: the AI's order is read first (orderPlaySa is a pure sort)
     * and recorded as order_triggers labels, all built before anything is
     * placed, as in decide mode. Entries are then handed to the AI one at a
     * time, in that order, which is exactly its own loop; a single-slot target
     * choice is recorded around each, built before the AI binds it and
     * labelled from what it bound.
     */
    private void observeSimultaneous(List<SpellAbility> sas) {
        List<SpellAbility> ai = super.orderSimultaneousSa(new java.util.ArrayList<>(sas));
        if (ai == null || ai.size() != sas.size() || !ai.containsAll(sas)) {
            // Not a permutation: the replay below would not be the AI's loop.
            super.orderAndPlaySimultaneousSa(sas);
            return;
        }
        if (ai.size() > 1) {
            orderTriggers(sas, ai);
        }
        for (SpellAbility sa : ai) {
            ObservedTarget ot = sa.isTrigger() && !sa.isCopied() ? openObservedTarget(sa) : null;
            try {
                playOneAsAi(sa);
            } finally {
                if (ot != null) {
                    ot.close();
                }
            }
        }
    }

    /**
     * Repeated pick-one ordering of stack entries: each decision picks the
     * entry put on the stack next, so the first pick ends bottommost and
     * resolves last. The last entry is forced and not asked.
     *
     * @param expert the AI's order (observe mode), or null to ask
     */
    private List<SpellAbility> orderTriggers(List<SpellAbility> sas, List<SpellAbility> expert) {
        List<SpellAbility> remaining = new java.util.ArrayList<>(sas);
        List<SpellAbility> out = new java.util.ArrayList<>();
        while (remaining.size() > 1) {
            List<Serializer.Opt> opts = new java.util.ArrayList<>();
            for (SpellAbility sa : remaining) {
                opts.add(entryOpt("put ", sa, " on the stack next"));
            }
            int pick;
            if (expert == null) {
                pick = askChoice("order_triggers", opts);
                if (pick < 0 || pick >= remaining.size()) {
                    pick = 0;
                }
            } else {
                int label = remaining.indexOf(expert.get(out.size()));
                closeChoice(openChoice("order_triggers", opts), label);
                pick = Math.max(0, label);
            }
            out.add(remaining.remove(pick));
        }
        out.addAll(remaining);
        return out;
    }

    /** Option naming a stack entry by its source card and its own text. */
    private Serializer.Opt entryOpt(String prefix, SpellAbility sa, String suffix) {
        forge.game.card.Card host = sa.getHostCard();
        boolean shown = host != null && shownTo(host, getPlayer());
        String text = stripIds(String.valueOf(sa));
        String name = shown ? host.getName() : "a hidden card";
        String what = sa.isTrigger() ? " trigger" : sa.isCopied() ? " copy" : " ability";
        return new Serializer.Opt(prefix + name + what + suffix + ": " + text)
                .ref(shown ? StateExtractor.cardRef(host.getId()) : null)
                .oracle(shown ? OracleIds.of(host) : null);
    }

    /**
     * prepareSingleSa of PlayerControllerAi with target choice routed: modal
     * (Charm) choices and TargetingPlayer handling as the AI does them, then
     * setupTargets if anything in the ability chain targets, else the AI's
     * doTrigger (which, for an untargeted trigger, decides nothing else).
     */
    private boolean prepareTrigger(SpellAbility sa, boolean mandatory) {
        SpellAbility s = sa;
        if (s.getApi() == forge.game.ability.ApiType.Charm) {
            if (!forge.game.ability.effects.CharmEffect.makeChoices(s)) {
                return false;
            }
            if (!s.hasParam("Random")) {
                // The AI stops here: its mode choice bound any targets. A
                // routed mode choice did not, so route those targets too.
                return !chainTargets(s) || setupTargetsRouted(s);
            }
            s = s.getSubAbility();
            if (s == null) {
                return true;
            }
        }
        if (chainTargets(s)) {
            return setupTargetsRouted(s);
        }
        return getAi().doTrigger(s, mandatory);
    }

    private static boolean chainTargets(SpellAbility sa) {
        for (SpellAbility s = sa; s != null; s = s.getSubAbility()) {
            if (s.usesTargeting()) {
                return true;
            }
        }
        return false;
    }

    private boolean setupTargetsRouted(SpellAbility sa) {
        boolean prev = ROUTE_TARGETS.get();
        ROUTE_TARGETS.set(Boolean.TRUE);
        try {
            return sa.setupTargets();
        } finally {
            ROUTE_TARGETS.set(prev);
        }
    }

    /**
     * One target slot at a time, kind {@code choose_target}: the engine's legal
     * candidates for this ability given the targets already picked
     * (TargetRestrictions.getAllCandidates(sa, false) applies SpellAbility.canTarget;
     * its boolean means players-only),
     * plus "choose no more targets" once the minimum is met. A slot with a
     * single forced candidate is not asked. Returns null to leave the ability
     * to Forge's AI -- divided, random and stack (spell/ability) targets --
     * which then decides all of it.
     */
    private Boolean chooseTargetsRouted(SpellAbility sa) {
        if (sa == null || !sa.usesTargeting()) {
            return null;
        }
        forge.game.spellability.TargetRestrictions tr = sa.getTargetRestrictions();
        if (tr == null || sa.isDividedAsYouChoose() || tr.isRandomTarget()
                || tr.isRandomNumTargets()
                || (tr.getZone() != null && tr.getZone().contains(forge.game.zone.ZoneType.Stack))) {
            return null;
        }
        forge.game.card.Card host = sa.getHostCard();
        int max = tr.getMaxTargets(host, sa);
        boolean asked = false;
        while (sa.getTargets().size() < max) {
            List<forge.game.GameEntity> cands = new java.util.ArrayList<>();
            for (forge.game.GameEntity e : tr.getAllCandidates(sa, false)) {
                if (!sa.getTargets().contains(e)) {
                    cands.add(e);
                }
            }
            boolean canStop = sa.isMinTargetChosen();
            if (cands.isEmpty()) {
                break;
            }
            if (cands.size() == 1 && !canStop) {
                if (!sa.getTargets().add(cands.get(0))) {
                    break;
                }
                continue;
            }
            List<Serializer.Opt> opts = targetOpts(sa, cands, canStop);
            int pick = askChoice("choose_target", opts);
            if (pick == ActionSource.DELEGATE) {
                if (asked) {
                    break;
                }
                sa.resetTargets();
                return null;
            }
            asked = true;
            int base = canStop ? 1 : 0;
            if (pick < base) {
                break;
            }
            sa.getTargets().add(cands.get(Math.min(pick - base, cands.size() - 1)));
        }
        return sa.isTargetNumberValid();
    }

    private List<Serializer.Opt> targetOpts(SpellAbility sa,
            List<forge.game.GameEntity> cands, boolean canStop) {
        Player me = getPlayer();
        forge.game.card.Card host = sa.getHostCard();
        String src = host != null && shownTo(host, me) ? host.getName() : "a hidden card";
        int slot = sa.getTargets().size() + 1;
        String what = src + " trigger" + (slot > 1 ? " (target " + slot + ")" : "");
        List<Serializer.Opt> opts = new java.util.ArrayList<>();
        if (canStop) {
            opts.add(new Serializer.Opt("choose no more targets for " + what));
        }
        for (forge.game.GameEntity e : cands) {
            if (e instanceof forge.game.card.Card) {
                forge.game.card.Card c = (forge.game.card.Card) e;
                boolean shown = shownTo(c, me);
                opts.add(new Serializer.Opt("target " + (shown ? c.getName() : "a hidden card")
                                + " with " + what)
                        .ref(shown ? StateExtractor.cardRef(c.getId()) : null)
                        .oracle(shown ? OracleIds.of(c) : null));
            } else if (e instanceof Player) {
                opts.add(new Serializer.Opt("target " + (e.equals(me) ? "yourself" : "opponent")
                                + " with " + what)
                        .ref(StateExtractor.entityRef(e)));
            } else {
                opts.add(new Serializer.Opt("target " + stripIds(String.valueOf(e))
                        + " with " + what));
            }
        }
        return opts;
    }

    /** An observe-mode choose_target record awaiting the AI's binding. */
    private final class ObservedTarget {
        final SpellAbility tgtSa;
        final List<forge.game.GameEntity> cands;
        final boolean canStop;
        final String body;

        ObservedTarget(SpellAbility tgtSa, List<forge.game.GameEntity> cands, boolean canStop,
                String body) {
            this.tgtSa = tgtSa;
            this.cands = cands;
            this.canStop = canStop;
            this.body = body;
        }

        void close() {
            int label = -1;
            try {
                forge.game.spellability.TargetChoices tc = tgtSa.getTargets();
                int n = tc == null ? 0 : tc.size();
                if (n == 0) {
                    label = canStop ? 0 : -1;
                } else if (n == 1) {
                    for (int i = 0; i < cands.size(); i++) {
                        if (tc.contains(cands.get(i))) {
                            label = (canStop ? 1 : 0) + i;
                            break;
                        }
                    }
                }
            } finally {
                closeChoice(body, label);
            }
        }
    }

    /**
     * The decide-mode decision that the AI's binding corresponds to, when that
     * is a single decision: exactly one targeting ability in the chain, one
     * target at most, a choice the routed path would ask (and not delegate).
     * Multi-slot and multi-ability targets are routed in decide mode but not
     * recorded here, since later slots' options depend on earlier picks.
     */
    private ObservedTarget openObservedTarget(SpellAbility sa) {
        if (sa.getApi() == forge.game.ability.ApiType.Charm || sa.hasParam("TargetingPlayer")) {
            return null;
        }
        SpellAbility tgtSa = null;
        for (SpellAbility s = sa; s != null; s = s.getSubAbility()) {
            if (s.usesTargeting()) {
                if (tgtSa != null) {
                    return null;
                }
                tgtSa = s;
            }
        }
        if (tgtSa == null) {
            return null;
        }
        forge.game.spellability.TargetRestrictions tr = tgtSa.getTargetRestrictions();
        if (tr == null || tgtSa.isDividedAsYouChoose() || tr.isRandomTarget()
                || tr.isRandomNumTargets() || tgtSa.hasParam("TargetingPlayer")
                || (tr.getZone() != null && tr.getZone().contains(forge.game.zone.ZoneType.Stack))) {
            return null;
        }
        forge.game.card.Card host = tgtSa.getHostCard();
        if (tr.getMaxTargets(host, tgtSa) != 1 || tgtSa.getTargets().size() != 0) {
            return null;
        }
        boolean canStop = tr.getMinTargets(host, tgtSa) == 0;
        List<forge.game.GameEntity> cands =
                new java.util.ArrayList<>(tr.getAllCandidates(tgtSa, false));
        if (cands.size() + (canStop ? 1 : 0) < 2) {
            return null;
        }
        String body = openChoice("choose_target", targetOpts(tgtSa, cands, canStop));
        return body == null ? null : new ObservedTarget(tgtSa, cands, canStop, body);
    }

    /**
     * Static triggers ("Static$ True") resolve immediately, without the stack:
     * TriggerHandler calls playTrigger, and the AI's version is prepareSingleSa
     * (which, for an optional one, also decides whether to use it) then
     * ComputerUtil.playNoStack. Routed: an optional one is first a
     * confirm_trigger decision; targets as for stack triggers.
     */
    @Override
    public boolean playTrigger(forge.game.card.Card host, forge.game.trigger.WrappedAbility wa,
            boolean isMandatory) {
        // Static triggers bypass the stack, so no cast event announces them: the
        // trigger event and the scope its consequences point at are opened here.
        HistoryHub hub = HistoryHub.of(getGame());
        if (hub != null && wa != null) {
            hub.beginStaticTrigger(host, wa);
        }
        boolean used = false;
        try {
            used = playTriggerInner(host, wa, isMandatory);
            return used;
        } finally {
            if (hub != null && wa != null) {
                hub.endStaticTrigger(used);
            }
        }
    }

    private boolean playTriggerInner(forge.game.card.Card host, forge.game.trigger.WrappedAbility wa,
            boolean isMandatory) {
        census("playTrigger");
        if (!routeChoices() || wa == null) {
            return super.playTrigger(host, wa, isMandatory);
        }
        List<Serializer.Opt> confirm = isMandatory ? null : java.util.Arrays.asList(
                new Serializer.Opt("decline"),
                new Serializer.Opt("accept: " + stripIds(String.valueOf(wa))));
        if (ACTIONS.decides()) {
            if (confirm != null && askChoice("confirm_trigger", confirm) == 0) {
                return false;
            }
            if (!prepareTrigger(wa, true)) {
                return false;
            }
            return forge.ai.ComputerUtil.playNoStack(wa.getActivatingPlayer(), wa, getGame(), true);
        }
        String body = confirm == null ? null : openChoice("confirm_trigger", confirm);
        // A declined optional trigger has no target decision in decide mode.
        ObservedTarget ot = isMandatory ? openObservedTarget(wa) : null;
        boolean used = false;
        try {
            used = super.playTrigger(host, wa, isMandatory);
            return used;
        } finally {
            closeChoice(body, used ? 1 : 0);
            if (ot != null) {
                ot.close();
            }
        }
    }

    /** Opt-in for the same reason as combat: it changes which decisions exist. */
    private static final boolean ROUTE_CHOICES = Boolean.getBoolean("magicnet.choices");

    private boolean routeChoices() {
        return ROUTE_CHOICES && (Serializer.enabled() || ACTIONS.needsRequest());
    }

}
