package magicnet.bridge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import forge.game.Game;
import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetRestrictions;

/**
 * Measures the cardinality of the *complete* legal action space at a priority
 * decision.
 *
 * <p>The candidate pool we already measure (mean 5.6-8.9 abilities) counts
 * abilities, not actions. A single ability expands into many distinct complete
 * actions once modes, X and target assignment are specified. This probe
 * estimates that expansion so we can decide between flat enumeration with an
 * attention head over actions and autoregressive factorization with
 * prefix-feasible masks.
 *
 * <p>Two numbers are reported per decision:
 * <ul>
 *   <li><b>raw</b> -- every distinct target assignment counted separately.</li>
 *   <li><b>dedup</b> -- target candidates collapsed into equivalence classes
 *       (see {@link #targetClass}), so that two identical untapped Goblin tokens
 *       contribute one choice rather than two. This is a sound, value-free
 *       reduction: it removes duplicates, not bad actions.</li>
 * </ul>
 *
 * <p>Everything here must be non-mutating. {@code canTarget}, {@code usesTargeting}
 * and the {@code TargetRestrictions} getters are pure. {@code setMaxXValue} is
 * NOT -- it calls {@code sa.setXManaCostPaid(...)} -- so the prior value is saved
 * and restored around the call.
 *
 * <p>Caveats, deliberately not papered over:
 * <ul>
 *   <li>Modes are not yet enumerated, so modal spells are undercounted.</li>
 *   <li>Payment plans (§13) are not enumerated, so consequential-payment
 *       branching is undercounted.</li>
 *   <li>Target legality is checked per-object; joint constraints across a
 *       multi-target assignment (maxTotalCMC, maxTotalPower, uniqueness beyond
 *       distinctness) are not applied, so multi-target counts are upper bounds.</li>
 * </ul>
 */
final class ActionCardinality {

    private ActionCardinality() {
    }

    /** Sample every Nth priority decision. 0 disables. */
    static final int EVERY = Integer.getInteger("magicnet.cardinality", 0);

    private static final java.util.concurrent.atomic.AtomicInteger TICK =
            new java.util.concurrent.atomic.AtomicInteger();

    /** Counts above this are clamped, and the decision is flagged saturated. */
    private static final long LIMIT = 10_000_000L;

    private static final List<Long> RAW = Collections.synchronizedList(new ArrayList<>());
    private static final List<Long> DEDUP = Collections.synchronizedList(new ArrayList<>());
    private static final List<Long> FLAT = Collections.synchronizedList(new ArrayList<>());
    private static final List<Long> PAY = Collections.synchronizedList(new ArrayList<>());
    private static final List<Long> PAYD = Collections.synchronizedList(new ArrayList<>());
    private static final List<Integer> POOL = Collections.synchronizedList(new ArrayList<>());

    /** Measure payment plans too (§13); off by default, it is not free. */
    private static final boolean PAY_ON =
            Integer.getInteger("magicnet.cardinality.pay", 0) > 0;

    private static volatile String worstPay = "(none)";
    private static volatile long worstPayCount = 0;

    private static final java.util.concurrent.atomic.AtomicInteger SATURATED =
            new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger TARGETING_SAS =
            new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger X_SAS =
            new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger MULTI_TARGET_SAS =
            new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.atomic.AtomicInteger MODAL_SAS =
            new java.util.concurrent.atomic.AtomicInteger();

    /** Worst single ability seen, for attributing the tail to specific cards. */
    private static volatile String worstAbility = "(none)";
    private static volatile long worstAbilityCount = 0;

    /** Worst flat-head ability, i.e. ignoring the target-set explosion. */
    private static volatile String worstFlat = "(none)";
    private static volatile long worstFlatCount = 0;

    private static final boolean DEBUG = Integer.getInteger("magicnet.cardinality.debug", 0) > 0;

    private static final java.util.concurrent.atomic.AtomicInteger DEBUG_SHOWN =
            new java.util.concurrent.atomic.AtomicInteger();

    static boolean shouldSample() {
        return EVERY > 0 && TICK.incrementAndGet() % EVERY == 0;
    }

    /**
     * Estimate the complete-action count for this decision.
     *
     * @param pool   the unfiltered candidate pool
     * @param game   live game, read-only here
     * @param me     the deciding player
     */
    static void measure(List<SpellAbility> pool, Game game, Player me) {
        if (pool == null || game == null || me == null) {
            return;
        }

        List<GameObject> universe = targetUniverse(game);

        long raw = 1;   // passing priority is always exactly one legal action
        long dedup = 1;
        long flat = 1;
        long pay = 1;
        long payd = 1;
        int playable = 0;

        for (SpellAbility sa : pool) {
            if (sa == null) {
                continue;
            }
            try {
                if (!sa.canPlay()) {
                    continue;
                }
            } catch (Throwable ignored) {
                continue;
            }
            playable++;

            long[] tgt = perAbility(sa, me, universe);
            long rawHere = tgt[0];
            long dedupHere = tgt[1];
            long flatHere = tgt[2];

            if (rawHere > worstAbilityCount) {
                worstAbilityCount = rawHere;
                worstAbility = describe(sa) + " (completions=" + rawHere
                        + ", dedup=" + dedupHere + ")";
            }
            if (flatHere > worstFlatCount) {
                worstFlatCount = flatHere;
                worstFlat = describe(sa) + " (flat=" + flatHere + ")";
            }

            if (PAY_ON) {
                long[] p = PaymentCardinality.plans(sa, me);
                if (p[0] > worstPayCount) {
                    worstPayCount = p[0];
                    worstPay = describe(sa) + " (plans=" + p[0] + ", dedup=" + p[1] + ")";
                }
                // Payment multiplies the flat head: every (ability, mode, X) tuple
                // can be paid in each of these ways.
                pay = sat(pay + sat(flatHere * p[0]));
                payd = sat(payd + sat(flatHere * p[1]));
            }

            raw = sat(raw + rawHere);
            dedup = sat(dedup + dedupHere);
            flat = sat(flat + flatHere);
        }

        if (raw >= LIMIT) {
            SATURATED.incrementAndGet();
        }
        RAW.add(raw);
        DEDUP.add(dedup);
        FLAT.add(flat);
        PAY.add(pay);
        PAYD.add(payd);
        POOL.add(playable);
    }

    /** Hard cap on enumerated X values, to bound pathological loops. */
    private static final int X_CAP = 20;

    private static final java.util.concurrent.atomic.AtomicInteger X_TRUNCATED =
            new java.util.concurrent.atomic.AtomicInteger();

    /**
     * Complete-action count for one ability, enumerating X and targets
     * <em>jointly</em>.
     *
     * <p>X and target legality are genuinely entangled: Indomitable Creativity
     * destroys <em>X</em> target permanents, so the number of targets to choose is
     * a function of X; March of Otherworldly Light exiles a permanent of mana
     * value X or less, so the target <em>set</em> grows with X. Computing the two
     * independently and multiplying is wrong in both directions. Instead we bind
     * each candidate X in turn and re-evaluate targeting under it, then sum.
     *
     * <p>Binding X mutates the ability, so the prior value is restored in a
     * {@code finally}.
     */
    private static long[] perAbility(SpellAbility sa, Player me, List<GameObject> universe) {
        boolean hasX;
        try {
            hasX = sa.costHasManaX();
        } catch (Throwable ignored) {
            hasX = false;
        }
        if (!hasX) {
            return modeAware(sa, universe);
        }

        X_SAS.incrementAndGet();
        int max = maxX(sa, me);
        if (max < 0) {
            return modeAware(sa, universe);
        }
        if (max > X_CAP) {
            max = X_CAP;
            X_TRUNCATED.incrementAndGet();
        }

        Integer saved;
        try {
            saved = sa.getXManaCostPaid();
        } catch (Throwable ignored) {
            return modeAware(sa, universe);
        }
        long raw = 0;
        long dedup = 0;
        long flat = 0;
        try {
            for (int x = 0; x <= max; x++) {
                sa.setXManaCostPaid(x);
                long[] t = modeAware(sa, universe);
                raw = sat(raw + t[0]);
                dedup = sat(dedup + t[1]);
                flat = sat(flat + t[2]);
            }
        } catch (Throwable ignored) {
            // partial sum still informative
        } finally {
            try {
                sa.setXManaCostPaid(saved);
            } catch (Throwable ignored) {
                // best effort
            }
        }
        return new long[] {raw, dedup, flat};
    }

    /** Max affordable X. {@code setMaxXValue} mutates, so we restore. */
    private static int maxX(SpellAbility sa, Player me) {
        Integer saved = null;
        boolean savedOk = false;
        try {
            saved = sa.getXManaCostPaid();
            savedOk = true;
            return forge.ai.ComputerUtilCost.setMaxXValue(sa, me, sa.isTrigger());
        } catch (Throwable ignored) {
            return -1;
        } finally {
            if (savedOk) {
                try {
                    sa.setXManaCostPaid(saved);
                } catch (Throwable ignored) {
                    // best effort
                }
            }
        }
    }

    /**
     * Number of distinct X values. {@code setMaxXValue} mutates, so we restore.
     */
    /**
     * Mode-aware choice count. For a non-modal ability this is just
     * {@link #targetChoices}. For a modal one ("choose one", "choose one or
     * more", "choose two"), the total is
     *
     * <pre>sum over k in [min,num] of sum over k-subsets S of prod_{m in S} targets(m)</pre>
     *
     * computed with a subset-product DP. A mode with no legal target contributes
     * a factor of 0 and so drops out automatically, which is the correct
     * behaviour -- an unchoosable mode is not a choice.
     *
     * <p>Read via {@code getAdditionalAbilityList("Choices")} rather than
     * {@code CharmEffect.makePossibleOptions}, because the latter calls
     * {@code AbilitySub.setSVar(...)} and would mutate live abilities.
     *
     * @return {raw, dedup}
     */
    private static long[] modeAware(SpellAbility sa, List<GameObject> universe) {
        List<forge.game.spellability.AbilitySub> choices;
        try {
            choices = sa.getAdditionalAbilityList("Choices");
        } catch (Throwable ignored) {
            return targetChoices(sa, universe);
        }
        if (choices == null || choices.isEmpty()) {
            return targetChoices(sa, universe);
        }

        MODAL_SAS.incrementAndGet();
        int n = choices.size();
        int num = intParam(sa, "CharmNum", 1);
        int min = intParam(sa, "MinCharmNum", num);
        if (num > n) {
            num = n;
        }
        if (min < 0) {
            min = 0;
        }
        if (min > num) {
            min = num;
        }

        long[] rawW = new long[n];
        long[] dedupW = new long[n];
        long[] flatW = new long[n];
        for (int i = 0; i < n; i++) {
            long[] t = targetChoices(choices.get(i), universe);
            rawW[i] = t[0];
            dedupW[i] = t[1];
            flatW[i] = t[2];
        }

        long raw = subsetProductSum(rawW, min, num);
        long dedup = subsetProductSum(dedupW, min, num);
        // With 0/1 weights this counts exactly the feasible mode-combinations.
        long flat = subsetProductSum(flatW, min, num);

        // The root ability may itself target in addition to its modes.
        long[] own = targetChoices(sa, universe);
        if (own[0] != 1 || own[1] != 1) {
            raw = sat(raw * own[0]);
            dedup = sat(dedup * own[1]);
            flat = sat(flat * own[2]);
        }
        return new long[] {raw, dedup, flat};
    }

    /** sum over k in [min,num] of sum over k-subsets S of prod_{i in S} w[i]. */
    private static long subsetProductSum(long[] w, int min, int num) {
        long[] dp = new long[num + 1];
        dp[0] = 1;
        for (long wi : w) {
            for (int j = num - 1; j >= 0; j--) {
                if (dp[j] != 0 && wi != 0) {
                    dp[j + 1] = sat(dp[j + 1] + sat(dp[j] * wi));
                }
            }
        }
        long total = 0;
        for (int k = Math.max(min, 1); k <= num; k++) {
            total = sat(total + dp[k]);
        }
        return total;
    }

    private static int intParam(SpellAbility sa, String key, int dflt) {
        try {
            if (!sa.hasParam(key)) {
                return dflt;
            }
            return Integer.parseInt(sa.getParam(key).trim());
        } catch (Throwable ignored) {
            return dflt;
        }
    }

    /**
     * @return {raw target assignments, dedup target assignments, feasible?}
     *     where the third element is 1 if at least one legal target assignment
     *     exists and 0 otherwise. It collapses the target-set choice, so summing
     *     it measures the "flat head": the number of
     *     {@code (ability, mode-combo, X)} tuples a model would have to choose
     *     among if target selection were delegated to a separate decoder.
     */
    private static long[] targetChoices(SpellAbility sa, List<GameObject> universe) {
        try {
            if (!sa.usesTargeting()) {
                return new long[] {1, 1, 1};
            }
        } catch (Throwable ignored) {
            return new long[] {1, 1, 1};
        }
        TARGETING_SAS.incrementAndGet();

        TargetRestrictions tr;
        try {
            tr = sa.getTargetRestrictions();
        } catch (Throwable ignored) {
            return new long[] {1, 1, 1};
        }
        if (tr == null) {
            return new long[] {1, 1, 1};
        }

        List<GameObject> legal = new ArrayList<>();
        for (GameObject g : universe) {
            try {
                if (sa.canTarget(g)) {
                    legal.add(g);
                }
            } catch (Throwable ignored) {
                // not targetable
            }
        }
        if (legal.isEmpty()) {
            if (DEBUG && DEBUG_SHOWN.incrementAndGet() <= 40) {
                System.out.println("[cardinality.dbg] ZERO targets: " + describe(sa)
                        + " universe=" + universe.size()
                        + " activator=" + safeActivator(sa)
                        + " valid=" + java.util.Arrays.toString(safeValidTgts(tr)));
            }
            return new long[] {0, 0, 0};
        }
        if (DEBUG && DEBUG_SHOWN.incrementAndGet() <= 40) {
            System.out.println("[cardinality.dbg] " + describe(sa)
                    + " legalTargets=" + legal.size()
                    + " universe=" + universe.size()
                    + " activator=" + safeActivator(sa)
                    + " valid=" + java.util.Arrays.toString(safeValidTgts(tr)));
        }

        int minT;
        int maxT;
        try {
            Card host = sa.getHostCard();
            minT = tr.getMinTargets(host, sa);
            maxT = tr.getMaxTargets(host, sa);
        } catch (Throwable ignored) {
            minT = 1;
            maxT = 1;
        }
        if (maxT <= 0) {
            maxT = 1;
        }
        if (minT < 0) {
            minT = 0;
        }
        int lo = Math.max(minT, 1);
        int hi = Math.min(maxT, legal.size());
        if (hi < lo) {
            return new long[] {0, 0, 0};
        }
        if (hi > lo) {
            MULTI_TARGET_SAS.incrementAndGet();
        }

        long raw = 0;
        for (int k = lo; k <= hi; k++) {
            raw = sat(raw + choose(legal.size(), k));
        }

        // Equivalence classes: multiplicities of indistinguishable targets.
        Map<String, Integer> classes = new HashMap<>();
        for (GameObject g : legal) {
            classes.merge(targetClass(g), 1, Integer::sum);
        }
        int[] mult = new int[classes.size()];
        int i = 0;
        for (Integer m : classes.values()) {
            mult[i++] = m;
        }
        long dedup = 0;
        long[] ways = boundedMultisetCounts(mult, hi);
        for (int k = lo; k <= hi; k++) {
            dedup = sat(dedup + ways[k]);
        }

        return new long[] {raw, dedup, raw > 0 ? 1 : 0};
    }

    /**
     * A canonical key for target equivalence. Two objects sharing a key are
     * interchangeable for the purpose of choosing a target.
     *
     * <p>This is intentionally conservative -- it only collapses objects that
     * agree on every attribute we represent. Over-collapsing would silently
     * delete real choices, which is far worse than under-collapsing.
     */
    private static String targetClass(GameObject g) {
        if (g instanceof Player) {
            return "P:" + ((Player) g).getName();
        }
        if (g instanceof Card) {
            Card c = (Card) g;
            StringBuilder sb = new StringBuilder(64);
            sb.append("C:").append(c.getName());
            try {
                sb.append('|').append(c.getZone() == null ? "?" : c.getZone().getZoneType());
                sb.append('|').append(c.getController() == null ? "?" : c.getController().getName());
                sb.append('|').append(c.isTapped() ? 'T' : 'u');
                sb.append('|').append(c.isSick() ? 'S' : '-');
                sb.append('|').append(c.getNetPower()).append('/').append(c.getNetToughness());
                sb.append('|').append(c.getDamage());
                sb.append('|').append(c.getCounters() == null ? "" : c.getCounters().toString());
                sb.append('|').append(c.isFaceDown() ? 'D' : '-');
                sb.append('|').append(c.getAttachedCards() == null
                        ? 0 : c.getAttachedCards().size());
                sb.append('|').append(c.getEnchantedBy() == null
                        ? 0 : c.getEnchantedBy().size());
            } catch (Throwable ignored) {
                // fall back to identity so we never over-collapse
                return "C!" + System.identityHashCode(c);
            }
            return sb.toString();
        }
        return "O:" + System.identityHashCode(g);
    }

    /**
     * ways[k] = number of size-k multisets drawn from classes with the given
     * multiplicities (items within a class are interchangeable).
     */
    private static long[] boundedMultisetCounts(int[] mult, int maxK) {
        long[] dp = new long[maxK + 1];
        dp[0] = 1;
        for (int m : mult) {
            long[] next = new long[maxK + 1];
            for (int j = 0; j <= maxK; j++) {
                if (dp[j] == 0) {
                    continue;
                }
                int top = Math.min(m, maxK - j);
                for (int c = 0; c <= top; c++) {
                    next[j + c] = sat(next[j + c] + dp[j]);
                }
            }
            dp = next;
        }
        return dp;
    }

    private static long choose(int n, int k) {
        if (k < 0 || k > n) {
            return 0;
        }
        if (k == 0 || k == n) {
            return 1;
        }
        int kk = Math.min(k, n - k);
        long r = 1;
        for (int i = 1; i <= kk; i++) {
            r = r * (n - kk + i) / i;
            if (r >= LIMIT) {
                return LIMIT;
            }
        }
        return r;
    }

    private static long sat(long v) {
        return v < 0 || v > LIMIT ? LIMIT : v;
    }

    private static List<GameObject> targetUniverse(Game game) {
        List<GameObject> out = new ArrayList<>();
        try {
            for (Card c : game.getCardsInGame()) {
                out.add(c);
            }
        } catch (Throwable ignored) {
            // partial universe still useful
        }
        try {
            out.addAll(game.getPlayers());
        } catch (Throwable ignored) {
            // ditto
        }
        return out;
    }

    private static String safeActivator(SpellAbility sa) {
        try {
            Player p = sa.getActivatingPlayer();
            return p == null ? "NULL" : p.getName();
        } catch (Throwable ignored) {
            return "?";
        }
    }

    private static String[] safeValidTgts(TargetRestrictions tr) {
        try {
            String[] v = tr.getValidTgts();
            return v == null ? new String[0] : v;
        } catch (Throwable ignored) {
            return new String[] {"?"};
        }
    }

    private static String describe(SpellAbility sa) {
        try {
            return String.valueOf(sa.getHostCard()) + " / " + sa.getApi();
        } catch (Throwable ignored) {
            return "?";
        }
    }

    private static long pct(List<Long> sorted, double p) {
        if (sorted.isEmpty()) {
            return 0;
        }
        int idx = (int) Math.ceil(p / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(idx, sorted.size() - 1)));
    }

    static void print() {
        if (EVERY <= 0) {
            return;
        }
        List<Long> raw;
        List<Long> dedup;
        List<Long> flat;
        List<Integer> pool;
        synchronized (RAW) {
            raw = new ArrayList<>(RAW);
        }
        synchronized (DEDUP) {
            dedup = new ArrayList<>(DEDUP);
        }
        synchronized (FLAT) {
            flat = new ArrayList<>(FLAT);
        }
        synchronized (POOL) {
            pool = new ArrayList<>(POOL);
        }
        if (raw.isEmpty()) {
            System.out.println("[cardinality] no samples");
            return;
        }
        Collections.sort(raw);
        Collections.sort(dedup);
        Collections.sort(flat);

        double rawMean = raw.stream().mapToLong(Long::longValue).average().orElse(0);
        double dedupMean = dedup.stream().mapToLong(Long::longValue).average().orElse(0);
        double flatMean = flat.stream().mapToLong(Long::longValue).average().orElse(0);
        double poolMean = pool.stream().mapToInt(Integer::intValue).average().orElse(0);

        System.out.println("[cardinality] samples=" + raw.size()
                + " sampledEvery=" + EVERY
                + " saturated=" + SATURATED.get() + " (clamp=" + LIMIT + ")");
        System.out.printf("[cardinality] playable abilities/decision: mean=%.2f%n", poolMean);
        System.out.printf(
                "[cardinality] RAW   complete actions: mean=%.1f p50=%d p90=%d p99=%d max=%d%n",
                rawMean, pct(raw, 50), pct(raw, 90), pct(raw, 99), raw.get(raw.size() - 1));
        System.out.printf(
                "[cardinality] DEDUP complete actions: mean=%.1f p50=%d p90=%d p99=%d max=%d%n",
                dedupMean, pct(dedup, 50), pct(dedup, 90), pct(dedup, 99),
                dedup.get(dedup.size() - 1));
        if (rawMean > 0) {
            System.out.printf("[cardinality] dedup/raw ratio (mean) = %.3f%n",
                    dedupMean / rawMean);
        }
        System.out.printf(
                "[cardinality] FLAT  (ability,mode,X) tuples: mean=%.1f p50=%d p90=%d p99=%d max=%d%n",
                flatMean, pct(flat, 50), pct(flat, 90), pct(flat, 99),
                flat.get(flat.size() - 1));
        System.out.println("[cardinality] worst flat ability: " + worstFlat
                + " -> " + worstFlatCount);
        if (PAY_ON) {
            List<Long> pay;
            List<Long> payd;
            synchronized (PAY) {
                pay = new ArrayList<>(PAY);
            }
            synchronized (PAYD) {
                payd = new ArrayList<>(PAYD);
            }
            Collections.sort(pay);
            Collections.sort(payd);
            double payMean = pay.stream().mapToLong(Long::longValue).average().orElse(0);
            double paydMean = payd.stream().mapToLong(Long::longValue).average().orElse(0);
            System.out.printf(
                    "[cardinality] PAY   flat x payment plans: mean=%.1f p50=%d p90=%d p99=%d max=%d%n",
                    payMean, pct(pay, 50), pct(pay, 90), pct(pay, 99),
                    pay.get(pay.size() - 1));
            System.out.printf(
                    "[cardinality] PAYD  flat x payment (dedup): mean=%.1f p50=%d p90=%d p99=%d max=%d%n",
                    paydMean, pct(payd, 50), pct(payd, 90), pct(payd, 99),
                    payd.get(payd.size() - 1));
            System.out.println("[cardinality] worst payment ability: " + worstPay
                    + " -> " + worstPayCount);
            System.out.println("[cardinality] payment limitations: "
                    + PaymentCardinality.LIMITATIONS);
        }
        System.out.println("[cardinality] targeting SAs=" + TARGETING_SAS.get()
                + " multi-target SAs=" + MULTI_TARGET_SAS.get()
                + " X SAs=" + X_SAS.get()
                + " modal SAs=" + MODAL_SAS.get()
                + " X-truncated=" + X_TRUNCATED.get() + " (cap=" + X_CAP + ")");
        System.out.println("[cardinality] worst single ability: " + worstAbility
                + " -> " + worstAbilityCount);
        System.out.println("[cardinality] NOTE: payment plans not enumerated"
                + " (undercount); multi-target joint constraints not applied (overcount).");
    }
}
