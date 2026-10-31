package magicnet.bridge;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.lang.reflect.Array;

/**
 * Thread-safe recorder for PlayerController callbacks.
 *
 * Every generated override in InstrumentedController funnels here. We track, per
 * decision type: invocation count, wall time spent inside Forge's own AI, and the
 * branching factor (how many options the engine handed us).
 *
 * Branching factor is the number that matters for action-space design: it is the
 * size of the legal set the engine pre-filtered for us, so it bounds the width of
 * whatever head has to consume that decision type.
 */
public final class DecisionLog {

    private DecisionLog() {}

    public static final class Stat {
        final String method;
        final AtomicLong count = new AtomicLong();
        final AtomicLong nanos = new AtomicLong();
        final AtomicLong optionSum = new AtomicLong();
        final AtomicLong optionObs = new AtomicLong();
        final AtomicLong optionMax = new AtomicLong();
        // Branching-factor histogram, bucketed: 0,1,2,3,4,5-8,9-16,17-32,33-64,65+
        final AtomicLong[] buckets = new AtomicLong[10];

        Stat(String method) {
            this.method = method;
            for (int i = 0; i < buckets.length; i++) {
                buckets[i] = new AtomicLong();
            }
        }
    }

    private static final Map<String, Stat> STATS = new ConcurrentHashMap<>();
    private static final AtomicLong GAMES = new AtomicLong();
    private static volatile boolean enabled = true;

    /** Per-decision event trace. Only populated when traceLimit > 0. */
    private static final List<String> TRACE = new ArrayList<>();
    private static volatile int traceLimit = 0;

    public static void setEnabled(boolean v) { enabled = v; }
    public static void setTraceLimit(int n) { traceLimit = n; }
    public static void noteGameFinished() { flushGameDecisions(); GAMES.incrementAndGet(); }
    public static long games() { return Math.max(1L, GAMES.get()); }

    /**
     * Called from every generated override.
     *
     * @param method  unique key: name/arity
     * @param startNs nanoTime captured before the super call
     * @param player  the acting player (may be null during construction)
     * @param args    the raw callback arguments
     */
    public static void record(String method, long startNs, Object player, Object[] args) {
        if (schemaSampling) {
            sampleSchema(method, args);
        }
        if (!isBookkeeping(method)) {
            countDecision(player, false);
        }
        recordWithOptions(method, startNs, player, branchingFactor(args));
    }

    // ---------------------------------------------------------------
    // Schema sampling
    //
    // Most of a callback's shape is recoverable from its Java type signature, so
    // it needs no observation. Two things are not:
    //   - Map<String,Object> "params" bags, whose keys are untyped
    //   - discriminator Strings, whose value set is not expressed in the type
    // Enums are statically enumerable and recorded only for completeness.
    // ---------------------------------------------------------------

    private static volatile boolean schemaSampling = false;
    private static final int MAX_DISTINCT = 24;
    private static final Map<String, Map<Integer, java.util.Set<String>>> SCHEMA =
            new ConcurrentHashMap<>();

    public static void setSchemaSampling(boolean v) { schemaSampling = v; }

    private static void sampleSchema(String method, Object[] args) {
        if (args == null) {
            return;
        }
        Map<Integer, java.util.Set<String>> perArg =
                SCHEMA.computeIfAbsent(method, k -> new ConcurrentHashMap<>());
        for (int i = 0; i < args.length; i++) {
            java.util.Set<String> seen = perArg.computeIfAbsent(i,
                    k -> java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<>()));
            if (seen.size() >= MAX_DISTINCT) {
                continue;
            }
            try {
                seen.add(describe(args[i]));
            } catch (Throwable ignored) {
                // Never let instrumentation break a game.
            }
        }
    }

    private static String describe(Object o) {
        if (o == null) {
            return "null";
        }
        if (o instanceof Map<?, ?> m) {
            java.util.TreeSet<String> keys = new java.util.TreeSet<>();
            for (Object k : m.keySet()) {
                keys.add(String.valueOf(k));
            }
            return "Map{" + String.join("|", keys) + "}";
        }
        if (o instanceof String s) {
            String t = s.length() > 60 ? s.substring(0, 57) + "..." : s;
            return "\"" + t.replace("\n", "\\n").replace(",", ";") + "\"";
        }
        if (o instanceof Enum<?> e) {
            return e.getClass().getSimpleName() + "." + e.name();
        }
        if (o instanceof Boolean || o instanceof Integer || o instanceof Long) {
            return o.getClass().getSimpleName() + ":" + o;
        }
        if (o instanceof forge.game.spellability.SpellAbility sa) {
            Object api = null;
            try {
                api = sa.getApi();
            } catch (Throwable ignored) {
                // getApi may be unset on some abilities.
            }
            return "SpellAbility[api=" + api + "]";
        }
        if (o instanceof Collection<?> c) {
            return "Collection<" + elemType(c.iterator()) + ">";
        }
        if (o instanceof Iterable<?> it) {
            return o.getClass().getSimpleName() + "<" + elemType(it.iterator()) + ">";
        }
        if (o.getClass().isArray()) {
            return o.getClass().getSimpleName();
        }
        return o.getClass().getSimpleName();
    }

    private static String elemType(java.util.Iterator<?> it) {
        if (it.hasNext()) {
            Object f = it.next();
            return f == null ? "?" : f.getClass().getSimpleName();
        }
        return "empty";
    }

    public static void writeSchema(PrintWriter out) {
        out.println("method,arg_index,distinct_observed,values");
        List<String> methods = new ArrayList<>(SCHEMA.keySet());
        java.util.Collections.sort(methods);
        for (String m : methods) {
            Map<Integer, java.util.Set<String>> perArg = SCHEMA.get(m);
            List<Integer> idx = new ArrayList<>(perArg.keySet());
            java.util.Collections.sort(idx);
            for (Integer i : idx) {
                java.util.Set<String> vals = perArg.get(i);
                synchronized (vals) {
                    out.printf("%s,%d,%d,%s%n", m, i, vals.size(),
                            "\"" + String.join(" ", vals).replace("\"", "'") + "\"");
                }
            }
        }
        out.flush();
    }

    /** Callbacks carrying an untyped Map params bag -- the only truly opaque shape. */
    public static void printOpaqueParams() {
        System.out.println("\n=== untyped Map<String,Object> params bags (keys only knowable by observation) ===");
        List<String> methods = new ArrayList<>(SCHEMA.keySet());
        java.util.Collections.sort(methods);
        int n = 0;
        for (String m : methods) {
            for (Map.Entry<Integer, java.util.Set<String>> e : SCHEMA.get(m).entrySet()) {
                java.util.Set<String> vals = e.getValue();
                boolean isMap;
                synchronized (vals) {
                    isMap = vals.stream().anyMatch(v -> v.startsWith("Map{"));
                }
                if (isMap) {
                    synchronized (vals) {
                        System.out.printf("  %-40s arg%d  %s%n", m, e.getKey(), String.join(" ", vals));
                    }
                    n++;
                }
            }
        }
        if (n == 0) {
            System.out.println("  (none observed)");
        }
    }

    /** Non-decision hooks: UI plumbing, profile setup, shuffling, flags. */
    private static final Set<String> BOOKKEEPING = new HashSet<>(Arrays.asList(
            "canPlayUnlimitedLands", "isOrderedZone", "autoPassCancel", "pilotsNonAggroDeck",
            "awaitNextInput", "cancelAwaitNextInput", "setupAutoProfile", "cheatShuffle",
            "complainCardsCantPlayWell", "revealUnsupported", "revealAISkipCards",
            "notifyOfValue", "notifyStackAddition", "notifyStackRemoval", "reveal",
            "resetAtEndOfTurn", "nextGameDecision", "getManaPool", "useMana",
            "declareManaPool", "autoYieldUntilEnd", "isGuiPlayer", "getRandomSeed"
    ));

    public static boolean isBookkeeping(String method) {
        int slash = method.indexOf('/');
        return BOOKKEEPING.contains(slash < 0 ? method : method.substring(0, slash));
    }

    private static final AtomicLong PRIORITY_PASSES = new AtomicLong();
    private static final AtomicLong PRIORITY_CALLS = new AtomicLong();
    private static final AtomicLong SOURCE_SUM = new AtomicLong();
    private static final AtomicLong SOURCE_OBS = new AtomicLong();

    /**
     * Per-game, per-player decision counts -- the budget a chess-clock analogue
     * would actually meter.
     *
     * <p>The real MTGO clock is ~20 minutes of wall-clock per player, which is far
     * too long to allow per episode. To pick a decision budget that is non-binding
     * for ordinary games we need the distribution, not the mean: the question is
     * where the tail of legitimate games sits.
     *
     * <p>Counted per player because a clock is per player -- that is what makes it
     * self-attributing when one player loops.
     */
    private static final Map<String, long[]> GAME_DECISIONS = new ConcurrentHashMap<>();
    private static final List<long[]> GAME_SAMPLES =
            java.util.Collections.synchronizedList(new ArrayList<>());

    private static void countDecision(Object player, boolean priority) {
        if (player == null) {
            return;
        }
        long[] c = GAME_DECISIONS.computeIfAbsent(String.valueOf(player), k -> new long[2]);
        synchronized (c) {
            c[0]++;
            if (priority) {
                c[1]++;
            }
        }
    }

    private static void flushGameDecisions() {
        for (long[] c : GAME_DECISIONS.values()) {
            synchronized (c) {
                GAME_SAMPLES.add(new long[] { c[0], c[1] });
            }
        }
        GAME_DECISIONS.clear();
    }

    private static long pct(List<Long> sorted, double p) {
        if (sorted.isEmpty()) {
            return 0;
        }
        int i = (int) Math.ceil(p / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, i)));
    }

    public static void printDecisionBudget() {
        List<long[]> s = new ArrayList<>(GAME_SAMPLES);
        if (s.isEmpty()) {
            return;
        }
        List<Long> all = new ArrayList<>();
        List<Long> pri = new ArrayList<>();
        for (long[] c : s) {
            all.add(c[0]);
            pri.add(c[1]);
        }
        java.util.Collections.sort(all);
        java.util.Collections.sort(pri);
        System.out.printf("%n=== decision budget (per player per game, n=%d) ===%n", s.size());
        System.out.printf("%-12s %7s %7s %7s %7s %7s %7s%n",
                "", "min", "mean", "p50", "p90", "p99", "max");
        long sumAll = 0;
        for (long v : all) {
            sumAll += v;
        }
        long sumPri = 0;
        for (long v : pri) {
            sumPri += v;
        }
        System.out.printf("%-12s %7d %7d %7d %7d %7d %7d%n", "all calls",
                all.get(0), sumAll / all.size(), pct(all, 50), pct(all, 90),
                pct(all, 99), all.get(all.size() - 1));
        System.out.printf("%-12s %7d %7d %7d %7d %7d %7d%n", "priority",
                pri.get(0), sumPri / pri.size(), pct(pri, 50), pct(pri, 90),
                pct(pri, 99), pri.get(pri.size() - 1));
    }

    /**
     * Special-cased recording for chooseSpellAbilityToPlay, whose branching factor
     * must be reconstructed by the controller rather than read off the arguments.
     */
    public static void recordPriority(long startNs, Object player, int candidates, boolean passed) {
        PRIORITY_CALLS.incrementAndGet();
        if (passed) {
            PRIORITY_PASSES.incrementAndGet();
        }
        countDecision(player, true);
        recordWithOptions("chooseSpellAbilityToPlay/0", startNs, player, candidates);
    }

    public static void recordSources(int n) {
        SOURCE_SUM.addAndGet(n);
        SOURCE_OBS.incrementAndGet();
    }

    private static final AtomicLong TGT_ABILITIES = new AtomicLong();
    private static final AtomicLong TGT_PREBOUND = new AtomicLong();
    private static final AtomicLong TGT_BOUND_SUM = new AtomicLong();

    /** Records whether a chosen targeted ability already had its targets bound. */
    public static void recordTargeting(int boundTargets) {
        TGT_ABILITIES.incrementAndGet();
        if (boundTargets > 0) {
            TGT_PREBOUND.incrementAndGet();
            TGT_BOUND_SUM.addAndGet(boundTargets);
        }
    }

    public static void printTargetingProbe() {
        long n = TGT_ABILITIES.get();
        System.out.println("\n=== where does target binding happen on the AI path? ===");
        if (n == 0) {
            System.out.println("  no targeted abilities were chosen in this sample");
            return;
        }
        long pre = TGT_PREBOUND.get();
        System.out.printf("  targeted abilities chosen:        %d%n", n);
        System.out.printf("  already bound when returned:      %d (%.1f%%)%n", pre, 100.0 * pre / n);
        System.out.printf("  mean targets on those:            %.2f%n",
                pre == 0 ? 0.0 : (double) TGT_BOUND_SUM.get() / pre);
        Stat cf = STATS.get("chooseTargetsFor/1");
        System.out.printf("  chooseTargetsFor callbacks:       %d%n", cf == null ? 0 : cf.count.get());
        System.out.println(pre > 0
                ? "  => targets are folded INTO the action; measured branching factor is a lower bound"
                : "  => targets are chosen separately, via the callback");
    }

    private static final AtomicLong TGT_STRIPPED_SA = new AtomicLong();
    private static final AtomicLong TGT_STRIPPED_SUM = new AtomicLong();
    private static final AtomicLong CB_CALLS = new AtomicLong();
    private static final AtomicLong CB_OK = new AtomicLong();
    private static final AtomicLong CB_CAND_SUM = new AtomicLong();
    private static final AtomicLong CB_CAND_OBS = new AtomicLong();
    private static final AtomicLong CB_CAND_MAX = new AtomicLong();
    private static final AtomicLong CB_BOUND_SUM = new AtomicLong();
    private static final AtomicLong ENGINE_FAIL = new AtomicLong();
    private static final java.util.Map<String, AtomicLong> ENGINE_FAIL_KINDS =
            new java.util.concurrent.ConcurrentHashMap<>();

    public static void recordTargetsStripped(int n) {
        TGT_STRIPPED_SA.incrementAndGet();
        TGT_STRIPPED_SUM.addAndGet(n);
    }

    public static void recordTargetCallback(int candidates, int bound, boolean ok) {
        CB_CALLS.incrementAndGet();
        if (ok) {
            CB_OK.incrementAndGet();
        }
        CB_BOUND_SUM.addAndGet(bound);
        if (candidates >= 0) {
            CB_CAND_SUM.addAndGet(candidates);
            CB_CAND_OBS.incrementAndGet();
            CB_CAND_MAX.accumulateAndGet(candidates, Math::max);
        }
    }

    private static final AtomicLong ENGINE_CAST_OK = new AtomicLong();
    private static final AtomicLong ENGINE_CAST_FALSE = new AtomicLong();
    private static final java.util.Map<String, AtomicLong> ENGINE_FALSE_NAMES =
            new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Distinguishes "the engine refused the cast and rolled back" (returns false)
     * from "the engine threw" (recordEngineCastFailure). The AI path cannot return
     * false, so a false here is the rewind firing -- and the caller re-choosing the
     * same ability is what livelocks the phase handler.
     */
    private static final AtomicLong CIP_IDENTITY = new AtomicLong();
    private static final AtomicLong CIP_SIGNATURE = new AtomicLong();
    private static final AtomicLong CIP_ABSENT = new AtomicLong();
    private static final java.util.Map<String, AtomicLong> CIP_ABSENT_NAMES =
            new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Records whether the AI's chosen ability was addressable as a pool index.
     * Absent entries are the ones that matter: they are actions the policy could
     * never express, and games containing them cannot be replayed.
     */
    public static void recordChosenInPool(int byIdentity, int bySignature, String desc) {
        if (byIdentity >= 0) {
            CIP_IDENTITY.incrementAndGet();
        } else if (bySignature >= 0) {
            CIP_SIGNATURE.incrementAndGet();
        } else {
            CIP_ABSENT.incrementAndGet();
            CIP_ABSENT_NAMES.computeIfAbsent(String.valueOf(desc), k -> new AtomicLong())
                    .incrementAndGet();
        }
    }

    public static void printChosenInPool() {
        long id = CIP_IDENTITY.get();
        long sig = CIP_SIGNATURE.get();
        long no = CIP_ABSENT.get();
        long tot = id + sig + no;
        if (tot == 0) {
            return;
        }
        System.out.println("\n=== chosen-action addressability ===");
        System.out.printf("  chosen abilities:        %d%n", tot);
        System.out.printf("  found by identity:       %d (%.2f%%)%n", id, 100.0 * id / tot);
        System.out.printf("  found by signature only: %d (%.2f%%)%n", sig, 100.0 * sig / tot);
        System.out.printf("  NOT IN POOL:             %d (%.2f%%)%n", no, 100.0 * no / tot);
        if (no > 0) {
            System.out.println("  unaddressable actions (top 15):");
            CIP_ABSENT_NAMES.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                    .limit(15)
                    .forEach(e -> System.out.printf("      %6d  %s%n",
                            e.getValue().get(), e.getKey()));
        }
    }

    private static final String POOL_ORDER_FILE = System.getProperty("magicnet.poolorder", "");
    private static final AtomicLong POOL_ORDER_N = new AtomicLong();
    private static java.io.PrintWriter POOL_ORDER_OUT;

    /**
     * Dumps the ordered, content-addressed contents of the offered ability pool.
     *
     * The action space hands the policy an *index* into this list, so if the order
     * is not reproducible across runs at a fixed seed, recorded actions cannot be
     * replayed and the index-based action encoding is unsound. Described by host
     * card name plus ability text only -- never object identity, which is exactly
     * the thing suspected of varying.
     */
    public static synchronized void recordPoolOrder(java.util.List<forge.game.spellability.SpellAbility> pool) {
        if (POOL_ORDER_FILE.isEmpty() || pool == null) {
            return;
        }
        long idx = POOL_ORDER_N.getAndIncrement();
        if (idx >= 400) {
            return;
        }
        try {
            if (POOL_ORDER_OUT == null) {
                POOL_ORDER_OUT = new java.io.PrintWriter(new java.io.FileWriter(POOL_ORDER_FILE));
            }
            StringBuilder sb = new StringBuilder();
            sb.append(idx).append('\t').append(pool.size()).append('\t');
            for (forge.game.spellability.SpellAbility sa : pool) {
                String host = "?";
                String desc = "?";
                try {
                    host = sa.getHostCard() == null ? "?" : sa.getHostCard().getName();
                    desc = String.valueOf(sa.toString());
                } catch (Throwable ignored) {
                    // Diagnostic only.
                }
                sb.append('[').append(host).append('|').append(desc).append(']');
            }
            POOL_ORDER_OUT.println(sb);
            POOL_ORDER_OUT.flush();
        } catch (Throwable ignored) {
            // Diagnostic only.
        }
    }

    private static final AtomicLong ANNOUNCE_TOTAL = new AtomicLong();
    private static final AtomicLong ANNOUNCE_NULL = new AtomicLong();
    private static final java.util.Map<String, AtomicLong> ANNOUNCE_NULL_API =
            new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Tracks how often the AI controller declines to answer an announce (X, modes),
     * which on the engine path is indistinguishable from the player cancelling.
     */
    public static void recordAnnounce(forge.game.spellability.SpellAbility sa,
            String announce, boolean wasNull) {
        ANNOUNCE_TOTAL.incrementAndGet();
        if (!wasNull) {
            return;
        }
        ANNOUNCE_NULL.incrementAndGet();
        String api = "?";
        try {
            api = sa == null || sa.getApi() == null ? "null-api" : sa.getApi().toString();
        } catch (Throwable ignored) {
            // Diagnostic only.
        }
        ANNOUNCE_NULL_API.computeIfAbsent(api + "/" + announce, k -> new AtomicLong())
                .incrementAndGet();
    }

    public static void recordEngineCastResult(boolean ok, String name) {
        if (ok) {
            ENGINE_CAST_OK.incrementAndGet();
        } else {
            ENGINE_CAST_FALSE.incrementAndGet();
            ENGINE_FALSE_NAMES.computeIfAbsent(String.valueOf(name), k -> new AtomicLong())
                    .incrementAndGet();
        }
    }

    public static void recordEngineCastFailure(Throwable t) {
        ENGINE_FAIL.incrementAndGet();
        String kind = t.getClass().getName() + ": " + String.valueOf(t.getMessage());
        ENGINE_FAIL_KINDS.computeIfAbsent(kind, k -> new AtomicLong()).incrementAndGet();
    }

    public static void printTwoStageProbe() {
        System.out.println("\n=== two-stage action space probe ===");
        System.out.printf("  mode:                             %s%n",
                System.getProperty("magicnet.targetmode", "baseline"));
        System.out.printf("  abilities stripped of targets:    %d (%d targets removed)%n",
                TGT_STRIPPED_SA.get(), TGT_STRIPPED_SUM.get());
        long calls = CB_CALLS.get();
        System.out.printf("  chooseTargetsFor callbacks:       %d%n", calls);
        if (calls > 0) {
            System.out.printf("  returned true (targets legal):    %d (%.1f%%)%n",
                    CB_OK.get(), 100.0 * CB_OK.get() / calls);
            System.out.printf("  mean targets bound by callback:   %.2f%n",
                    (double) CB_BOUND_SUM.get() / calls);
            if (CB_CAND_OBS.get() > 0) {
                System.out.printf("  legal target candidates: mean %.2f  max %d  (n=%d)%n",
                        (double) CB_CAND_SUM.get() / CB_CAND_OBS.get(),
                        CB_CAND_MAX.get(), CB_CAND_OBS.get());
            }
        }
        if (ANNOUNCE_TOTAL.get() > 0) {
            System.out.printf("  announce callbacks: %d (%d AI declined -> defaulted)%n",
                    ANNOUNCE_TOTAL.get(), ANNOUNCE_NULL.get());
            ANNOUNCE_NULL_API.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                    .limit(10)
                    .forEach(e -> System.out.printf("      %6d  %s%n",
                            e.getValue().get(), e.getKey()));
        }
        if (ENGINE_CAST_OK.get() > 0 || ENGINE_CAST_FALSE.get() > 0) {
            long ok = ENGINE_CAST_OK.get();
            long no = ENGINE_CAST_FALSE.get();
            System.out.printf("  engine casts: %d ok, %d refused+rolled back (%.2f%%)%n",
                    ok, no, 100.0 * no / Math.max(1, ok + no));
            ENGINE_FALSE_NAMES.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                    .limit(10)
                    .forEach(e -> System.out.printf("      %6d  %s%n",
                            e.getValue().get(), e.getKey()));
        }
        if (ENGINE_FAIL.get() > 0) {
            System.out.printf("  engine-cast failures (fell back): %d%n", ENGINE_FAIL.get());
            ENGINE_FAIL_KINDS.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                    .limit(5)
                    .forEach(e -> System.out.printf("      %6d  %s%n", e.getValue().get(), e.getKey()));
        }
        if (TGT_STRIPPED_SA.get() > 0 && calls == 0) {
            System.out.println("  => targets were discarded and never requested back:"
                    + " no two-stage factorization on this path");
        } else if (calls > 0) {
            System.out.println("  => engine re-requests targets and supplies the legal set:"
                    + " two-stage factorization available");
        }
    }

    private static final AtomicLong PA_SAMPLES = new AtomicLong();
    private static final AtomicLong PA_NARROW = new AtomicLong();
    private static final AtomicLong PA_COSTONLY = new AtomicLong();
    private static final AtomicLong PA_WIDE = new AtomicLong();
    private static final AtomicLong PA_MISSING = new AtomicLong();
    private static final AtomicLong PA_MISSING_LEGAL = new AtomicLong();
    private static final java.util.Map<String, AtomicLong> PA_BY_ZONE =
            new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.Map<String, AtomicLong> PA_LEGAL_EXAMPLES =
            new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.Map<String, AtomicLong> PA_COST_EXAMPLES =
            new java.util.concurrent.ConcurrentHashMap<>();

    public static void recordPoolAudit(int narrow, int costOnly, int wide, int missing, int missingLegal) {
        PA_SAMPLES.incrementAndGet();
        PA_NARROW.addAndGet(narrow);
        PA_COSTONLY.addAndGet(costOnly);
        PA_WIDE.addAndGet(wide);
        PA_MISSING.addAndGet(missing);
        PA_MISSING_LEGAL.addAndGet(missingLegal);
    }

    public static void recordPoolGap(String kind, String zone, Object sa, boolean legalNow) {
        PA_BY_ZONE.computeIfAbsent(kind + " / " + zone, k -> new AtomicLong()).incrementAndGet();
        if ("cost-variant".equals(kind)) {
            bump(PA_COST_EXAMPLES, describeSa(sa));
        } else if (legalNow) {
            bump(PA_LEGAL_EXAMPLES, zone + "  " + describeSa(sa));
        }
    }

    private static void bump(java.util.Map<String, AtomicLong> m, String k) {
        if (m.size() < 4000 || m.containsKey(k)) {
            m.computeIfAbsent(k, x -> new AtomicLong()).incrementAndGet();
        }
    }

    private static String describeSa(Object sa) {
        try {
            forge.game.spellability.SpellAbility s = (forge.game.spellability.SpellAbility) sa;
            String host = s.getHostCard() == null ? "?" : s.getHostCard().getName();
            String d = String.valueOf(s.getDescription());
            if (d.length() > 70) {
                d = d.substring(0, 70) + "...";
            }
            return host + " :: " + s.getApi() + " :: " + (s.getPayCosts() == null ? "-" : s.getPayCosts().toString()) + " :: " + d;
        } catch (Throwable t) {
            return "?";
        }
    }

    public static void printPoolAudit() {
        long n = PA_SAMPLES.get();
        if (n == 0) {
            return;
        }
        System.out.println("\n=== ability pool audit: does the bot API hide legal plays? ===");
        System.out.printf("  sampled priority decisions:       %d%n", n);
        System.out.printf("  narrow pool (bot API):     mean %.2f%n", (double) PA_NARROW.get() / n);
        System.out.printf("  wide pool (engine):        mean %.2f%n", (double) PA_WIDE.get() / n);
        System.out.printf("  in wide, not in narrow:    mean %.2f%n", (double) PA_MISSING.get() / n);
        System.out.printf("  ...and legal right now:    mean %.2f  (total %d)%n",
                (double) PA_MISSING_LEGAL.get() / n, PA_MISSING_LEGAL.get());
        System.out.printf("  cost variants added by expansion: mean %.2f%n",
                (double) PA_COSTONLY.get() / n);

        System.out.println("\n  gaps by kind / zone:");
        PA_BY_ZONE.entrySet().stream()
                .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                .limit(15)
                .forEach(e -> System.out.printf("    %9d  %s%n", e.getValue().get(), e.getKey()));

        if (!PA_LEGAL_EXAMPLES.isEmpty()) {
            System.out.println("\n  LEGAL plays the bot API never exposes (top 25 by frequency):");
            PA_LEGAL_EXAMPLES.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                    .limit(25)
                    .forEach(e -> System.out.printf("    %7d  %s%n", e.getValue().get(), e.getKey()));
        } else {
            System.out.println("\n  no legal play was ever absent from the bot API pool");
        }

        if (!PA_COST_EXAMPLES.isEmpty()) {
            System.out.println("\n  cost variants only visible after expansion (top 15):");
            PA_COST_EXAMPLES.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
                    .limit(15)
                    .forEach(e -> System.out.printf("    %7d  %s%n", e.getValue().get(), e.getKey()));
        }
    }

    private static void recordWithOptions(String method, long startNs, Object player, int options) {        if (!enabled) {
            return;
        }
        long elapsed = System.nanoTime() - startNs;
        Stat s = STATS.computeIfAbsent(method, Stat::new);
        s.count.incrementAndGet();
        s.nanos.addAndGet(elapsed);

        if (options >= 0) {
            s.optionSum.addAndGet(options);
            s.optionObs.incrementAndGet();
            s.optionMax.accumulateAndGet(options, Math::max);
            s.buckets[bucketOf(options)].incrementAndGet();
        }

        if (traceLimit > 0) {
            synchronized (TRACE) {
                if (TRACE.size() < traceLimit) {
                    TRACE.add(method + "," + options + "," + (elapsed / 1000L) + "," + context(player));
                }
            }
        }
    }

    /**
     * Largest collection/array/map argument, which is the engine-supplied legal set
     * for this decision. Returns -1 when no argument carries a cardinality, i.e. the
     * decision is not a "pick from this list" shape.
     */
    private static int branchingFactor(Object[] args) {
        int best = -1;
        if (args == null) {
            return -1;
        }
        for (Object a : args) {
            if (a == null) {
                continue;
            }
            int n = -1;
            if (a instanceof Collection<?> c) {
                n = c.size();
            } else if (a instanceof Map<?, ?> m) {
                n = m.size();
            } else if (a.getClass().isArray()) {
                n = Array.getLength(a);
            } else if (a instanceof Iterable<?> it) {
                // Forge's FCollectionView / CardCollectionView are Iterable but not Collection.
                int k = 0;
                for (java.util.Iterator<?> i = it.iterator(); i.hasNext() && k < 10000; i.next()) {
                    k++;
                }
                n = k;
            }
            if (n > best) {
                best = n;
            }
        }
        return best;
    }

    private static int bucketOf(int n) {
        if (n <= 4) {
            return n;          // 0,1,2,3,4
        }
        if (n <= 8) {
            return 5;
        }
        if (n <= 16) {
            return 6;
        }
        if (n <= 32) {
            return 7;
        }
        if (n <= 64) {
            return 8;
        }
        return 9;
    }

    private static String context(Object player) {
        try {
            if (player == null) {
                return ",";
            }
            forge.game.player.Player p = (forge.game.player.Player) player;
            forge.game.Game g = p.getGame();
            if (g == null) {
                return ",";
            }
            return g.getPhaseHandler().getTurn() + "," + g.getPhaseHandler().getPhase();
        } catch (Exception e) {
            return ",";
        }
    }

    private static final String[] BUCKET_NAMES = {
        "n0", "n1", "n2", "n3", "n4", "n5_8", "n9_16", "n17_32", "n33_64", "n65plus"
    };

    public static void writeSummary(PrintWriter out) {
        List<Stat> all = new ArrayList<>(STATS.values());
        all.sort(Comparator.comparingLong((Stat s) -> s.count.get()).reversed());

        long total = 0;
        for (Stat s : all) {
            total += s.count.get();
        }
        long g = games();

        out.print("method,kind,count,pct_of_calls,per_game,total_ms,mean_us,mean_options,max_options");
        for (String b : BUCKET_NAMES) {
            out.print("," + b);
        }
        out.println();

        for (Stat s : all) {
            long c = s.count.get();
            long obs = s.optionObs.get();
            String meanOpt = obs == 0 ? "" : String.format("%.2f", (double) s.optionSum.get() / obs);
            String maxOpt = obs == 0 ? "" : Long.toString(s.optionMax.get());
            out.printf("%s,%s,%d,%.3f,%.2f,%.1f,%.1f,%s,%s",
                    s.method,
                    isBookkeeping(s.method) ? "bookkeeping" : "decision",
                    c,
                    total == 0 ? 0.0 : 100.0 * c / total,
                    (double) c / g,
                    s.nanos.get() / 1e6,
                    c == 0 ? 0.0 : s.nanos.get() / 1000.0 / c,
                    meanOpt,
                    maxOpt);
            for (AtomicLong b : s.buckets) {
                out.print("," + b.get());
            }
            out.println();
        }
        out.flush();
    }

    public static void writeTrace(PrintWriter out) {
        out.println("method,options,micros,turn,phase");
        synchronized (TRACE) {
            for (String t : TRACE) {
                out.println(t);
            }
        }
        out.flush();
    }

    public static void printTopToConsole(int n) {
        List<Stat> all = new ArrayList<>(STATS.values());
        all.sort(Comparator.comparingLong((Stat s) -> s.count.get()).reversed());

        List<Stat> decisions = new ArrayList<>();
        long totalDecision = 0;
        long totalBook = 0;
        for (Stat s : all) {
            if (isBookkeeping(s.method)) {
                totalBook += s.count.get();
            } else {
                decisions.add(s);
                totalDecision += s.count.get();
            }
        }
        long g = games();

        System.out.println();
        System.out.println("=== PlayerController decision callbacks (bookkeeping hooks excluded) ===");
        System.out.printf("distinct decision types: %d    decision calls: %d (%.1f/game)    "
                        + "bookkeeping calls filtered: %d%n",
                decisions.size(), totalDecision, (double) totalDecision / g, totalBook);

        long pc = PRIORITY_CALLS.get();
        if (pc > 0) {
            System.out.printf("chooseSpellAbilityToPlay: %.1f%% of calls were a pass "
                            + "(mean %.1f source cards enumerated)%n",
                    100.0 * PRIORITY_PASSES.get() / pc,
                    SOURCE_OBS.get() == 0 ? 0.0 : (double) SOURCE_SUM.get() / SOURCE_OBS.get());
        }
        System.out.println();
        System.out.printf("%-40s %9s %7s %9s %9s %9s %9s%n",
                "METHOD", "COUNT", "PCT", "PER_GAME", "MEAN_OPT", "MAX_OPT", "MEAN_US");

        int shown = 0;
        double cum = 0;
        for (Stat s : decisions) {
            if (shown++ >= n) {
                break;
            }
            long c = s.count.get();
            long obs = s.optionObs.get();
            double pct = totalDecision == 0 ? 0 : 100.0 * c / totalDecision;
            cum += pct;
            System.out.printf("%-40s %9d %6.2f%% %9.2f %9s %9s %9.1f%n",
                    s.method,
                    c,
                    pct,
                    (double) c / g,
                    obs == 0 ? "-" : String.format("%.1f", (double) s.optionSum.get() / obs),
                    obs == 0 ? "-" : Long.toString(s.optionMax.get()),
                    c == 0 ? 0.0 : s.nanos.get() / 1000.0 / c);
        }
        System.out.printf("%ntop-%d accounts for %.1f%% of decision callbacks%n",
                Math.min(n, decisions.size()), cum);

        // Branching-factor histogram for the main priority decision drives the
        // width of the action head, so surface it directly.
        Stat pri = STATS.get("chooseSpellAbilityToPlay/0");
        if (pri != null && pri.optionObs.get() > 0) {
            System.out.println("\n=== chooseSpellAbilityToPlay branching factor (candidate pool) ===");
            long obs = pri.optionObs.get();
            long cumc = 0;
            for (int i = 0; i < BUCKET_NAMES.length; i++) {
                long v = pri.buckets[i].get();
                cumc += v;
                System.out.printf("  %-8s %8d  %5.1f%%   cum %5.1f%%%n",
                        BUCKET_NAMES[i], v, 100.0 * v / obs, 100.0 * cumc / obs);
            }
        }
    }

    public static int distinctFired() {
        return STATS.size();
    }

    // ---- Affordability audit ------------------------------------------------

    private static final java.util.List<int[]> AFFORD =
            java.util.Collections.synchronizedList(new java.util.ArrayList<>());

    public static void recordAffordability(int total, int timingOk, int payable) {
        AFFORD.add(new int[]{total, timingOk, payable});
    }

    public static void printAffordability() {
        if (AFFORD.isEmpty()) {
            return;
        }
        long t = 0;
        long k = 0;
        long p = 0;
        int zeroPayable = 0;
        int zeroTiming = 0;
        synchronized (AFFORD) {
            for (int[] a : AFFORD) {
                t += a[0];
                k += a[1];
                p += a[2];
                if (a[2] == 0) {
                    zeroPayable++;
                }
                if (a[1] == 0) {
                    zeroTiming++;
                }
            }
        }
        int n = AFFORD.size();
        System.out.println();
        System.out.println("=== action pool affordability ===");
        System.out.printf("priority decisions sampled: %d%n", n);
        System.out.printf("mean candidates (unfiltered): %.1f%n", (double) t / n);
        System.out.printf("mean passing canPlay():       %.1f  (%.1f%% of pool)%n",
                (double) k / n, 100.0 * k / Math.max(1, t));
        System.out.printf("mean also payable:            %.1f  (%.1f%% of pool)%n",
                (double) p / n, 100.0 * p / Math.max(1, t));
        System.out.printf("decisions with 0 playable:    %d (%.1f%%)%n",
                zeroTiming, 100.0 * zeroTiming / n);
        System.out.printf("decisions with 0 payable:     %d (%.1f%%)  <- true forced-pass rate%n",
                zeroPayable, 100.0 * zeroPayable / n);
    }

    private static int chosenTotal;
    private static int chosenTimingFail;
    private static int chosenPayFail;
    private static final java.util.Map<String, Integer> CHOSEN_ALL =
            new java.util.concurrent.ConcurrentHashMap<>();
    private static final java.util.Map<String, Integer> CHOSEN_FAIL =
            new java.util.concurrent.ConcurrentHashMap<>();

    public static synchronized void recordChosenLegality(boolean timing, boolean payable,
                                                         String desc) {
        chosenTotal++;
        CHOSEN_ALL.merge(desc.split(" :: ")[0], 1, Integer::sum);
        if (!timing) {
            chosenTimingFail++;
            CHOSEN_FAIL.merge("TIMING  " + desc, 1, Integer::sum);
        } else if (!payable) {
            chosenPayFail++;
            CHOSEN_FAIL.merge("PAYABLE " + desc, 1, Integer::sum);
        }
    }

    public static synchronized void printFilterSoundness() {
        if (chosenTotal == 0) {
            return;
        }
        System.out.println();
        System.out.println("=== filter soundness (abilities the AI actually chose) ===");
        System.out.printf("chosen abilities sampled: %d%n", chosenTotal);
        System.out.printf("rejected by canPlay():    %d (%.2f%%)%n",
                chosenTimingFail, 100.0 * chosenTimingFail / chosenTotal);
        System.out.printf("rejected by canPayMana(): %d (%.2f%%)%n",
                chosenPayFail, 100.0 * chosenPayFail / chosenTotal);
        if (chosenTimingFail + chosenPayFail == 0) {
            System.out.println("ok: no false negatives observed");
            System.out.println("cards exercised (what the predicate was actually tested on):");
            CHOSEN_ALL.entrySet().stream()
                    .sorted((a, b) -> b.getValue() - a.getValue())
                    .limit(20)
                    .forEach(e -> System.out.printf("  %4dx %s%n", e.getValue(), e.getKey()));
        } else {
            System.out.println("UNSOUND -- these would be masked away from the policy:");
            CHOSEN_FAIL.entrySet().stream()
                    .sorted((a, b) -> b.getValue() - a.getValue())
                    .limit(15)
                    .forEach(e -> System.out.printf("  %4dx %s%n", e.getValue(), e.getKey()));
        }
    }
}
