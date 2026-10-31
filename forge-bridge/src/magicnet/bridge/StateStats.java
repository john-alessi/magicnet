package magicnet.bridge;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Aggregates token-count statistics across states.
 *
 * The question this exists to answer is whether the state fits in a transformer
 * context at all, and if not, which zone is responsible. Means are nearly useless
 * for that -- a model has to handle the worst case, not the average -- so
 * everything here is reported as a distribution with tail percentiles.
 */
public final class StateStats {

    private static final List<Integer> TOTALS = Collections.synchronizedList(new ArrayList<>());
    private static final List<Integer> TOTALS_NO_HISTORY =
            Collections.synchronizedList(new ArrayList<>());
    private static final List<Integer> COLLAPSED =
            Collections.synchronizedList(new ArrayList<>());
    private static final List<Integer> EDGES = Collections.synchronizedList(new ArrayList<>());
    private static final Map<String, List<Integer>> BY_ZONE =
            Collections.synchronizedMap(new TreeMap<>());
    private static final List<Integer> UNCOMPRESSED_LIB =
            Collections.synchronizedList(new ArrayList<>());
    private static final List<Integer> HISTORY_AT_END =
            Collections.synchronizedList(new ArrayList<>());

    /** History length after run-length encoding consecutive identical events. */
    private static final List<Integer> HISTORY_RLE =
            Collections.synchronizedList(new ArrayList<>());

    /**
     * History length after collapsing repeated k-gram cycles, which is what an
     * infinite combo actually produces: not a run of one repeated event, but a
     * repeated multi-event sequence.
     */
    private static final List<Integer> HISTORY_CYCLE =
            Collections.synchronizedList(new ArrayList<>());

    private StateStats() {
    }

    public static void recordUncompressedLibrary(int n) {
        UNCOMPRESSED_LIB.add(n);
    }

    public static void recordHistoryAtGameEnd(int n) {
        HISTORY_AT_END.add(n);
    }

    /** Total hidden-card redactions applied across both viewers, per game. */
    private static final List<Integer> REDACTIONS =
            Collections.synchronizedList(new ArrayList<>());

    public static void recordRedactions(int n) {
        REDACTIONS.add(n);
    }

    public static int totalRedactions() {
        int t = 0;
        synchronized (REDACTIONS) {
            for (int n : REDACTIONS) {
                t += n;
            }
        }
        return t;
    }

    /**
     * Records one extracted state.
     *
     * The collapsed count models the compression available for fully exchangeable
     * objects: a wide board of identical tokens (say forty Squirrel tokens) is
     * replaced by one token carrying a count. Two tokens may only be collapsed if
     * they are identical in every visible property AND carry no edges -- an edge
     * makes an object individuated (it is the thing being targeted, or the thing
     * blocking), so collapsing it would destroy information the model needs.
     */
    public static void record(List<StateExtractor.Tok> toks) {
        int total = toks.size();
        int history = 0;
        int edges = 0;
        Map<String, Integer> zoneCounts = new HashMap<>();

        Set<String> collapsibleSigs = new HashSet<>();
        int collapsibleTokens = 0;
        Map<String, Integer> sigCounts = new HashMap<>();

        for (StateExtractor.Tok t : toks) {
            edges += t.edgeCount();
            boolean isHistory = "history".equals(t.type());
            if (isHistory) {
                history++;
            }
            String z = t.zone() == null ? t.type() : t.zone();
            zoneCounts.merge(z, 1, Integer::sum);

            // History is a SEQUENCE: two identical events at different points in
            // the game are not interchangeable, because their order is the
            // information. Collapsing them would be measuring a compression that
            // is not actually available. Only unordered board objects qualify.
            if (t.edgeCount() == 0 && !isHistory) {
                sigCounts.merge(t.signature(), 1, Integer::sum);
            }
        }

        for (Map.Entry<String, Integer> e : sigCounts.entrySet()) {
            if (e.getValue() > 1) {
                collapsibleSigs.add(e.getKey());
                collapsibleTokens += e.getValue();
            }
        }
        int collapsed = total - collapsibleTokens + collapsibleSigs.size();

        // History compression, measured two ways. Both are lossy in general but
        // near-lossless for a deterministic loop, whose net effect is already fully
        // observable in the board state (life total, token count) -- history exists
        // to carry what the board does not.
        List<String> hist = new ArrayList<>();
        for (StateExtractor.Tok t : toks) {
            if ("history".equals(t.type())) {
                hist.add(t.signature());
            }
        }
        if (!hist.isEmpty()) {
            HISTORY_RLE.add(rleLength(hist));
            HISTORY_CYCLE.add(cycleCollapsedLength(hist));
        }

        TOTALS.add(total);
        TOTALS_NO_HISTORY.add(total - history);
        COLLAPSED.add(collapsed);
        EDGES.add(edges);
        for (Map.Entry<String, Integer> e : zoneCounts.entrySet()) {
            BY_ZONE.computeIfAbsent(e.getKey(), k ->
                    Collections.synchronizedList(new ArrayList<>())).add(e.getValue());
        }
    }

    /** Length after collapsing runs of the same event into one token plus a count. */
    private static int rleLength(List<String> xs) {
        int n = 0;
        for (int i = 0; i < xs.size(); i++) {
            if (i == 0 || !xs.get(i).equals(xs.get(i - 1))) {
                n++;
            }
        }
        return n;
    }

    /**
     * Greedy collapse of repeated k-gram cycles, k = 1..8.
     *
     * A combo loop emits a repeating multi-event sequence (cast, flicker, return,
     * recast...), which plain RLE cannot see because no two adjacent events match.
     *
     * The repeat count is a *property* of the body tokens, not a token of its own --
     * same treatment as the interchangeable-object count. That matters: an earlier
     * version charged k+1 tokens per cycle, which made a simple run cost 2 tokens
     * where RLE charges 1, so cycle collapse scored *worse* than plain RLE. With
     * the count as a property, k=1 degenerates exactly to RLE and this is a strict
     * improvement by construction.
     *
     * Greedy selection maximises tokens *saved*, not span covered; those differ
     * whenever a long weak repetition overlaps a short strong one.
     */
    private static int cycleCollapsedLength(List<String> xs) {
        int i = 0;
        int n = 0;
        int sz = xs.size();
        while (i < sz) {
            int bestK = 1;
            int bestReps = 1;
            int bestSaved = 0;
            for (int k = 1; k <= 8 && i + 2 * k <= sz; k++) {
                int reps = 1;
                while (i + (reps + 1) * k <= sz && matches(xs, i, i + reps * k, k)) {
                    reps++;
                }
                if (reps > 1) {
                    int saved = (reps * k) - k;
                    if (saved > bestSaved) {
                        bestSaved = saved;
                        bestK = k;
                        bestReps = reps;
                    }
                }
            }
            // The cycle body costs k tokens; the repeat count rides as a property.
            n += bestK;
            i += bestK * bestReps;
        }
        return n;
    }

    private static boolean matches(List<String> xs, int a, int b, int k) {
        for (int j = 0; j < k; j++) {
            if (!xs.get(a + j).equals(xs.get(b + j))) {
                return false;
            }
        }
        return true;
    }

    private static int pct(List<Integer> xs, double p) {
        if (xs.isEmpty()) {
            return 0;
        }
        List<Integer> c = new ArrayList<>(xs);
        Collections.sort(c);
        int i = (int) Math.ceil(p / 100.0 * c.size()) - 1;
        return c.get(Math.max(0, Math.min(c.size() - 1, i)));
    }

    private static double mean(List<Integer> xs) {
        if (xs.isEmpty()) {
            return 0;
        }
        long s = 0;
        for (int x : xs) {
            s += x;
        }
        return (double) s / xs.size();
    }

    private static String row(String label, List<Integer> xs) {
        return String.format("  %-28s n=%-7d mean=%-8.1f p50=%-6d p90=%-6d p99=%-6d max=%d",
                label, xs.size(), mean(xs), pct(xs, 50), pct(xs, 90), pct(xs, 99),
                xs.isEmpty() ? 0 : Collections.max(xs));
    }

    public static void print() {
        if (TOTALS.isEmpty()) {
            System.out.println("\n=== token stats: no states recorded "
                    + "(enable with -Dmagicnet.tokens=1) ===");
            return;
        }
        System.out.println("\n=== state token counts ===");
        System.out.println(row("total (with history)", TOTALS));
        System.out.println(row("total (no history)", TOTALS_NO_HISTORY));
        System.out.println(row("collapsed (board only)", COLLAPSED));
        System.out.println(row("edges", EDGES));
        if (!UNCOMPRESSED_LIB.isEmpty()) {
            System.out.println(row("library cards (if expanded)", UNCOMPRESSED_LIB));
        }
        if (!HISTORY_AT_END.isEmpty()) {
            System.out.println(row("history tokens at game end", HISTORY_AT_END));
        }
        if (!HISTORY_RLE.isEmpty()) {
            System.out.println(row("history after RLE", HISTORY_RLE));
            System.out.println(row("history after cycle collapse", HISTORY_CYCLE));
        }

        System.out.println("\n=== by zone ===");
        synchronized (BY_ZONE) {
            for (Map.Entry<String, List<Integer>> e : BY_ZONE.entrySet()) {
                System.out.println(row(e.getKey(), e.getValue()));
            }
        }

        double saved = mean(TOTALS_NO_HISTORY) <= 0 ? 0
                : 100.0 * (1 - mean(COLLAPSED) / mean(TOTALS));
        System.out.printf("%nexchangeable-object collapse saves %.1f%% of tokens on average%n",
                saved);
    }

    public static void writeCsv(PrintWriter w) {
        w.println("metric,n,mean,p50,p90,p99,max");
        writeRow(w, "total_with_history", TOTALS);
        writeRow(w, "total_no_history", TOTALS_NO_HISTORY);
        writeRow(w, "collapsed", COLLAPSED);
        writeRow(w, "edges", EDGES);
        writeRow(w, "library_cards", UNCOMPRESSED_LIB);
        writeRow(w, "history_at_game_end", HISTORY_AT_END);
        writeRow(w, "history_rle", HISTORY_RLE);
        writeRow(w, "history_cycle_collapsed", HISTORY_CYCLE);
        synchronized (BY_ZONE) {
            for (Map.Entry<String, List<Integer>> e : BY_ZONE.entrySet()) {
                writeRow(w, "zone_" + e.getKey(), e.getValue());
            }
        }
    }

    private static void writeRow(PrintWriter w, String name, List<Integer> xs) {
        if (xs.isEmpty()) {
            return;
        }
        w.printf("%s,%d,%.2f,%d,%d,%d,%d%n", name, xs.size(), mean(xs),
                pct(xs, 50), pct(xs, 90), pct(xs, 99), Collections.max(xs));
    }
}
