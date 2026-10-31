package magicnet.bridge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Offline validation of the payment mask against exhaustive enumeration.
 *
 * <p>This is test scaffolding, not inference infrastructure. It never runs in
 * the hot path; it exists only to check that {@link MaskCost#extendable} -- the
 * O(V*E) matching predicate that does ship -- accepts exactly those states from
 * which a complete payment exists.
 *
 * <p>The brute force deliberately shares no logic with the code under test. It
 * does not call {@code apply()} and does not use matching. It enumerates the
 * underlying model directly: choose an option per used source, then allocate
 * each produced mana to a coloured shard it can satisfy or to generic. A shared
 * helper would make a shared bug invisible, which is the usual way a
 * differential test quietly becomes worthless.
 *
 * <p>Two independent properties are checked.
 *
 * <ul>
 *   <li><b>A -- mask exactness.</b> For every reachable state,
 *       {@code extendable(residual, remaining)} must equal
 *       {@code truth(residual, remaining)}. Both are handed identical inputs, so
 *       a mismatch isolates the matching algorithm. A false positive walks the
 *       decoder into a dead end; a false negative silently deletes a legal
 *       payment from the policy's action space.
 *   <li><b>B -- transition completeness.</b> From every solvable state there
 *       must exist a pick whose successor <i>under {@code apply()}</i> is also
 *       solvable. This checks the state transition rather than the predicate:
 *       {@code apply()} chooses greedily which shard a mana pays, and a wrong
 *       greedy choice can strand a payment even when the mask is perfect.
 * </ul>
 *
 * <p>Boards are capped at {@link #MAX_SOURCES} because the brute force is
 * exponential. The cap binds only this test -- {@code extendable()} itself is
 * polynomial and runs uncapped. Coverage is bought with adversarial colour
 * structure (hybrids, multi-mana sources, tight Hall near-misses) rather than
 * with board size, since matching has no size-dependent branches.
 *
 * <p>Run: {@code java -cp build:... magicnet.bridge.PrefixOracle [trials] [seed]}
 */
public final class PrefixOracle {

    private PrefixOracle() {
    }

    /** Brute force is exponential in this; see class javadoc. */
    private static final int MAX_SOURCES = 5;

    private static final int COLOURLESS = 5;

    /** A payment requirement: coloured shard masks plus a generic count. */
    private static final class Need {
        final List<Byte> shards;
        final int generic;

        Need(List<Byte> shards, int generic) {
            this.shards = shards;
            this.generic = generic;
        }
    }

    /** One test board: a requirement and the sources available to pay it. */
    private static final class Board {
        final String name;
        final Need need;
        final List<PaymentCardinality.Src> srcs;

        Board(String name, Need need, List<PaymentCardinality.Src> srcs) {
            this.name = name;
            this.need = need;
            this.srcs = srcs;
        }
    }

    // ---------------------------------------------------------------- truth

    /**
     * Does some complete payment exist from the unused sources?
     *
     * <p>Exhaustive over (subset of sources) x (option per source) x
     * (allocation of each produced mana). No matching, no {@code apply()}.
     */
    private static boolean truth(List<Byte> shards, int generic,
            List<PaymentCardinality.Src> srcs, boolean[] used) {
        return pick(0, new int[6], shards, generic, srcs, used);
    }

    /** Recursively choose, for each unused source, to skip it or tap it. */
    private static boolean pick(int idx, int[] counts, List<Byte> shards,
            int generic, List<PaymentCardinality.Src> srcs, boolean[] used) {
        if (idx == srcs.size()) {
            return covers(counts, shards, 0, generic);
        }
        if (used[idx] && pick(idx + 1, counts, shards, generic, srcs, used)) {
            return true;
        }
        if (used[idx]) {
            return false;
        }
        // Skip this source: a payment need not tap everything.
        if (pick(idx + 1, counts, shards, generic, srcs, used)) {
            return true;
        }
        for (int[] opt : srcs.get(idx).options) {
            int c = opt[0];
            int amt = Math.max(1, opt[1]);
            if (c < 0 || c > COLOURLESS) {
                continue;
            }
            counts[c] += amt;
            boolean ok = pick(idx + 1, counts, shards, generic, srcs, used);
            counts[c] -= amt;
            if (ok) {
                return true;
            }
        }
        return false;
    }

    /**
     * Can this produced mana cover the requirement?
     *
     * <p>Each coloured shard needs one mana whose colour is in its mask; a
     * colourless shard (mask 0) needs colourless mana specifically. Anything
     * left over, of any colour, pays generic.
     */
    private static boolean covers(int[] counts, List<Byte> shards, int i, int generic) {
        if (i == shards.size()) {
            int left = 0;
            for (int c = 0; c <= COLOURLESS; c++) {
                left += counts[c];
            }
            return left >= generic;
        }
        byte mask = shards.get(i);
        for (int c = 0; c <= COLOURLESS; c++) {
            if (counts[c] == 0) {
                continue;
            }
            boolean fits = mask == 0
                    ? c == COLOURLESS
                    : c < 5 && (mask & (byte) (1 << c)) != 0;
            if (!fits) {
                continue;
            }
            counts[c]--;
            boolean ok = covers(counts, shards, i + 1, generic);
            counts[c]++;
            if (ok) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------ traversal

    private static long statesSeen;
    private static long maskFalsePos;
    private static long maskFalseNeg;
    private static long transitionDeadEnds;
    private static final List<String> FINDINGS = new ArrayList<>();

    /**
     * Visit every state reachable by the decoder and check both properties.
     *
     * <p>States are keyed by (used sources, residual shards, generic). That is
     * exactly the argument list {@code extendable()} receives, so two prefixes
     * with the same key are indistinguishable to the predicate and one
     * representative suffices -- the collapse is lossless for this test.
     */
    private static void walk(Board b, List<Byte> shards, int generic,
            boolean[] used, Set<String> seen, List<String> path) {
        String key = key(used, shards, generic);
        if (!seen.add(key)) {
            return;
        }
        statesSeen++;

        List<PaymentCardinality.Src> remaining = new ArrayList<>();
        for (int i = 0; i < b.srcs.size(); i++) {
            if (!used[i]) {
                remaining.add(b.srcs.get(i));
            }
        }

        boolean expect = truth(shards, generic, b.srcs, used);
        boolean got = MaskCost.extendable(new ArrayList<>(shards), generic, remaining);
        if (expect != got) {
            if (got) {
                maskFalsePos++;
            } else {
                maskFalseNeg++;
            }
            record(b, path, shards, generic,
                    got ? "MASK TOO LOOSE (dead end reachable)"
                        : "MASK TOO STRICT (legal payment deleted)");
        }

        boolean done = shards.isEmpty() && generic <= 0;
        if (done) {
            return;
        }

        // Property B: from a solvable state, some pick must stay solvable.
        boolean anyLiveSuccessor = false;

        for (int i = 0; i < b.srcs.size(); i++) {
            if (used[i]) {
                continue;
            }
            for (int[] opt : b.srcs.get(i).options) {
                List<Byte> rs = new ArrayList<>(shards);
                int rg = MaskCost.apply(rs, generic, opt[0], opt[1]);
                used[i] = true;
                if (truth(rs, rg, b.srcs, used)) {
                    anyLiveSuccessor = true;
                }
                path.add(label(i, opt));
                walk(b, rs, rg, used, seen, path);
                path.remove(path.size() - 1);
                used[i] = false;
            }
        }

        if (expect && !anyLiveSuccessor) {
            transitionDeadEnds++;
            record(b, path, shards, generic,
                    "TRANSITION DEAD END (apply() strands a feasible payment)");
        }
    }

    private static void record(Board b, List<String> path, List<Byte> shards,
            int generic, String what) {
        if (FINDINGS.size() >= 12) {
            return;
        }
        FINDINGS.add(String.format("  %-46s %s%n      board=%s%n"
                        + "      prefix=%s  residual=%s+%dgeneric",
                what, b.name, describe(b), path.isEmpty() ? "<empty>" : path,
                render(shards), generic));
    }

    // ------------------------------------------------------------- plumbing

    private static String key(boolean[] used, List<Byte> shards, int generic) {
        List<Byte> s = new ArrayList<>(shards);
        java.util.Collections.sort(s);
        return Arrays.toString(used) + s + "/" + Math.max(0, generic);
    }

    private static String label(int srcIdx, int[] opt) {
        return "s" + srcIdx + "->" + "WUBRGC".charAt(opt[0])
                + (Math.max(1, opt[1]) > 1 ? "x" + opt[1] : "");
    }

    private static String render(List<Byte> shards) {
        if (shards.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder();
        for (byte m : shards) {
            sb.append('{');
            if (m == 0) {
                sb.append('C');
            } else {
                for (int c = 0; c < 5; c++) {
                    if ((m & (1 << c)) != 0) {
                        sb.append("WUBRG".charAt(c));
                    }
                }
            }
            sb.append('}');
        }
        return sb.toString();
    }

    private static String describe(Board b) {
        StringBuilder sb = new StringBuilder();
        sb.append(render(b.need.shards)).append('+').append(b.need.generic)
                .append("generic from ");
        for (PaymentCardinality.Src s : b.srcs) {
            sb.append('[');
            for (int[] o : s.options) {
                sb.append("WUBRGC".charAt(o[0]));
                if (Math.max(1, o[1]) > 1) {
                    sb.append('x').append(o[1]);
                }
            }
            sb.append(']');
        }
        return sb.toString();
    }

    private static PaymentCardinality.Src src(int[]... options) {
        PaymentCardinality.Src s = new PaymentCardinality.Src();
        for (int[] o : options) {
            s.options.add(o);
        }
        return s;
    }

    private static int[] o(char colour, int amount) {
        return new int[] {"WUBRGC".indexOf(colour), amount};
    }

    private static byte mask(String colours) {
        byte m = 0;
        for (char c : colours.toCharArray()) {
            m |= (byte) (1 << "WUBRG".indexOf(c));
        }
        return m;
    }

    private static Need need(int generic, String... shards) {
        List<Byte> out = new ArrayList<>();
        for (String s : shards) {
            out.add(s.equals("C") ? (byte) 0 : mask(s));
        }
        return new Need(out, generic);
    }

    // --------------------------------------------------------------- boards

    /** Hand-built cases aimed at the colour structure where bugs actually live. */
    private static List<Board> curated() {
        List<Board> out = new ArrayList<>();

        // Hybrid ordering: greedy apply() may spend the Plains on the {W/U}.
        out.add(new Board("hybrid-order", need(0, "WU", "W"),
                Arrays.asList(src(o('W', 1)), src(o('U', 1)))));

        // One source producing two colourless against two {C} shards.
        out.add(new Board("sol-ring-CC", need(0, "C", "C"),
                Arrays.asList(src(o('C', 2)))));

        // Nykthos-style: one big green source against three {G}.
        out.add(new Board("nykthos-GGG", need(0, "G", "G", "G"),
                Arrays.asList(src(o('G', 1)), src(o('G', 3)))));

        // Steam Vents dead end from the worked example.
        out.add(new Board("steam-vents", need(2, "U", "U"),
                Arrays.asList(src(o('U', 1)), src(o('U', 1), o('R', 1)),
                        src(o('R', 1)), src(o('W', 1)))));

        // Tri-land against a tight three-colour cost.
        out.add(new Board("triland-WUB", need(0, "W", "U", "B"),
                Arrays.asList(src(o('W', 1), o('U', 1), o('B', 1)),
                        src(o('U', 1)), src(o('W', 1)))));

        // Hall near-miss: three shards, two blue sources.
        out.add(new Board("hall-miss", need(0, "U", "U", "U"),
                Arrays.asList(src(o('U', 1)), src(o('U', 1)), src(o('R', 1)))));

        // Colourless shard that coloured mana must not be allowed to pay.
        out.add(new Board("colourless-strict", need(0, "C"),
                Arrays.asList(src(o('R', 1)), src(o('G', 1)))));

        // Multi-mana coloured source covering a shard plus generic.
        out.add(new Board("surplus-to-generic", need(2, "R"),
                Arrays.asList(src(o('R', 3)))));

        return out;
    }

    /** Randomised boards over the same adversarial vocabulary. */
    private static Board random(Random rng, int n) {
        List<PaymentCardinality.Src> srcs = new ArrayList<>();
        int count = 2 + rng.nextInt(MAX_SOURCES - 1);
        for (int i = 0; i < count; i++) {
            PaymentCardinality.Src s = new PaymentCardinality.Src();
            int kind = rng.nextInt(10);
            if (kind < 4) {
                s.options.add(new int[] {rng.nextInt(5), 1});          // basic
            } else if (kind < 7) {
                int a = rng.nextInt(5);
                int b = (a + 1 + rng.nextInt(4)) % 5;
                s.options.add(new int[] {a, 1});                       // dual
                s.options.add(new int[] {b, 1});
            } else if (kind < 8) {
                for (int c = 0; c < 5; c++) {
                    s.options.add(new int[] {c, 1});                   // any colour
                }
            } else if (kind < 9) {
                s.options.add(new int[] {COLOURLESS, 1 + rng.nextInt(2)});
            } else {
                s.options.add(new int[] {rng.nextInt(5), 2});          // multi-mana
            }
            srcs.add(s);
        }
        List<Byte> shards = new ArrayList<>();
        int ns = rng.nextInt(4);
        for (int i = 0; i < ns; i++) {
            int kind = rng.nextInt(8);
            if (kind < 5) {
                shards.add((byte) (1 << rng.nextInt(5)));
            } else if (kind < 7) {
                int a = rng.nextInt(5);
                int b = (a + 1 + rng.nextInt(4)) % 5;
                shards.add((byte) ((1 << a) | (1 << b)));
            } else {
                shards.add((byte) 0);
            }
        }
        Board b = new Board("random#" + n, new Need(shards, rng.nextInt(3)), srcs);
        return b;
    }

    // ----------------------------------------------------------------- main

    public static void main(String[] args) {
        int trials = args.length > 0 ? Integer.parseInt(args[0]) : 4000;
        long seed = args.length > 1 ? Long.parseLong(args[1]) : 42L;
        Random rng = new Random(seed);

        List<Board> boards = new ArrayList<>(curated());
        int curatedCount = boards.size();
        for (int i = 0; i < trials; i++) {
            boards.add(random(rng, i));
        }

        Map<String, long[]> byClass = new HashMap<>();
        long t0 = System.nanoTime();
        for (Board b : boards) {
            long beforeLoose = maskFalsePos;
            long beforeStrict = maskFalseNeg;
            long beforeDead = transitionDeadEnds;
            walk(b, new ArrayList<>(b.need.shards), b.need.generic,
                    new boolean[b.srcs.size()], new HashSet<>(), new ArrayList<>());
            String cls = b.name.startsWith("random") ? "random" : b.name;
            long[] agg = byClass.computeIfAbsent(cls, k -> new long[3]);
            agg[0] += maskFalsePos - beforeLoose;
            agg[1] += maskFalseNeg - beforeStrict;
            agg[2] += transitionDeadEnds - beforeDead;
        }
        double ms = (System.nanoTime() - t0) / 1e6;

        System.out.println();
        System.out.println("=== payment prefix oracle ===");
        System.out.printf("boards      %d (%d curated, %d random, seed %d)%n",
                boards.size(), curatedCount, trials, seed);
        System.out.printf("states      %d checked in %.0f ms%n", statesSeen, ms);
        System.out.println();
        System.out.println("A  mask exactness");
        System.out.printf("     too loose (dead end reachable)      %d%n", maskFalsePos);
        System.out.printf("     too strict (legal payment deleted)  %d%n", maskFalseNeg);
        System.out.println("B  transition completeness");
        System.out.printf("     apply() strands feasible payment    %d%n",
                transitionDeadEnds);

        if (!FINDINGS.isEmpty()) {
            System.out.println();
            System.out.println("first findings:");
            for (String f : FINDINGS) {
                System.out.println(f);
            }
            System.out.println();
            System.out.println("by case:");
            for (Map.Entry<String, long[]> e : byClass.entrySet()) {
                long[] v = e.getValue();
                if (v[0] + v[1] + v[2] > 0) {
                    System.out.printf("  %-22s loose=%d strict=%d stranded=%d%n",
                            e.getKey(), v[0], v[1], v[2]);
                }
            }
        }
        System.out.println();
        boolean clean = maskFalsePos + maskFalseNeg + transitionDeadEnds == 0;
        System.out.println(clean ? "PASS" : "FAIL");
        System.exit(clean ? 0 : 1);
    }
}
