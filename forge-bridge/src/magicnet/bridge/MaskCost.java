package magicnet.bridge;

import java.util.ArrayList;
import java.util.List;

import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.game.card.Card;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

/**
 * Measures the cost of <em>prefix masking</em> for the payment decoder.
 *
 * <p>An autoregressive payment decoder picks one mana source at a time. For the
 * mask to be exact, at each step and for each candidate source we must answer:
 * <em>if I commit this source to this colour, is the residual cost still payable
 * from the sources that remain?</em> That predicate is what makes legal dead ends
 * unreachable, and its cost is what decides whether masked decoding is affordable.
 *
 * <p>Two predicates are timed:
 *
 * <ul>
 *   <li><b>own</b> -- bipartite matching of residual shards against remaining
 *       sources, plus a capacity check for generic. Pure, and has the right
 *       semantics because it is told which sources are still unspent.
 *   <li><b>forge</b> -- {@code ComputerUtilMana.canPayManaCost(ManaCostBeingPaid,
 *       sa, player, effect)}. Timed for reference, but <b>semantically wrong for
 *       this purpose</b>: it re-derives available sources from the player, so it
 *       cannot know that part of the mana base is already committed to the prefix.
 *       It answers "payable from scratch", not "payable from what is left", and so
 *       over-reports feasibility. Recorded here to quantify both the latency and
 *       the disagreement rate.
 * </ul>
 *
 * <p>Also acts as a mutation test: battlefield tap state, mana-pool size and the
 * ability's own flags are snapshotted before and after, because
 * {@code payManaCost} (which Forge's predicate delegates to) contains
 * {@code ManaPool.payManaCostFromPool}, {@code SpellAbility.setSkip} and
 * {@code MagicStack.addAndUnfreeze} behind branches this probe does not try to
 * prove unreachable by reading bytecode.
 *
 * <p>Enabled with {@code -Dmagicnet.maskcost=N} (every Nth decision).
 */
public final class MaskCost {

    private MaskCost() {
    }

    static final int EVERY = Integer.getInteger("magicnet.maskcost", 0);

    private static final boolean DEBUG =
            Integer.getInteger("magicnet.maskcost.debug", 0) > 0;

    private static final int SOURCE_CAP = 28;

    /** Per-action totals. */
    private static final List<Integer> CALLS = new ArrayList<>();
    private static final List<Long> OWN_NS = new ArrayList<>();
    private static final List<Long> FORGE_NS = new ArrayList<>();
    private static final List<Integer> WIDTH = new ArrayList<>();
    private static final List<Integer> STEPS = new ArrayList<>();

    private static long ownCalls;
    private static long forgeCalls;
    private static long disagree;
    private static long forgeOverReports;
    private static long ownOverReports;
    private static long zeroAgree;
    private static long zeroForgeYesOwnNo;
    private static long zeroOwnYesForgeNo;
    /** Forge-declines-but-rules-allow: man-land self-tap, see selfTappableHost. */
    private static long zeroManlandExempt;
    /** Payment decisions reached with mana already floating (see measure). */
    private static long zeroPoolNonEmpty;
    /** Mana abilities offered as standalone choices at priority. */
    private static final java.util.concurrent.atomic.AtomicLong MANA_AT_PRIORITY =
            new java.util.concurrent.atomic.AtomicLong();

    static void noteManaAbilityOffered() {
        MANA_AT_PRIORITY.incrementAndGet();
    }

    /** Mana sources the AI helper offers that engine-side enumeration does not. */
    private static long srcAiOnly;
    /** Mana sources engine-side enumeration finds that the AI helper omits. */
    private static long srcEngineOnly;
    private static long srcSamples;

    /**
     * Compares the AI helper's mana-source list against enumerating the battlefield
     * directly. {@code ComputerUtilMana.getAvailableManaSources} lives in forge.ai,
     * so it is free to prune for the AI's convenience; the engine-side equivalent is
     * every controlled permanent with a mana ability that {@code canPlay()}. If the
     * two agree, replacing our shim with the raw engine API is a lateral move rather
     * than a simplification.
     */
    private static void auditManaSources(Player me) {
        try {
            java.util.Set<Integer> ai = new java.util.HashSet<>();
            for (Card c : forge.ai.ComputerUtilMana.getAvailableManaSources(me, true)) {
                if (c != null) {
                    ai.add(c.getId());
                }
            }
            java.util.Set<Integer> eng = new java.util.HashSet<>();
            for (Card c : me.getCardsIn(forge.game.zone.ZoneType.Battlefield)) {
                if (c == null) {
                    continue;
                }
                for (SpellAbility ma : c.getManaAbilities()) {
                    boolean ok;
                    try {
                        ok = ma != null && ma.canPlay();
                    } catch (Throwable t) {
                        ok = false;
                    }
                    if (ok) {
                        eng.add(c.getId());
                        break;
                    }
                }
            }
            srcSamples++;
            for (Integer id : ai) {
                if (!eng.contains(id)) {
                    srcAiOnly++;
                }
            }
            for (Integer id : eng) {
                if (!ai.contains(id)) {
                    srcEngineOnly++;
                }
            }
        } catch (Throwable ignored) {
            // best-effort instrumentation
        }
    }

    private static long mutations;
    private static int dbg;
    private static int zdbg;

    private static final java.util.concurrent.atomic.AtomicInteger SEEN =
            new java.util.concurrent.atomic.AtomicInteger();

    static boolean due() {
        return EVERY > 0 && SEEN.incrementAndGet() % EVERY == 0;
    }

    /**
     * Simulates a masked autoregressive payment decode for one ability and
     * records the cost of the mask.
     */
    static void measure(SpellAbility sa, Player me) {
        ManaCost cost;
        int amountOfX;
        try {
            if (sa.getPayCosts() == null || sa.getPayCosts().getCostMana() == null) {
                return;
            }
            cost = sa.getPayCosts().getCostMana().getMana();
            amountOfX = sa.getPayCosts().getCostMana().getAmountOfX();
        } catch (Throwable ignored) {
            return;
        }
        if (cost == null || cost.isZero()) {
            return;
        }

        auditManaSources(me);

        List<PaymentCardinality.Src> srcs;
        try {
            srcs = PaymentCardinality.sources(me, sa);
        } catch (Throwable ignored) {
            return;
        }
        if (srcs == null || srcs.isEmpty() || srcs.size() > SOURCE_CAP) {
            return;
        }

        // Residual requirement: coloured shard masks + generic.
        List<Byte> shards = new ArrayList<>();
        int generic;
        try {
            generic = cost.getGenericCost();
            for (ManaCostShard s : cost) {
                // See PaymentCardinality: {X} iterates as a mask-0 shard and would
                // be double-counted against the amountOfX * X generic added below.
                if (s != null && s != ManaCostShard.X) {
                    shards.add(s.getColorMask());
                }
            }
            if (amountOfX > 0) {
                Integer x = sa.getXManaCostPaid();
                if (x != null && x > 0) {
                    generic += amountOfX * x;
                }
            }
        } catch (Throwable ignored) {
            return;
        }
        if (shards.isEmpty() && generic <= 0) {
            return;
        }

        // Does the player already have mana floating? Our requirement is built from
        // the printed cost alone, so a non-empty pool would make the mask too
        // strict. Counted to tell a live gap from a theoretical one.
        try {
            int pooled = 0;
            for (Object ignored : me.getManaPool()) {
                pooled++;
            }
            if (pooled > 0) {
                zeroPoolNonEmpty++;
            }
        } catch (Throwable ignored) {
            // best-effort
        }

        String before = snapshot(sa, me);

        ManaCostBeingPaid residualForForge;
        try {
            residualForForge = new ManaCostBeingPaid(cost);
            if (amountOfX > 0) {
                Integer x = sa.getXManaCostPaid();
                if (x != null && x > 0) {
                    residualForForge.increaseGenericMana(amountOfX * x);
                }
            }
        } catch (Throwable ignored) {
            return;
        }

        // Empty-prefix differential test. With nothing committed, "payable from the
        // remaining sources" and "payable from scratch" are the SAME question, so
        // the two predicates must agree exactly. Any disagreement here is a bug in
        // the own predicate, not a semantic gap.
        try {
            boolean ownFull = extendable(new ArrayList<>(shards), generic,
                    new ArrayList<>(srcs));
            boolean forgeFull = forge.ai.ComputerUtilMana.canPayManaCost(
                    new ManaCostBeingPaid(residualForForge), sa, me, false);
            if (ownFull == forgeFull) {
                zeroAgree++;
            } else if (forgeFull) {
                zeroForgeYesOwnNo++;
                if (DEBUG && zdbg < 15) {
                    zdbg++;
                    System.out.println("[maskcost] ZERO-PREFIX own=NO forge=YES "
                            + sa.getHostCard() + " cost=" + cost
                            + " shards=" + shards.size() + " generic=" + generic
                            + " sources=" + srcs.size() + " " + renderSrcs(srcs)
                            + " board=" + availNames(me) + " WHY " + diagnose(me, sa));
                }
            } else {
                if (selfTappableHost(sa, me) || lifeCostManaAbility(me)) {
                    zeroManlandExempt++;
                } else {
                    zeroOwnYesForgeNo++;
                }
                if (DEBUG && zdbg < 15) {
                    zdbg++;
                    System.out.println("[maskcost] ZERO-PREFIX own=YES forge=NO "
                            + sa.getHostCard() + " cost=" + cost
                            + " shards=" + shards.size() + " generic=" + generic
                            + " sources=" + srcs.size() + " " + renderSrcs(srcs)
                            + " board=" + availNames(me) + " WHY " + diagnose(me, sa));
                }
            }
        } catch (Throwable ignored) {
            // not counted
        }

        List<PaymentCardinality.Src> remaining = new ArrayList<>(srcs);
        int calls = 0;
        int steps = 0;
        int maxWidth = 0;
        long ownNs = 0;
        long forgeNs = 0;

        // Decode until the cost is covered or we run out of sources.
        while ((!shards.isEmpty() || generic > 0) && !remaining.isEmpty()) {
            steps++;
            int width = 0;
            int chosenSrc = -1;
            int chosenColor = -1;

            for (int i = 0; i < remaining.size(); i++) {
                PaymentCardinality.Src s = remaining.get(i);
                for (int[] opt : s.options) {
                    width++;
                    calls++;

                    // Candidate: commit source i producing colour opt[0].
                    List<Byte> rs = new ArrayList<>(shards);
                    int rg = generic;
                    int applied = apply(rs, rg, opt[0], opt[1]);
                    rg = applied;

                    List<PaymentCardinality.Src> rem2 = new ArrayList<>(remaining);
                    rem2.remove(i);

                    long t0 = System.nanoTime();
                    boolean okOwn = extendable(rs, rg, rem2);
                    ownNs += System.nanoTime() - t0;
                    ownCalls++;

                    boolean okForge = okOwn;
                    try {
                        ManaCostBeingPaid probe = new ManaCostBeingPaid(residualForForge);
                        applyMcbp(probe, opt[0], opt[1]);
                        long t1 = System.nanoTime();
                        okForge = forge.ai.ComputerUtilMana.canPayManaCost(
                                probe, sa, me, false);
                        forgeNs += System.nanoTime() - t1;
                        forgeCalls++;
                    } catch (Throwable ignored) {
                        // leave okForge aligned; latency simply not recorded
                    }
                    if (okOwn != okForge) {
                        disagree++;
                        if (okForge) {
                            forgeOverReports++;
                        } else {
                            ownOverReports++;
                        }
                    }

                    if (okOwn && chosenSrc < 0) {
                        chosenSrc = i;
                        chosenColor = opt[0];
                    }
                }
            }
            maxWidth = Math.max(maxWidth, width);

            if (chosenSrc < 0) {
                break; // dead end: no extendable candidate
            }
            int[] pick = null;
            for (int[] opt : remaining.get(chosenSrc).options) {
                if (opt[0] == chosenColor) {
                    pick = opt;
                    break;
                }
            }
            generic = apply(shards, generic, chosenColor, pick == null ? 1 : pick[1]);
            applyMcbp(residualForForge, chosenColor, pick == null ? 1 : pick[1]);
            remaining.remove(chosenSrc);
        }

        String after = snapshot(sa, me);
        if (!before.equals(after)) {
            mutations++;
            if (DEBUG && dbg < 5) {
                dbg++;
                System.out.println("[maskcost] MUTATION on " + sa.getHostCard()
                        + "\n  before=" + before + "\n  after =" + after);
            }
        }

        if (calls > 0) {
            CALLS.add(calls);
            OWN_NS.add(ownNs);
            FORGE_NS.add(forgeNs);
            WIDTH.add(maxWidth);
            STEPS.add(steps);
            if (DEBUG && dbg < 25) {
                dbg++;
                System.out.println("[maskcost] " + sa.getHostCard() + " cost=" + cost
                        + " sources=" + srcs.size() + " steps=" + steps
                        + " maxWidth=" + maxWidth + " calls=" + calls
                        + " own=" + (ownNs / 1000) + "us forge=" + (forgeNs / 1000) + "us");
            }
        }
    }

    /** Same as {@link #apply} but against a Forge {@link ManaCostBeingPaid}. */
    private static void applyMcbp(ManaCostBeingPaid m, int colorIdx, int amount) {
        int amt = Math.max(1, amount);
        byte bit = colorIdx >= 0 && colorIdx < 5 ? (byte) (1 << colorIdx) : 0;
        for (int n = 0; n < amt; n++) {
            try {
                ManaCostShard hit = null;
                for (ManaCostShard sh : m.getDistinctShards()) {
                    if (m.getUnpaidShards(sh) <= 0) {
                        continue;
                    }
                    byte mm = sh.getColorMask();
                    if (mm == 0 ? bit == 0 : (mm & bit) != 0) {
                        hit = sh;
                        break;
                    }
                }
                if (hit != null) {
                    m.decreaseShard(hit, 1);
                } else if (m.getGenericManaAmount() > 0) {
                    m.decreaseGenericMana(1);
                }
            } catch (Throwable ignored) {
                return;
            }
        }
    }

    /**
     * Pays one mana of the given colour against the requirement, preferring a
     * coloured shard it can satisfy and falling back to generic. Mutates
     * {@code shards}; returns the new generic count.
     */
    static int apply(List<Byte> shards, int generic, int colorIdx, int amount) {
        int amt = Math.max(1, amount);
        byte bit = colorIdx >= 0 && colorIdx < 5 ? (byte) (1 << colorIdx) : 0;
        for (int n = 0; n < amt; n++) {
            int hit = -1;
            for (int i = 0; i < shards.size(); i++) {
                byte m = shards.get(i);
                if (m == 0 ? bit == 0 : (m & bit) != 0) {
                    hit = i;
                    break;
                }
            }
            if (hit >= 0) {
                shards.remove(hit);
            } else if (generic > 0) {
                generic--;
            }
        }
        return generic;
    }

    /** Beyond this many expanded unit sources, stop refining and answer loosely. */
    private static final int UNIT_CAP = 64;

    /** Beyond this many multi-mana option combinations, answer loosely. */
    private static final int COMBO_CAP = 256;

    /**
     * Is the residual requirement still satisfiable from the remaining sources?
     *
     * <p>Maximum bipartite matching of coloured shards against sources that can
     * produce them; generic is then covered by whatever capacity is left over.
     *
     * <p>A source producing {@code a} mana is first expanded into {@code a} unit
     * sources, because matching is one-to-one and would otherwise let a single
     * Nykthos- or Sol Ring-style source cover only one shard. That was a real
     * bug: {@code PrefixOracle} found 278 states where this predicate rejected a
     * payment that exists, which for a mask means silently deleting a legal
     * action rather than merely mismeasuring one.
     *
     * <p>Expansion needs a committed option, since colour and amount travel
     * together, so option choices are enumerated -- but only for sources that
     * actually produce more than one mana. Those are rare (usually none), and
     * when there are none this reduces exactly to the original single pass.
     *
     * <p>If either cap is exceeded the answer is {@code true}. That direction is
     * deliberate: a too-loose mask announces itself when Forge rejects the
     * payment, whereas a too-strict one removes the action with no symptom.
     */
    static boolean extendable(List<Byte> shards, int generic,
            List<PaymentCardinality.Src> remaining) {
        if (shards.isEmpty() && generic <= 0) {
            return true;
        }
        List<Integer> multi = new ArrayList<>();
        long combos = 1;
        for (int j = 0; j < remaining.size(); j++) {
            for (int[] opt : remaining.get(j).options) {
                if (Math.max(1, opt[1]) > 1) {
                    multi.add(j);
                    combos *= Math.max(1, remaining.get(j).options.size());
                    break;
                }
            }
        }
        if (combos > COMBO_CAP) {
            return true;
        }
        return choose(0, new int[multi.size()], multi, shards, generic, remaining);
    }

    /** Enumerate the committed option for each multi-mana source, then test. */
    private static boolean choose(int at, int[] choice, List<Integer> multi,
            List<Byte> shards, int generic, List<PaymentCardinality.Src> remaining) {
        if (at == multi.size()) {
            List<PaymentCardinality.Src> units = expand(choice, multi, remaining);
            return units != null && feasible(shards, generic, units);
        }
        int opts = remaining.get(multi.get(at)).options.size();
        for (int k = 0; k < Math.max(1, opts); k++) {
            choice[at] = k;
            if (choose(at + 1, choice, multi, shards, generic, remaining)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Rewrite the source list so every entry produces exactly one mana.
     *
     * <p>Single-mana sources pass through with all their colour options intact;
     * a multi-mana source contributes one unit per mana of its chosen colour.
     * Returns null if the expansion would exceed {@link #UNIT_CAP}.
     */
    private static List<PaymentCardinality.Src> expand(int[] choice,
            List<Integer> multi, List<PaymentCardinality.Src> remaining) {
        List<PaymentCardinality.Src> units = new ArrayList<>();
        for (int j = 0; j < remaining.size(); j++) {
            int slot = multi.indexOf(j);
            if (slot < 0) {
                units.add(remaining.get(j));
                continue;
            }
            List<int[]> opts = remaining.get(j).options;
            if (opts.isEmpty()) {
                continue;
            }
            int[] opt = opts.get(Math.min(choice[slot], opts.size() - 1));
            int amount = Math.max(1, opt[1]);
            if (units.size() + amount > UNIT_CAP) {
                return null;
            }
            for (int n = 0; n < amount; n++) {
                PaymentCardinality.Src unit = new PaymentCardinality.Src();
                unit.options.add(new int[] {opt[0], 1});
                unit.key = remaining.get(j).key;
                units.add(unit);
            }
        }
        return units;
    }

    /** Matching plus a generic-capacity check over unit-valued sources. */
    private static boolean feasible(List<Byte> shards, int generic,
            List<PaymentCardinality.Src> units) {
        int ns = shards.size();
        int nr = units.size();
        int[] matchOf = new int[nr];
        java.util.Arrays.fill(matchOf, -1);
        int matched = 0;
        for (int i = 0; i < ns; i++) {
            boolean[] seen = new boolean[nr];
            if (augment(i, shards, units, matchOf, seen)) {
                matched++;
            }
        }
        if (matched < ns) {
            return false;
        }
        // Every source now yields one mana, so each unmatched one covers one
        // point of generic and each matched one is already spent.
        return nr - matched >= generic;
    }

    private static boolean augment(int shardIdx, List<Byte> shards,
            List<PaymentCardinality.Src> remaining, int[] matchOf, boolean[] seen) {
        byte need = shards.get(shardIdx);
        for (int j = 0; j < remaining.size(); j++) {
            if (seen[j]) {
                continue;
            }
            boolean can = false;
            for (int[] opt : remaining.get(j).options) {
                int ci = opt[0];
                byte bit = ci >= 0 && ci < 5 ? (byte) (1 << ci) : 0;
                if (need == 0 ? bit == 0 : (need & bit) != 0) {
                    can = true;
                    break;
                }
            }
            if (!can) {
                continue;
            }
            seen[j] = true;
            if (matchOf[j] < 0
                    || augment(matchOf[j], shards, remaining, matchOf, seen)) {
                matchOf[j] = shardIdx;
                return true;
            }
        }
        return false;
    }

    /**
     * Is this an ability whose own host can legally be tapped to help pay it?
     *
     * <p>Man-land animation abilities ({@code Restless Anchorage} {1}{W}{U},
     * {@code Hall of Storm Giants} {5}{U}, {@code Mutavault} {1}) carry no tap
     * symbol, so tapping the land for mana and then animating it is legal -- and
     * not even always a misplay: animating Mutavault can turn on an effect that
     * cares about controlling a creature type.
     *
     * <p>Forge's {@code canPayManaCost} is an AI helper, not a rules oracle, and
     * declines these. We are the correct side, so they are bucketed separately
     * rather than counted as failures. Left in the failure column they would
     * become permanent known-bad entries, and a differential test with standing
     * failures stops detecting new ones.
     */
    private static boolean selfTappableHost(SpellAbility sa, Player me) {
        try {
            if (sa.getPayCosts() != null && sa.getPayCosts().hasTapCost()) {
                return false;
            }
            Card host = sa.getHostCard();
            if (host == null || !host.isLand()) {
                return false;
            }
            for (Card c : forge.ai.ComputerUtilMana.getAvailableManaSources(me, true)) {
                if (c == host) {
                    return true;
                }
            }
        } catch (Throwable ignored) {
            // fall through
        }
        return false;
    }

    /** Per-card, per-ability verdict for the three filters in {@code sources()}. */
    private static String diagnose(Player me, SpellAbility forSa) {
        StringBuilder sb = new StringBuilder();
        try {
            for (Card c : forge.ai.ComputerUtilMana.getAvailableManaSources(me, true)) {
                StringBuilder per = new StringBuilder();
                for (SpellAbility ma : c.getManaAbilities()) {
                    if (ma == null) {
                        continue;
                    }
                    boolean play;
                    try {
                        play = ma.canPlay();
                    } catch (Throwable t) {
                        play = true;
                    }
                    forge.game.spellability.AbilityManaPart mp = ma.getManaPart();
                    boolean unres = true;
                    if (mp != null && forSa != null) {
                        try {
                            unres = mp.meetsManaRestrictions(forSa);
                        } catch (Throwable t) {
                            unres = true;
                        }
                    }
                    per.append(per.length() == 0 ? "" : ";")
                            .append(play ? "play" : "NOPLAY")
                            .append('/')
                            .append(unres ? "free" : "RESTRICTED")
                            .append('/')
                            .append(mp == null ? "nomp" : mp.getOrigProduced())
                            .append('/')
                            .append(abilityCost(ma));
                }
                sb.append(sb.length() == 0 ? "" : "  ")
                        .append(c.getName()).append('=').append(per);
            }
        } catch (Throwable t) {
            return "?";
        }
        return sb.toString();
    }

    /**
     * Is some available mana ability gated behind a life payment?
     *
     * <p>Starting Town's second ability costs {@code {T}, Pay 1 life}. Paying life
     * is legal whenever the player's life total is at least the amount paid -- you
     * may go to zero and only lose at the next state-based check -- so a plan that
     * spends it is rules-legal. Forge's mana planner is an AI helper and declines
     * to pay life for mana as a matter of policy, so it reports such costs as
     * unpayable. We are right and Forge is conservative, which is the same
     * situation as {@link #selfTappableHost}; bucket it rather than let a
     * permanent known-good divergence mask new regressions.
     */
    private static boolean lifeCostManaAbility(Player me) {
        try {
            for (Card c : forge.ai.ComputerUtilMana.getAvailableManaSources(me, true)) {
                for (SpellAbility ma : c.getManaAbilities()) {
                    if (ma == null || ma.getPayCosts() == null) {
                        continue;
                    }
                    for (forge.game.cost.CostPart p : ma.getPayCosts().getCostParts()) {
                        if (p instanceof forge.game.cost.CostPayLife) {
                            return true;
                        }
                    }
                }
            }
        } catch (Throwable ignored) {
            return false;
        }
        return false;
    }

    /** The activation cost of a mana ability, to spot non-tap additional costs. */
    private static String abilityCost(SpellAbility ma) {
        try {
            return ma.getPayCosts() == null ? "-" : ma.getPayCosts().toString();
        } catch (Throwable t) {
            return "?";
        }
    }

    /** Render each source's colour options, for debugging disagreements. */
    private static String renderSrcs(List<PaymentCardinality.Src> srcs) {
        StringBuilder sb = new StringBuilder();
        for (PaymentCardinality.Src s : srcs) {
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

    /** The permanents Forge considers available mana sources, by name. */
    private static String availNames(Player me) {
        try {
            StringBuilder sb = new StringBuilder();
            for (Card c : forge.ai.ComputerUtilMana.getAvailableManaSources(me, true)) {
                sb.append(sb.length() == 0 ? "" : ", ").append(c.getName());
                if (c.isTapped()) {
                    sb.append("(tapped)");
                }
            }
            return "{" + sb + "}";
        } catch (Throwable t) {
            return "{?}";
        }
    }

    /** Tap state, pool size and ability flags -- enough to catch a mutation. */    private static String snapshot(SpellAbility sa, Player me) {
        StringBuilder sb = new StringBuilder();
        try {
            for (Card c : me.getGame().getCardsIn(forge.game.zone.ZoneType.Battlefield)) {
                sb.append(c.getId()).append(c.isTapped() ? 'T' : 'u').append(',');
            }
            sb.append("|pool=");
            int n = 0;
            for (Object ignored : me.getManaPool()) {
                n++;
            }
            sb.append(n);
            sb.append("|skip=").append(sa.isSkip());
            sb.append("|x=").append(sa.getXManaCostPaid());
            sb.append("|castFrom=").append(sa.getHostCard().getCastFrom());
        } catch (Throwable t) {
            sb.append("snapshot-failed");
        }
        return sb.toString();
    }

    static void print() {
        if (EVERY <= 0 || CALLS.isEmpty()) {
            return;
        }
        System.out.println("[maskcost] actions measured: " + CALLS.size());
        System.out.println("[maskcost] predicate calls per action: " + stats(CALLS));
        System.out.println("[maskcost] decode steps per action:    " + stats(STEPS));
        System.out.println("[maskcost] max mask width per action:  " + stats(WIDTH));
        System.out.println("[maskcost] own predicate:   total_calls=" + ownCalls
                + " mean=" + meanUs(OWN_NS, ownCalls) + "us/call"
                + " per_action=" + meanUsPerAction(OWN_NS) + "us");
        System.out.println("[maskcost] forge predicate: total_calls=" + forgeCalls
                + " mean=" + meanUs(FORGE_NS, forgeCalls) + "us/call"
                + " per_action=" + meanUsPerAction(FORGE_NS) + "us");
        System.out.println("[maskcost] own-vs-forge disagreements: " + disagree
                + " of " + forgeCalls
                + (forgeCalls > 0
                        ? String.format(" (%.2f%%)", 100.0 * disagree / forgeCalls) : "")
                + "  [forge says yes/own says no: " + forgeOverReports
                + " | own says yes/forge says no: " + ownOverReports + "]");
        long zt = zeroAgree + zeroForgeYesOwnNo + zeroOwnYesForgeNo + zeroManlandExempt;
        System.out.println("[maskcost] EMPTY-PREFIX agreement (exactness test): "
                + zeroAgree + "/" + zt
                + (zt > 0 ? String.format(" (%.2f%%)", 100.0 * zeroAgree / zt) : "")
                + "  [own too strict: " + zeroForgeYesOwnNo
                + " | own too loose: " + zeroOwnYesForgeNo
                + " | forge-conservative, we are correct: " + zeroManlandExempt + "]");
        System.out.println("[maskcost] MUTATIONS DETECTED: " + mutations
                + (mutations == 0 ? " (forge predicate appears pure)" : " <-- UNSAFE"));
        System.out.println("[maskcost] MANA-SOURCE pool, ai-helper vs engine: "
                + srcSamples + " samples, ai-only=" + srcAiOnly
                + ", engine-only=" + srcEngineOnly);
        System.out.println("[maskcost] FLOATING-MANA blind spot: "
                + zeroPoolNonEmpty + "/" + zt + " payment decisions had a non-empty "
                + "mana pool; mana abilities offered at priority: "
                + MANA_AT_PRIORITY.get());
    }

    private static String meanUs(List<Long> ns, long calls) {
        if (calls <= 0) {
            return "n/a";
        }
        long t = 0;
        for (long v : ns) {
            t += v;
        }
        return String.format("%.3f", t / 1000.0 / calls);
    }

    private static String meanUsPerAction(List<Long> ns) {
        if (ns.isEmpty()) {
            return "n/a";
        }
        long t = 0;
        for (long v : ns) {
            t += v;
        }
        return String.format("%.1f", t / 1000.0 / ns.size());
    }

    private static String stats(List<Integer> xs) {
        if (xs.isEmpty()) {
            return "n/a";
        }
        List<Integer> s = new ArrayList<>(xs);
        java.util.Collections.sort(s);
        long sum = 0;
        for (int v : s) {
            sum += v;
        }
        return "mean=" + String.format("%.1f", (double) sum / s.size())
                + " p50=" + s.get(s.size() / 2)
                + " p90=" + s.get(Math.min(s.size() - 1, (int) (s.size() * 0.9)))
                + " max=" + s.get(s.size() - 1);
    }
}
