package magicnet.bridge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.AbilityManaPart;
import forge.game.spellability.SpellAbility;

/**
 * Counts the distinct ways to pay an ability's mana cost.
 *
 * <p>§14.4 established that the flat {@code (ability, mode, X)} head stays small
 * (max 96) while raw complete-action counts reach millions, and that the whole
 * explosion lives in target-set selection. Payment is the remaining threat to that
 * conclusion, because unlike modes -- which are bounded by the card -- the number of
 * ways to tap for a cost scales with the *board*, and it attaches to the flat head
 * rather than to the target decoder.
 *
 * <p>Two counts, mirroring §14.1:
 *
 * <ul>
 *   <li><b>raw</b> -- every distinct selection of {@code (source, colour produced)}
 *       counted separately, so two untapped Forests are two plans.
 *   <li><b>dedup</b> -- sources collapsed into consequence-equivalence classes, so
 *       two Forests are one plan but a Forest and a Llanowar Elves are two (tapping
 *       the creature costs you an attacker and is affected by summoning sickness).
 * </ul>
 *
 * <p>A "plan" here is an <em>exact</em> payment: the produced mana equals the cost.
 * Plans that float surplus mana are legal but are not counted; see
 * {@link #LIMITATIONS}.
 *
 * <p><b>Cost reduction is not applied.</b> {@code CostPartMana.getMana()} returns the
 * printed cost; {@code ReduceCost} statics (Domain, affinity) are applied later, to a
 * {@code ManaCostBeingPaid}. The two Forge entry points that would apply them both
 * mutate live state -- {@code ComputerUtilMana.calculateManaCost} calls
 * {@code Card.setCastFrom} / {@code SpellAbility.setMaxWaterbend} /
 * {@code setXManaCostPaid}, and the 6-arg {@code CostAdjustment.adjust} drives the
 * Delve exile machinery ({@code addDelved}, {@code setExiledWith},
 * {@code CardZoneTable.put}) [both verified, bytecode]. Neither is safe to call on a
 * live game, so reduced-cost cards are counted at their printed cost and are large
 * overcounts.
 */
final class PaymentCardinality {

    private PaymentCardinality() {
    }

    /** Fallback for producers whose colours cannot be resolved statically. */
    private static final List<Integer> ANY_COLOUR =
            java.util.Collections.unmodifiableList(
                    java.util.Arrays.asList(0, 1, 2, 3, 4));

    /** Colour index order: W U B R G C. Masks follow {@code MagicColor}. */
    private static final byte[] MASK = {1, 2, 4, 8, 16, 0};

    private static final int NCOL = 6;

    /** Above this many sources we stop rather than risk a pathological DP. */
    private static final int SOURCE_CAP = 28;

    /** Per-ability clamp. */
    private static final long LIMIT = 10_000_000L;

    static final String LIMITATIONS =
            "exact payments only (no surplus-floating plans); cost REDUCTION statics "
            + "(Domain/affinity, e.g. Leyline Binding) not applied, so reduced-cost "
            + "cards are large overcounts; Phyrexian, Convoke, Improvise and Delve not "
            + "enumerated; monocolour-hybrid shards treated as coloured only; mana "
            + "abilities with a non-literal Amount counted as 1; mana abilities with "
            + "ADDITIONAL activation costs (Starting Town: '{T}, Pay 1 life') are "
            + "treated as freely usable, since the cost is almost always payable -- "
            + "a sacrifice-cost producer would be a too-loose gap";

    /**
     * @return {raw plans, dedup plans}; {1,1} if the ability has no mana cost
     *     (exactly one way to pay nothing), {0,0} if it cannot be paid.
     */
    static long[] plans(SpellAbility sa, Player me) {
        ManaCost cost;
        int amountOfX = 0;
        try {
            if (sa.getPayCosts() == null || sa.getPayCosts().getCostMana() == null) {
                return new long[] {1, 1};
            }
            cost = sa.getPayCosts().getCostMana().getMana();
            amountOfX = sa.getPayCosts().getCostMana().getAmountOfX();
        } catch (Throwable ignored) {
            return new long[] {1, 1};
        }
        if (cost == null || cost.isZero()) {
            return new long[] {1, 1};
        }

        // Requirements: one entry per coloured shard (mask of acceptable colours),
        // plus a generic count that accepts anything.
        List<Byte> shardMasks = new ArrayList<>();
        int generic;
        try {
            generic = cost.getGenericCost();
            for (ManaCostShard s : cost) {
                if (s == null) {
                    continue;
                }
                // {X} appears in the shard iteration with colorMask 0, which is
                // indistinguishable from true colourless {C}. Its contribution is
                // already added below as amountOfX * X generic, so counting it here
                // would both double-count it and impose a bogus colourless-only
                // requirement. [verified: ManaCost("X R R R") yields 4 shards]
                if (s == ManaCostShard.X) {
                    continue;
                }
                byte m = s.getColorMask();
                if (m == 0) {
                    // {C} -- colourless, only payable by colourless mana.
                    shardMasks.add((byte) 0);
                } else {
                    shardMasks.add(m);
                }
            }
        } catch (Throwable ignored) {
            return new long[] {1, 1};
        }

        // X contributes generic mana once bound.
        if (amountOfX > 0) {
            try {
                Integer x = sa.getXManaCostPaid();
                if (x != null && x > 0) {
                    generic += amountOfX * x;
                }
            } catch (Throwable ignored) {
                // leave generic as printed
            }
        }

        int need = generic + shardMasks.size();
        if (need <= 0) {
            return new long[] {1, 1};
        }

        List<Src> sources = sources(me, sa);
        if (sources == null) {
            return new long[] {0, 0};
        }
        if (sources.isEmpty()) {
            return new long[] {0, 0};
        }

        long raw = countRaw(sources, need, shardMasks, generic);
        long dedup = countDedup(sources, need, shardMasks, generic);
        if (DEBUG && DBG.incrementAndGet() <= 25) {
            StringBuilder sb = new StringBuilder();
            for (Src s : sources) {
                sb.append('[');
                for (int[] o : s.options) {
                    sb.append("WUBRGC".charAt(o[0])).append(o[1] > 1 ? "x" + o[1] : "");
                }
                sb.append(']');
            }
            System.out.println("[pay.dbg] " + sa.getHostCard() + " cost=" + cost
                    + (generic != cost.getGenericCost() ? " (+X gen=" + generic + ")" : "")
                    + " need=" + need + " sources=" + sources.size()
                    + " " + sb + " -> raw=" + raw + " dedup=" + dedup);
        }
        return new long[] {raw, dedup};
    }

    private static final boolean DEBUG =
            Integer.getInteger("magicnet.cardinality.pay.debug", 0) > 0;

    private static final java.util.concurrent.atomic.AtomicInteger DBG =
            new java.util.concurrent.atomic.AtomicInteger();

    /** One available mana source and the colours it can produce. */
    static final class Src {
        /** For each colour index, the amount produced if this option is chosen. */
        final List<int[]> options = new ArrayList<>(); // {colourIndex, amount}
        String key = "";
    }

    static List<Src> sources(Player me, SpellAbility forSa) {
        forge.game.card.CardCollection avail;
        try {
            avail = forge.ai.ComputerUtilMana.getAvailableManaSources(me, true);
        } catch (Throwable ignored) {
            return null;
        }
        if (avail == null) {
            return null;
        }
        if (avail.size() > SOURCE_CAP) {
            return null;
        }

        List<Src> out = new ArrayList<>();
        // If the ability taps its own host as part of the cost, the host cannot also
        // produce mana to pay for it. getAvailableManaSources does not know about the
        // ability being paid, so it still offers the host.
        Card selfTapped = null;
        try {
            if (forSa != null && forSa.getPayCosts() != null
                    && forSa.getPayCosts().hasTapCost()) {
                selfTapped = forSa.getHostCard();
            }
        } catch (Throwable ignored) {
            selfTapped = null;
        }
        for (Card c : avail) {
            if (c == null || (selfTapped != null && c == selfTapped)) {
                continue;
            }
            Src s = new Src();
            java.util.Set<Integer> seen = new java.util.HashSet<>();
            try {
                for (SpellAbility ma : c.getManaAbilities()) {
                    if (ma == null || !playable(ma)) {
                        continue;
                    }
                    AbilityManaPart mp = ma.getManaPart();
                    if (mp == null || !unrestricted(mp, forSa)) {
                        continue;
                    }
                    int amount = amount(ma);
                    List<Integer> colours = producible(mp, ma);
                    if (colours.isEmpty() && reflects(ma)) {
                        colours = reflected(ma);
                        if (colours.isEmpty()) {
                            // Mox Amber controlling no legendary permanents. The
                            // ability is still activatable -- it simply adds no
                            // mana. Zero production is a determined fact, not an
                            // unknown, so drop this ability rather than guessing.
                            continue;
                        }
                    }
                    if (colours.isEmpty()) {
                        // Playable and unrestricted, but we cannot tell what it
                        // makes. Assume any colour rather than dropping the
                        // source: too loose is announced when Forge rejects the
                        // payment, too strict removes the action with no symptom.
                        colours = ANY_COLOUR;
                    }
                    for (int ci : colours) {
                        int sig = ci * 64 + Math.min(amount, 15);
                        if (seen.add(sig)) {
                            s.options.add(new int[] {ci, amount});
                        }
                    }
                }
            } catch (Throwable ignored) {
                continue;
            }
            if (s.options.isEmpty()) {
                continue;
            }
            s.key = classKey(c, s);
            out.add(s);
        }
        return out;
    }

    /** Does this ability mirror other permanents' colours rather than declare its own? */
    private static boolean reflects(SpellAbility ma) {
        try {
            return ma.getApi() == forge.game.ability.ApiType.ManaReflected;
        } catch (Throwable ignored) {
            return false;
        }
    }

    /**
     * Colours a reflected-mana ability can actually produce right now.
     *
     * <p>{@code ManaReflected} (Mox Amber, Reflecting Pool) declares no fixed
     * colour -- it mirrors whatever some other set of permanents can make, so the
     * answer is a property of the board rather than of the script. Forge resolves
     * this with {@code CardUtil.getReflectableManaColors}, which returns colour
     * names; purity is asserted behaviourally by the snapshot test in
     * {@code MaskCost}, not by reading the implementation.
     */
    private static List<Integer> reflected(SpellAbility ma) {
        List<Integer> out = new ArrayList<>();
        try {
            for (String name : forge.game.card.CardUtil.getReflectableManaColors(ma)) {
                byte b = forge.card.MagicColor.fromName(name);
                for (int i = 0; i < 5; i++) {
                    if (MASK[i] == b && !out.contains(i)) {
                        out.add(i);
                    }
                }
            }
        } catch (Throwable ignored) {
            return out;
        }
        return out;
    }

    /**
     * Can this mana ability's output legally pay for {@code forSa}?
     *
     * <p>Restricted mana ("spend this mana only to cast a creature spell of the
     * chosen type") is resolvable **statically** here, because by the time payment
     * begins the spell, mode, targets and declared cost are all fixed (CR 601.2f
     * precedes 601.2h). So a restriction is not a constraint the payment decoder
     * has to carry -- it is a filter applied once, before decoding starts, and
     * {@code extendable()} needs no knowledge of it.
     *
     * <p>Filtering is per ability, not per card, which gives the right answer for
     * split sources: Cavern of Souls keeps its unrestricted {@code {T}: Add {C}}
     * and loses only its any-colour ability when the spell is not a creature of
     * the named type.
     */
    private static boolean unrestricted(AbilityManaPart mp, SpellAbility forSa) {
        if (forSa == null) {
            return true;
        }
        try {
            return mp.meetsManaRestrictions(forSa);
        } catch (Throwable ignored) {
            return true;
        }
    }

    /**
     * Is this mana ability actually activatable right now?
     *
     * <p>Reading the produced colours off the script is not enough: conditional
     * mana abilities declare a colour they can only sometimes make. The Verge
     * cycle is the case that exposed this -- Floodfarm Verge scripts {W} and {U}
     * but the {U} ability activates only while you control a Plains or an
     * Island, so on a board of Fountainport plus Floodfarm Verge we claimed
     * {1}{U} was payable and Forge, correctly, did not.
     *
     * <p>On failure the answer is {@code true}, keeping the error in the
     * self-announcing direction (Forge rejects a too-loose proposal; a
     * too-strict one deletes the action with no symptom).
     */
    private static boolean playable(SpellAbility ma) {
        try {
            return ma.canPlay();
        } catch (Throwable ignored) {
            return true;
        }
    }

    private static int amount(SpellAbility ma) {
        try {
            if (ma.hasParam("Amount")) {
                return Math.max(1, Integer.parseInt(ma.getParam("Amount").trim()));
            }
        } catch (Throwable ignored) {
            // dynamic amount -- counted as 1, see LIMITATIONS
        }
        return 1;
    }

    private static List<Integer> producible(AbilityManaPart mp, SpellAbility ma) {
        List<Integer> out = new ArrayList<>();
        try {
            if (mp.isAnyMana()) {
                for (int i = 0; i < 5; i++) {
                    out.add(i);
                }
                return out;
            }
            String s = mp.isComboMana() ? mp.getComboColors(ma) : mp.mana(ma);
            if (s == null) {
                return out;
            }
            for (String tok : s.trim().split("\\s+")) {
                if (tok.isEmpty()) {
                    continue;
                }
                int ci = colorIndex(tok.charAt(0));
                if (ci >= 0 && !out.contains(ci)) {
                    out.add(ci);
                }
            }
        } catch (Throwable ignored) {
            // treated as producing nothing
        }
        return out;
    }

    private static int colorIndex(char ch) {
        switch (Character.toUpperCase(ch)) {
            case 'W': return 0;
            case 'U': return 1;
            case 'B': return 2;
            case 'R': return 3;
            case 'G': return 4;
            case 'C': return 5;
            default: return -1;
        }
    }

    /**
     * Consequence-equivalence key (§13.3). Two sources share a key only if tapping
     * either has the same downstream consequences: same colours produced, same card
     * name, same permanent types that other effects count (artifact, creature), and
     * same spend-triggered behaviour.
     */
    private static String classKey(Card c, Src s) {
        StringBuilder sb = new StringBuilder(48);
        try {
            sb.append(c.getName());
            sb.append('|').append(c.isArtifact() ? 'A' : '-');
            sb.append(c.isCreature() ? 'C' : '-');
            sb.append(c.isLand() ? 'L' : '-');
            sb.append('|').append(c.isSick() ? 'S' : '-');
            sb.append('|').append(c.isTapped() ? 'T' : 'u');
            boolean trig = false;
            boolean restricted = false;
            for (SpellAbility ma : c.getManaAbilities()) {
                AbilityManaPart mp = ma.getManaPart();
                if (mp == null) {
                    continue;
                }
                trig |= mp.getTriggersWhenSpent();
                String r = mp.getManaRestrictions();
                restricted |= r != null && !r.isEmpty();
            }
            sb.append('|').append(trig ? 'X' : '-').append(restricted ? 'R' : '-');
        } catch (Throwable ignored) {
            return "id:" + System.identityHashCode(c);
        }
        List<String> opts = new ArrayList<>();
        for (int[] o : s.options) {
            opts.add(o[0] + "x" + o[1]);
        }
        Collections.sort(opts);
        sb.append('|').append(opts);
        return sb.toString();
    }

    // ---- counting -------------------------------------------------------

    /** Encode a 6-vector of small counts into a long, 5 bits each. */
    private static long enc(int[] v) {
        long k = 0;
        for (int i = 0; i < NCOL; i++) {
            k = (k << 5) | Math.min(v[i], 31);
        }
        return k;
    }

    private static int[] dec(long k) {
        int[] v = new int[NCOL];
        for (int i = NCOL - 1; i >= 0; i--) {
            v[i] = (int) (k & 31);
            k >>= 5;
        }
        return v;
    }

    private static int total(int[] v) {
        int t = 0;
        for (int x : v) {
            t += x;
        }
        return t;
    }

    /** DP over individual sources: two Forests are two distinct plans. */
    private static long countRaw(List<Src> sources, int need,
            List<Byte> shardMasks, int generic) {
        Map<Long, Long> cur = new HashMap<>();
        cur.put(0L, 1L);
        for (Src s : sources) {
            Map<Long, Long> next = new HashMap<>();
            for (Map.Entry<Long, Long> e : cur.entrySet()) {
                int[] v = dec(e.getKey());
                long w = e.getValue();
                merge(next, e.getKey(), w);          // skip this source
                for (int[] o : s.options) {
                    int[] nv = v.clone();
                    nv[o[0]] += o[1];
                    if (total(nv) > need) {
                        continue;
                    }
                    merge(next, enc(nv), w);
                }
            }
            cur = next;
            if (cur.size() > 2_000_000) {
                return LIMIT;
            }
        }
        return tally(cur, need, shardMasks, generic);
    }

    /**
     * DP over equivalence classes: a class of m identical sources contributes a
     * <em>composition</em> (how many produce each colour), counted once regardless
     * of which physical permanents are chosen.
     */
    private static long countDedup(List<Src> sources, int need,
            List<Byte> shardMasks, int generic) {
        Map<String, List<Src>> classes = new HashMap<>();
        for (Src s : sources) {
            classes.computeIfAbsent(s.key, k -> new ArrayList<>()).add(s);
        }

        Map<Long, Long> cur = new HashMap<>();
        cur.put(0L, 1L);
        for (Map.Entry<String, List<Src>> cls : classes.entrySet()) {
            int m = cls.getValue().size();
            Src proto = cls.getValue().get(0);
            List<int[]> deltas = compositions(proto.options, m, need);
            Map<Long, Long> next = new HashMap<>();
            for (Map.Entry<Long, Long> e : cur.entrySet()) {
                int[] v = dec(e.getKey());
                long w = e.getValue();
                for (int[] d : deltas) {
                    int[] nv = v.clone();
                    boolean ok = true;
                    for (int i = 0; i < NCOL; i++) {
                        nv[i] += d[i];
                    }
                    if (total(nv) > need) {
                        ok = false;
                    }
                    if (ok) {
                        merge(next, enc(nv), w);
                    }
                }
            }
            cur = next;
            if (cur.size() > 2_000_000) {
                return LIMIT;
            }
        }
        return tally(cur, need, shardMasks, generic);
    }

    /**
     * All distinct colour-count vectors obtainable by using up to {@code m} members
     * of one class, each choosing one of {@code options}.
     */
    private static List<int[]> compositions(List<int[]> options, int m, int need) {
        List<int[]> acc = new ArrayList<>();
        java.util.Set<Long> seen = new java.util.HashSet<>();
        int[] v = new int[NCOL];
        comp(options, 0, m, need, v, acc, seen);
        return acc;
    }

    private static void comp(List<int[]> options, int oi, int remaining, int need,
            int[] v, List<int[]> acc, java.util.Set<Long> seen) {
        if (total(v) > need) {
            return;
        }
        if (oi == options.size()) {
            if (seen.add(enc(v))) {
                acc.add(v.clone());
            }
            return;
        }
        int[] o = options.get(oi);
        for (int k = 0; k <= remaining; k++) {
            int[] nv = v.clone();
            nv[o[0]] += o[1] * k;
            if (total(nv) > need) {
                break;
            }
            comp(options, oi + 1, remaining - k, need, nv, acc, seen);
        }
    }

    private static void merge(Map<Long, Long> m, long k, long w) {
        m.merge(k, w, (a, b) -> Math.min(LIMIT, a + b));
    }

    /** Sum the weights of produced-mana vectors that can actually cover the cost. */
    private static long tally(Map<Long, Long> states, int need,
            List<Byte> shardMasks, int generic) {
        long out = 0;
        for (Map.Entry<Long, Long> e : states.entrySet()) {
            int[] v = dec(e.getKey());
            if (total(v) != need) {
                continue;
            }
            if (feasible(v, shardMasks, generic)) {
                out = Math.min(LIMIT, out + e.getValue());
            }
        }
        return out;
    }

    /**
     * Can a produced-mana multiset {@code v} cover the coloured shards, with generic
     * absorbing the rest? Checked by Hall's condition over the distinct shard masks;
     * since {@code total(v) == need}, generic is satisfied automatically whenever the
     * coloured demands are.
     */
    private static boolean feasible(int[] v, List<Byte> shardMasks, int generic) {
        if (shardMasks.isEmpty()) {
            return true;
        }
        Map<Byte, Integer> demand = new HashMap<>();
        for (Byte m : shardMasks) {
            demand.merge(m, 1, Integer::sum);
        }
        List<Byte> masks = new ArrayList<>(demand.keySet());
        int n = masks.size();
        for (int sub = 1; sub < (1 << n); sub++) {
            int d = 0;
            byte union = 0;
            boolean needColorless = false;
            for (int i = 0; i < n; i++) {
                if ((sub & (1 << i)) == 0) {
                    continue;
                }
                byte mk = masks.get(i);
                d += demand.get(mk);
                if (mk == 0) {
                    needColorless = true;
                } else {
                    union |= mk;
                }
            }
            int supply = 0;
            for (int ci = 0; ci < 5; ci++) {
                if ((union & MASK[ci]) != 0) {
                    supply += v[ci];
                }
            }
            if (needColorless) {
                supply += v[5];
            }
            if (d > supply) {
                return false;
            }
        }
        return true;
    }
}
