package magicnet.bridge;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import forge.card.mana.ManaAtom;
import forge.card.mana.ManaCostShard;
import forge.game.card.Card;
import forge.game.mana.Mana;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.mana.ManaPool;
import forge.game.player.Player;
import forge.game.spellability.AbilityManaPart;
import forge.game.spellability.SpellAbility;

/**
 * Schema 17: mana payment as a sequence of {@code pay_mana} pick-one decisions
 * ({@code -Dmagicnet.payment=true}).
 *
 * <p>One decision per payment step until the cost is paid. Candidates are option
 * tokens: a mana ability of a permanent the payer controls, one per colour it
 * can be told to make ({@code activate_mana}); mana floating in the pool
 * ({@code use_pool}; plain units of one colour collapse into one option, a unit
 * with a rider -- Cavern of Souls' "can't be countered" -- is its own option);
 * 2 life for a Phyrexian shard ({@code pay_life}). Only candidates after which
 * the rest of the cost is still payable are offered ({@link #feasible}, built on
 * {@link MaskCost#extendable}); a single legal candidate is taken without asking
 * and not recorded, like a forced target.
 *
 * <p><b>Decide mode</b> pays like {@code PlayerControllerHuman}: {@code payManaCost}
 * goes through the engine's {@code PlaySpellAbility.payManaCost} (cost
 * adjustment, Convoke/Delve/Improvise callbacks, X), which calls back
 * {@code applyManaToCost}, answered here. A step is executed as
 * {@code InputPayMana.activateManaAbility} does: express colour choice, the
 * mana ability through {@code PlaySpellAbility.playSpellAbility} (costs,
 * resolution, mana triggers), then {@code ManaPool.payManaFromAbility}; pool
 * mana by {@code ManaPool.tryPayCostWithMana}; Phyrexian life as
 * {@code InputPayManaOfCostPayment} does (shard now, life once paid).
 *
 * <p><b>Observe mode</b> leaves Forge's AI paying exactly as it does (its
 * {@code ComputerUtilMana.payManaCost(Cost, ...)} split into the two public calls
 * it makes) and reconstructs the steps from {@code SpellAbility.getPayingMana()},
 * whose order is the AI's payment order. Each step's observation is taken when
 * the AI taps that step's source, with that one card shown untapped.
 */
final class ManaPayment {

    private ManaPayment() {
    }

    /** Opt-in: routing payment changes which decisions exist. */
    static final boolean ROUTE = Boolean.getBoolean("magicnet.payment");

    /** Diagnostic: compare the first-step mask with canPayManaCost (~1 ms per payment). */
    private static final boolean AUDIT = Boolean.getBoolean("magicnet.payment.audit");

    /** Test hook: corrupt observe-mode labels, to show the replay check can fail. */
    private static final boolean FAULT = Boolean.getBoolean("magicnet.payment.fault");

    static final String KIND = "pay_mana";
    static final String VERB_ACTIVATE = "activate_mana";
    static final String VERB_POOL = "use_pool";
    static final String VERB_LIFE = "pay_life";

    /** Vocab {@code mana_color}, indexed as {@link MaskCost}'s colour indices. */
    static final String[] COLOURS = Schema.MANA_COLORS.toArray(new String[0]);
    private static final byte[] BYTES = {1, 2, 4, 8, 16, (byte) ManaAtom.COLORLESS};

    private static final int ABILITY = 0;
    private static final int POOL = 1;
    private static final int LIFE = 2;

    // ---- counters ----------------------------------------------------------
    private static final Map<String, AtomicLong> COUNTS = new java.util.concurrent.ConcurrentHashMap<>();

    static void count(String what) {
        COUNTS.computeIfAbsent(what, k -> new AtomicLong()).incrementAndGet();
    }

    static void count(String what, long n) {
        COUNTS.computeIfAbsent(what, k -> new AtomicLong()).addAndGet(n);
    }

    private static int logged;

    private static void log(String msg) {
        if (logged++ < 40) {
            System.out.println("[payment] " + msg);
        }
    }

    static void printSummary() {
        if (COUNTS.isEmpty()) {
            return;
        }
        System.out.println("\n=== mana payment routing (-Dmagicnet.payment) ===");
        new java.util.TreeMap<>(COUNTS).forEach((k, v) -> System.out.printf("  %-36s %d%n", k, v.get()));
    }

    /** Whether this payer's mana payment is decided by the action source. */
    static boolean decides() {
        return ROUTE && InstrumentedController.ACTIONS.decides()
                && (Serializer.enabled() || InstrumentedController.ACTIONS.needsRequest());
    }

    /** Whether Forge's AI pays and the steps are recorded as labels. */
    static boolean observes() {
        return ROUTE && !InstrumentedController.ACTIONS.decides() && Serializer.enabled();
    }

    // ---- candidates ----------------------------------------------------------

    /** One step a payer could take next. */
    static final class Cand {
        final int type;
        Card card;
        SpellAbility ma;
        /** Colour index (0-5) shown on the option; for a fixed multi-colour bundle, the first. */
        int ci;
        /** Mana bytes one activation adds (ability). */
        byte[] produce;
        /** Whether the ability makes a colour choice that must be steered. */
        boolean choice;
        /** Pool: interchangeable units (a unit with a rider is alone). */
        final List<Mana> units = new ArrayList<>();
        boolean restricted;
        boolean snow;
        String rider;
        /** Schema 18: the ability's own mana cost (filters, Signets); null if none. */
        forge.card.mana.ManaCost cost;
        /** Schema 18: "any combination" ability; its colours are chosen by pay_mana_combo. */
        boolean combo;
        List<Integer> comboCols;
        /** Schema 18: activations available without {T} (sacrifice-another, life, counters). */
        int uses = 1;
        /** Schema 18: its cost taps an untapped creature that is itself a mana source (Springleaf Drum). */
        boolean tapsSource;
        List<Card> fodder;
        /** Creatures its tap cost could tap that are not mana sources themselves. */
        List<Card> idleFodder;

        Cand(int type) {
            this.type = type;
        }

        boolean same(Cand o) {
            if (o == null || o.type != type) {
                return false;
            }
            if (type == ABILITY) {
                return o.card == card && o.ma == ma && o.ci == ci;
            }
            if (type == POOL) {
                return !o.units.isEmpty() && !units.isEmpty()
                        && o.units.get(0).getColor() == units.get(0).getColor()
                        && o.restricted == restricted && o.snow == snow
                        && java.util.Objects.equals(o.rider, rider) && o.card == card;
            }
            return true;
        }
    }

    static int colourIndex(byte b) {
        for (int i = 0; i < 5; i++) {
            if (b == BYTES[i]) {
                return i;
            }
        }
        return 5;
    }

    /** Spending this pool mana changes the spell (or triggers), beyond paying. */
    private static boolean hasRider(Mana m, SpellAbility sa) {
        try {
            return m.addsNoCounterMagic(sa) || m.addsCounters(sa) || m.addsKeywords(sa)
                    || m.triggersWhenSpent();
        } catch (RuntimeException e) {
            return false;
        }
    }

    private static String riderText(Mana m, forge.game.Game game) {
        try {
            AbilityManaPart mp = m.getManaAbility();
            SpellAbility src = mp == null ? null : mp.getSourceSA();
            if (src != null) {
                String t = Serializer.optionText(String.valueOf(src), src, game);
                if (t != null) {
                    return t;
                }
            }
        } catch (RuntimeException ignored) {
            // fall through
        }
        return "spending this mana has an additional effect";
    }

    /** Units a mana ability adds per activation, from its script. */
    private static int amountParam(SpellAbility ma) {
        try {
            if (ma.hasParam("Amount")) {
                return Math.max(1, Integer.parseInt(ma.getParam("Amount").trim()));
            }
        } catch (RuntimeException ignored) {
            count("silent: dynamic mana amount (counted as 1)");
        }
        return 1;
    }

    /**
     * Every candidate step on the live state, before masking. Pool mana that
     * can pay this spell; every playable, unrestricted mana ability of a
     * battlefield permanent the payer controls (not {@code exclude}d, not the
     * host of a {T} ability being paid for), per colour it can be told to make.
     */
    static List<Cand> enumerate(Player me, SpellAbility sa, Set<Card> exclude) {
        List<Cand> out = new ArrayList<>();
        List<Mana> floating = new ArrayList<>();
        for (Mana m : me.getManaPool()) {
            floating.add(m);
        }
        out.addAll(poolCands(floating, sa, me.getGame()));

        Card host = sa.getHostCard();
        boolean hostTaps = false;
        try {
            hostTaps = sa.getPayCosts() != null && sa.getPayCosts().hasTapCost();
        } catch (RuntimeException ignored) {
            // treat as no tap cost
        }
        SpellAbility root = sa.getRootAbility();
        List<Card> cards = new ArrayList<>(me.getCardsIn(forge.game.zone.ZoneType.Battlefield));
        // Schema 18: mana abilities usable from other zones (Spirit Guides in hand).
        for (forge.game.zone.ZoneType z : OFF_ZONES) {
            for (Card c : me.getCardsIn(z)) {
                if (!c.getManaAbilities().isEmpty()) {
                    cards.add(c);
                }
            }
        }
        for (Card c : cards) {
            if (c == null || exclude.contains(c) || (hostTaps && c == host)) {
                continue;
            }
            for (SpellAbility ma : c.getManaAbilities()) {
                if (ma == null || ma == sa || ma == root) {
                    continue;
                }
                try {
                    if (ma.getActivatingPlayer() != me) {
                        ma.setActivatingPlayer(me);
                    }
                    if (!PaymentCardinality.playable(ma)) {
                        continue;
                    }
                    AbilityManaPart mp = ma.getManaPart();
                    if (mp == null || !PaymentCardinality.unrestricted(mp, sa)) {
                        continue;
                    }
                    forge.card.mana.ManaCost own = null;
                    if (ma.getPayCosts() != null && ma.getPayCosts().hasManaCost()) {
                        // Schema 18: filters/Signets, paid by a nested payment
                        // when taken (its own pay_mana steps).
                        own = ma.getPayCosts().getTotalMana();
                    }
                    int before = out.size();
                    addAbility(out, c, ma, mp);
                    int uses = uses(ma, c, me);
                    List<Card> fodder = tapFodder(ma, c, me);
                    for (int i = before; i < out.size(); i++) {
                        out.get(i).cost = own;
                        out.get(i).uses = uses;
                        out.get(i).fodder = fodder;
                    }
                } catch (RuntimeException e) {
                    count("enumerate: ability threw");
                }
            }
        }
        markTapSources(out);
        return out;
    }

    /** Creatures a "tap an untapped creature" cost of {@code ma} could tap; null when it has none. */
    private static List<Card> tapFodder(SpellAbility ma, Card host, Player me) {
        forge.game.cost.Cost cost = ma.getPayCosts();
        if (cost == null) {
            return null;
        }
        for (forge.game.cost.CostPart part : cost.getCostParts()) {
            if (part instanceof forge.game.cost.CostTapType && !ma.isCrew()) {
                List<Card> out = new ArrayList<>();
                for (Card x : forge.game.card.CardLists.filter(forge.game.card.CardLists.getValidCards(
                        me.getCardsIn(forge.game.zone.ZoneType.Battlefield), part.getType().split(";"), me, host, ma),
                        forge.game.card.CardPredicates.CAN_TAP)) {
                    if (x != host) {
                        out.add(x);
                    }
                }
                return out;
            }
        }
        return null;
    }

    /**
     * A tap-a-creature mana cost consumes a mana source only when every
     * creature it could tap is one ({@link Cand#tapsSource}).
     */
    private static void markTapSources(List<Cand> out) {
        Set<Card> sources = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (Cand c : out) {
            if (c.type == ABILITY) {
                sources.add(c.card);
            }
        }
        for (Cand c : out) {
            if (c.fodder != null) {
                // the creature it taps is a cost shard matched by a fodder Src:
                // a mana source's, or a fodder-only Src for one that is not
                c.tapsSource = true;
                c.idleFodder = new ArrayList<>();
                for (Card f : c.fodder) {
                    if (!sources.contains(f)) {
                        c.idleFodder.add(f);
                    }
                }
            }
        }
    }

    private static final forge.game.zone.ZoneType[] OFF_ZONES = {forge.game.zone.ZoneType.Hand,
            forge.game.zone.ZoneType.Graveyard, forge.game.zone.ZoneType.Exile, forge.game.zone.ZoneType.Command};

    private static final int USES_CAP = 4;

    /**
     * How many times an ability can be activated for this payment: once with
     * {T} or a self-sacrifice/exile; else bounded by what its cost consumes
     * (another permanent to sacrifice, life, counters on the source).
     */
    private static int uses(SpellAbility ma, Card c, Player me) {
        forge.game.cost.Cost cost = ma.getPayCosts();
        if (cost == null || cost.hasTapCost() || cost.hasManaCost() || !c.isInPlay()) {
            return 1;
        }
        int n = USES_CAP;
        for (forge.game.cost.CostPart part : cost.getCostParts()) {
            int amt = Math.max(1, part.getAbilityAmount(ma));
            if (part.payCostFromSource() && (part instanceof forge.game.cost.CostSacrifice
                    || part instanceof forge.game.cost.CostExile || part instanceof forge.game.cost.CostReturn)) {
                return 1;
            } else if (part instanceof forge.game.cost.CostSacrifice) {
                n = Math.min(n, forge.game.card.CardLists.getValidCards(me.getCardsIn(forge.game.zone.ZoneType.Battlefield),
                        part.getType().split(";"), me, c, ma).size() / amt);
            } else if (part instanceof forge.game.cost.CostPayLife) {
                n = Math.min(n, Math.max(0, me.getLife() - 1) / amt);
            } else if (part instanceof forge.game.cost.CostRemoveCounter rc && part.payCostFromSource()) {
                n = Math.min(n, c.getCounters(rc.counter) / amt);
            } else if (!(part instanceof forge.game.cost.CostPartMana)) {
                n = Math.min(n, 1);
            }
        }
        if (n > 1) {
            count("mask: repeatable mana ability (" + n + " uses)");
        }
        return Math.max(1, n);
    }

    private static void addAbility(List<Cand> out, Card c, SpellAbility ma, AbilityManaPart mp) {
        int amount = amountParam(ma);
        List<Integer> cols;
        boolean choice;
        if (mp.isComboMana() && amount > 1 && !ma.hasParam("Each")) {
            // Schema 18: one option; its colours are pay_mana_combo decisions.
            cols = PaymentCardinality.producible(mp, ma);
            if (cols.isEmpty()) {
                return;
            }
            Cand k = new Cand(ABILITY);
            k.card = c;
            k.ma = ma;
            k.ci = cols.get(0);
            k.combo = true;
            k.comboCols = cols;
            k.produce = new byte[amount];
            java.util.Arrays.fill(k.produce, BYTES[cols.get(0)]);
            out.add(k);
            return;
        }
        if (mp.isAnyMana() || mp.isComboMana()) {
            cols = PaymentCardinality.producible(mp, ma);
            choice = true;
        } else if (PaymentCardinality.reflects(ma)) {
            cols = PaymentCardinality.reflected(ma);
            if (cols.isEmpty()) {
                return; // Mox Amber with nothing to reflect: adds no mana
            }
            choice = true;
        } else {
            String s = mp.mana(ma);
            List<Integer> bundle = new ArrayList<>();
            if (s != null) {
                for (String tok : s.trim().split("\\s+")) {
                    int ci = tok.isEmpty() ? -1 : "WUBRGC".indexOf(Character.toUpperCase(tok.charAt(0)));
                    if (ci >= 0) {
                        bundle.add(ci);
                    }
                }
            }
            if (bundle.isEmpty()) {
                count("silent: unreadable mana production (offered as any colour)");
                cols = PaymentCardinality.ANY_COLOUR;
                choice = true;
            } else {
                Cand k = new Cand(ABILITY);
                k.card = c;
                k.ma = ma;
                k.ci = bundle.get(0);
                k.produce = new byte[bundle.size() * amount];
                for (int i = 0; i < k.produce.length; i++) {
                    k.produce[i] = BYTES[bundle.get(i % bundle.size())];
                }
                if (new java.util.HashSet<>(bundle).size() > 1) {
                    count("note: fixed multi-colour bundle");
                }
                out.add(k);
                return;
            }
        }
        for (int ci : cols) {
            Cand k = new Cand(ABILITY);
            k.card = c;
            k.ma = ma;
            k.ci = ci;
            k.choice = choice && cols.size() > 1;
            k.produce = new byte[amount];
            java.util.Arrays.fill(k.produce, BYTES[ci]);
            out.add(k);
        }
    }

    // ---- the mask ------------------------------------------------------------

    /**
     * Srcs of the mask model: one per pool unit; per card, one Src whose
     * options are the union of its plain abilities (alt 0), and one alt per
     * ability with a mana cost (filters, Signets; its cost on the Src) or an
     * "any combination" production (one unit Src per mana, every colour).
     * Repeatable abilities ({@link Cand#uses}) contribute that many copies.
     */
    private static List<PaymentCardinality.Src> model(List<Cand> all, Cand without, int poolUnitsToDrop) {
        List<PaymentCardinality.Src> out = new ArrayList<>();
        Map<Card, PaymentCardinality.Src> byCard = new IdentityHashMap<>();
        Map<Card, Integer> alts = new IdentityHashMap<>();
        Set<Card> idle = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (Cand c : all) {
            if (c.type == ABILITY && c.idleFodder != null) {
                idle.addAll(c.idleFodder);
            }
        }
        for (Card f : idle) {
            PaymentCardinality.Src u = new PaymentCardinality.Src();
            u.fodder = true;
            u.key = "fodder";
            out.add(u);
        }
        Map<SpellAbility, Integer> altOf = new IdentityHashMap<>();
        for (Cand c : all) {
            if (c.type == POOL) {
                int n = c.units.size() - (c == without ? poolUnitsToDrop : 0);
                for (int i = 0; i < n; i++) {
                    PaymentCardinality.Src u = unit(c.ci);
                    u.snow = c.snow;
                    out.add(u);
                }
            } else if (c.type == ABILITY) {
                if (without != null && without.type == ABILITY && c.card == without.card) {
                    continue;
                }
                boolean snow = c.card.isSnow();
                if (c.cost != null || c.combo || c.tapsSource || distinct(c.produce) > 1) {
                    Integer alt = altOf.get(c.ma);
                    boolean fresh = alt == null;
                    if (fresh) {
                        alt = alts.merge(c.card, 1, Integer::sum);
                        altOf.put(c.ma, alt);
                    }
                    PaymentCardinality.Src first = null;
                    for (PaymentCardinality.Src x : out) {
                        if (x.group == c.card && x.alt == alt) {
                            first = x;
                            break;
                        }
                    }
                    boolean bundle = !c.combo && distinct(c.produce) > 1;
                    if (bundle && fresh) {
                        for (int k = 0; k < c.produce.length; k++) {
                            PaymentCardinality.Src u = unit(colourIndex(c.produce[k]));
                            tag(u, c, alt, snow, k == 0);
                            out.add(u);
                        }
                    } else if (bundle) {
                        count("mask: second option of a fixed bundle ability (ignored)");
                    } else if (c.combo) {
                        for (int k = 0; k < c.produce.length * Math.max(1, c.uses); k++) {
                            PaymentCardinality.Src u = new PaymentCardinality.Src();
                            for (int ci : c.comboCols) {
                                u.options.add(new int[] {ci, 1});
                            }
                            tag(u, c, alt, snow, k == 0);
                            out.add(u);
                        }
                    } else if (first == null) {
                        first = new PaymentCardinality.Src();
                        tag(first, c, alt, snow, true);
                        first.options.add(new int[] {c.ci, c.produce.length});
                        out.add(first);
                    } else {
                        first.options.add(new int[] {c.ci, c.produce.length});
                    }
                    continue;
                }
                PaymentCardinality.Src s = byCard.get(c.card);
                if (s == null) {
                    s = new PaymentCardinality.Src();
                    s.group = c.card;
                    s.snow = snow;
                    s.fodder = isFodder(c.card);
                    byCard.put(c.card, s);
                    out.add(s);
                    for (int k = 1; k < c.uses; k++) {
                        PaymentCardinality.Src copy = new PaymentCardinality.Src();
                        copy.group = c.card;
                        copy.snow = snow;
                        copy.key = "copy";
                        out.add(copy);
                    }
                }
                boolean dup = false;
                for (int[] o : s.options) {
                    if (o[0] == c.ci && o[1] == c.produce.length) {
                        dup = true;
                    }
                }
                if (!dup) {
                    s.options.add(new int[] {c.ci, c.produce.length});
                    for (PaymentCardinality.Src x : out) {
                        if (x.group == c.card && x.alt == 0 && x != s && "copy".equals(x.key)) {
                            x.options.add(new int[] {c.ci, c.produce.length});
                        }
                    }
                }
            }
        }
        return out;
    }

    private static boolean isFodder(Card c) {
        try {
            return c.isCreature() && c.isInPlay() && !c.isTapped();
        } catch (RuntimeException e) {
            return false;
        }
    }

    private static int distinct(byte[] b) {
        java.util.Set<Byte> s = new java.util.HashSet<>();
        for (byte x : b) {
            s.add(x);
        }
        return s.size();
    }

    /** Marks a Src as alt {@code alt} of its card; the first Src of an alt carries the ability's cost. */
    private static void tag(PaymentCardinality.Src u, Cand c, int alt, boolean snow, boolean carriesCost) {
        u.group = c.card;
        u.alt = alt;
        u.snow = snow;
        u.filterOut = c.cost != null;
        u.fodder = isFodder(c.card);
        if (carriesCost && (c.cost != null || c.tapsSource)) {
            u.costShards = new ArrayList<>();
            int[] g = {0};
            if (c.cost != null) {
                shardsOf(c.cost, u.costShards, g);
            }
            if (c.tapsSource) {
                u.costShards.add(MaskCost.CREATURE);
            }
            u.costGeneric = g[0];
        }
    }

    /** Coloured/colourless shard masks of {@code cost} into {@code out}; generic (and X) into {@code generic[0]}. */
    private static void shardsOf(forge.card.mana.ManaCost cost, List<Byte> out, int[] generic) {
        generic[0] += cost.getGenericCost();
        for (ManaCostShard sh : cost) {
            if (sh == ManaCostShard.GENERIC) {
                continue;
            }
            if (sh == ManaCostShard.X || sh == ManaCostShard.COLORED_X || sh.isOr2Generic()) {
                generic[0] += sh.isOr2Generic() ? 2 : 0;
            } else if (sh.isSnow()) {
                out.add(MaskCost.SNOW);
            } else {
                out.add(sh.getColorMask());
            }
        }
    }

    private static PaymentCardinality.Src unit(int ci) {
        PaymentCardinality.Src s = new PaymentCardinality.Src();
        s.options.add(new int[] {ci, 1});
        return s;
    }

    /** How many Phyrexian shards life can still pay for, on top of {@code lifePaid}. */
    static int lifeShards(Player me, SpellAbility sa, ManaCostBeingPaid m, int lifePaid, boolean effect) {
        int phy = 0;
        for (ManaCostShard sh : m.getDistinctShards()) {
            if (sh.isPhyrexian()) {
                phy += m.getUnpaidShards(sh);
            }
        }
        int k = 0;
        try {
            while (k < phy && me.canPayLife(lifePaid + 2 * (k + 1), effect, sa)) {
                k++;
            }
        } catch (RuntimeException ignored) {
            // no life payment
        }
        return k;
    }

    private static final int SPLIT_CAP = 8;

    /**
     * Can {@code m} still be paid from {@code srcs}, with life covering up to
     * {@code lifeShards} Phyrexian shards? Each Phyrexian shard is paid by
     * mana or life and each monocoloured hybrid ({@code {2/W}}) by its colour
     * or two generic; every split is tried against {@link MaskCost#extendable}.
     */
    static boolean feasible(ManaCostBeingPaid m, List<PaymentCardinality.Src> srcs, int lifeShards) {
        return feasible(m, srcs, lifeShards, null);
    }

    /**
     * As {@link #feasible(ManaCostBeingPaid, List, int)}, plus {@code ownCost}:
     * the cost of a mana ability just taken (a filter), which its own output
     * ({@code filterOut} Srcs) cannot pay.
     */
    static boolean feasible(ManaCostBeingPaid m, List<PaymentCardinality.Src> srcs, int lifeShards,
            forge.card.mana.ManaCost ownCost) {
        return feasible(m, srcs, lifeShards, ownCost, null);
    }

    static boolean feasible(ManaCostBeingPaid m, List<PaymentCardinality.Src> srcs, int lifeShards,
            forge.card.mana.ManaCost ownCost, List<Byte> extraCost) {
        List<Byte> shards = new ArrayList<>();
        if (extraCost != null) {
            shards.addAll(extraCost);
        }
        if (ownCost != null) {
            List<Byte> cs = new ArrayList<>();
            int[] g = {0};
            shardsOf(ownCost, cs, g);
            for (byte b : cs) {
                shards.add((byte) (b | MaskCost.COST_FLAG));
            }
            for (int k = 0; k < g[0]; k++) {
                shards.add((byte) (MaskCost.ANY | MaskCost.COST_FLAG));
            }
        }
        List<Byte> phy = new ArrayList<>();
        List<Byte> two = new ArrayList<>();
        int generic = m.getGenericManaAmount();
        for (ManaCostShard sh : m.getDistinctShards()) {
            if (sh == ManaCostShard.GENERIC) {
                continue;
            }
            int n = m.getUnpaidShards(sh);
            for (int i = 0; i < n; i++) {
                if (sh.isPhyrexian()) {
                    phy.add(sh.getColorMask());
                } else if (sh.isOr2Generic()) {
                    two.add(sh.getColorMask());
                } else if (sh == ManaCostShard.X || sh == ManaCostShard.COLORED_X) {
                    count("mask: unpaid X shard (as generic)");
                    generic++;
                } else if (sh.isSnow()) {
                    shards.add(MaskCost.SNOW);
                } else {
                    shards.add(sh.getColorMask());
                }
            }
        }
        int n = phy.size() + two.size();
        if (n > SPLIT_CAP) {
            count("mask: split cap exceeded (loose)");
            return true;
        }
        for (int bits = 0; bits < (1 << n); bits++) {
            List<Byte> s = new ArrayList<>(shards);
            int g = generic;
            int life = 0;
            for (int i = 0; i < n; i++) {
                boolean alt = (bits & (1 << i)) != 0;
                if (i < phy.size()) {
                    if (alt) {
                        life++;
                    } else {
                        s.add(phy.get(i));
                    }
                } else if (alt) {
                    g += 2;
                } else {
                    s.add(two.get(i - phy.size()));
                }
            }
            if (life > lifeShards) {
                continue;
            }
            if (extendableAlts(s, g, srcs)) {
                return true;
            }
        }
        return false;
    }

    /** Most MaskCost.extendable calls one alternative search may make before answering loosely. */
    private static final int ALT_BUDGET = 512;

    /**
     * {@link MaskCost#extendable} with per-card alternatives (Srcs sharing a
     * group): one alt per card is used -- the plain one (0), or an ability
     * with a mana cost or bundle; an ability with a cost may also be left
     * unused. A used ability's cost joins the requirement as cost shards,
     * which its own output (and other filters' output) cannot pay.
     * Assignments are tried by how many cards leave their default (plain, or
     * unused for a card with only costed abilities, or its only costless
     * alternative), after a relaxed necessary check (every alternative's
     * output, no costs). Over {@link #ALT_BUDGET} the answer is loose.
     */
    static boolean extendableAlts(List<Byte> shards, int generic, List<PaymentCardinality.Src> srcs) {
        Map<Object, List<Integer>> choices = new LinkedHashMap<>();
        for (PaymentCardinality.Src x : srcs) {
            if (x.group != null && x.alt > 0) {
                choices.computeIfAbsent(x.group, k -> new ArrayList<>());
                if (!choices.get(x.group).contains(x.alt)) {
                    choices.get(x.group).add(x.alt);
                }
            }
        }
        if (choices.isEmpty()) {
            return MaskCost.extendable(shards, generic, new ArrayList<>(srcs));
        }
        if (!MaskCost.extendable(shards, generic, new ArrayList<>(srcs))) {
            return false; // even every alternative at once, free, is not enough
        }
        List<Object> groups = new ArrayList<>(choices.keySet());
        for (Object g : groups) {
            boolean plain = false;
            boolean costed = false;
            for (PaymentCardinality.Src x : srcs) {
                plain |= x.group == g && x.alt == 0;
                costed |= x.group == g && x.costShards != null;
            }
            List<Integer> l = choices.get(g);
            if (plain) {
                l.add(0, 0);
            } else if (costed || l.size() > 1) {
                l.add(0, -1);
            }
        }
        int[] budget = {ALT_BUDGET};
        for (int k = 0; k <= groups.size(); k++) {
            Boolean r = tryAlts(shards, generic, srcs, groups, choices, new int[groups.size()], 0, k, budget);
            if (r != null) {
                return r;
            }
        }
        if (budget[0] <= 0) {
            count("mask: alternative budget exceeded (loose)");
            return true;
        }
        return false;
    }

    /** Assignments with exactly {@code left} more non-default picks from group {@code at}; null: none feasible. */
    private static Boolean tryAlts(List<Byte> shards, int generic, List<PaymentCardinality.Src> srcs,
            List<Object> groups, Map<Object, List<Integer>> choices, int[] pick, int at, int left, int[] budget) {
        if (budget[0] <= 0) {
            return null;
        }
        if (at == groups.size()) {
            if (left > 0) {
                return null;
            }
            budget[0]--;
            Map<Object, Integer> chosen = new IdentityHashMap<>();
            for (int i = 0; i < groups.size(); i++) {
                chosen.put(groups.get(i), choices.get(groups.get(i)).get(pick[i]));
            }
            List<Byte> s = new ArrayList<>(shards);
            List<PaymentCardinality.Src> use = new ArrayList<>();
            for (PaymentCardinality.Src x : srcs) {
                Integer a = x.group == null ? null : chosen.get(x.group);
                if (a != null && a != x.alt) {
                    continue;
                }
                use.add(x);
                if (x.costShards != null) {
                    for (byte b : x.costShards) {
                        s.add((byte) (b | MaskCost.COST_FLAG));
                    }
                    for (int k = 0; k < x.costGeneric; k++) {
                        s.add((byte) (MaskCost.ANY | MaskCost.COST_FLAG));
                    }
                }
            }
            return MaskCost.extendable(s, generic, use) ? Boolean.TRUE : null;
        }
        int n = choices.get(groups.get(at)).size();
        pick[at] = 0;
        if (groups.size() - at - 1 >= left) {
            Boolean r = tryAlts(shards, generic, srcs, groups, choices, pick, at + 1, left, budget);
            if (r != null) {
                return r;
            }
        }
        if (left > 0) {
            for (int o = 1; o < n; o++) {
                pick[at] = o;
                Boolean r = tryAlts(shards, generic, srcs, groups, choices, pick, at + 1, left - 1, budget);
                if (r != null) {
                    return r;
                }
            }
        }
        pick[at] = 0;
        return null;
    }

    /**
     * The residual after taking {@code c} (on a copy), or null if {@code c}
     * pays nothing. Unpaid output of an ability floats: {@code surplus} gets
     * its colour indices.
     */
    private static ManaCostBeingPaid after(Cand c, ManaCostBeingPaid m, Player me, SpellAbility sa,
            List<Integer> surplus) {
        ManaCostBeingPaid copy = new ManaCostBeingPaid(m);
        ManaPool pool = me.getManaPool();
        try {
            if (c.type == LIFE) {
                return copy.payPhyrexian() ? copy : null;
            }
            if (c.type == POOL) {
                Mana u = c.units.get(0);
                return sa.allowsPayingWithShard(u.getSourceCard(), u.getColor())
                        && copy.isNeeded(u, pool) && copy.payMana(u, pool) ? copy : null;
            }
            int paid = 0;
            AbilityManaPart mp = c.ma.getManaPart();
            for (byte b : c.produce) {
                Mana u = new Mana(b, c.card, mp, me);
                if (sa.allowsPayingWithShard(c.card, b) && copy.isNeeded(u, pool) && copy.payMana(u, pool)) {
                    paid++;
                } else if (surplus != null) {
                    surplus.add(colourIndex(b));
                }
            }
            return paid > 0 ? copy : null;
        } catch (RuntimeException e) {
            count("mask: simulation threw");
            return null;
        }
    }

    /** The legal candidates: those after which the rest of the cost is still payable. */
    static List<Cand> legal(List<Cand> all, ManaCostBeingPaid m, Player me, SpellAbility sa, int lifeShards) {
        return legal(all, m, me, sa, lifeShards, null);
    }

    /**
     * {@code outer}: the residual of the payment this one is nested in (the
     * spell a filter is being activated for), which the rest must also cover.
     */
    static List<Cand> legal(List<Cand> all, ManaCostBeingPaid m, Player me, SpellAbility sa, int lifeShards,
            forge.card.mana.ManaCost outer) {
        List<Cand> out = new ArrayList<>();
        for (Cand c : all) {
            for (byte[] produce : productions(c)) {
                if (legalWith(c, produce, all, m, me, sa, lifeShards, outer)) {
                    out.add(c);
                    break;
                }
            }
        }
        return out;
    }

    /** What one activation of {@code c} may add: its fixed output, or every colour multiset of a combo. */
    private static List<byte[]> productions(Cand c) {
        if (c.type != ABILITY || !c.combo) {
            return java.util.Collections.singletonList(c.produce);
        }
        List<byte[]> out = new ArrayList<>();
        multisets(c.comboCols, c.produce.length, 0, new byte[c.produce.length], 0, out);
        return out;
    }

    private static void multisets(List<Integer> cols, int n, int from, byte[] cur, int at, List<byte[]> out) {
        if (out.size() >= 64) {
            return;
        }
        if (at == n) {
            out.add(cur.clone());
            return;
        }
        for (int i = from; i < cols.size(); i++) {
            cur[at] = BYTES[cols.get(i)];
            multisets(cols, n, i, cur, at + 1, out);
        }
    }

    private static boolean legalWith(Cand c, byte[] produce, List<Cand> all, ManaCostBeingPaid m, Player me,
            SpellAbility sa, int lifeShards, forge.card.mana.ManaCost outer) {
        List<Integer> surplus = new ArrayList<>();
        byte[] keep = c.produce;
        ManaCostBeingPaid next;
        try {
            c.produce = produce;
            next = after(c, m, me, sa, surplus);
        } finally {
            c.produce = keep;
        }
        if (next == null) {
            return false;
        }
        if (outer != null) {
            next.addManaCost(outer);
        }
        List<PaymentCardinality.Src> rem = model(all, c, 1);
        for (int ci : surplus) {
            PaymentCardinality.Src u = unit(ci);
            u.snow = c.type == ABILITY && c.card.isSnow();
            u.filterOut = c.cost != null;
            rem.add(u);
        }
        if (c.type == ABILITY && c.tapsSource) {
            List<Byte> extra = new ArrayList<>();
            extra.add((byte) (MaskCost.CREATURE | MaskCost.COST_FLAG));
            return feasible(next, rem, lifeShards, c.cost, extra);
        }
        return feasible(next, rem, lifeShards - (c.type == LIFE ? 1 : 0), c.type == ABILITY ? c.cost : null);
    }

    /** All candidates plus {@code pay_life} when a Phyrexian shard is unpaid and life allows. */
    private static List<Cand> withLife(List<Cand> all, int lifeShards) {
        if (lifeShards <= 0) {
            return all;
        }
        List<Cand> out = new ArrayList<>(all);
        out.add(new Cand(LIFE));
        return out;
    }

    // ---- the record ------------------------------------------------------------

    static Serializer.Prompt prompt(InstrumentedController ctl, SpellAbility sa, ManaCostBeingPaid m,
            List<Mana> spent, int lifePaid, List<Card> used) {
        Serializer.Prompt p = ctl.promptFor(KIND, sa);
        int[] rem = new int[6];
        for (ManaCostShard sh : m.getDistinctShards()) {
            if (sh == ManaCostShard.GENERIC) {
                continue;
            }
            int n = m.getUnpaidShards(sh);
            byte mask = sh.getColorMask();
            if (mask == 0) {
                if (sh.isOfKind(ManaAtom.COLORLESS)) {
                    rem[5] += n;
                }
                continue;
            }
            for (int i = 0; i < 5; i++) {
                if ((mask & BYTES[i]) != 0) {
                    rem[i] += n;
                }
            }
        }
        int[] sp = new int[6];
        int snow = 0;
        for (Mana x : spent) {
            sp[colourIndex(x.getColor())]++;
            if (x.isSnow()) {
                snow++;
            }
        }
        p.remainingCost(costText(m)).remaining(m.getGenericManaAmount(), rem).spent(sp, snow, lifePaid);
        for (Card c : used) {
            p.select(c);
        }
        return p;
    }

    private static String costText(ManaCostBeingPaid m) {
        try {
            String s = m.toManaCost().toString();
            return s == null || s.isBlank() ? null : s;
        } catch (RuntimeException e) {
            return null;
        }
    }

    static List<Serializer.Cand> options(List<Cand> legal, forge.game.Game game) {
        List<Serializer.Cand> out = new ArrayList<>();
        for (Cand c : legal) {
            if (c.type == LIFE) {
                out.add(Serializer.Cand.option(VERB_LIFE).cost("2 life"));
            } else if (c.type == POOL) {
                Serializer.Cand o = Serializer.Cand.option(VERB_POOL).manaColor(COLOURS[c.ci])
                        .manaAmount(c.units.size()).restricted(c.restricted).snow(c.snow);
                if (c.rider != null) {
                    o.rider(c.rider).edge(Schema.SOURCE_IS, c.card);
                }
                out.add(o);
            } else {
                String text = Serializer.optionText(String.valueOf(c.ma), c.ma, game);
                Serializer.Cand o = Serializer.Cand.option(VERB_ACTIVATE)
                        .ability(text != null ? text : "Add mana");
                if (!c.combo) {
                    o.manaColor(COLOURS[c.ci]);
                }
                o.manaAmount(c.produce.length).edge(Schema.SOURCE_IS, c.card);
                if (c.cost != null) {
                    o.cost(c.cost.toString());
                }
                out.add(o);
            }
        }
        return out;
    }

    // ---- decide mode -------------------------------------------------------------

    /**
     * {@code applyManaToCost} for a routed payer: ask one step at a time until
     * {@code m} is paid. Falls back to Forge's AI for the remainder (counted)
     * if no candidate is legal, which the mask should make impossible.
     */
    static boolean payInteractive(InstrumentedController ctl, ManaCostBeingPaid m, SpellAbility sa,
            boolean effect) {
        Player me = ctl.getPlayer();
        count("decide: payments");
        // Cost adjustment (Delve/Convoke) is over; nested checks are not ours.
        Object[] adjusting = beginAdjust(null, null);
        forge.card.mana.ManaCost outer = OUTER.get();
        OUTER.remove();
        Combo outerCombo = COMBO.get();
        COMBO.remove();
        me.pushPaidForSA(sa);
        sa.setManaCostBeingPaid(m);
        int lifePaid = 0;
        List<Card> used = new ArrayList<>();
        Set<Card> exclude = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        List<Cand> failed = new ArrayList<>();
        boolean ok = false;
        try {
            int guard = 0;
            while (!m.isPaid()) {
                if (++guard > 100) {
                    count("decide: step guard tripped");
                    break;
                }
                int lifeShards = lifeShards(me, sa, m, lifePaid, effect);
                List<Cand> all = withLife(enumerate(me, sa, exclude), lifeShards);
                all.removeIf(c -> failed.stream().anyMatch(f -> f.same(c)));
                List<Cand> legal = legal(all, m, me, sa, lifeShards, outer);
                boolean first = guard == 1;
                if (first && AUDIT) {
                    agreement(m, sa, me, effect, !legal.isEmpty());
                }
                if (legal.isEmpty()) {
                    // At the first step this is an unaffordable cast (the
                    // priority pool is unfiltered, section 11.1), not a mask
                    // failure; Forge's AI then trying anyway tells which.
                    boolean ai = forge.ai.ComputerUtilMana.payManaCost(m, sa, me, effect);
                    if (first) {
                        count(ai ? "decide: UNPAYABLE PER MASK BUT AI PAID (mask too strict)"
                                : "decide: unaffordable at first step (AI agrees)");
                    } else {
                        count("decide: DEAD END mid-payment");
                        count(ai ? "decide: dead end, AI fallback paid" : "decide: dead end, AI fallback failed");
                    }
                    if (!first) {
                        StringBuilder d = new StringBuilder();
                        for (Cand x : all) {
                            d.append(" [").append(describe(x)).append(x.type == ABILITY ? " " + x.ma : "")
                                    .append(x.cost != null ? " cost " + x.cost : "").append(" uses ").append(x.uses)
                                    .append(x.tapsSource ? " tapsSource" : "").append(']');
                        }
                        log("dead end detail: used " + used + " spent " + sa.getPayingMana() + " candidates" + d);
                    }
                    if (!first || ai) {
                        log((first ? "too strict" : "dead end") + " paying " + sa.getHostCard()
                                + (outer != null ? " (nested; outer residual " + outer + ")" : "")
                                + " residual " + costText(m) + " shards " + shardDump(m) + " paid=" + m.isPaid()
                                + " candidates " + all.size()
                                + " life " + lifePaid);
                    }
                    break;
                }
                int pick;
                count("decide: steps");
                if (legal.size() == 1) {
                    count("decide: forced steps (not asked)");
                    pick = 0;
                } else {
                    pick = ctl.askChoice(prompt(ctl, sa, m, sa.getPayingMana(), lifePaid, used),
                            options(legal, me.getGame()));
                    if (pick == ActionSource.DELEGATE || pick < 0 || pick >= legal.size()) {
                        count("decide: source delegated (AI pays the rest)");
                        forge.ai.ComputerUtilMana.payManaCost(m, sa, me, effect);
                        break;
                    }
                    count("decide: asked");
                    count("decide: width", legal.size());
                }
                Cand c = legal.get(pick);
                int[] life = {lifePaid};
                if (execute(ctl, c, m, sa, me, effect, life, all, lifeShards)) {
                    lifePaid = life[0];
                    if (c.type == ABILITY && !used.contains(c.card)) {
                        used.add(c.card);
                        exclude.add(c.card);
                    }
                    count(c.type == ABILITY ? "decide: step activate_mana"
                            : c.type == POOL ? "decide: step use_pool" : "decide: step pay_life");
                } else {
                    count("decide: STEP FAILED (excluded, re-asked)");
                    log("step failed: " + describe(c) + " paying " + sa.getHostCard()
                            + " residual " + costText(m) + " shards " + shardDump(m)
                            + " unit " + (c.type == POOL ? c.units.get(0) : null));
                    failed.add(c);
                    if (c.type == ABILITY) {
                        exclude.add(c.card);
                    }
                }
            }
            ok = m.isPaid();
            if (ok && lifePaid > 0 && !me.payLife(lifePaid, sa, effect)) {
                count("decide: Phyrexian life payment refused");
                ok = false;
            }
            return ok;
        } finally {
            endAdjust(adjusting);
            if (outer != null) {
                OUTER.set(outer);
            }
            if (outerCombo != null) {
                COMBO.set(outerCombo);
            }
            sa.setManaCostBeingPaid(null);
            me.popPaidForSA();
            if (!ok) {
                count("decide: payment returned false (engine rolls back)");
            }
        }
    }

    /**
     * At the first step, our mask and Forge's own affordability helper answer
     * the same question; a disagreement is either a gap in our model or Forge's
     * AI declining something the rules allow (section 14.10).
     */
    private static void agreement(ManaCostBeingPaid m, SpellAbility sa, Player me, boolean effect,
            boolean own) {
        boolean theirs;
        try {
            theirs = forge.ai.ComputerUtilMana.canPayManaCost(new ManaCostBeingPaid(m), sa, me, effect);
        } catch (RuntimeException e) {
            count("decide: first-step check threw");
            return;
        }
        count(own == theirs ? "decide: first step agrees with canPayManaCost"
                : own ? "decide: first step own=yes forge=no" : "decide: first step own=no forge=yes");
        if (own != theirs) {
            StringBuilder srcs = new StringBuilder();
            for (Cand c : enumerate(me, sa, java.util.Collections.emptySet())) {
                srcs.append(" [").append(describe(c)).append(c.type == ABILITY ? " " + c.ma : "").append(']');
            }
            log("first-step disagreement own=" + own + " forge=" + theirs + " paying " + sa.getHostCard()
                    + " residual " + costText(m) + " life " + me.getLife() + " sources" + srcs);
        }
    }

    private static String shardDump(ManaCostBeingPaid m) {
        StringBuilder b = new StringBuilder();
        for (ManaCostShard sh : m.getDistinctShards()) {
            b.append(sh.name()).append('x').append(m.getUnpaidShards(sh)).append(' ');
        }
        return b.toString().trim();
    }

    private static String describe(Cand c) {
        if (c.type == LIFE) {
            return "pay 2 life";
        }
        if (c.type == POOL) {
            return "pool " + COLOURS[c.ci] + (c.rider != null ? " (rider)" : "");
        }
        return c.card + " -> " + COLOURS[c.ci] + "x" + c.produce.length;
    }

    /** One step, the way the human controller's InputPayMana takes it. */
    private static boolean execute(InstrumentedController ctl, Cand c, ManaCostBeingPaid m, SpellAbility sa,
            Player me, boolean effect, int[] lifePaid, List<Cand> all, int lifeShards) {
        ManaPool pool = me.getManaPool();
        // Progress is mana spent on this cost, not fewer unpaid shards: paying
        // {2/W} with non-white mana turns the shard into {1}, same count.
        int before = sa.getPayingMana().size();
        try {
            if (c.type == LIFE) {
                if (!me.canPayLife(lifePaid[0] + 2, effect, sa) || !m.payPhyrexian()) {
                    return false;
                }
                sa.setSpendPhyrexianMana(true);
                lifePaid[0] += 2;
                return true;
            }
            if (c.type == POOL) {
                // ManaPool.tryPayCostWithColor (InputPayMana.useManaFromPool)
                // for this one unit: it would pick the unit of that colour itself.
                Mana u = c.units.get(0);
                if (!u.meetsManaRestrictions(sa) || !sa.allowsPayingWithShard(u.getSourceCard(), u.getColor())
                        || !pool.tryPayCostWithMana(sa, m, u, false)) {
                    return false;
                }
                sa.getPayingMana().add(u);
                return true;
            }
            SpellAbility ma = c.ma;
            ma.setActivatingPlayer(me);
            if (c.combo) {
                for (AbilityManaPart mp : ma.getAllManaParts()) {
                    mp.clearExpressChoice();
                }
            } else if (c.choice && c.ci < 5) {
                ma.setManaExpressChoice(forge.card.ColorSet.fromMask(BYTES[c.ci]));
            }
            int poolBefore = pool.totalMana();
            forge.card.mana.ManaCost prevOuter = OUTER.get();
            Combo prevCombo = COMBO.get();
            if (c.cost != null) {
                // the nested payment of the filter's cost must leave the rest payable
                // (a combination filter: the residual of a production that works)
                ManaCostBeingPaid res = null;
                byte[] keep = c.produce;
                try {
                    for (byte[] p : productions(c)) {
                        c.produce = p;
                        ManaCostBeingPaid r = after(c, m, me, sa, null);
                        if (r != null && (res == null || legalWith(c, p, all, m, me, sa, lifeShards, null))) {
                            res = r;
                            if (legalWith(c, p, all, m, me, sa, lifeShards, null)) {
                                break;
                            }
                        }
                    }
                } finally {
                    c.produce = keep;
                }
                OUTER.set((res != null ? res : m).toManaCost());
                count("decide: step with a nested payment (filter)");
            }
            if (c.combo) {
                Combo cx = new Combo(m, sa, c.card, ma);
                cx.step = all;
                COMBO.set(cx);
            }
            boolean played;
            Object[] prevStep = STEP.get();
            STEP.set(new Object[] {c, m, sa, lifeShards, all});
            try {
                played = forge.game.player.PlaySpellAbility.playSpellAbility(ctl, me, ma);
            } finally {
                if (prevStep == null) {
                    STEP.remove();
                } else {
                    STEP.set(prevStep);
                }
                if (prevOuter == null) {
                    OUTER.remove();
                } else {
                    OUTER.set(prevOuter);
                }
                if (prevCombo == null) {
                    COMBO.remove();
                } else {
                    COMBO.set(prevCombo);
                }
            }
            if (!played) {
                count("decide: mana ability activation refused");
                return false;
            }
            boolean meets = true;
            for (AbilityManaPart mp : ma.getAllManaParts()) {
                if (!mp.meetsManaRestrictions(sa)) {
                    meets = false;
                    break;
                }
            }
            if (meets) {
                pool.payManaFromAbility(sa, m, ma);
            }
            try {
                java.util.Collection<Mana> made = ma.getManaPart().getLastManaProduced();
                for (Mana x : made) {
                    if (!c.combo && colourIndex(x.getColor()) != c.ci && c.produce.length > 0
                            && java.util.Arrays.binarySearch(sortedCopy(c.produce), x.getColor()) < 0) {
                        count("decide: produced colour differs from the option");
                        break;
                    }
                }
            } catch (RuntimeException ignored) {
                // diagnostic only
            }
            if (sa.getPayingMana().size() == before && pool.totalMana() > poolBefore) {
                count("decide: step produced mana that paid nothing (floats)");
                log("step floated: " + describe(c) + " for " + sa.getHostCard() + " residual " + costText(m)
                        + " pool " + pool.totalMana() + " meets " + meets + " produced "
                        + ma.getManaPart().getLastManaProduced());
            }
            return sa.getPayingMana().size() > before || pool.totalMana() > poolBefore;
        } catch (RuntimeException e) {
            count("decide: step threw");
            log("step threw " + e + " on " + describe(c));
            return false;
        }
    }

    private static byte[] sortedCopy(byte[] b) {
        byte[] s = b.clone();
        java.util.Arrays.sort(s);
        return s;
    }

    // ---- nested contexts and "any combination" colours (schema 18) -------------------

    /** Residual of the payment a filter's nested cost payment is made for. */
    private static final ThreadLocal<forge.card.mana.ManaCost> OUTER = new ThreadLocal<>();

    /** The routed payment step whose mana ability is being activated: {Cand, m, sa, lifeShards}. */
    private static final ThreadLocal<Object[]> STEP = new ThreadLocal<>();

    /**
     * Mask for a card chosen to pay a mana ability's own non-mana cost
     * (Springleaf Drum's tap, Ashnod's Altar's sacrifice) during a routed
     * step: after losing {@code f} as a source, is the payment the step is
     * for still payable? True outside a step.
     */
    static boolean stepKeepsPayable(Player me, Card f) {
        Object[] st = STEP.get();
        if (st == null) {
            return true;
        }
        Cand c = (Cand) st[0];
        ManaCostBeingPaid m = (ManaCostBeingPaid) st[1];
        SpellAbility sa = (SpellAbility) st[2];
        int life = (Integer) st[3];
        try {
            // the step's own candidate list (re-enumerating now would see
            // sources as unactivatable while this ability is mid-activation)
            List<Cand> all = new ArrayList<>();
            @SuppressWarnings("unchecked")
            List<Cand> step = (List<Cand>) st[4];
            for (Cand x : step) {
                if (!(x.type == ABILITY && x.card == f)) {
                    all.add(x);
                }
            }
            // f pays the tap cost itself
            boolean taps = c.tapsSource;
            List<Card> idle = c.idleFodder;
            c.tapsSource = false;
            c.idleFodder = null;
            for (Cand x : all) {
                if (x.idleFodder != null && x.idleFodder.contains(f)) {
                    x.idleFodder = new ArrayList<>(x.idleFodder);
                    x.idleFodder.remove(f);
                }
            }
            try {
                for (byte[] p : productions(c)) {
                    if (legalWith(c, p, all, m, me, sa, life, null)) {
                        return true;
                    }
                }
                return false;
            } finally {
                c.tapsSource = taps;
                c.idleFodder = idle;
            }
        } catch (RuntimeException e) {
            count("mask: step cost check threw");
            return true;
        }
    }

    static void debugStep(Player me, List<Card> left) {
        Object[] st = STEP.get();
        if (st == null) {
            log("cost cards all kept, no step: " + left);
            return;
        }
        Cand c = (Cand) st[0];
        StringBuilder d = new StringBuilder();
        @SuppressWarnings("unchecked")
        List<Cand> step = (List<Cand>) st[4];
        for (Cand x : step) {
            d.append(" [").append(describe(x)).append(x.cost != null ? " cost " + x.cost : "")
                    .append(x.tapsSource ? " tapsSource" : "").append(x.fodder != null ? " fodder " + x.fodder : "")
                    .append(']');
        }
        log("cost cards all kept: step " + describe(c) + " residual " + costText((ManaCostBeingPaid) st[1]) + " cost cards " + left
                + " step cands" + d);
    }

    /** The routed payment step whose combination ability is resolving. */
    private static final ThreadLocal<Combo> COMBO = new ThreadLocal<>();

    static final String KIND_COMBO = "pay_mana_combo";
    static final String VERB_COLOR = "choose_color";

    private static final class Combo {
        final ManaCostBeingPaid m;
        final SpellAbility sa;
        final Card card;
        final SpellAbility ma;
        /** Observe mode: the other candidates as they were (null: the live state). */
        List<Cand> rest;
        /** Decide mode: the step's candidates, refreshed when the colours are asked. */
        List<Cand> step;
        int lifeShards = -1;

        Combo(ManaCostBeingPaid m, SpellAbility sa, Card card, SpellAbility ma) {
            this.m = m;
            this.sa = sa;
            this.card = card;
            this.ma = ma;
        }
    }

    /** Colour bytes {@code options} allows for the next unit, in WUBRG order. */
    private static List<Byte> comboColours(forge.card.ColorSet options, List<Byte> chosen, boolean different) {
        List<Byte> out = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            if ((options.getColor() & BYTES[i]) != 0 && !(different && chosen.contains(BYTES[i]))) {
                out.add(BYTES[i]);
            }
        }
        return out;
    }

    /** Some completion of {@code chosen} to {@code amount} units leaves the payment payable. */
    private static boolean comboFeasible(Combo x, Player me, List<Byte> chosen, forge.card.ColorSet options,
            int amount, boolean different) {
        if (chosen.size() == amount) {
            Cand c = new Cand(ABILITY);
            c.card = x.card;
            c.ma = x.ma;
            c.produce = new byte[amount];
            for (int i = 0; i < amount; i++) {
                c.produce[i] = chosen.get(i);
            }
            List<Integer> surplus = new ArrayList<>();
            ManaCostBeingPaid next = after(c, x.m, me, x.sa, surplus);
            if (next == null) {
                return false;
            }
            Set<Card> ex = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            ex.add(x.card);
            List<Cand> rest = x.rest;
            if (rest == null && x.step != null) {
                // the step's sources still usable (its own cost may have used some)
                List<Mana> floating = new ArrayList<>();
                for (Mana u : me.getManaPool()) {
                    floating.add(u);
                }
                rest = poolCands(floating, x.sa, me.getGame());
                for (Cand k : x.step) {
                    if (k.type == ABILITY && k.card != x.card && !(k.card.isTapped() && k.ma.getPayCosts() != null
                            && k.ma.getPayCosts().hasTapCost()) && k.card.getZone() != null) {
                        rest.add(k);
                    } else if (k.type == LIFE) {
                        rest.add(k);
                    }
                }
            }
            List<PaymentCardinality.Src> rem = model(rest != null ? rest : enumerate(me, x.sa, ex), null, 0);
            for (int ci : surplus) {
                PaymentCardinality.Src u = unit(ci);
                u.snow = x.card.isSnow();
                rem.add(u);
            }
            return feasible(next, rem, x.lifeShards >= 0 ? x.lifeShards : lifeShards(me, x.sa, next, 0, false));
        }
        for (byte b : comboColours(options, chosen, different)) {
            chosen.add(b);
            boolean ok = comboFeasible(x, me, chosen, options, amount, different);
            chosen.remove(chosen.size() - 1);
            if (ok) {
                return true;
            }
        }
        return false;
    }

    /** Whether some unordered completion works: colours chosen so far may be any order. */
    private static boolean comboLegal(Combo x, Player me, List<Byte> chosen, byte b, forge.card.ColorSet options,
            int amount, boolean different) {
        if (amount > 3) {
            return true;
        }
        List<Byte> base = new ArrayList<>(chosen);
        base.add(b);
        try {
            return comboFeasible(x, me, base, options, amount, different);
        } catch (RuntimeException e) {
            count("mask: combo check threw");
            return true;
        }
    }

    private static List<Byte> legalComboColours(Combo x, Player me, List<Byte> chosen,
            forge.card.ColorSet options, int amount, boolean different) {
        List<Byte> all = comboColours(options, chosen, different);
        if (x == null) {
            return all;
        }
        List<Byte> out = new ArrayList<>();
        for (byte b : all) {
            if (comboLegal(x, me, chosen, b, options, amount, different)) {
                out.add(b);
            }
        }
        if (out.isEmpty()) {
            count("mask: combo no colour helps (all offered)");
            return all;
        }
        return out;
    }

    private static Serializer.Prompt comboPrompt(InstrumentedController ctl, SpellAbility ma, Combo x,
            List<Byte> chosen) {
        Serializer.Prompt p = ctl.promptFor(KIND_COMBO, x != null ? x.sa : ma).selectedOther(chosen.size());
        if (x != null) {
            ManaCostBeingPaid sim = new ManaCostBeingPaid(x.m);
            for (byte b : chosen) {
                Mana u = new Mana(b, x.card, x.ma.getManaPart(), ctl.getPlayer());
                if (sim.isNeeded(u, ctl.getPlayer().getManaPool())) {
                    sim.payMana(u, ctl.getPlayer().getManaPool());
                }
            }
            remainingInto(p, sim);
            p.edge(Schema.OBJ, x.card);
        } else if (ma != null) {
            p.edge(Schema.OBJ, ma.getHostCard());
        }
        return p;
    }

    private static List<Serializer.Cand> comboOpts(List<Byte> legal) {
        List<Serializer.Cand> out = new ArrayList<>();
        for (byte b : legal) {
            out.add(Serializer.Cand.option(VERB_COLOR).manaColor(COLOURS[colourIndex(b)]));
        }
        return out;
    }

    /**
     * specifyManaCombo for a routed payer: one colour per unit
     * ({@code pay_mana_combo}), masked against the payment it is made for when
     * inside a routed step; any allowed colour otherwise (Manamorphose).
     * Null when the source delegated.
     */
    static Map<Byte, Integer> combo(InstrumentedController ctl, SpellAbility ma, forge.card.ColorSet options,
            int amount, boolean different) {
        Combo x = COMBO.get();
        if (x != null && x.ma != ma && (ma == null || x.card != ma.getHostCard())) {
            x = null;
        }
        count(x != null ? "decide: combo colours in a payment" : "decide: combo colours outside a payment");
        List<Byte> chosen = new ArrayList<>();
        for (int k = 0; k < amount; k++) {
            List<Byte> legal = legalComboColours(x, ctl.getPlayer(), chosen, options, amount, different);
            if (legal.isEmpty()) {
                return null;
            }
            int pick;
            if (legal.size() == 1) {
                pick = 0;
                count("decide: combo colour forced");
            } else {
                pick = ctl.askChoice(comboPrompt(ctl, ma, x, chosen), comboOpts(legal));
                if (pick < 0 || pick >= legal.size()) {
                    count("decide: combo delegated");
                    return null;
                }
                count("decide: asked pay_mana_combo");
            }
            chosen.add(legal.get(pick));
        }
        Map<Byte, Integer> out = new LinkedHashMap<>();
        for (byte b : chosen) {
            out.merge(b, 1, Integer::sum);
        }
        return out;
    }

    // ---- Convoke / Improvise / Delve -------------------------------------------------
    //
    // These are not steps of the mana payment in Forge: CostAdjustment.adjust
    // asks for them up front (Delve, then Convoke/Improvise), before
    // applyManaToCost, through PlayerController.chooseCardsToDelve and
    // chooseCardsForConvokeOrImprovise. They are routed there as their own
    // sequential decisions (pay_delve, pay_convoke), masked like pay_mana:
    // stopping is offered only if the rest is payable from mana sources, a
    // card only if the rest stays payable. CostAdjustment also calls both in
    // its test mode (affordability checks), so they are routed only inside a
    // payment this class started (ADJUSTING).

    static final String KIND_CONVOKE = "pay_convoke";
    static final String KIND_DELVE = "pay_delve";

    /** The cost and ability being adjusted by a payment we started; null outside one. */
    private static final ThreadLocal<Object[]> ADJUSTING = new ThreadLocal<>();

    static Object[] beginAdjust(forge.card.mana.ManaCost cost, SpellAbility sa) {
        Object[] prev = ADJUSTING.get();
        if (cost == null) {
            ADJUSTING.remove();
        } else {
            ADJUSTING.set(new Object[] {cost, sa});
        }
        return prev;
    }

    static void endAdjust(Object[] prev) {
        if (prev == null) {
            ADJUSTING.remove();
        } else {
            ADJUSTING.set(prev);
        }
    }

    static boolean adjusting() {
        return ADJUSTING.get() != null;
    }

    /** The ability whose cost is being adjusted (chooseCardsToDelve is not told). */
    static SpellAbility adjustingSa() {
        Object[] a = ADJUSTING.get();
        return a == null ? null : (SpellAbility) a[1];
    }

    /**
     * Emerge/Offering mask: is the cost being adjusted still payable after
     * sacrificing {@code c} (Emerge: generic reduced by its mana value;
     * Offering: its mana cost subtracted), without {@code c}'s own mana?
     * Other reductions of the same adjustment are not seen (loose or strict
     * only by them).
     */
    static boolean payableAfterSacrifice(Player me, SpellAbility sa, Card c, boolean emerge) {
        Object[] adj = ADJUSTING.get();
        forge.card.mana.ManaCost cost = adj == null ? null : (forge.card.mana.ManaCost) adj[0];
        if (cost == null) {
            return true;
        }
        try {
            ManaCostBeingPaid m = new ManaCostBeingPaid(cost);
            if (emerge) {
                m.decreaseGenericMana(Math.min(c.getCMC(), m.getGenericManaAmount()));
            } else {
                m.subtractManaCost(c.getManaCost());
            }
            Set<Card> ex = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            ex.add(c);
            List<Cand> all = enumerate(me, sa, ex);
            return feasible(m, model(all, null, 0), lifeShards(me, sa, m, 0, false));
        } catch (RuntimeException e) {
            count("mask: emerge check threw");
            return true;
        }
    }

    /** One Convoke/Improvise choice: tap {@code card} to pay {@code shard}. */
    private static final class Tap {
        final Card card;
        final ManaCostShard shard;
        final boolean improvise;

        Tap(Card card, ManaCostShard shard, boolean improvise) {
            this.card = card;
            this.shard = shard;
            this.improvise = improvise;
        }
    }

    private static List<Tap> tapOptions(ManaCostBeingPaid res, List<Card> cards, boolean artifacts,
            boolean creatures, Integer water) {
        List<Tap> out = new ArrayList<>();
        for (Card c : cards) {
            if (water != null) {
                // Waterbend: each artifact or creature pays {1}
                if (res.getGenericManaAmount() > 0 && (c.isArtifact() || c.isCreature())) {
                    out.add(new Tap(c, ManaCostShard.GENERIC, false));
                }
                continue;
            }
            java.util.Set<ManaCostShard> seen = new java.util.LinkedHashSet<>();
            boolean convoke = creatures && c.isCreature();
            boolean improvise = artifacts && c.isArtifact();
            if (convoke) {
                byte colours = c.getColor().getColor();
                for (ManaCostShard sh : res.getDistinctShards()) {
                    if (sh != ManaCostShard.GENERIC && res.getUnpaidShards(sh) > 0
                            && (sh.getColorMask() & colours) != 0) {
                        seen.add(sh);
                    }
                }
            }
            if ((convoke || improvise) && res.getGenericManaAmount() > 0) {
                seen.add(ManaCostShard.GENERIC);
            }
            for (ManaCostShard sh : seen) {
                out.add(new Tap(c, sh, improvise && !convoke));
            }
        }
        return out;
    }

    /** A Convoke/Improvise card as a mask source: its colours (convoke) or generic only. */
    private static PaymentCardinality.Src tapSrc(Card c, boolean artifacts, boolean creatures, Integer water) {
        PaymentCardinality.Src s = new PaymentCardinality.Src();
        if (water == null && creatures && c.isCreature()) {
            byte colours = c.getColor().getColor();
            for (int i = 0; i < 5; i++) {
                if ((colours & BYTES[i]) != 0) {
                    s.options.add(new int[] {i, 1});
                }
            }
        }
        if (s.options.isEmpty()) {
            // generic only: colour index 5 also matches a {C} shard, a looseness
            // the mask accepts (Improvise/Convoke cannot pay {C})
            s.options.add(new int[] {5, 1});
        }
        return s;
    }

    /** Mana sources that stay available once {@code tapped} are tapped for the cost. */
    private static List<PaymentCardinality.Src> manaModel(Player me, SpellAbility sa, java.util.Collection<Card> tapped) {
        Set<Card> ex = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        ex.addAll(tapped);
        return model(enumerate(me, sa, ex), null, 0);
    }

    /**
     * Whether {@code m} can be paid for {@code sa} from scratch: the payment
     * mask ({@link #feasible}) on an empty prefix, over every source the
     * payer has now (floating mana, mana abilities, life for Phyrexian
     * shards). Used by {@link PlayFilter} before anything is announced.
     */
    static boolean affordable(Player me, SpellAbility sa, ManaCostBeingPaid m) {
        return feasible(m, manaModel(me, sa, java.util.Collections.emptyList()), lifeShards(me, sa, m, 0, false));
    }

    private static String shardText(ManaCostShard sh) {
        return sh == ManaCostShard.GENERIC ? "{1}" : sh.toString();
    }

    /**
     * The legal Convoke/Improvise choices at {@code res}, stop first (index 0)
     * when legal; {@code stop[0]} says whether it is.
     */
    private static List<Tap> legalTaps(Player me, SpellAbility sa, ManaCostBeingPaid res, List<Card> pool,
            java.util.Collection<Card> chosen, boolean artifacts, boolean creatures, boolean[] stop, Integer water) {
        List<Card> left = new ArrayList<>();
        for (Card c : pool) {
            if (!chosen.contains(c)) {
                left.add(c);
            }
        }
        int life = lifeShards(me, sa, res, 0, false);
        stop[0] = feasible(res, manaModel(me, sa, chosen), life);
        List<Tap> out = new ArrayList<>();
        if (water != null && chosen.size() >= water) {
            return new ArrayList<>();
        }
        int more = water == null ? Integer.MAX_VALUE : water - chosen.size() - 1;
        for (Tap t : tapOptions(res, left, artifacts, creatures, water)) {
            ManaCostBeingPaid next = new ManaCostBeingPaid(res);
            next.decreaseShard(t.shard, 1);
            List<Card> now = new ArrayList<>(chosen);
            now.add(t.card);
            List<PaymentCardinality.Src> srcs = manaModel(me, sa, now);
            int added = 0;
            for (Card o : left) {
                if (o != t.card && added < more) {
                    srcs.add(tapSrc(o, artifacts, creatures, water));
                    added++;
                }
            }
            if (feasible(next, srcs, lifeShards(me, sa, next, 0, false))) {
                out.add(t);
            }
        }
        return out;
    }

    private static Serializer.Prompt tapPrompt(InstrumentedController ctl, SpellAbility sa, ManaCostBeingPaid res,
            java.util.Collection<Card> chosen, boolean stop) {
        Serializer.Prompt p = ctl.promptFor(KIND_CONVOKE, sa).optional(stop);
        remainingInto(p, res);
        for (Card c : chosen) {
            p.select(c);
        }
        return p;
    }

    private static void remainingInto(Serializer.Prompt p, ManaCostBeingPaid m) {
        int[] rem = new int[6];
        for (ManaCostShard sh : m.getDistinctShards()) {
            if (sh == ManaCostShard.GENERIC) {
                continue;
            }
            int n = m.getUnpaidShards(sh);
            byte mask = sh.getColorMask();
            if (mask == 0) {
                if (sh.isOfKind(ManaAtom.COLORLESS)) {
                    rem[5] += n;
                }
                continue;
            }
            for (int i = 0; i < 5; i++) {
                if ((mask & BYTES[i]) != 0) {
                    rem[i] += n;
                }
            }
        }
        p.remainingCost(costText(m)).remaining(m.getGenericManaAmount(), rem);
    }

    private static List<Serializer.Cand> tapOpts(List<Tap> taps, boolean stop, boolean water) {
        List<Serializer.Cand> out = new ArrayList<>();
        if (stop) {
            out.add(Serializer.Cand.option("choose_nothing"));
        }
        for (Tap t : taps) {
            Serializer.Cand o = Serializer.Cand.option(water ? VERB_WATERBEND
                    : t.improvise ? VERB_IMPROVISE : VERB_CONVOKE)
                    .pays(shardText(t.shard)).edge(Schema.SOURCE_IS, t.card);
            out.add(o);
        }
        return out;
    }

    static final String VERB_CONVOKE = "convoke";
    static final String VERB_IMPROVISE = "improvise";
    static final String VERB_WATERBEND = "waterbend";

    /** chooseCardsForConvokeOrImprovise for a routed payer, one card at a time. */
    static Map<Card, ManaCostShard> convoke(InstrumentedController ctl, SpellAbility sa,
            forge.card.mana.ManaCost cost, forge.game.card.CardCollectionView untapped,
            boolean artifacts, boolean creatures, Integer water) {
        Player me = ctl.getPlayer();
        ManaCostBeingPaid res = new ManaCostBeingPaid(cost);
        List<Card> pool = new ArrayList<>(untapped);
        Map<Card, ManaCostShard> chosen = new LinkedHashMap<>();
        count("decide: convoke/improvise payments");
        while (true) {
            boolean[] stop = {false};
            List<Tap> legal = legalTaps(me, sa, res, pool, chosen.keySet(), artifacts, creatures, stop, water);
            int n = legal.size() + (stop[0] ? 1 : 0);
            if (n == 0) {
                // Nothing chosen yet: unaffordable even with Convoke (the cast
                // fails at applyManaToCost and rolls back); otherwise a dead end.
                count(chosen.isEmpty() ? "decide: convoke unaffordable at first step"
                        : "decide: convoke DEAD END mid-choice");
                break;
            }
            int pick;
            if (n == 1) {
                pick = 0;
                count("decide: convoke forced");
            } else {
                pick = ctl.askChoice(tapPrompt(ctl, sa, res, chosen.keySet(), stop[0]), tapOpts(legal, stop[0], water != null));
                if (pick < 0 || pick >= n) {
                    count("decide: convoke delegated (stopping)");
                    break;
                }
                count("decide: convoke asked");
            }
            if (stop[0] && pick == 0) {
                break;
            }
            Tap t = legal.get(pick - (stop[0] ? 1 : 0));
            chosen.put(t.card, t.shard);
            res.decreaseShard(t.shard, 1);
            count(t.improvise ? "decide: step improvise" : "decide: step convoke");
        }
        return chosen;
    }

    /** Observe mode: the AI's Convoke/Improvise map, recorded as pay_convoke steps (map order). */
    static void observeConvoke(InstrumentedController ctl, SpellAbility sa, forge.card.mana.ManaCost cost,
            forge.game.card.CardCollectionView untapped, boolean artifacts, boolean creatures,
            Map<Card, ManaCostShard> ai, Snap pre, Integer water) {
        Player me = ctl.getPlayer();
        ManaCostBeingPaid res = new ManaCostBeingPaid(cost);
        List<Card> pool = new ArrayList<>(untapped);
        Map<Card, ManaCostShard> chosen = new LinkedHashMap<>();
        List<Map.Entry<Card, ManaCostShard>> steps = new ArrayList<>(ai == null ? java.util.Collections.emptySet()
                : ai.entrySet());
        count("observe: convoke/improvise payments");
        for (int k = 0; k <= steps.size(); k++) {
            boolean[] stop = {false};
            List<Tap> legal = legalTaps(me, sa, res, pool, chosen.keySet(), artifacts, creatures, stop, water);
            int n = legal.size() + (stop[0] ? 1 : 0);
            int label = -1;
            if (k == steps.size()) {
                label = stop[0] ? 0 : -1;
            } else {
                Map.Entry<Card, ManaCostShard> e = steps.get(k);
                for (int i = 0; i < legal.size(); i++) {
                    if (legal.get(i).card == e.getKey() && legal.get(i).shard == e.getValue()) {
                        label = i + (stop[0] ? 1 : 0);
                    }
                }
            }
            if (label < 0 && n > 0) {
                count("observe: convoke AI step not among legal candidates");
            }
            if (n >= 2 && canRecord(pre, me)) {
                Serializer.Built body = Serializer.buildChoiceBody(me.getGame(), me, pre.toks, pre.histSize,
                        tapPrompt(ctl, sa, res, chosen.keySet(), stop[0]), tapOpts(legal, stop[0], water != null), false);
                ctl.writeBuilt(body, label);
                count("observe: convoke recorded");
            }
            if (k < steps.size()) {
                Map.Entry<Card, ManaCostShard> e = steps.get(k);
                chosen.put(e.getKey(), e.getValue());
                res.decreaseShard(e.getValue(), 1);
            }
            if (k == steps.size() || n == 0) {
                break;
            }
        }
    }

    /** The cost Delve is asked against: the adjusting cost's coloured part plus {@code generic}. */
    private static ManaCostBeingPaid delveResidual(int generic) {
        Object[] adj = ADJUSTING.get();
        forge.card.mana.ManaCost cost = adj == null ? null : (forge.card.mana.ManaCost) adj[0];
        ManaCostBeingPaid res = new ManaCostBeingPaid(cost == null ? forge.card.mana.ManaCost.ZERO : cost);
        res.decreaseGenericMana(res.getGenericManaAmount());
        for (ManaCostShard sh : new ArrayList<>(res.getUnpaidShards())) {
            if (sh == ManaCostShard.X || sh == ManaCostShard.COLORED_X) {
                res.decreaseShard(sh, 1);
            }
        }
        res.increaseGenericMana(generic);
        return res;
    }

    private static Serializer.Prompt delvePrompt(InstrumentedController ctl, SpellAbility sa, ManaCostBeingPaid res,
            List<Card> chosen, boolean stop) {
        Serializer.Prompt p = ctl.promptFor(KIND_DELVE, sa).optional(stop);
        remainingInto(p, res);
        for (Card c : chosen) {
            p.select(c);
        }
        return p;
    }

    private static List<Serializer.Cand> delveOpts(List<Card> left, boolean stop) {
        List<Serializer.Cand> out = new ArrayList<>();
        if (stop) {
            out.add(Serializer.Cand.option("choose_nothing"));
        }
        for (Card c : left) {
            out.add(Serializer.Cand.ref(c, "choose_object"));
        }
        return out;
    }

    /**
     * Delve legality: every graveyard card pays the same {1}, so a card is
     * legal iff the cost stays payable with the rest of the graveyard delved
     * too, and stopping iff it is payable from mana sources alone.
     */
    private static boolean delveStop(Player me, SpellAbility sa, ManaCostBeingPaid res) {
        return feasible(res, manaModel(me, sa, java.util.Collections.emptyList()), lifeShards(me, sa, res, 0, false));
    }

    private static boolean delveMore(Player me, SpellAbility sa, ManaCostBeingPaid res, int left) {
        ManaCostBeingPaid next = new ManaCostBeingPaid(res);
        next.decreaseGenericMana(Math.min(left, next.getGenericManaAmount()));
        return feasible(next, manaModel(me, sa, java.util.Collections.emptyList()), lifeShards(me, sa, next, 0, false));
    }

    static forge.game.card.CardCollectionView delve(InstrumentedController ctl, SpellAbility sa, int generic,
            forge.game.card.CardCollection grave) {
        Player me = ctl.getPlayer();
        ManaCostBeingPaid res = delveResidual(generic);
        List<Card> left = new ArrayList<>(grave);
        List<Card> chosen = new ArrayList<>();
        count("decide: delve payments");
        while (res.getGenericManaAmount() > 0 && !left.isEmpty()) {
            boolean stop = delveStop(me, sa, res);
            boolean more = delveMore(me, sa, res, left.size());
            int n = (stop ? 1 : 0) + (more ? left.size() : 0);
            if (n == 0) {
                count(chosen.isEmpty() ? "decide: delve unaffordable at first step"
                        : "decide: delve DEAD END mid-choice");
                break;
            }
            int pick;
            if (n == 1) {
                pick = 0;
                count("decide: delve forced");
            } else {
                pick = ctl.askChoice(delvePrompt(ctl, sa, res, chosen, stop),
                        delveOpts(more ? left : java.util.Collections.emptyList(), stop));
                if (pick < 0 || pick >= n) {
                    count("decide: delve delegated (stopping)");
                    break;
                }
                count("decide: delve asked");
            }
            if (stop && pick == 0) {
                break;
            }
            Card c = left.remove(pick - (stop ? 1 : 0));
            chosen.add(c);
            res.decreaseGenericMana(1);
            count("decide: step delve");
        }
        return new forge.game.card.CardCollection(chosen);
    }

    /** Observe mode: the AI's delved cards, recorded as pay_delve steps (its order). */
    static void observeDelve(InstrumentedController ctl, SpellAbility sa, int generic,
            forge.game.card.CardCollection grave, forge.game.card.CardCollectionView ai, Snap pre) {
        Player me = ctl.getPlayer();
        ManaCostBeingPaid res = delveResidual(generic);
        List<Card> left = new ArrayList<>(grave);
        List<Card> chosen = new ArrayList<>();
        List<Card> steps = ai == null ? new ArrayList<>() : new ArrayList<>(ai);
        count("observe: delve payments");
        for (int k = 0; k <= steps.size(); k++) {
            if (res.getGenericManaAmount() <= 0 || left.isEmpty()) {
                break;
            }
            boolean stop = delveStop(me, sa, res);
            boolean more = delveMore(me, sa, res, left.size());
            int n = (stop ? 1 : 0) + (more ? left.size() : 0);
            int label = -1;
            if (k == steps.size()) {
                label = stop ? 0 : -1;
            } else if (more) {
                int i = left.indexOf(steps.get(k));
                label = i < 0 ? -1 : i + (stop ? 1 : 0);
            }
            if (label < 0 && n > 0) {
                count("observe: delve AI step not among legal candidates");
            }
            if (n >= 2 && canRecord(pre, me)) {
                Serializer.Built body = Serializer.buildChoiceBody(me.getGame(), me, pre.toks, pre.histSize,
                        delvePrompt(ctl, sa, res, chosen, stop),
                        delveOpts(more ? left : java.util.Collections.emptyList(), stop), false);
                ctl.writeBuilt(body, label);
                count("observe: delve recorded");
            }
            if (k == steps.size()) {
                break;
            }
            Card c = steps.get(k);
            left.remove(c);
            chosen.add(c);
            res.decreaseGenericMana(1);
        }
    }

    /**
     * Whether a record built now from {@code s} keeps the history delta order:
     * no observe-mode record is open around this payment (it would be written
     * after ours), and no record was written since {@code s} was taken.
     */
    static boolean canRecord(Snap s, Player me) {
        if (s == null) {
            return false;
        }
        if (InstrumentedController.OPEN_BODIES.get()[0] > 0) {
            count("observe: inside an open observed record (not recorded)");
            return false;
        }
        if (s.histSize < Serializer.emittedFor(me)) {
            count("observe: snapshot older than the last record (not recorded)");
            return false;
        }
        return true;
    }

    /** Observation before an observed callback, when the dump wants it. */
    static Snap preSnap(InstrumentedController ctl) {
        return Serializer.enabled() ? snap(ctl, ctl.getPlayer()) : null;
    }

    // ---- observe mode --------------------------------------------------------------

    /** Observation tokens at a point in time, for a record built later. */
    static final class Snap {
        final List<StateExtractor.Tok> toks;
        final int histSize;

        Snap(List<StateExtractor.Tok> toks, int histSize) {
            this.toks = toks;
            this.histSize = histSize;
        }
    }

    /** The observe-mode payment in progress (game thread only). */
    private static final ThreadLocal<Observed> ACTIVE = new ThreadLocal<>();

    private static final class Observed {
        final InstrumentedController ctl;
        final Player me;
        final SpellAbility sa;
        final ManaCostBeingPaid live;
        final List<Cand> all0;
        final int life0;
        final Map<Integer, Snap> atTap = new java.util.HashMap<>();
        final Set<Card> tapped = java.util.Collections.newSetFromMap(new IdentityHashMap<>());

        Observed(InstrumentedController ctl, Player me, SpellAbility sa, ManaCostBeingPaid live,
                List<Cand> all0, int life0) {
            this.ctl = ctl;
            this.me = me;
            this.sa = sa;
            this.live = live;
            this.all0 = all0;
            this.life0 = life0;
        }

        /** A source of this payment was just tapped: snapshot if the step it starts is a choice. */
        void onTap(int cardId) {
            Card src = null;
            for (Cand c : all0) {
                if (c.type == ABILITY && c.card.getId() == cardId) {
                    src = c.card;
                    break;
                }
            }
            if (src == null || atTap.containsKey(cardId)) {
                return;
            }
            List<Cand> rest = new ArrayList<>();
            for (Cand c : all0) {
                if (c.type == ABILITY && tapped.contains(c.card)) {
                    continue;
                }
                rest.add(c);
            }
            tapped.add(src);
            if (rest.isEmpty() || legal(rest, live, me, sa, life0).size() < 2) {
                return;
            }
            Snap s = snap(ctl, me);
            if (s != null) {
                untap(s, cardId);
                atTap.put(cardId, s);
            }
        }
    }

    /** Clears the {@code tapped} flag of the card the AI just tapped for this step. */
    private static void untap(Snap s, int cardId) {
        String ref = StateExtractor.cardRef(cardId);
        for (StateExtractor.Tok t : s.toks) {
            if (ref.equals(t.props.get("id"))) {
                t.props.remove("tapped");
            }
        }
    }

    private static Snap snap(InstrumentedController ctl, Player me) {
        List<StateExtractor.Tok> hist = InstrumentedController.history(me.getGame(), me);
        List<StateExtractor.Tok> toks = ctl.extractTokens(null, hist);
        return toks == null ? null : new Snap(toks, hist == null ? 0 : hist.size());
    }

    /** Forwards tap events to the observed payment, if any. One per game. */
    public static final class TapListener {
        @com.google.common.eventbus.Subscribe
        public void onTap(forge.game.event.GameEventCardTapped ev) {
            Observed o = ACTIVE.get();
            if (o != null && ev.tapped() && ev.card() != null) {
                try {
                    o.onTap(ev.card().getId());
                } catch (RuntimeException e) {
                    count("observe: tap capture threw");
                }
            }
        }
    }

    private static final Map<forge.game.Game, Boolean> SUBSCRIBED =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /**
     * {@code payManaCost} for an observed payer: exactly Forge's AI
     * ({@code PlayerControllerAi.payManaCost} is {@code ComputerUtilMana.payManaCost(
     * new Cost(cost, effect), player, sa, effect)}, whose private body is these two
     * public calls with the same flags), then the reconstructed steps are written.
     */
    static boolean payObserved(InstrumentedController ctl, forge.card.mana.ManaCost cost, SpellAbility sa,
            boolean effect) {
        Player me = ctl.getPlayer();
        forge.game.Game game = me.getGame();
        if (SUBSCRIBED.putIfAbsent(game, Boolean.TRUE) == null) {
            game.subscribeToEvents(new TapListener());
        }
        count("observe: payments");
        Object[] prev = beginAdjust(cost, sa);
        ManaCostBeingPaid m;
        try {
            m = forge.ai.ComputerUtilMana.calculateManaCost(
                    new forge.game.cost.Cost(cost, effect), sa, me, false, 0, effect);
        } finally {
            endAdjust(prev);
        }
        ManaCostBeingPaid start = new ManaCostBeingPaid(m);
        List<Mana> pool0 = new ArrayList<>();
        for (Mana x : me.getManaPool()) {
            pool0.add(x);
        }
        int life0 = lifeShards(me, sa, start, 0, effect);
        List<Cand> all0 = withLife(enumerate(me, sa, java.util.Collections.emptySet()), life0);
        Snap pre = null;
        boolean poolUsable = false;
        for (Cand c : all0) {
            poolUsable |= c.type == POOL;
        }
        if (poolUsable && legal(all0, start, me, sa, life0).size() >= 2) {
            pre = snap(ctl, me);
        }
        Observed o = new Observed(ctl, me, sa, m, all0, life0);
        Observed outer = ACTIVE.get();
        ACTIVE.set(o);
        boolean ok;
        try {
            ok = forge.ai.ComputerUtilMana.payManaCost(m, sa, me, effect);
        } finally {
            ACTIVE.set(outer);
        }
        if (!ok) {
            count("observe: AI payment failed (not recorded)");
            return false;
        }
        try {
            reconstruct(ctl, o, start, pool0, pre);
        } catch (RuntimeException e) {
            count("observe: reconstruction threw");
            log("reconstruction threw " + e);
        }
        return ok;
    }

    /** One step of the AI's payment, recovered from getPayingMana(). */
    private static final class AiStep {
        final int type;
        Card card;
        AbilityManaPart part;
        final List<Mana> paid = new ArrayList<>();

        AiStep(int type) {
            this.type = type;
        }
    }

    private static List<AiStep> aiSteps(SpellAbility sa, List<Mana> pool0, ManaCostBeingPaid start, Player me) {
        List<AiStep> out = new ArrayList<>();
        Set<Mana> fromPool = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        fromPool.addAll(pool0);
        AiStep cur = null;
        ManaCostBeingPaid sim = new ManaCostBeingPaid(start);
        for (Mana x : sa.getPayingMana()) {
            if (fromPool.contains(x)) {
                AiStep s = new AiStep(POOL);
                s.paid.add(x);
                out.add(s);
                cur = null;
            } else if (cur != null && sameCard(cur.card, x.getSourceCard()) && cur.part == x.getManaAbility()) {
                cur.paid.add(x);
            } else {
                cur = new AiStep(ABILITY);
                cur.card = x.getSourceCard();
                cur.part = x.getManaAbility();
                cur.paid.add(x);
                out.add(cur);
            }
            sim.payMana(x, me.getManaPool());
        }
        // Phyrexian shards left unpaid by mana were paid with life. Their place
        // in the AI's order is not recoverable from getPayingMana(); appended.
        for (ManaCostShard sh : sim.getDistinctShards()) {
            if (sh.isPhyrexian()) {
                for (int i = 0; i < sim.getUnpaidShards(sh); i++) {
                    out.add(new AiStep(LIFE));
                }
            }
        }
        return out;
    }

    /** Mana.getSourceCard() can be a last-known-information copy: compare by id. */
    private static boolean sameCard(Card a, Card b) {
        return a == b || (a != null && b != null && a.getId() == b.getId());
    }

    private static void reconstruct(InstrumentedController ctl, Observed o, ManaCostBeingPaid start,
            List<Mana> pool0, Snap pre) {
        Player me = o.me;
        SpellAbility sa = o.sa;
        List<AiStep> steps = aiSteps(sa, pool0, start, me);
        // Resolve each step's (possibly LKI) source card to the live candidate card.
        for (AiStep st : steps) {
            if (st.type == ABILITY) {
                for (Cand c : o.all0) {
                    if (c.type == ABILITY && sameCard(c.card, st.card)) {
                        st.card = c.card;
                        break;
                    }
                }
            }
        }
        count("observe: AI steps", steps.size());
        ManaCostBeingPaid sim = new ManaCostBeingPaid(start);
        List<Card> used = new ArrayList<>();
        List<Mana> spent = new ArrayList<>();
        List<Mana> poolLeft = new ArrayList<>(pool0);
        int lifePaid = 0;
        int lifeShards = o.life0;
        // replay: the steps the recorded labels (or forced picks) name
        List<Card> replayed = new ArrayList<>();
        List<Card> aiCards = new ArrayList<>();
        Snap last = pre;
        for (AiStep st : steps) {
            if (st.type == ABILITY && !aiCards.contains(st.card)) {
                aiCards.add(st.card);
            }
            List<Cand> all = new ArrayList<>();
            for (Cand c : o.all0) {
                if (c.type == ABILITY && !used.contains(c.card)) {
                    all.add(c);
                }
            }
            all.addAll(0, poolCands(poolLeft, sa, me.getGame()));
            if (lifeShards > 0 && sim.containsPhyrexianMana()) {
                all.add(new Cand(LIFE));
            }
            List<Cand> legal = legal(all, sim, me, sa, lifeShards);
            int label = -1;
            for (int i = 0; i < legal.size(); i++) {
                if (matches(legal.get(i), st)) {
                    label = i;
                    break;
                }
            }
            if (FAULT && legal.size() > 1 && label >= 0) {
                label = (label + 1) % legal.size();
            }
            if (label < 0) {
                count("observe: AI step not among legal candidates");
                log("unmatched AI step " + (st.type == ABILITY ? st.card + " " + st.paid
                        : st.type == POOL ? "pool " + st.paid : "life") + " paying " + sa.getHostCard()
                        + " residual " + costText(sim) + " legal " + legal.size());
            } else {
                Cand pick = legal.get(label);
                if (pick.type == ABILITY) {
                    replayed.add(pick.card);
                }
            }
            if (legal.size() >= 2) {
                Snap s = st.type == ABILITY && st.card != null ? o.atTap.get(st.card.getId()) : null;
                if (s == null) {
                    s = last != null ? last : null;
                    count(st.type == ABILITY ? "observe: no tap snapshot (earlier one used)"
                            : st.type == POOL ? "observe: pool step (pre-payment snapshot)"
                            : "observe: life step (latest snapshot)");
                }
                if (s == null) {
                    s = snap(ctl, me);
                    count("observe: post-payment snapshot used");
                }
                if (canRecord(s, me)) {
                    Serializer.Built body = Serializer.buildChoiceBody(me.getGame(), me, s.toks, s.histSize,
                            prompt(ctl, sa, sim, spent, lifePaid, used), options(legal, me.getGame()), false);
                    ctl.writeBuilt(body, label);
                    count("observe: recorded");
                }
            } else {
                count("observe: forced steps (not recorded)");
            }
            if (label >= 0 && legal.get(label).combo) {
                try {
                    observeCombo(ctl, legal.get(label), st, sim, sa, all, lifeShards, last != null ? last : pre);
                } catch (RuntimeException e) {
                    count("observe: combo reconstruction threw");
                }
            }
            if (st.type == ABILITY && st.card != null && o.atTap.get(st.card.getId()) != null) {
                last = o.atTap.get(st.card.getId());
            }
            // advance the simulation by what the AI actually did
            if (st.type == LIFE) {
                sim.payPhyrexian();
                lifePaid += 2;
                lifeShards--;
            } else {
                for (Mana x : st.paid) {
                    sim.payMana(x, me.getManaPool());
                    spent.add(x);
                    poolLeft.remove(x);
                }
                if (st.type == ABILITY && st.card != null && !used.contains(st.card)) {
                    used.add(st.card);
                }
            }
        }
        if (!sim.isPaid()) {
            count("observe: reconstruction did not reach paid");
        }
        if (replayed.equals(aiCards)) {
            count("observe: replay of labels reproduces AI sources");
        } else {
            count("observe: REPLAY MISMATCH");
            log("replay mismatch paying " + sa.getHostCard() + ": labels " + replayed + " AI " + aiCards);
        }
    }

    /** Observe mode: the colours the AI's combination ability made, as pay_mana_combo steps. */
    private static void observeCombo(InstrumentedController ctl, Cand c, AiStep st, ManaCostBeingPaid sim,
            SpellAbility sa, List<Cand> all, int lifeShards, Snap s) {
        Player me = ctl.getPlayer();
        if (st.paid.size() != c.produce.length) {
            count("observe: combo output partly floating (colours not recorded)");
            return;
        }
        Combo x = new Combo(new ManaCostBeingPaid(sim), sa, c.card, c.ma);
        x.rest = new ArrayList<>();
        for (Cand k : all) {
            if (!(k.type == ABILITY && k.card == c.card)) {
                x.rest.add(k);
            }
        }
        x.lifeShards = lifeShards;
        byte mask = 0;
        for (int ci : c.comboCols) {
            mask |= BYTES[ci];
        }
        forge.card.ColorSet options = forge.card.ColorSet.fromMask(mask);
        boolean different = c.ma.getManaPart() != null && c.ma.getManaPart().getOrigProduced().contains("Different");
        List<Byte> chosen = new ArrayList<>();
        boolean replayOk = true;
        for (Mana u : st.paid) {
            List<Byte> legal = legalComboColours(x, me, chosen, options, c.produce.length, different);
            int label = legal.indexOf(u.getColor());
            if (label < 0) {
                count("observe: AI combo colour not among candidates");
                replayOk = false;
            }
            if (FAULT && label >= 0 && legal.size() > 1) {
                label = (label + 1) % legal.size();
            }
            if (label >= 0 && legal.get(label) != u.getColor()) {
                replayOk = false;
            }
            if (legal.size() >= 2 && s != null && canRecord(s, me)) {
                ctl.writeBuilt(Serializer.buildChoiceBody(me.getGame(), me, s.toks, s.histSize,
                        comboPrompt(ctl, c.ma, x, chosen), comboOpts(legal), false), label);
                count("observe: recorded pay_mana_combo");
            }
            chosen.add(u.getColor());
        }
        count(replayOk ? "observe: replay of combo labels reproduces AI colours" : "observe: REPLAY MISMATCH (combo)");
    }

    /**
     * Pool candidates: units that can pay {@code sa}, collapsed when
     * interchangeable -- same colour, restriction and snow, and either no
     * rider or the same rider from the same source. Plain units first.
     */
    private static List<Cand> poolCands(List<Mana> units, SpellAbility sa, forge.game.Game game) {
        Map<String, Cand> plain = new LinkedHashMap<>();
        Map<String, Cand> ridden = new LinkedHashMap<>();
        for (Mana m : units) {
            try {
                if (!m.meetsManaRestrictions(sa)) {
                    continue;
                }
            } catch (RuntimeException e) {
                continue;
            }
            boolean rider = hasRider(m, sa);
            String text = rider ? riderText(m, game) : null;
            String key = m.getColor() + "|" + m.isRestricted() + "|" + m.isSnow()
                    + (rider ? "|" + (m.getSourceCard() == null ? -1 : m.getSourceCard().getId()) + "|" + text : "");
            Map<String, Cand> into = rider ? ridden : plain;
            Cand c = into.get(key);
            if (c == null) {
                c = new Cand(POOL);
                c.ci = colourIndex(m.getColor());
                c.restricted = m.isRestricted();
                c.snow = m.isSnow();
                c.rider = text;
                c.card = rider ? m.getSourceCard() : null;
                into.put(key, c);
            }
            c.units.add(m);
        }
        List<Cand> res = new ArrayList<>(plain.values());
        res.addAll(ridden.values());
        return res;
    }

    private static boolean matches(Cand c, AiStep st) {
        if (c.type != st.type) {
            return false;
        }
        if (st.type == LIFE) {
            return true;
        }
        Mana first = st.paid.get(0);
        if (st.type == POOL) {
            for (Mana u : c.units) {
                if (u == first) {
                    return true;
                }
            }
            return false;
        }
        if (c.card == st.card && c.ma != null && c.ma.getManaPart() == st.part
                && (c.combo || distinct(c.produce) > 1)) {
            return true;
        }
        return c.card == st.card && c.ma != null && c.ma.getManaPart() == st.part
                && c.ci == colourIndex(first.getColor());
    }
}
