package magicnet.bridge;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityMode;
import forge.game.zone.ZoneType;

/**
 * Schema 19: removes from the priority candidate list what cannot be played
 * right now ({@code -Dmagicnet.filter}; on by default when an external action
 * source decides, see {@link InstrumentedController#FILTER}).
 *
 * <p>{@code ComputerUtilAbility.getSpellAbilities} keeps every ability that is
 * merely <em>possible</em> (zone and activator restrictions): an activated
 * ability of a tapped permanent, a sorcery-speed ability at instant speed, a
 * spell the player cannot pay for. A policy picks them, the engine refuses
 * them, and the policy is asked again. Three checks, all read-only:
 *
 * <ol>
 *   <li><b>Activation</b> (abilities and land plays, not spells):
 *       {@code SpellAbility.canPlay()}, the exact check the engine path refuses
 *       on ({@link InstrumentedController#playChosenSpellAbility}) and
 *       {@code PlaySpellAbility} applies to lands. It needs the activating
 *       player, which {@code Card.getAllPossibleAbilities} already set while
 *       building the pool -- so nothing is set here; an ability whose activator
 *       is not this player is kept. Spells need no such check: the pool only
 *       holds spells for which Forge's {@code canPlay(true)} held (their
 *       {@code isPossible} is {@code canPlay}).</li>
 *   <li><b>Targets</b> (CR 601.2c): a spell or ability whose own targeting
 *       needs a target and has no legal candidate ({@link #noLegalTargets}).</li>
 *   <li><b>Mana</b>: the printed mana cost (X = 0, optional costs unpaid, so
 *       the cheapest form) through the payment mask on an empty prefix
 *       ({@link ManaPayment#affordable}, built on {@link MaskCost#extendable},
 *       which agrees with Forge's {@code canPayManaCost} on every measured
 *       empty prefix). Skipped -- the candidate is kept -- whenever the cost
 *       could be reduced at payment time, because modelling the reduction would
 *       mean running {@code CostAdjustment.adjust}, which asks the controller
 *       (Delve, Convoke, which static to apply): any applicable
 *       {@code ReduceCost}/{@code SetCost} static, Delve, Convoke, Improvise,
 *       Assist, Emerge, Offering, Waterbend, power-up, pip reductions.
 *       Cost increases are ignored, which can only keep more.</li>
 * </ol>
 *
 * Anything that throws is kept: the filter may leave an unplayable candidate
 * in, never take a playable one out.
 */
final class PlayFilter {

    private PlayFilter() {
    }

    private static final Map<String, AtomicLong> COUNTS = new java.util.concurrent.ConcurrentHashMap<>();

    private static void count(String what) {
        COUNTS.computeIfAbsent(what, k -> new AtomicLong()).incrementAndGet();
    }

    static void printSummary() {
        if (COUNTS.isEmpty()) {
            return;
        }
        System.out.println("\n=== priority candidate filter (-Dmagicnet.filter) ===");
        new java.util.TreeMap<>(COUNTS).forEach((k, v) -> System.out.printf("  %-44s %d%n", k, v.get()));
    }

    /** The playable part of {@code pool}, in pool order. */
    static List<SpellAbility> filter(List<SpellAbility> pool, Player me) {
        List<SpellAbility> out = new ArrayList<>(pool.size());
        for (SpellAbility sa : pool) {
            String why = unplayable(sa, me);
            count("offered");
            if (why == null) {
                out.add(sa);
            } else {
                count("removed: " + why);
            }
        }
        return out;
    }

    /** Observe mode: the AI played something the filter had removed. */
    static void recordHidden(SpellAbility sa, Player me) {
        count("HIDDEN: AI played a removed candidate");
        String why = unplayable(sa, me);
        System.out.println("[filter] AI played a filtered-out candidate (" + why + "): "
                + describe(sa));
    }

    static void recordKept() {
        count("AI choice kept in the filtered pool");
    }

    private static String describe(SpellAbility sa) {
        try {
            return sa.getHostCard() + " :: " + sa;
        } catch (RuntimeException e) {
            return "?";
        }
    }

    /** Why {@code sa} cannot be played now, or null if it may be (or that is unknown). */
    static String unplayable(SpellAbility sa, Player me) {
        if (sa == null || me == null) {
            return null;
        }
        try {
            if (sa.getActivatingPlayer() != me) {
                count("kept: activator not set to the player");
                return null;
            }
            if (!sa.isSpell() && !sa.canPlay()) {
                return sa.isLandAbility() ? "land play not allowed" : "ability cannot be activated";
            }
            if (sa.isLandAbility()) {
                return null;
            }
            if (noLegalTargets(sa)) {
                return "no legal targets";
            }
            if (noTargetsAtAnyX(sa)) {
                return "no legal targets at any X";
            }
            Cost cost = sa.getPayCosts();
            if (cost == null || !cost.hasManaCost()) {
                return null;
            }
            ManaCost mana = cost.getTotalMana();
            if (mana == null || mana.isNoCost() || mana.isZero()) {
                return null;
            }
            ManaCostBeingPaid m = new ManaCostBeingPaid(mana);
            int x = m.getUnpaidShards(ManaCostShard.X);
            if (x > 0) {
                m.decreaseShard(ManaCostShard.X, x);
            }
            int cx = m.getUnpaidShards(ManaCostShard.COLORED_X);
            if (cx > 0) {
                m.decreaseShard(ManaCostShard.COLORED_X, cx);
            }
            String reduction = reduction(sa, me);
            if (reduction != null) {
                // Schema 20: a reduction that only takes off generic mana
                // (ReduceCost statics without a colour) cannot pay the
                // coloured part, so that part alone must be payable.
                if (genericReduction(sa, me) == null) {
                    count("kept: cost may be reduced (" + reduction + ")");
                    return null;
                }
                m.decreaseGenericMana(m.getGenericManaAmount());
                if (m.isPaid() || ManaPayment.affordable(me, sa, m)) {
                    count("kept: cost may be reduced (" + reduction + ")");
                    return null;
                }
                return "coloured mana cannot be paid (generic reduction only)";
            }
            if (m.isPaid()) {
                return null;
            }
            return ManaPayment.affordable(me, sa, m) ? null : "mana cost cannot be paid";
        } catch (Throwable t) {
            count("kept: check threw");
            return null;
        }
    }

    /**
     * CR 601.2c: a spell or ability whose own (top-level) targeting needs at
     * least one target and has no legal candidate now cannot be cast -- the
     * engine refuses it at {@code setupTargets}. Candidates are counted as
     * the option tokens' {@code target_candidates} are
     * ({@code TargetRestrictions.getNumCandidates}: stack objects plus
     * players and cards). Not decided -- kept -- when the restriction or the
     * cost mentions X (validity is evaluated with X unannounced, i.e. 0) or the
     * minimum depends on a choice not made yet; sub-abilities are not checked
     * (their targeting can be conditional or modal).
     */
    private static boolean noLegalTargets(SpellAbility sa) {
        forge.game.spellability.TargetRestrictions tgt = sa.getTargetRestrictions();
        if (tgt == null || !sa.usesTargeting()) {
            return false;
        }
        Card host = sa.getHostCard();
        if (host == null) {
            return false;
        }
        String valid = sa.getParam("ValidTgts");
        if (valid != null && valid.contains("X")) {
            count("kept: targets depend on X");
            return false;
        }
        try {
            if (sa.getPayCosts() != null && sa.getPayCosts().getCostMana() != null
                    && sa.getPayCosts().getCostMana().getAmountOfX() > 0) {
                count("kept: targets depend on X");
                return false;
            }
        } catch (RuntimeException e) {
            return false;
        }
        if (tgt.getMinTargets(host, sa) <= 0) {
            return false;
        }
        return tgt.getNumCandidates(sa) == 0;
    }

    /**
     * Schema 20: targeting that depends on X ({@code ValidTgts} or the cost
     * mentions X) with no legal target for any X in 0..{@link Announce#SPAN}
     * nor a very large X (X set on the ability for the check and restored).
     */
    private static boolean noTargetsAtAnyX(SpellAbility sa) {
        forge.game.spellability.TargetRestrictions tgt = sa.getTargetRestrictions();
        if (tgt == null || !sa.usesTargeting() || sa.getHostCard() == null || !Announce.announcesX(sa)) {
            return false;
        }
        if (Announce.targetsPossible(sa, 99)) {
            return false;
        }
        for (int x = 0; x <= Announce.SPAN; x++) {
            if (Announce.targetsPossible(sa, x)) {
                return false;
            }
        }
        return true;
    }

    private static java.lang.reflect.Method checkRequirement;
    private static boolean checkRequirementLooked;

    /** {@link #reduction}, for the announce mask. */
    static String reductionOf(SpellAbility sa, Player me) {
        return reduction(sa, me);
    }

    /**
     * Upper bound on the generic mana the applicable {@code ReduceCost}
     * statics take off {@code sa} (their {@code Amount}s, summed, read-only),
     * or null when the reduction is of another kind or not just generic.
     */
    static Integer genericReduction(SpellAbility sa, Player me) {
        String why = reduction(sa, me);
        if (why == null) {
            return 0;
        }
        if (!"ReduceCost static".equals(why)) {
            return null;
        }
        Game game = me.getGame();
        Card host = sa.getHostCard();
        List<Card> cards = new ArrayList<>();
        for (ZoneType z : new ZoneType[] {ZoneType.Battlefield, ZoneType.Stack, ZoneType.Command}) {
            for (Card c : game.getCardsIn(z)) {
                cards.add(c);
            }
        }
        if (host != null && !cards.contains(host)) {
            cards.add(host);
        }
        int total = 0;
        for (Card c : cards) {
            for (StaticAbility st : c.getStaticAbilities()) {
                if (!st.checkMode(StaticAbilityMode.ReduceCost) || !applies(sa, st)) {
                    continue;
                }
                if (st.hasParam("Color") || st.hasParam("ForEachShard") || st.hasParam("Cost")) {
                    return null;
                }
                int amount = forge.game.ability.AbilityUtils.calculateAmount(st.getHostCard(),
                        st.getParamOrDefault("Amount", "1"), st);
                total += Math.max(0, amount);
            }
        }
        return total;
    }

    /** A reason the cost of {@code sa} may be reduced when it is paid, or null if it cannot be. */
    private static String reduction(SpellAbility sa, Player me) {
        Card host = sa.getHostCard();
        if (sa.hasParam("ReduceCost") || sa.hasParam("TapCreaturesForMana")) {
            return "ability parameter";
        }
        if (sa.isPowerUp() || sa.isOffering() || sa.isEmerge()) {
            return "power-up/offering/emerge";
        }
        Integer water = sa.getMaxWaterbend();
        if (water != null && water > 0) {
            return "waterbend";
        }
        if (sa.isSpell()) {
            List<String> pips = sa.getPipsToReduce();
            if (pips != null && !pips.isEmpty()) {
                return "pips";
            }
            if (host != null && (host.hasKeyword(Keyword.DELVE) || host.hasKeyword(Keyword.CONVOKE)
                    || host.hasKeyword(Keyword.IMPROVISE) || host.hasKeyword(Keyword.ASSIST))) {
                return "delve/convoke/improvise/assist";
            }
        }
        Game game = me.getGame();
        List<Card> cards = new ArrayList<>();
        for (ZoneType z : new ZoneType[] {ZoneType.Battlefield, ZoneType.Stack, ZoneType.Command}) {
            for (Card c : game.getCardsIn(z)) {
                cards.add(c);
            }
        }
        if (host != null && !cards.contains(host)) {
            cards.add(host);
        }
        for (Card c : cards) {
            for (StaticAbility st : c.getStaticAbilities()) {
                if (st.checkMode(StaticAbilityMode.SetCost)) {
                    return "SetCost static";
                }
                if (st.checkMode(StaticAbilityMode.ReduceCost) && applies(sa, st)) {
                    return "ReduceCost static";
                }
            }
        }
        return null;
    }

    /**
     * Forge's own applicability test for a cost-reduction static
     * ({@code CostAdjustment.checkRequirement}, private: conditions and
     * valid-card/spell/activator parameters, all reads). True when it cannot
     * be called.
     */
    private static boolean applies(SpellAbility sa, StaticAbility st) {
        synchronized (PlayFilter.class) {
            if (!checkRequirementLooked) {
                checkRequirementLooked = true;
                try {
                    checkRequirement = forge.game.cost.CostAdjustment.class.getDeclaredMethod(
                            "checkRequirement", SpellAbility.class, StaticAbility.class);
                    checkRequirement.setAccessible(true);
                } catch (Throwable t) {
                    checkRequirement = null;
                }
            }
        }
        if (checkRequirement == null) {
            return true;
        }
        try {
            return Boolean.TRUE.equals(checkRequirement.invoke(null, sa, st));
        } catch (Throwable t) {
            return true;
        }
    }
}
