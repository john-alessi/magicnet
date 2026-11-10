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
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetRestrictions;
import forge.game.zone.ZoneType;

/**
 * Schema 20: announcements ({@code PlayerController.announceRequirements}: the
 * value of X, and other {@code Announce$} variables) as decision kind
 * {@code announce} ({@code -Dmagicnet.choices=true}).
 *
 * <p>The engine asks while casting ({@code PlaySpellAbility.announceValuesLikeX},
 * the spell already in the Stack zone, {@code casting: true}). Options are
 * {@code number} tokens ({@code value}) for the legal values in
 * {@code [min, max]}, ascending. Masked, with read-only checks: an X that
 * makes the mana cost unpayable ({@link ManaPayment#affordable}; affordability
 * is monotone in X, so the largest affordable X is found by search) and an X
 * that leaves the spell's own targeting without enough legal candidates (X is
 * set on the ability for the check and restored). Undecidable cases (X in a
 * non-mana cost part, a coloured X, a cost that may be reduced at payment) are
 * kept. Ranges wider than {@link #SPAN} offer {@code min..min+SPAN} plus the
 * largest legal value. A single legal value is taken without asking.
 *
 * <p>Observe mode: Forge's AI never announces -- its cast path
 * ({@code ComputerUtil.handlePlayingSpellAbility}) takes X from its own
 * evaluation and fixes it while paying ({@code ComputerUtilMana}: the
 * ability's X SVar, {@code Count$xPaid}). The same record is built when the AI's
 * cast moves the spell to the stack (its state: the spell {@code casting},
 * its {@code x_paid} removed), labelled with the X the AI's payment will use;
 * after the cast the label is checked against the X actually paid.
 */
final class Announce {

    private Announce() {
    }

    static final String KIND = "announce";
    /** Values offered above the minimum before the range is capped. */
    static final int SPAN = 20;
    /** Bound on the affordable-X search when the engine gives no maximum. */
    private static final int SEARCH_CAP = 1 << 12;

    private static final Map<String, AtomicLong> COUNTS = new java.util.concurrent.ConcurrentHashMap<>();

    static void count(String what) {
        COUNTS.computeIfAbsent(what, k -> new AtomicLong()).incrementAndGet();
    }

    static void printSummary() {
        if (COUNTS.isEmpty()) {
            return;
        }
        System.out.println("\n=== announce routing (-Dmagicnet.choices) ===");
        new java.util.TreeMap<>(COUNTS).forEach((k, v) -> System.out.printf("  %-52s %d%n", k, v.get()));
    }

    // ---- legal values ----------------------------------------------------------------

    /** Values to offer, ascending; never empty. */
    static List<Integer> values(SpellAbility sa, Player me, int min, int max, String announce,
            boolean additionalMask) {
        if (max < min) {
            max = min;
        }
        boolean isX = "X".equalsIgnoreCase(announce);
        int hi = max;
        if (isX) {
            Integer afford = maxAffordableX(sa, me, min, max);
            if (afford != null) {
                hi = Math.min(hi, afford);
                if (afford < max) {
                    count("mask: X capped by affordability");
                }
            } else if (max == Integer.MAX_VALUE) {
                count("kept: affordability undecidable (unbounded, capped at min+" + SPAN + ")");
                hi = min + SPAN;
            } else {
                count("kept: affordability undecidable");
            }
        } else if (max == Integer.MAX_VALUE) {
            hi = min + SPAN;
        }
        List<Integer> vals = new ArrayList<>();
        for (int v = min; v <= hi && v <= min + SPAN; v++) {
            vals.add(v);
        }
        if (hi > min + SPAN) {
            vals.add(hi);
            count("range capped (" + SPAN + " above min, plus the largest)");
        }
        if (!isX && additionalMask) {
            List<Integer> ok = new ArrayList<>();
            for (int v : vals) {
                if (additionalPayable(sa, me, announce, v)) {
                    ok.add(v);
                }
            }
            if (ok.isEmpty()) {
                count("kept: every value masked by additional costs (all offered)");
            } else {
                if (ok.size() < vals.size()) {
                    count("mask: values removed by additional costs (" + announce + ")");
                }
                vals = ok;
            }
        }
        if (isX) {
            List<Integer> ok = new ArrayList<>();
            for (int v : vals) {
                if (targetsPossible(sa, v)) {
                    ok.add(v);
                }
            }
            if (ok.isEmpty()) {
                count("kept: every value masked by targets (all offered)");
            } else {
                if (ok.size() < vals.size()) {
                    count("mask: values removed by targets");
                }
                vals = ok;
            }
        }
        return vals;
    }

    /**
     * Largest X in [min, max] for which the mana cost stays payable, or null
     * if that cannot be decided read-only. If even {@code min} is unaffordable
     * the cast is doomed anyway; undecided, kept.
     */
    static Integer maxAffordableX(SpellAbility sa, Player me, int min, int max) {
        try {
            Cost cost = sa.getPayCosts();
            if (cost == null || !cost.hasManaCost() || cost.getCostMana() == null) {
                count("undecidable: no mana cost");
                return null;
            }
            int xs = cost.getCostMana().getAmountOfX();
            if (xs <= 0) {
                count("undecidable: X not in the mana cost");
                return null;
            }
            for (forge.game.cost.CostPart part : cost.getCostParts()) {
                if (!(part instanceof forge.game.cost.CostPartMana) && String.valueOf(part.getAmount()).contains("X")) {
                    count("undecidable: X in a non-mana cost part");
                    return null;
                }
            }
            if (sa.getXColor() != null) {
                count("undecidable: coloured X");
                return null;
            }
            String reduction = PlayFilter.reductionOf(sa, me);
            Integer cut = PlayFilter.genericReduction(sa, me);
            if (cut == null) {
                count("undecidable: cost may be reduced (" + reduction + ")");
                return null;
            }
            if (cut > 0) {
                count("bounded: generic ReduceCost reduction assumed in full");
            }
            ManaCost mana = cost.getTotalMana();
            if (mana == null || mana.getShardCount(ManaCostShard.COLORED_X) > 0) {
                count("undecidable: coloured X shard");
                return null;
            }
            if (!affordableAt(sa, me, mana, min, cut)) {
                count("kept: minimum X unaffordable");
                return null;
            }
            int top = max == Integer.MAX_VALUE ? min + SEARCH_CAP : max;
            int lo = min;
            int step = 1;
            int hi = -1;
            while (true) {
                int probe = (int) Math.min((long) lo + step, top);
                if (probe <= lo) {
                    return lo;
                }
                if (affordableAt(sa, me, mana, probe, cut)) {
                    lo = probe;
                    if (probe == top) {
                        return lo;
                    }
                    step *= 2;
                } else {
                    hi = probe;
                    break;
                }
            }
            while (hi - lo > 1) {
                int mid = lo + (hi - lo) / 2;
                if (affordableAt(sa, me, mana, mid, cut)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
        } catch (Throwable t) {
            count("kept: affordability check threw");
            return null;
        }
    }

    /** {@code cut}: generic mana reductions take off (an upper bound: keeps more). */
    private static boolean affordableAt(SpellAbility sa, Player me, ManaCost mana, int x, int cut) {
        ManaCostBeingPaid m = new ManaCostBeingPaid(mana);
        if (m.getXcounter() > 0) {
            m.setXManaCostPaid(x, "1");
        }
        int left = m.getUnpaidShards(ManaCostShard.X);
        if (left > 0) {
            m.decreaseShard(ManaCostShard.X, left);
            m.increaseGenericMana(x * left);
        }
        if (cut > 0) {
            m.decreaseGenericMana(Math.min(cut, m.getGenericManaAmount()));
        }
        return m.isPaid() || ManaPayment.affordable(me, sa, m);
    }

    /**
     * CR 601.2c with X = {@code x}: the ability's own targeting can still get
     * its minimum number of targets. True when it does not target or the check
     * fails. X is set on the ability for the check and restored.
     */
    static boolean targetsPossible(SpellAbility sa, int x) {
        TargetRestrictions tgt;
        try {
            tgt = sa.getTargetRestrictions();
            if (tgt == null || !sa.usesTargeting() || sa.getHostCard() == null) {
                return true;
            }
        } catch (Throwable t) {
            return true;
        }
        Integer prev = sa.getXManaCostPaid();
        try {
            sa.setXManaCostPaid(x);
            int need = tgt.getMinTargets(sa.getHostCard(), sa);
            return need <= 0 || tgt.getNumCandidates(sa) >= need;
        } catch (Throwable t) {
            count("kept: target check threw");
            return true;
        } finally {
            sa.setXManaCostPaid(prev);
        }
    }

    /**
     * A non-X announced value (March of Otherworldly Light's "Exiled": white
     * cards to exile) feeds the additional costs a {@code RaiseCost} static
     * adds: with it set on the ability and its card, can the adjusted cost's
     * non-mana parts be paid ({@code CostAdjustment.adjust},
     * {@code CostPayment.canPayAdditionalCosts})? Decide mode only (adjust
     * records the waterbend maximum, which payment recomputes). The value is
     * restored; true when the check fails.
     */
    static boolean additionalPayable(SpellAbility sa, Player me, String var, int v) {
        Card host = sa.getHostCard();
        if (host == null || sa.getPayCosts() == null) {
            return true;
        }
        String prevSa = sa.getSVars().get(var);
        String prevCard = host.getSVars().get(var);
        try {
            sa.setSVar(var, String.valueOf(v));
            host.setSVar(var, String.valueOf(v));
            Cost c = forge.game.cost.CostAdjustment.adjust(sa.getPayCosts(), sa, false);
            return forge.game.cost.CostPayment.canPayAdditionalCosts(c, sa, false, me);
        } catch (Throwable t) {
            count("kept: additional cost check threw");
            return true;
        } finally {
            if (prevSa == null) {
                sa.removeSVar(var);
            } else {
                sa.setSVar(var, prevSa);
            }
            if (prevCard == null) {
                host.removeSVar(var);
            } else {
                host.setSVar(var, prevCard);
            }
        }
    }

    private static List<Serializer.Cand> options(List<Integer> vals) {
        List<Serializer.Cand> opts = new ArrayList<>(vals.size());
        for (int v : vals) {
            opts.add(Serializer.Cand.option("number").value(v));
        }
        return opts;
    }

    private static Serializer.Prompt prompt(InstrumentedController ctl, SpellAbility sa, String announce,
            int min, List<Integer> vals) {
        return ctl.promptFor(KIND, sa).announce(announce).min(min).max(vals.get(vals.size() - 1));
    }

    // ---- decide mode -----------------------------------------------------------------

    /** The policy's value, or null to fall back to the existing behaviour. */
    static Integer decide(InstrumentedController ctl, SpellAbility sa, int min, int max, String announce) {
        Player me = ctl.getPlayer();
        if ("X".equalsIgnoreCase(announce)) {
            // Not announced yet: a value left from a refused earlier attempt of
            // this ability must not show on the casting spell.
            sa.setXManaCostPaid(null);
        }
        List<Integer> vals = values(sa, me, min, max, announce, true);
        if (vals.size() == 1) {
            count("decide: single legal value (not asked)");
            return vals.get(0);
        }
        int pick = ctl.askChoice(prompt(ctl, sa, announce, min, vals), options(vals));
        if (pick == ActionSource.DELEGATE) {
            return null;
        }
        count("decide: asked");
        return vals.get(Math.max(0, Math.min(pick, vals.size() - 1)));
    }

    // ---- observe mode ----------------------------------------------------------------

    /** Whether the engine would ask {@code announceRequirements} for X on this cast. */
    static boolean announcesX(SpellAbility sa) {
        try {
            if (sa == null || sa.isCopied() || sa.isWrapper() || sa.isLandAbility()) {
                return false;
            }
            String announce = sa.getParam("Announce");
            if (announce != null) {
                for (String v : announce.split(",")) {
                    if ("X".equalsIgnoreCase(v.trim())) {
                        return true;
                    }
                }
            }
            Cost cost = sa.getPayCosts();
            if (cost == null || !cost.hasXInAnyCostPart()) {
                return false;
            }
            String sVar = sa.getParamOrDefault("XAlternative", sa.getSVar("X"));
            return "Count$xPaid".equals(sVar) || sVar.isEmpty();
        } catch (Throwable t) {
            return false;
        }
    }

    /** The X Forge's AI will pay (ComputerUtilMana: the X SVar, at least 1 under XMin). */
    private static int aiX(SpellAbility sa) {
        Integer x = sa.getXManaCostPaid();
        if (x != null) {
            return x;
        }
        int v = forge.game.ability.AbilityUtils.calculateAmount(sa.getHostCard(),
                sa.getParamOrDefault("XAlternative", "X"), sa);
        Cost cost = sa.getPayCosts();
        if (v < 1 && cost != null && cost.getCostMana() != null && cost.getCostMana().getXMin() > 0) {
            v = 1;
        }
        return v;
    }

    private static final class Pending {
        final InstrumentedController ctl;
        final SpellAbility sa;
        final int hostId;
        Integer label;
        boolean done;

        Pending(InstrumentedController ctl, SpellAbility sa, int hostId) {
            this.ctl = ctl;
            this.sa = sa;
            this.hostId = hostId;
        }
    }

    private static final ThreadLocal<Pending> PENDING = new ThreadLocal<>();

    private static final Map<Game, Boolean> SUBSCRIBED =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    /**
     * Forwards the move of the observed spell to the stack. One per game.
     * Subscribed to {@code GameEvent} like {@link HistoryHub} and registered
     * after it, so the hub has tracked the move before the record is built
     * (Guava dispatches subscribers of the concrete event class first).
     */
    public static final class StackListener {
        @com.google.common.eventbus.Subscribe
        public void onEvent(forge.game.event.GameEvent event) {
            if (!(event instanceof forge.game.event.GameEventCardChangeZone ev)) {
                return;
            }
            Pending p = PENDING.get();
            if (p == null || p.done || ev.card() == null || ev.to() == null
                    || ev.to().zoneType() != ZoneType.Stack || ev.card().getId() != p.hostId) {
                return;
            }
            p.done = true;
            try {
                record(p);
            } catch (RuntimeException e) {
                count("observe: record threw");
            }
        }
    }

    /**
     * Before Forge's AI casts {@code sa} (observe mode). A spell is recorded when
     * it reaches the stack; an activated ability (no move) right away.
     */
    static void observeBegin(InstrumentedController ctl, SpellAbility sa) {
        if (!announcesX(sa)) {
            return;
        }
        Card host = sa.getHostCard();
        if (host == null) {
            return;
        }
        Game game = host.getGame();
        Pending p = new Pending(ctl, sa, host.getId());
        if (sa.isSpell() && !host.isCopiedSpell() && !host.isInZone(ZoneType.Stack)) {
            if (SUBSCRIBED.putIfAbsent(game, Boolean.TRUE) == null) {
                game.subscribeToEvents(new StackListener());
            }
            PENDING.set(p);
            return;
        }
        p.done = true;
        PENDING.set(p);
        record(p);
    }

    /** After the AI's cast: checks the label against the X it paid. */
    static void observeEnd(SpellAbility sa) {
        Pending p = PENDING.get();
        if (p == null || p.sa != sa) {
            return;
        }
        PENDING.remove();
        if (!p.done) {
            count("observe: spell never reached the stack (not recorded)");
            return;
        }
        if (p.label == null) {
            return;
        }
        Integer paid = sa.getXManaCostPaid();
        if (sa.isSkip()) {
            count("observe: AI cast failed after the record");
        } else if (paid != null && paid.intValue() == p.label) {
            count("observe: label equals the X the AI paid");
        } else {
            count("observe: LABEL MISMATCH (AI paid X=" + paid + ")");
        }
    }

    private static void record(Pending p) {
        InstrumentedController ctl = p.ctl;
        SpellAbility sa = p.sa;
        Player me = ctl.getPlayer();
        Game game = me.getGame();
        int aiX = aiX(sa);
        int min;
        int max;
        try {
            org.apache.commons.lang3.Range<Integer> r =
                    forge.game.ability.AbilityUtils.getAnnouncementBounds(sa, "X");
            min = r.getMinimum();
            max = r.getMaximum();
        } catch (Throwable t) {
            count("observe: bounds threw (not recorded)");
            return;
        }
        List<Integer> vals = values(sa, me, min, max, "X", false);
        p.label = aiX;
        if (vals.size() == 1) {
            count("observe: single legal value (not recorded)");
            return;
        }
        if (!Serializer.enabled() || InstrumentedController.OPEN_BODIES.get()[0] > 0) {
            count("observe: inside an open observed record (not recorded)");
            return;
        }
        List<StateExtractor.Tok> hist = InstrumentedController.history(game, me);
        List<StateExtractor.Tok> toks = ctl.extractTokens(null, hist);
        if (toks == null) {
            return;
        }
        // The AI fixed X before casting; the policy has not announced it yet.
        String ref = StateExtractor.cardRef(p.hostId);
        for (StateExtractor.Tok t : toks) {
            if (ref.equals(t.props.get("id"))) {
                t.props.remove("x_paid");
            }
        }
        Serializer.Built body = Serializer.buildChoiceBody(game, me, toks, hist == null ? 0 : hist.size(),
                prompt(ctl, sa, "X", min, vals), options(vals), false);
        int label = vals.indexOf(aiX);
        if (label < 0) {
            count("observe: AI's X not among the offered values (unlabelled)");
        }
        ctl.writeBuilt(body, label);
        count("observe: recorded");
    }
}
