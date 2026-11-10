package magicnet.bridge;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.card.CardPredicates;
import forge.game.cost.Cost;
import forge.game.cost.CostDiscard;
import forge.game.cost.CostExile;
import forge.game.cost.CostPart;
import forge.game.cost.CostReturn;
import forge.game.cost.CostReveal;
import forge.game.cost.CostSacrifice;
import forge.game.cost.CostTapType;
import forge.game.cost.CostUntapType;
import forge.game.cost.PaymentDecision;
import forge.game.player.Player;
import forge.game.spellability.OptionalCost;
import forge.game.spellability.OptionalCostValue;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

/**
 * Schema 18: the non-mana cost choices {@code PlayerControllerHuman} asks and
 * Forge's AI answered silently (behind {@code -Dmagicnet.payment=true}).
 *
 * <ul>
 * <li>Card-choosing cost parts ({@code pay_sacrifice}, {@code pay_discard},
 * {@code pay_exile}, {@code pay_tap}, {@code pay_untap}, {@code pay_return},
 * {@code pay_reveal}). The engine pays a cost through
 * {@code CostPayment.payCost(controller.getCostDecisionMaker(...))}: the human
 * gets a {@code HumanCostDecision} (one {@code InputSelectCardsFromList} per
 * part), the AI an {@code AiCostDecision}. Deciding, the controller hands out
 * {@link Routed}, an {@code AiCostDecision} whose visits for those parts ask
 * one card at a time from the list {@code HumanCostDecision} builds; branches
 * it does not mirror stay with the AI (counted). Forge's AI path builds its
 * {@code AiCostDecision} itself, so observing, {@link #observeCast} replays
 * {@code PlayerControllerAi.playChosenSpellAbility} with an observing
 * {@link Routed} that takes the AI's decision and records it.</li>
 * <li>Optional costs ({@code choose_optional_cost}: kicker, buyback, entwine,
 * ...; and keyword costs: casualty, conspire, offspring, multikicker,
 * replicate, squad), asked by {@code PlaySpellAbility.chooseOptionalAdditionalCosts}
 * / {@code GameActionUtil.addExtraKeywordCost}.</li>
 * <li>Alternative additional costs ({@code choose_additional_cost}, Lightning
 * Axe's "discard a card or pay {5}"), via {@code getAbilityToPlay}, where
 * Forge's AI takes the first.</li>
 * <li>Emerge / Offering sacrifice ({@code pay_sacrifice} with cost "Emerge" /
 * "Offering"), via {@code CostAdjustment}'s {@code choosePermanentsToSacrifice}.</li>
 * </ul>
 */
final class CostChoices {

    private CostChoices() {
    }

    static final String KIND_SACRIFICE = "pay_sacrifice";
    static final String KIND_DISCARD = "pay_discard";
    static final String KIND_EXILE = "pay_exile";
    static final String KIND_TAP = "pay_tap";
    static final String KIND_UNTAP = "pay_untap";
    static final String KIND_RETURN = "pay_return";
    static final String KIND_REVEAL = "pay_reveal";
    static final String KIND_OPTIONAL = "choose_optional_cost";
    static final String KIND_ALTERNATIVE = "choose_additional_cost";
    static final String VERB_OPTIONAL = "optional_cost";

    /** Test hook: corrupt observe-mode labels, to show the replay check can fail. */
    private static final boolean FAULT = Boolean.getBoolean("magicnet.payment.fault");

    static boolean decides() {
        return ManaPayment.decides();
    }

    static boolean observes() {
        return ManaPayment.observes();
    }

    static void count(String what) {
        ManaPayment.count("cost: " + what);
    }

    /** Depth of engine casts started by the controller (PlaySpellAbility.playSpellAbility). */
    static final ThreadLocal<int[]> ENGINE_CAST = ThreadLocal.withInitial(() -> new int[1]);

    static boolean inEngineCast() {
        return ENGINE_CAST.get()[0] > 0;
    }

    // ---- card-choosing cost parts ------------------------------------------------

    /** "Choose {@code count} of {@code list}" (or, crew, total power {@code power}). */
    static final class Pick {
        final String kind;
        final String text;
        final List<Card> list;
        final int count;
        final int power;
        /** Stop (choose_nothing) offered before the first card (Emerge/Offering: never). */
        final boolean optional;

        Pick(String kind, String text, List<Card> list, int count, int power, boolean optional) {
            this.kind = kind;
            this.text = text;
            this.list = list;
            this.count = count;
            this.power = power;
            this.optional = optional;
        }

        boolean done(List<Card> chosen) {
            if (power > 0) {
                int p = 0;
                for (Card c : chosen) {
                    p += Math.max(0, c.getNetPower());
                }
                return p >= power;
            }
            return chosen.size() >= count;
        }

        int remainingPower(List<Card> chosen) {
            int p = power;
            for (Card c : chosen) {
                p -= Math.max(0, c.getNetPower());
            }
            return p;
        }
    }

    private static String partText(CostPart part, forge.game.Game game) {
        String s;
        try {
            s = part.toString();
        } catch (RuntimeException e) {
            s = null;
        }
        s = s == null ? null : EventTokenizer.normalize(s, game);
        return s == null || s.isBlank() ? null : s.trim();
    }

    private static int amount(CostPart part, SpellAbility sa) {
        try {
            return part.getAbilityAmount(sa);
        } catch (RuntimeException e) {
            return -1;
        }
    }

    /** HumanCostDecision.visit(CostSacrifice), main branch. Null: not mirrored. */
    static Pick sacrifice(CostSacrifice cost, Player p, SpellAbility sa, boolean effect) {
        String type = cost.getType();
        if (cost.payCostFromSource() || "OriginalHost".equals(type) || "All".equals(cost.getAmount())
                || type.contains("+WithDifferentNames")) {
            return null;
        }
        CardCollection list = CardLists.filter(p.getCardsIn(ZoneType.Battlefield),
                CardPredicates.canBeSacrificedBy(sa, effect));
        list = CardLists.getValidCards(list, type.split(";"), p, sa.getHostCard(), sa);
        int c = amount(cost, sa);
        return c <= 0 || list.size() < c ? null
                : new Pick(KIND_SACRIFICE, partText(cost, p.getGame()), list, c, 0, false);
    }

    /** HumanCostDecision.visit(CostDiscard), the typed branch. */
    static Pick discard(CostDiscard cost, Player p, SpellAbility sa) {
        String type = cost.getType();
        if (cost.payCostFromSource() || type.equals("Hand") || type.equals("LastDrawn")
                || type.equals("Random") || type.contains("+With")) {
            return null;
        }
        CardCollection hand = CardLists.getValidCards(p.getCardsIn(ZoneType.Hand), type.split(";"), p,
                sa.getHostCard(), sa);
        int c = amount(cost, sa);
        return c <= 0 || hand.size() < c ? null
                : new Pick(KIND_DISCARD, partText(cost, p.getGame()), hand, c, 0, false);
    }

    /** HumanCostDecision.visit(CostExile), the plain "exile N" branches (not library, not same-owner). */
    static Pick exile(CostExile cost, Player p, SpellAbility sa, boolean effect) {
        String type = cost.getType();
        if (cost.payCostFromSource() || type.equals("OriginalHost") || type.equals("All")
                || type.contains("FromTopGrave") || type.contains("+with") || cost.from.isEmpty()
                || cost.from.contains(ZoneType.Library) || cost.zoneRestriction == 0) {
            return null;
        }
        CardCollection list = cost.zoneRestriction != 1 ? new CardCollection(p.getGame().getCardsIn(cost.from))
                : new CardCollection(p.getCardsIn(cost.from));
        list = CardLists.getValidCards(list, type.split(";"), p, sa.getHostCard(), sa);
        list = CardLists.filter(list, CardPredicates.canExiledBy(sa, effect));
        int c = amount(cost, sa);
        return c <= 0 || list.size() < c ? null
                : new Pick(KIND_EXILE, partText(cost, p.getGame()), list, c, 0, false);
    }

    /** HumanCostDecision.visit(CostTapType): N creatures, or crew's total power. */
    static Pick tap(CostTapType cost, Player p, SpellAbility sa) {
        String type = cost.getType();
        if (type.equals("OriginalHost") || type.contains("sharesCreatureTypeWith")) {
            return null;
        }
        int power = 0;
        if (type.contains("+withTotalPowerGE")) {
            String totalP = type.split("withTotalPowerGE")[1];
            type = type.replace("+withTotalPowerGE" + totalP, "");
            try {
                power = Integer.parseInt(totalP.trim());
            } catch (NumberFormatException e) {
                return null;
            }
        } else if ("Any".equals(cost.getAmount())) {
            return null;
        }
        CardCollection list = CardLists.getValidCards(p.getCardsIn(ZoneType.Battlefield), type.split(";"), p,
                sa.getHostCard(), sa);
        list = CardLists.filter(list, sa.isCrew() ? CardPredicates.CAN_CREW : CardPredicates.CAN_TAP);
        String text = partText(cost, p.getGame());
        if (power > 0) {
            int total = 0;
            for (Card c : list) {
                total += Math.max(0, c.getNetPower());
            }
            return total < power ? null : new Pick(KIND_TAP, text, list, 0, power, false);
        }
        int c = amount(cost, sa);
        return c <= 0 || list.size() < c ? null : new Pick(KIND_TAP, text, list, c, 0, false);
    }

    /** HumanCostDecision.visit(CostUntapType). */
    static Pick untap(CostUntapType cost, Player p, SpellAbility sa) {
        CardCollection list = CardLists.getValidCards(p.getGame().getCardsIn(ZoneType.Battlefield),
                cost.getType().split(";"), p, sa.getHostCard(), sa);
        list = CardLists.filter(list, x -> x.canUntap(null, false)
                && (x.getCounters(forge.game.card.CounterEnumType.STUN) == 0
                        || x.canRemoveCounters(forge.game.card.CounterEnumType.STUN)));
        int c = amount(cost, sa);
        return c <= 0 || list.size() < c ? null
                : new Pick(KIND_UNTAP, partText(cost, p.getGame()), list, c, 0, false);
    }

    /** HumanCostDecision.visit(CostReturn), not from source. */
    static Pick returning(CostReturn cost, Player p, SpellAbility sa) {
        if (cost.payCostFromSource() || sa.getActivatingPlayer() == null) {
            return null;
        }
        CardCollection list = CardLists.getValidCards(sa.getActivatingPlayer().getCardsIn(ZoneType.Battlefield),
                cost.getType().split(";"), p, sa.getHostCard(), sa);
        int c = amount(cost, sa);
        return c <= 0 || list.size() < c ? null
                : new Pick(KIND_RETURN, partText(cost, p.getGame()), list, c, 0, false);
    }

    /** HumanCostDecision.visit(CostReveal), the typed branch. */
    static Pick reveal(CostReveal cost, Player p, SpellAbility sa) {
        String type = cost.getType();
        if (cost.payCostFromSource() || type.equals("Hand") || type.equals("SameColor")) {
            return null;
        }
        CardCollection list = CardLists.getValidCards(p.getCardsIn(cost.getRevealFrom()), type.split(";"), p,
                sa.getHostCard(), sa);
        int c = amount(cost, sa);
        return c <= 0 || list.size() < c ? null
                : new Pick(KIND_REVEAL, partText(cost, p.getGame()), list, c, 0, false);
    }

    private static Serializer.Prompt prompt(InstrumentedController ctl, SpellAbility sa, Pick p, List<Card> chosen,
            boolean stop) {
        Serializer.Prompt q = ctl.promptFor(p.kind, sa).cost(p.text).optional(stop);
        if (p.power > 0) {
            q.remainingPower(p.remainingPower(chosen));
        } else {
            q.min(p.count - chosen.size()).max(p.count - chosen.size());
        }
        for (Card c : chosen) {
            q.select(c);
        }
        return q;
    }

    private static List<Serializer.Cand> opts(List<Card> left, boolean stop) {
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
     * Whether the next pick is a real choice: more cards than still needed
     * (a count part with exactly as many cards as it needs takes them all), or
     * a stop on offer.
     */
    private static boolean choice(Pick p, List<Card> left, List<Card> chosen, boolean stop) {
        if (stop) {
            return !left.isEmpty();
        }
        if (p.power > 0) {
            return left.size() >= 2;
        }
        return left.size() > p.count - chosen.size();
    }

    /** Asks the picks of {@code p} one card at a time; null if the source delegated. */
    static List<Card> ask(InstrumentedController ctl, SpellAbility sa, Pick p) {
        List<Card> left = new ArrayList<>(p.list);
        if (sa != null && sa.isManaAbility() && p.power == 0) {
            // paying a mana ability's own cost inside a routed mana step: keep
            // the cards whose loss as a mana source leaves that payment payable
            List<Card> keep = new ArrayList<>();
            for (Card c : left) {
                if (ManaPayment.stepKeepsPayable(ctl.getPlayer(), c)) {
                    keep.add(c);
                }
            }
            if (keep.size() >= p.count) {
                if (keep.size() < left.size()) {
                    count("mask: mana-ability cost cards removed");
                }
                left = keep;
            } else {
                count("mask: mana-ability cost cards all kept (too few pass)");
                ManaPayment.debugStep(ctl.getPlayer(), left);
            }
        }
        List<Card> chosen = new ArrayList<>();
        count("decide: " + p.kind);
        boolean first = true;
        while (!p.done(chosen) && !left.isEmpty()) {
            boolean stop = first && p.optional;
            int pick;
            if (!choice(p, left, chosen, stop)) {
                pick = 0;
                count("decide: forced pick");
            } else {
                pick = ctl.askChoice(prompt(ctl, sa, p, chosen, stop), opts(left, stop));
                if (pick < 0 || pick >= left.size() + (stop ? 1 : 0)) {
                    count("decide: delegated");
                    return null;
                }
                count("decide: asked " + p.kind);
            }
            first = false;
            if (stop) {
                if (pick == 0) {
                    break;
                }
                pick--;
            }
            chosen.add(left.remove(pick));
        }
        return chosen;
    }

    /** Observe mode: the AI's cards for {@code p}, recorded as the same sequence of picks. */
    static void observe(InstrumentedController ctl, SpellAbility sa, Pick p, List<Card> ai, ManaPayment.Snap pre) {
        Player me = ctl.getPlayer();
        List<Card> left = new ArrayList<>(p.list);
        List<Card> chosen = new ArrayList<>();
        List<Card> replay = new ArrayList<>();
        List<Card> steps = ai == null ? new ArrayList<>() : new ArrayList<>(ai);
        count("observe: " + p.kind);
        boolean first = true;
        for (int k = 0; k <= steps.size(); k++) {
            if (p.done(chosen) || left.isEmpty()) {
                if (k < steps.size()) {
                    count("observe: AI chose beyond the requirement");
                }
                break;
            }
            boolean stop = first && p.optional;
            int label;
            if (k == steps.size()) {
                label = stop ? 0 : -1;
            } else {
                int i = indexOf(left, steps.get(k));
                label = i < 0 ? -1 : i + (stop ? 1 : 0);
            }
            if (label < 0) {
                count("observe: AI card not among candidates");
            }
            int n = left.size() + (stop ? 1 : 0);
            if (FAULT && label >= 0 && n > 1) {
                label = (label + 1) % n;
            }
            if (choice(p, left, chosen, stop) && ManaPayment.canRecord(pre, me)) {
                Serializer.Built body = Serializer.buildChoiceBody(me.getGame(), me, pre.toks, pre.histSize,
                        prompt(ctl, sa, p, chosen, stop), opts(left, stop), false);
                ctl.writeBuilt(body, label);
                count("observe: recorded " + p.kind);
            }
            if (label >= 0 && !(stop && label == 0)) {
                replay.add(left.get(label - (stop ? 1 : 0)));
            }
            first = false;
            if (k == steps.size()) {
                break;
            }
            Card c = steps.get(k);
            int i = indexOf(left, c);
            if (i >= 0) {
                chosen.add(left.remove(i));
            }
        }
        if (sameCards(replay, steps)) {
            count("observe: replay of labels reproduces AI cards");
        } else {
            count("observe: REPLAY MISMATCH");
        }
    }

    private static int indexOf(List<Card> list, Card c) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == c || (c != null && list.get(i).getId() == c.getId())) {
                return i;
            }
        }
        return -1;
    }

    private static boolean sameCards(List<Card> a, List<Card> b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i).getId() != b.get(i).getId()) {
                return false;
            }
        }
        return true;
    }

    /**
     * The cost decision maker of a routed (deciding) or observed payer: Forge's
     * AI, except for the parts {@link Pick} mirrors from HumanCostDecision.
     */
    static final class Routed extends forge.ai.AiCostDecision {
        private final InstrumentedController ctl;
        private final boolean observe;

        Routed(InstrumentedController ctl, Player p, SpellAbility sa, boolean effect, boolean observe) {
            super(p, sa, effect);
            this.ctl = ctl;
            this.observe = observe;
        }

        private PaymentDecision run(CostPart part, Pick p, String what, Supplier<PaymentDecision> ai) {
            if (p == null) {
                if (!part.payCostFromSource()) {
                    count("silent: " + what + " (branch not mirrored, AI decides)");
                }
                return ai.get();
            }
            if (observe) {
                ManaPayment.Snap pre = choice(p, p.list, new ArrayList<>(), p.optional)
                        ? ManaPayment.preSnap(ctl) : null;
                PaymentDecision d = ai.get();
                try {
                    observe(ctl, ability, p, d == null ? null : d.cards, pre);
                } catch (RuntimeException e) {
                    count("observe: threw " + e.getClass().getSimpleName());
                }
                return d;
            }
            List<Card> got = ask(ctl, ability, p);
            if (got == null) {
                return ai.get();
            }
            if (!p.done(got)) {
                count("decide: DEAD END (requirement not met)");
                return null;
            }
            return PaymentDecision.card(got);
        }

        @Override
        public PaymentDecision visit(CostSacrifice cost) {
            return run(cost, sacrifice(cost, player, ability, isEffect()), "sacrifice", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(CostDiscard cost) {
            return run(cost, discard(cost, player, ability), "discard", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(CostExile cost) {
            return run(cost, exile(cost, player, ability, isEffect()), "exile", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(CostTapType cost) {
            return run(cost, tap(cost, player, ability), "tap", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(CostUntapType cost) {
            return run(cost, untap(cost, player, ability), "untap", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(CostReturn cost) {
            return run(cost, returning(cost, player, ability), "return", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(CostReveal cost) {
            return run(cost, reveal(cost, player, ability), "reveal", () -> super.visit(cost));
        }

        @Override
        public PaymentDecision visit(forge.game.cost.CostRemoveCounter cost) {
            if (!cost.payCostFromSource()) {
                count("silent: remove counters from other permanents (AI decides)");
            }
            return super.visit(cost);
        }

        @Override
        public PaymentDecision visit(forge.game.cost.CostCollectEvidence cost) {
            count("silent: collect evidence (AI decides)");
            return super.visit(cost);
        }

        @Override
        public PaymentDecision visit(forge.game.cost.CostPutCardToLib cost) {
            count("silent: put card to library cost (AI decides)");
            return super.visit(cost);
        }
    }

    /** Whether {@code sa}'s cost has a part {@link Routed} may ask about. */
    static boolean hasCardChoiceCost(SpellAbility sa) {
        Cost c = sa == null ? null : sa.getPayCosts();
        if (c == null) {
            return false;
        }
        for (CostPart part : c.getCostParts()) {
            if (part instanceof CostSacrifice || part instanceof CostDiscard || part instanceof CostExile
                    || part instanceof CostTapType || part instanceof CostUntapType
                    || part instanceof CostReturn || part instanceof CostReveal) {
                return true;
            }
        }
        return false;
    }

    /**
     * Observe mode: {@code PlayerControllerAi.playChosenSpellAbility} for a
     * non-land ability, i.e. {@code ComputerUtil.handlePlayingSpellAbility(ai,
     * sa, null)} line for line, with an observing {@link Routed} in place of the
     * {@code new AiCostDecision(ai, sa, false)} it builds. Only called when that
     * method's deferred-targeting action would be null (no TargetingPlayer).
     */
    static boolean observeCast(InstrumentedController ctl, SpellAbility sa) {
        Player ai = ctl.getPlayer();
        count("observe: casts replayed");
        Card source = sa.getHostCard();
        forge.game.Game game = source.getGame();
        Card host = sa.getHostCard();
        forge.game.zone.Zone hz = host.isCopiedSpell() ? null : host.getZone();
        source.setSplitStateToPlayAbility(sa);
        if (sa.isSpell() && !source.isCopiedSpell()) {
            sa = forge.game.ability.AbilityUtils.addSpliceEffects(sa);
            if (sa.getSplicedCards() != null && !sa.getSplicedCards().isEmpty() && ai.getController().isAI()) {
                sa.resetTargets();
                if (((forge.ai.PlayerControllerAi) ai.getController()).getAi().canPlaySa(sa)
                        != forge.ai.AiPlayDecision.WillPlay) {
                    return true;
                }
            }
            sa.setHostCard(game.getAction().moveToStack(source, sa));
        }
        if (!sa.isCopied()) {
            sa.resetPaidHash();
            sa.setPaidLife(0);
        }
        sa = forge.game.GameActionUtil.addExtraKeywordCost(sa);
        if (sa.getApi() == forge.game.ability.ApiType.Charm
                && !forge.game.ability.effects.CharmEffect.makeChoices(sa)) {
            return true;
        }
        Cost cost = sa.getPayCosts();
        game.getStack().freezeStack(sa);
        forge.game.cost.CostPayment pay = new forge.game.cost.CostPayment(cost, sa);
        if (pay.payComputerCosts(new Routed(ctl, ai, sa, false, true))) {
            game.getStack().addAndUnfreeze(sa);
            if (sa.getSplicedCards() != null && !sa.getSplicedCards().isEmpty()) {
                game.getAction().reveal(sa.getSplicedCards(), ai, true, "Computer reveals spliced cards from ");
            }
        } else {
            System.out.println("[" + sa.getActivatingPlayer() + "] AI failed to play " + sa.getHostCard() + " ["
                    + sa.getHostCard().getZone() + "]");
            sa.setSkip(true);
            if (host != null && hz != null && hz.is(ZoneType.Stack)) {
                Card c = game.getAction().moveTo(hz.getZoneType(), host, null, null);
                for (SpellAbility csa : c.getSpellAbilities()) {
                    csa.setSkip(true);
                }
            }
        }
        return true;
    }

    /** No ability in the chain has a TargetingPlayer (the AI's deferred-targeting action is null). */
    static boolean noTargetingPlayer(SpellAbility sa) {
        for (SpellAbility s = sa; s != null; s = s.getSubAbility()) {
            if (s.hasParam("TargetingPlayer")) {
                return false;
            }
        }
        return true;
    }

    // ---- optional costs ------------------------------------------------------------

    private static String costText(Cost cost, forge.game.Game game) {
        String s;
        try {
            s = cost.toSimpleString();
        } catch (RuntimeException e) {
            s = null;
        }
        if (s == null || s.isEmpty()) {
            s = String.valueOf(cost);
        }
        s = EventTokenizer.normalize(s, game);
        if (s != null && s.regionMatches(true, 0, "pay ", 0, 4)) {
            s = s.substring(4).trim();
        }
        return s == null || s.isBlank() ? null : s;
    }

    /**
     * Can {@code sa} be cast with {@code extra} added to its cost? Checked on a
     * Cost copy (never a SpellAbility copy, which would draw an ability id and
     * shift observe-mode records), with Forge's own affordability helper.
     */
    static boolean affordable(SpellAbility sa, Player me, List<Cost> extra) {
        try {
            Cost c = sa.getPayCosts() == null ? new Cost("", false) : sa.getPayCosts().copy();
            for (Cost x : extra) {
                c.add(x);
            }
            return forge.ai.ComputerUtilCost.canPayCost(c, sa, me, false);
        } catch (RuntimeException e) {
            count("mask: affordability check threw");
            return false;
        }
    }

    /** An optional cost is offered only if castable here and the total stays affordable. */
    private static boolean optionalLegal(SpellAbility sa, Player me, List<OptionalCostValue> picked,
            OptionalCostValue v) {
        OptionalCost t = v.getType();
        Card host = sa.getHostCard();
        boolean inGrave = host != null && host.isInZone(ZoneType.Graveyard);
        if ((t == OptionalCost.Retrace || t == OptionalCost.Jumpstart) && !inGrave) {
            return false;
        }
        List<Cost> extra = new ArrayList<>();
        for (OptionalCostValue x : picked) {
            if (x.getType() != OptionalCost.Offering) {
                extra.add(x.getCost());
            }
        }
        if (t != OptionalCost.Offering) {
            extra.add(v.getCost());
        }
        return affordable(sa, me, extra);
    }

    private static Serializer.Prompt optionalPrompt(InstrumentedController ctl, SpellAbility sa, int picked) {
        return ctl.promptFor(KIND_OPTIONAL, sa).optional(true).selectedOther(picked);
    }

    private static List<Serializer.Cand> optionalOpts(List<OptionalCostValue> legal, forge.game.Game game) {
        List<Serializer.Cand> out = new ArrayList<>();
        out.add(Serializer.Cand.option("choose_nothing"));
        for (OptionalCostValue v : legal) {
            out.add(Serializer.Cand.option(VERB_OPTIONAL).cost(costText(v.getCost(), game))
                    .ability(EventTokenizer.normalize(v.toString(), game)));
        }
        return out;
    }

    private static List<OptionalCostValue> legalOptional(SpellAbility sa, Player me, List<OptionalCostValue> left,
            List<OptionalCostValue> picked) {
        List<OptionalCostValue> out = new ArrayList<>();
        for (OptionalCostValue v : left) {
            if (optionalLegal(sa, me, picked, v)) {
                out.add(v);
            }
        }
        return out;
    }

    /** chooseOptionalCosts for a routed caster: add one optional cost at a time, or stop. */
    static List<OptionalCostValue> optional(InstrumentedController ctl, SpellAbility sa,
            List<OptionalCostValue> list) {
        Player me = ctl.getPlayer();
        List<OptionalCostValue> left = new ArrayList<>(list);
        List<OptionalCostValue> picked = new ArrayList<>();
        count("decide: optional cost choices");
        while (!left.isEmpty()) {
            List<OptionalCostValue> legal = legalOptional(sa, me, left, picked);
            if (legal.isEmpty()) {
                count("decide: optional costs none affordable (not asked)");
                break;
            }
            int pick = ctl.askChoice(optionalPrompt(ctl, sa, picked.size()), optionalOpts(legal, me.getGame()));
            if (pick < 0 || pick > legal.size()) {
                count("decide: optional delegated");
                return null;
            }
            count("decide: asked choose_optional_cost");
            if (pick == 0) {
                break;
            }
            OptionalCostValue v = legal.get(pick - 1);
            picked.add(v);
            left.remove(v);
        }
        return picked;
    }

    /**
     * Observe mode: the AI chose a variant of a pool ability with optional
     * costs already added (its getOriginalAndAltCostAbilities); record the
     * routed sequence against the pool ability it came from, labelled with
     * the variant's optional costs (in the list's order, then stop).
     */
    static void observeOptional(InstrumentedController ctl, SpellAbility chosen, List<SpellAbility> pool) {
        if (chosen == null || !chosen.isSpell() || pool == null) {
            return;
        }
        Player me = ctl.getPlayer();
        SpellAbility base = baseOf(chosen, pool);
        if (base == null) {
            count("observe: optional base ability not found");
            return;
        }
        List<OptionalCostValue> list;
        try {
            Card src = base.getHostCard();
            if (src == null || src.isInPlay() || base.getAlternateHost(src) != null) {
                return;
            }
            list = forge.game.GameActionUtil.getOptionalCostValues(base);
        } catch (RuntimeException e) {
            count("observe: optional cost values threw");
            return;
        }
        if (list.isEmpty()) {
            return;
        }
        List<OptionalCost> aiTypes = new ArrayList<>();
        for (OptionalCost t : chosen.getOptionalCosts()) {
            aiTypes.add(t);
        }
        count("observe: optional cost choices");
        ManaPayment.Snap pre = ManaPayment.preSnap(ctl);
        List<OptionalCostValue> left = new ArrayList<>(list);
        List<OptionalCostValue> picked = new ArrayList<>();
        List<OptionalCost> replay = new ArrayList<>();
        while (true) {
            List<OptionalCostValue> legal = legalOptional(base, me, left, picked);
            int label = 0;
            for (int i = 0; i < legal.size(); i++) {
                if (aiTypes.contains(legal.get(i).getType())) {
                    label = i + 1;
                    break;
                }
            }
            if (legal.isEmpty()) {
                break;
            }
            if (FAULT) {
                label = (label + 1) % (legal.size() + 1);
            }
            if (ManaPayment.canRecord(pre, me)) {
                Serializer.Built body = Serializer.buildChoiceBody(me.getGame(), me, pre.toks, pre.histSize,
                        optionalPrompt(ctl, base, picked.size()), optionalOpts(legal, me.getGame()), false);
                ctl.writeBuilt(body, label);
                count("observe: recorded choose_optional_cost");
            }
            if (label == 0) {
                break;
            }
            OptionalCostValue v = legal.get(label - 1);
            replay.add(v.getType());
            picked.add(v);
            left.remove(v);
        }
        if (replay.size() == aiTypes.size() && aiTypes.containsAll(replay)) {
            count("observe: replay of labels reproduces AI optional costs");
        } else {
            count("observe: REPLAY MISMATCH (optional costs)");
        }
    }

    private static SpellAbility baseOf(SpellAbility chosen, List<SpellAbility> pool) {
        for (SpellAbility s : pool) {
            if (s == chosen) {
                return s;
            }
        }
        Card h = chosen.getHostCard();
        for (SpellAbility s : pool) {
            if (s != null && s.isSpell() && s.getHostCard() != null && h != null
                    && s.getHostCard().getId() == h.getId() && s.getApi() == chosen.getApi()
                    && !s.getOptionalCosts().iterator().hasNext()) {
                return s;
            }
        }
        return null;
    }

    // ---- keyword costs (Casualty, Conspire, Offspring, Multikicker, Replicate, Squad) ----

    private static final int KEYWORD_CAP = 8;

    private static Serializer.Prompt keywordPrompt(InstrumentedController ctl, SpellAbility sa, String text) {
        return ctl.promptFor(KIND_OPTIONAL, sa).optional(true).text(text, ctl.getGame());
    }

    /** Options: stop (0 times), then paying the keyword cost 1..k times while affordable. */
    private static List<Serializer.Cand> keywordOpts(SpellAbility sa, Player me, Cost cost, int max,
            String label, forge.game.Game game) {
        List<Serializer.Cand> out = new ArrayList<>();
        out.add(Serializer.Cand.option("choose_nothing"));
        List<Cost> extra = new ArrayList<>();
        for (int n = 1; n <= Math.min(max, KEYWORD_CAP); n++) {
            extra.add(cost);
            if (!affordable(sa, me, extra)) {
                break;
            }
            Serializer.Cand c = Serializer.Cand.option(VERB_OPTIONAL).cost(costText(cost, game)).ability(label);
            if (max > 1) {
                c.value(n);
            }
            out.add(c);
        }
        return out;
    }

    private static String keywordLabel(forge.game.keyword.KeywordInterface ki, forge.game.Game game) {
        try {
            String s = ki == null ? null : ki.getTitle();
            return s == null ? null : EventTokenizer.normalize(s, game);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** chooseNumberForKeywordCost, routed; -1 when not asked. */
    static int keyword(InstrumentedController ctl, SpellAbility sa, Cost cost,
            forge.game.keyword.KeywordInterface ki, String text, int max) {
        Player me = ctl.getPlayer();
        List<Serializer.Cand> opts = keywordOpts(sa, me, cost, max, keywordLabel(ki, me.getGame()), me.getGame());
        if (opts.size() < 2) {
            count("decide: keyword cost unaffordable (0, not asked)");
            return 0;
        }
        int pick = ctl.askChoice(keywordPrompt(ctl, sa, text), opts);
        if (pick < 0 || pick >= opts.size()) {
            count("decide: keyword delegated");
            return -1;
        }
        count("decide: asked keyword cost");
        return pick;
    }

    /** Observe mode: record the AI's keyword-cost answer. */
    static void observeKeyword(InstrumentedController ctl, SpellAbility sa, Cost cost,
            forge.game.keyword.KeywordInterface ki, String text, int max, int aiAnswer, ManaPayment.Snap pre) {
        Player me = ctl.getPlayer();
        List<Serializer.Cand> opts = keywordOpts(sa, me, cost, max, keywordLabel(ki, me.getGame()), me.getGame());
        if (opts.size() < 2) {
            return;
        }
        int label = aiAnswer >= 0 && aiAnswer < opts.size() ? aiAnswer : -1;
        if (label < 0) {
            count("observe: AI keyword answer beyond the mask");
        }
        if (FAULT && label >= 0) {
            label = (label + 1) % opts.size();
        }
        if (ManaPayment.canRecord(pre, me)) {
            ctl.writeBuilt(Serializer.buildChoiceBody(me.getGame(), me, pre.toks, pre.histSize,
                    keywordPrompt(ctl, sa, text), opts, false), label);
            count("observe: recorded keyword cost");
        }
    }

    // ---- alternative additional costs (getAbilityToPlay) -------------------------------

    /** getAbilityToPlay over a spell's AlternateAdditionalCost variants, routed. */
    static SpellAbility alternative(InstrumentedController ctl, Card host, List<SpellAbility> abilities) {
        Player me = ctl.getPlayer();
        List<SpellAbility> legal = new ArrayList<>();
        for (SpellAbility s : abilities) {
            if (s.getActivatingPlayer() == null) {
                s.setActivatingPlayer(me);
            }
            if (affordable(s, me, new ArrayList<>())) {
                legal.add(s);
            }
        }
        if (legal.isEmpty()) {
            return null;
        }
        if (legal.size() == 1) {
            count("decide: additional cost forced");
            return legal.get(0);
        }
        List<Serializer.Cand> opts = new ArrayList<>();
        for (SpellAbility s : legal) {
            opts.add(Serializer.Cand.option(VERB_OPTIONAL).cost(costText(s.getPayCosts(), me.getGame())));
        }
        int pick = ctl.askChoice(ctl.promptFor(KIND_ALTERNATIVE, abilities.get(0)), opts);
        if (pick < 0 || pick >= legal.size()) {
            count("decide: additional cost delegated");
            return null;
        }
        count("decide: asked choose_additional_cost");
        return legal.get(pick);
    }

    // ---- Emerge / Offering (CostAdjustment's choosePermanentsToSacrifice) --------------

    static boolean isEmergeOrOffering(SpellAbility sa, int min, int max) {
        return sa != null && min == 0 && max == 1 && (sa.isEmerge() || sa.isOffering());
    }

    /** The sacrifice that reduces the cost; required (the cast fails without one). */
    static Pick emergePick(InstrumentedController ctl, SpellAbility sa,
            forge.game.card.CardCollectionView list) {
        List<Card> legal = new ArrayList<>();
        for (Card c : list) {
            if (ManaPayment.payableAfterSacrifice(ctl.getPlayer(), sa, c, sa.isEmerge())) {
                legal.add(c);
            }
        }
        if (legal.isEmpty()) {
            count("mask: emerge/offering none feasible (all offered)");
            legal.addAll(list);
        }
        return new Pick(KIND_SACRIFICE, sa.isEmerge() ? "Emerge" : "Offering", legal, 1, 0, false);
    }
}
