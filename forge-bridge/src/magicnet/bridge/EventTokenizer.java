package magicnet.bridge;

import java.util.Collection;
import java.util.Map;

import com.google.common.collect.Multimap;

import forge.game.GameEntityView;
import forge.game.card.CardView;
import forge.game.event.GameEvent;
import forge.game.event.GameEventAttackersDeclared;
import forge.game.event.GameEventBlockersDeclared;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardDestroyed;
import forge.game.event.GameEventCardForetold;
import forge.game.event.GameEventCardModeChosen;
import forge.game.event.GameEventCardPhased;
import forge.game.event.GameEventCardRegenerated;
import forge.game.event.GameEventCardSacrificed;
import forge.game.event.GameEventFlipCoin;
import forge.game.event.GameEventLandPlayed;
import forge.game.event.GameEventMulligan;
import forge.game.event.GameEventPlayerCounters;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventRollDie;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventSurveil;
import forge.game.event.GameEventTokenCreated;
import forge.game.event.IGameEventVisitor;
import forge.game.player.PlayerView;
import forge.game.spellability.SpellAbilityView;
import forge.game.zone.ZoneView;

/**
 * Turns a Forge game event into a history token, redacted for one viewer.
 *
 * <h2>Why a visitor</h2>
 * Forge ships {@code IGameEventVisitor.Base<T>}, which supplies a null-returning
 * default for all 58 event types, so only the types we actually keep need an
 * override and each one gets compile-time-checked typed field access. That is
 * strictly better than reflecting over field names, which would break silently
 * whenever Forge renames a field.
 *
 * <h2>Why this is per-viewer</h2>
 * Events carry {@link CardView}, which is full-information. Writing card names
 * into a history shared by both controllers would leak hidden information: a
 * library-to-hand {@code CardChangeZone} would expose the card the opponent just
 * drew. Each player therefore gets its own tokenizer and its own history.
 *
 * <h2>Why visibility is snapshotted here</h2>
 * Redaction is decided when the event fires, not when the state is later
 * extracted, because observation is immutable. Deciding at extraction time would
 * be wrong in both directions: a card drawn face-down but revealed later would
 * retroactively leak backwards through the log, and a card that was revealed on
 * the stack but has since returned to hand would be erased even though the
 * opponent legitimately saw it.
 */
public final class EventTokenizer extends IGameEventVisitor.Base<StateExtractor.Tok> {

    private final PlayerView viewer;

    /**
     * Current turn, set immediately before dispatch. The visitor signature is
     * fixed by Forge's interface so the turn cannot be passed as an argument.
     * Safe because events are dispatched synchronously on the single game thread.
     */
    private int turn;

    /** Counts redactions actually applied, as evidence the boundary holds. */
    private int redactions;

    public EventTokenizer(PlayerView viewer) {
        this.viewer = viewer;
    }

    public StateExtractor.Tok tokenize(GameEvent ev, int turn, String fallbackVerb) {
        this.turn = turn;
        StateExtractor.Tok t = null;
        try {
            t = ev.visit(this);
        } catch (Throwable ignored) {
            // A malformed event must never fail a game.
        }
        return t != null ? t : ev(fallbackVerb);
    }

    public int redactions() {
        return redactions;
    }

    private StateExtractor.Tok ev(String verb) {
        return new StateExtractor.Tok(Schema.HISTORY, Schema.HISTORY)
                .p("turn", turn)
                .p("event", verb);
    }

    /**
     * Attaches a card under a role, omitting the name (and oracle id) when the
     * viewer may not see it: an unseen card has no name, and absence means
     * unknown, as everywhere else in the schema. The id is always kept: it is a join key for debugging and for edge
     * resolution, and by the earlier argument it never needs to reach the
     * embedding, since two objects identical in every visible property and edge
     * are genuinely interchangeable.
     */
    private StateExtractor.Tok card(StateExtractor.Tok t, String role, CardView c) {
        return card(t, role, c, false);
    }

    private StateExtractor.Tok card(StateExtractor.Tok t, String role, CardView c, boolean seen) {
        if (c == null) {
            return t;
        }
        t.p(role + "_id", StateExtractor.cardRef(c.getId()));
        boolean visible = seen;
        try {
            visible = visible || c.canBeShownTo(viewer);
        } catch (Throwable ignored) {
            // keep what the caller established
        }
        if (visible) {
            t.p(role, c.getName());
            t.p(role + "_oracle_id", OracleIds.of(c));
        } else {
            redactions++;
        }
        return t;
    }

    private StateExtractor.Tok player(StateExtractor.Tok t, String role, PlayerView p) {
        if (p != null) {
            t.p(role, StateExtractor.playerRef(p.getId()));
        }
        return t;
    }

    private StateExtractor.Tok entity(StateExtractor.Tok t, String role, GameEntityView e) {
        if (e == null) {
            return t;
        }
        if (e instanceof CardView) {
            return card(t, role, (CardView) e);
        }
        if (e instanceof PlayerView) {
            return player(t, role, (PlayerView) e);
        }
        return t.p(role, e.getName());
    }

    private StateExtractor.Tok zone(StateExtractor.Tok t, String role, ZoneView z) {
        if (z == null) {
            return t;
        }
        t.p(role, String.valueOf(z.zoneType()));
        return player(t, role + "_player", z.player());
    }

    /**
     * A spell is identified by its host card plus which ability was used, because
     * the same card can offer several (Brazen Borrower's body vs Petty Theft).
     */
    private StateExtractor.Tok spell(StateExtractor.Tok t, SpellAbilityView sa) {
        if (sa == null) {
            return t;
        }
        card(t, "source", sa.getHostCard());
        return t.p("ability", withoutObjectIds(sa.getDescription()));
    }

    /**
     * Forge renders cards inside ability text with {@code Card.toString()}, i.e.
     * {@code "Name (id)"}: the "Name (105): Prowess ..." prefix of keyword
     * triggers, a trigger's "[Zone Changer: Name (170)]" context, "Play land by
     * Name (108)". Object ids are not reproducible across runs of one seed, so
     * they are removed. There is no id-free rendering reachable from the
     * {@code SpellAbilityView} the events carry: its description is
     * {@code toUnsuppressedString()}, which is where the ids come from.
     */
    static String withoutObjectIds(String s) {
        return s == null ? null : OBJECT_ID.matcher(s).replaceAll("");
    }

    /** " (123)" as Card.toString() appends it, before punctuation, space or the end. */
    private static final java.util.regex.Pattern OBJECT_ID =
            java.util.regex.Pattern.compile(" \\(\\d+\\)(?=[:,\\]')\\s]|$)");

    /**
     * Closes a casting attempt. See {@link GameHistory.Attempt} for outcomes.
     * Names are redacted as of now, so a card a rollback returned to hand is
     * hidden from the opponent here even though they saw it announced.
     */
    public StateExtractor.Tok attemptEnd(int turn, int attempt, String outcome, String reached,
            forge.game.spellability.SpellAbility sa, java.util.List<forge.game.GameEntity> targets,
            Map<String, Integer> announced) {
        this.turn = turn;
        StateExtractor.Tok t = ev("AttemptEnd")
                .p("attempt", attempt)
                .p("outcome", outcome)
                .p("reached", reached);
        try {
            if (sa != null) {
                spell(t, sa.getView());
                if (sa.getActivatingPlayer() != null) {
                    player(t, "actor", sa.getActivatingPlayer().getView());
                }
            }
        } catch (Throwable ignored) {
            // keep the outcome even without the proposal
        }
        for (int i = 0; i < targets.size(); i++) {
            forge.game.GameEntity e = targets.get(i);
            if (e instanceof forge.game.card.Card) {
                card(t, "target_" + i, ((forge.game.card.Card) e).getView());
            } else if (e instanceof forge.game.player.Player) {
                player(t, "target_" + i, ((forge.game.player.Player) e).getView());
            }
        }
        for (Map.Entry<String, Integer> e : announced.entrySet()) {
            t.p("announce_" + e.getKey(), e.getValue());
        }
        return t;
    }

    /**
     * One card of a reveal, as seen by this history's viewer.
     *
     * <p>Not a Forge event: reveals arrive only as a controller callback on each
     * player being shown, so the caller posts one of these into every viewer's
     * history. The recipient gets the name because the callback was delivered to
     * them; anyone else gets it only if Forge already lets them see the card
     * (e.g. it is in their own hand), so a bystander learns that a reveal
     * happened and how big it was, not what it showed.
     */
    public StateExtractor.Tok reveal(int turn, forge.game.card.Card c,
            forge.game.zone.ZoneType zone, forge.game.player.Player owner,
            forge.game.player.Player shownTo, int group, int size) {
        this.turn = turn;
        StateExtractor.Tok t = ev("Reveal");
        t.p("card_id", StateExtractor.cardRef(c.getId()));
        boolean recipient = viewer != null && shownTo != null && viewer.getId() == shownTo.getId();
        boolean visible = recipient;
        if (!visible) {
            try {
                visible = c.getView().canBeShownTo(viewer);
            } catch (Throwable ignored) {
                visible = false;
            }
        }
        if (visible) {
            t.p("card", c.getName());
            t.p("card_oracle_id", OracleIds.of(c));
        } else {
            redactions++;
        }
        // Not "zone": every history token already carries zone="history", and a
        // duplicate JSON key silently overwrote it.
        t.p("revealed_from", String.valueOf(zone));
        if (owner != null) {
            player(t, "owner", owner.getView());
        }
        if (shownTo != null) {
            player(t, "shown_to", shownTo.getView());
        }
        return t.p("reveal", group).p("reveal_size", size);
    }

    // ---- zone movement and casting -------------------------------------

    @Override
    public StateExtractor.Tok visit(GameEventCardChangeZone e) {
        StateExtractor.Tok t = ev("CardChangeZone");
        // canBeShownTo reads the card where it is NOW, so a card moving from a
        // public zone into a hidden one would be redacted although everyone saw
        // which card moved. Decide from what the viewer saw as it left.
        card(t, "card", e.card(), RevealKnowledge.sawItLeave(e.card(), e.from(), viewer));
        try {
            // Public: everyone sees that a face-down card moved. Lets the leak
            // check refuse a face-down departure as a public-origin cause.
            if (e.card() != null && e.card().isFaceDown()) {
                t.p("face_down", true);
            }
        } catch (Throwable ignored) {
            // omit rather than guess
        }
        zone(t, "from", e.from());
        return zone(t, "to", e.to());
    }

    @Override
    public StateExtractor.Tok visit(GameEventSpellAbilityCast e) {
        StateExtractor.Tok t = ev("SpellAbilityCast");
        // Forge's target description is a rendered string with raw card ids
        // ("Bolt (105)"), which are run-unstable join keys. Targets are recorded
        // on the attempt's AttemptEnd as target_<i> refs instead.
        return spell(t, e.sa());
    }

    @Override
    public StateExtractor.Tok visit(GameEventSpellResolved e) {
        StateExtractor.Tok t = ev("SpellResolved");
        spell(t, e.spell());
        // A fizzle is a materially different outcome from a resolution and is
        // public information, so it must not be collapsed into the same token.
        return t.flag("fizzled", e.hasFizzled());
    }

    @Override
    public StateExtractor.Tok visit(GameEventLandPlayed e) {
        StateExtractor.Tok t = ev("LandPlayed");
        player(t, "actor", e.player());
        return card(t, "card", e.land());
    }

    // ---- removal and damage --------------------------------------------

    /** Carries no fields at all; the card that died comes from CardChangeZone. */
    @Override
    public StateExtractor.Tok visit(GameEventCardDestroyed e) {
        return ev("CardDestroyed");
    }

    @Override
    public StateExtractor.Tok visit(GameEventCardSacrificed e) {
        return card(ev("CardSacrificed"), "card", e.card());
    }

    @Override
    public StateExtractor.Tok visit(GameEventCardDamaged e) {
        StateExtractor.Tok t = ev("CardDamaged");
        card(t, "card", e.card());
        card(t, "source", e.source());
        return t.p("amount", e.amount()).p("damage_type", String.valueOf(e.type()));
    }

    @Override
    public StateExtractor.Tok visit(GameEventPlayerDamaged e) {
        StateExtractor.Tok t = ev("PlayerDamaged");
        player(t, "target", e.target());
        card(t, "source", e.source());
        return t.p("amount", e.amount()).flag("combat", e.combat()).flag("infect", e.infect());
    }

    @Override
    public StateExtractor.Tok visit(GameEventCardRegenerated e) {
        StateExtractor.Tok t = ev("CardRegenerated");
        Collection<CardView> cs = e.cards();
        if (cs != null) {
            int i = 0;
            for (CardView c : cs) {
                card(t, "card" + (i++), c);
            }
            t.p("count", cs.size());
        }
        return t;
    }

    // ---- counters and totals -------------------------------------------

    @Override
    public StateExtractor.Tok visit(GameEventCardCounters e) {
        StateExtractor.Tok t = ev("CardCounters");
        card(t, "card", e.card());
        return t.p("counter", String.valueOf(e.type()))
                .p("from", e.oldValue())
                .p("to", e.newValue());
    }

    @Override
    public StateExtractor.Tok visit(GameEventPlayerCounters e) {
        StateExtractor.Tok t = ev("PlayerCounters");
        player(t, "target", e.receiver());
        return t.p("counter", String.valueOf(e.type()))
                .p("from", e.oldValue())
                .p("amount", e.amount());
    }

    @Override
    public StateExtractor.Tok visit(GameEventPlayerLivesChanged e) {
        StateExtractor.Tok t = ev("PlayerLivesChanged");
        player(t, "target", e.player());
        return t.p("from", e.oldLives()).p("to", e.newLives());
    }

    @Override
    public StateExtractor.Tok visit(GameEventPlayerPoisoned e) {
        StateExtractor.Tok t = ev("PlayerPoisoned");
        player(t, "target", e.receiver());
        player(t, "source", e.source());
        return t.p("from", e.oldValue()).p("amount", e.amount());
    }

    // ---- combat ---------------------------------------------------------

    @Override
    public StateExtractor.Tok visit(GameEventAttackersDeclared e) {
        StateExtractor.Tok t = ev("AttackersDeclared");
        player(t, "actor", e.player());
        Multimap<GameEntityView, CardView> m = e.attackersMap();
        int n = 0;
        if (m != null) {
            for (Map.Entry<GameEntityView, CardView> en : m.entries()) {
                t.edge(Schema.ATTACKS, StateExtractor.cardRef(en.getValue().getId()));
                t.edge(Schema.DEFENDER, refOf(en.getKey()));
                n++;
            }
        }
        return t.p("count", n);
    }

    /**
     * Forge builds this map as attacker -> blockers, but when an attacker is
     * unblocked it stores {@code List.of(attacker)} as a sentinel rather than an
     * empty collection. A self-pair therefore means *unblocked*, and emitting it
     * as a block edge would teach the model that creatures block themselves.
     *
     * Pair edges are emitted first, strictly as (blocks, blocked) adjacent pairs,
     * followed by any unblocked attackers, so downstream expansion can read the
     * paired region by position without ambiguity.
     */
    @Override
    public StateExtractor.Tok visit(GameEventBlockersDeclared e) {
        StateExtractor.Tok t = ev("BlockersDeclared");
        player(t, "actor", e.defendingPlayer());
        Map<GameEntityView, Multimap<CardView, CardView>> m = e.blockers();
        java.util.List<Integer> unblocked = new java.util.ArrayList<>();
        int pairs = 0;
        if (m != null) {
            for (Multimap<CardView, CardView> byAttacker : m.values()) {
                for (Map.Entry<CardView, CardView> en : byAttacker.entries()) {
                    int attacker = en.getKey().getId();
                    int blocker = en.getValue().getId();
                    if (attacker == blocker) {
                        unblocked.add(attacker);
                        continue;
                    }
                    t.edge(Schema.BLOCKS, StateExtractor.cardRef(blocker));
                    t.edge(Schema.BLOCKED, StateExtractor.cardRef(attacker));
                    pairs++;
                }
            }
        }
        for (int id : unblocked) {
            t.edge(Schema.UNBLOCKED, StateExtractor.cardRef(id));
        }
        return t.p("blocks_count", pairs).p("unblocked_count", unblocked.size());
    }

    /** A combat defender may be a player, a planeswalker, or a battle. */
    private static String refOf(GameEntityView e) {
        if (e instanceof CardView) {
            return StateExtractor.cardRef(((CardView) e).getId());
        }
        if (e instanceof PlayerView) {
            return StateExtractor.playerRef(((PlayerView) e).getId());
        }
        return null;
    }

    // ---- misc ------------------------------------------------------------

    @Override
    public StateExtractor.Tok visit(GameEventCardAttachment e) {
        StateExtractor.Tok t = ev("CardAttachment");
        card(t, "attachment", e.equipment());
        entity(t, "from", e.oldEntity());
        return entity(t, "to", e.newTarget());
    }

    @Override
    public StateExtractor.Tok visit(GameEventCardPhased e) {
        StateExtractor.Tok t = ev("CardPhased");
        card(t, "card", e.card());
        return t.p("phased_out", e.phaseState());
    }

    @Override
    public StateExtractor.Tok visit(GameEventCardModeChosen e) {
        StateExtractor.Tok t = ev("CardModeChosen");
        player(t, "actor", e.player());
        return t.p("card", e.cardName()).p("mode", e.mode());
    }

    /**
     * Scry and surveil leak nothing by themselves: the counts are public, the
     * card identities are not and are not carried on the event.
     */
    @Override
    public StateExtractor.Tok visit(GameEventScry e) {
        StateExtractor.Tok t = ev("Scry");
        player(t, "actor", e.player());
        return t.p("to_top", e.toTop()).p("to_bottom", e.toBottom());
    }

    @Override
    public StateExtractor.Tok visit(GameEventSurveil e) {
        StateExtractor.Tok t = ev("Surveil");
        player(t, "actor", e.player());
        return t.p("to_library", e.toLibrary()).p("to_graveyard", e.toGraveyard());
    }

    @Override
    public StateExtractor.Tok visit(GameEventMulligan e) {
        return player(ev("Mulligan"), "actor", e.player());
    }

    @Override
    public StateExtractor.Tok visit(GameEventCardForetold e) {
        return player(ev("CardForetold"), "actor", e.activatingPlayer());
    }

    /** Both carry no fields; the outcome is not exposed on the event. */
    @Override
    public StateExtractor.Tok visit(GameEventFlipCoin e) {
        return ev("FlipCoin");
    }

    @Override
    public StateExtractor.Tok visit(GameEventRollDie e) {
        return ev("RollDie");
    }

    @Override
    public StateExtractor.Tok visit(GameEventTokenCreated e) {
        return ev("TokenCreated");
    }
}
