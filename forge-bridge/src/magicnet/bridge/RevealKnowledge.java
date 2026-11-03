package magicnet.bridge;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEvent;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.IGameEventVisitor;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import forge.game.zone.ZoneView;

/**
 * Per-viewer knowledge of specific cards in an opponent's hand, learned from a
 * reveal.
 *
 * <p>Forge has no reveal {@link GameEvent}. A reveal reaches the bridge only as
 * {@link forge.game.player.PlayerController#reveal} on the controller of each
 * player being <em>shown</em> the cards ({@code GameAction.reveal} loops over the
 * recipients, skipping the owner when {@code dontRevealToOwner}). The AI
 * controller remembers the cards in its private {@code AiCardMemory}; nothing in
 * the game state records that the opponent now knows them, and {@code
 * CardView.canBeShownTo} stays false. So, as with {@link LibraryKnowledge}, the
 * information exists only for the duration of a controller callback, and the
 * callback's receiver is the one and only viewer it may be attributed to.
 *
 * <h2>Identity, validated against the live zone</h2>
 *
 * <p>We store {@code (card id, game timestamp)} per viewer, never a name or a
 * slot. Extraction only consults this for cards it is enumerating out of the
 * opponent's live hand, and accepts the fact only if the timestamp still
 * matches. Forge keeps a card's id across zone changes but gives it a fresh
 * game timestamp on every move except onto the stack ({@code
 * GameAction.changeZone}), so a card that leaves the hand and later comes back
 * no longer matches -- it is unknown again unless re-revealed. Knowledge is
 * additionally dropped eagerly when a {@link GameEventCardChangeZone} moves the
 * card out of a hand (the hand-to-stack move keeps its timestamp, so the event
 * is what covers cast-then-bounced). Either mechanism alone suffices for the
 * cases measured; both together fail closed.
 *
 * <p>Shuffle-hand-into-library, discard, mulligan and cast are all zone changes
 * out of the hand, so they need no special handling.
 *
 * <h2>Other sources</h2>
 *
 * <p>Two non-callback sources feed the same store, both from {@link
 * GameEventCardChangeZone}: a card entering a hand from a public zone is known
 * to every viewer who could see it leave ({@link #sawItLeave}; bounce,
 * regrowth), and a card revealed outside a hand and then moved into one by the
 * same resolving effect is known to the reveal's recipient ({@link #pend};
 * digs such as Silundi Vision, whose reveal precedes the move).
 */
public final class RevealKnowledge {

    /** viewer id -> (card id -> game timestamp at reveal). */
    private static final Map<Integer, Map<Integer, Long>> KNOWN = new HashMap<>();

    private static final Set<Integer> ATTACHED = new HashSet<>();

    private static int learned;
    private static int ignoredOwn;
    private static int ownOnlyCallbacks;
    private static int suppressed;
    private static int ignoredNotInHand;
    private static int droppedByMove;
    private static int emittedKnown;

    private RevealKnowledge() {
    }

    /**
     * Identity-level core, separated from {@link Card} so it can be checked
     * without a running game.
     *
     * @return whether a fact was recorded; viewing one's own card teaches nothing
     */
    static synchronized boolean learn(int viewerId, int ownerId, int cardId, long timestamp) {
        if (viewerId == ownerId) {
            ignoredOwn++;
            return false;
        }
        Long prev = KNOWN.computeIfAbsent(viewerId, k -> new HashMap<>()).put(cardId, timestamp);
        if (prev == null || prev != timestamp) {
            learned++;
        }
        return true;
    }

    static synchronized boolean knows(int viewerId, int cardId, long timestamp) {
        Map<Integer, Long> m = KNOWN.get(viewerId);
        Long ts = m == null ? null : m.get(cardId);
        return ts != null && ts == timestamp;
    }

    /** The card left a hand: nobody's reveal knowledge of it survives. */
    static synchronized void forget(int cardId) {
        for (Map<Integer, Long> m : KNOWN.values()) {
            if (m.remove(cardId) != null) {
                droppedByMove++;
            }
        }
    }

    /**
     * Records that {@code viewer} was shown {@code card}. Must only be called from
     * a reveal callback delivered to {@code viewer}'s own controller.
     */
    public static boolean observe(Player viewer, Card card) {
        if (viewer == null || card == null) {
            return false;
        }
        try {
            if (card.getZone() == null || card.getZone().getZoneType() != ZoneType.Hand) {
                synchronized (RevealKnowledge.class) {
                    ignoredNotInHand++;
                }
                return false;
            }
            Player owner = card.getZone().getPlayer();
            return learn(viewer.getId(), owner == null ? -1 : owner.getId(),
                    card.getId(), card.getGameTimestamp());
        } catch (Throwable ignored) {
            return false;
        }
    }

    /** Whether {@code viewer} knows {@code card}, which the caller found in a hand. */
    public static boolean knows(Player viewer, Card card) {
        if (viewer == null || card == null) {
            return false;
        }
        try {
            boolean k = knows(viewer.getId(), card.getId(), card.getGameTimestamp());
            if (k) {
                synchronized (RevealKnowledge.class) {
                    emittedKnown++;
                }
            }
            return k;
        } catch (Throwable ignored) {
            return false;
        }
    }

    /** A reveal callback that showed the receiver only their own cards. */
    public static synchronized void noteOwnOnly() {
        ownOnlyCallbacks++;
    }

    /** A Forge reveal callback the card's rules text does not justify; see InstrumentedController. */
    public static synchronized void noteSuppressed() {
        suppressed++;
    }

    /**
     * Whether {@code viewer} could see a card as it left {@code from}.
     *
     * <p>Public zones only, and never a face-down card unless Forge already let
     * this viewer look at it (its controller, or a "may look" grant). That is
     * deliberately stricter than CR 708.9, which reveals face-down permanents and
     * spells as they leave: we do not see that reveal happen, so we do not
     * assume it. Shared with EventTokenizer so the history token and the hand
     * knowledge are redacted by the same rule.
     */
    static boolean publicOrigin(ZoneType from, boolean faceDown, boolean mayLook) {
        if (from == null) {
            return false;
        }
        switch (from) {
            case Battlefield:
            case Graveyard:
            case Stack:
            case Exile:
            case Command:
                return !faceDown || mayLook;
            default:
                return false;
        }
    }

    /**
     * {@link #publicOrigin} for a zone-change event. {@code old} must be the
     * pre-move card's view: Forge fires GameEventCardChangeZone with the old
     * Card object, and every Card owns its own CardView, so its face-down state
     * is what the viewer saw before the move.
     */
    public static boolean sawItLeave(forge.game.card.CardView old, ZoneView from,
            forge.game.player.PlayerView viewer) {
        if (old == null || from == null || viewer == null) {
            return false;
        }
        try {
            boolean faceDown = old.isFaceDown();
            boolean mayLook = faceDown && (viewer.equals(old.getController()) || old.mayPlayerLook(viewer));
            return publicOrigin(from.zoneType(), faceDown, mayLook);
        } catch (Throwable ignored) {
            return false;
        }
    }

    /** A card shown to a viewer outside a hand, awaiting a same-effect move into one. */
    private static final class Pending {
        final int viewerId;
        final ZoneType zone;
        final Object resolving;

        Pending(int viewerId, ZoneType zone, Object resolving) {
            this.viewerId = viewerId;
            this.zone = zone;
            this.resolving = resolving;
        }
    }

    /** card id -> reveals of it that may still be followed by a move to hand. */
    private static final Map<Integer, java.util.List<Pending>> PENDING = new HashMap<>();

    /**
     * Records that {@code viewerId} was shown card {@code cardId} in {@code zone}
     * (not a hand) while {@code resolving} was resolving.
     *
     * <p>Silundi Vision and other digs reveal the card while it is still in the
     * library and move it afterwards (DigEffect; measured), so the reveal cannot
     * attach to a hand card yet, and the move gives the card a new timestamp.
     * The link is made only by {@link #moved} for the card's very next move,
     * when that move goes from the revealed zone into a hand while the SAME
     * ability is still on top of the stack. Anything else -- the card stays put
     * and is drawn later, a different effect moves it, nothing was resolving --
     * drops it. Reveals that arrive after the move (Granted) need none of
     * this: the card is already in hand and {@link #observe} records it.
     */
    static synchronized void pend(int viewerId, int ownerId, int cardId, ZoneType zone, Object resolving) {
        if (viewerId == ownerId || resolving == null || zone == null || zone == ZoneType.Hand) {
            return;
        }
        PENDING.computeIfAbsent(cardId, k -> new java.util.ArrayList<>())
                .add(new Pending(viewerId, zone, resolving));
    }

    /**
     * Applies one zone change: leaving a hand forgets the card; entering a hand
     * teaches each viewer who saw it leave a public zone, and the recipient of a
     * pending same-effect reveal.
     */
    static synchronized void moved(int cardId, ZoneType from, ZoneType to, int toOwnerId,
            long newTimestamp, Object resolving, int[] viewerIds,
            java.util.function.IntPredicate sawItLeave) {
        if (from == ZoneType.Hand) {
            forget(cardId);
        }
        java.util.List<Pending> mine = PENDING.remove(cardId);
        PENDING.values().removeIf(l -> {
            l.removeIf(p -> p.resolving != resolving);
            return l.isEmpty();
        });
        if (to != ZoneType.Hand) {
            return;
        }
        if (mine != null) {
            for (Pending p : mine) {
                boolean fresh = !knows(p.viewerId, cardId, newTimestamp);
                if (p.zone == from && p.resolving == resolving
                        && learn(p.viewerId, toOwnerId, cardId, newTimestamp) && fresh) {
                    learnedFromRevealThenMove++;
                }
            }
        }
        for (int v : viewerIds) {
            boolean fresh = !knows(v, cardId, newTimestamp);
            if (v != toOwnerId && sawItLeave.test(v) && learn(v, toOwnerId, cardId, newTimestamp) && fresh) {
                learnedFromPublicMove++;
            }
        }
    }

    /** Queues a non-hand reveal delivered to {@code viewer}'s controller; see {@link #pend}. */
    public static void observePending(Player viewer, Card card, Object resolving) {
        if (viewer == null || card == null) {
            return;
        }
        try {
            if (card.getZone() == null || card.getZone().getZoneType() == ZoneType.Hand) {
                return;
            }
            pend(viewer.getId(), card.getOwner() == null ? -1 : card.getOwner().getId(),
                    card.getId(), card.getZone().getZoneType(), resolving);
        } catch (Throwable ignored) {
            // never let instrumentation break the game
        }
    }

    private static int forgotByUnseenDeparture;

    /**
     * A card left a hand and {@code viewerId} could not tell which: it no
     * longer knows whether the cards it knew there (all of them if the card
     * went somewhere unseen, as with Brainstorm's put-back; only those of the
     * same name if it was seen arriving face up) are still in that hand.
     * Schemas 13-14; the tracking handles are lost by the same rule (GameHistory).
     */
    static synchronized void unseenDeparture(int viewerId, java.util.Collection<Integer> handIds) {
        Map<Integer, Long> m = KNOWN.get(viewerId);
        if (m == null) {
            return;
        }
        for (Integer id : handIds) {
            if (m.remove(id) != null) {
                forgotByUnseenDeparture++;
            }
        }
    }

    private static int learnedFromPublicMove;
    private static int learnedFromRevealThenMove;

    /** Subscribes the zone-change listener (expiry and move-based learning), once per game. */
    public static synchronized void attach(Game game) {
        if (game == null || !ATTACHED.add(System.identityHashCode(game))) {
            return;
        }
        final Game g = game;
        game.subscribeToEvents(new Object() {
            @com.google.common.eventbus.Subscribe
            @SuppressWarnings("unused")
            public void onEvent(GameEvent ev) {
                try {
                    GameEventCardChangeZone e = ev.visit(new IGameEventVisitor.Base<GameEventCardChangeZone>() {
                        @Override
                        public GameEventCardChangeZone visit(GameEventCardChangeZone e) {
                            return e;
                        }
                    });
                    if (e == null || e.card() == null) {
                        return;
                    }
                    onZoneChange(g, e);
                } catch (Throwable ignored) {
                    // never let instrumentation break the game
                }
            }
        });
    }

    private static void onZoneChange(Game g, GameEventCardChangeZone e) {
        int cardId = e.card().getId();
        ZoneType from = e.from() == null ? null : e.from().zoneType();
        ZoneType to = e.to() == null ? null : e.to().zoneType();
        int toOwner = -1;
        long ts = Long.MIN_VALUE;
        if (to == ZoneType.Hand && e.to().player() != null) {
            toOwner = e.to().player().getId();
            for (Player p : g.getPlayers()) {
                if (p.getId() != toOwner) {
                    continue;
                }
                for (Card c : p.getCardsIn(ZoneType.Hand)) {
                    if (c.getId() == cardId) {
                        ts = c.getGameTimestamp();
                    }
                }
            }
            if (ts == Long.MIN_VALUE) {
                // Not found in the hand it supposedly entered: learn nothing.
                to = null;
            }
        }
        Object resolving = null;
        try {
            resolving = g.getStack().peekAbility();
        } catch (Throwable ignored) {
            // no resolution context; pending reveals cannot match
        }
        java.util.List<Player> players = new java.util.ArrayList<>();
        for (Player p : g.getPlayers()) {
            players.add(p);
        }
        int[] ids = new int[players.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = players.get(i).getId();
        }
        moved(cardId, from, to, toOwner, ts, resolving, ids, v -> {
            for (Player p : players) {
                if (p.getId() == v) {
                    return sawItLeave(e.card(), e.from(), p.getView());
                }
            }
            return false;
        });
        if (from == ZoneType.Hand && e.from() != null && e.from().player() != null) {
            int handOwner = e.from().player().getId();
            ZoneType dest = e.to() == null ? null : e.to().zoneType();
            for (Player p : players) {
                if (p.getId() == handOwner) {
                    continue;
                }
                // Seen arriving face up: only known cards of the same name could
                // have been it (schema 14: the viewer cannot tell which copy).
                String sameName = seesArrival(g, e, dest, p) ? e.card().getName() : null;
                java.util.List<Integer> hand = new java.util.ArrayList<>();
                for (Player o : players) {
                    if (o.getId() == handOwner) {
                        for (Card c : o.getCardsIn(ZoneType.Hand)) {
                            if (sameName == null || sameName.equals(c.getName())) {
                                hand.add(c.getId());
                            }
                        }
                    }
                }
                unseenDeparture(p.getId(), hand);
            }
        }
    }

    /** Whether {@code viewer} sees, face up, the card arrive where it went. */
    private static boolean seesArrival(Game g, GameEventCardChangeZone e, ZoneType dest, Player viewer) {
        if (dest == null) {
            return false;
        }
        if (dest == ZoneType.Hand && e.to().player() != null && e.to().player().getId() == viewer.getId()) {
            return true;
        }
        if (!publicOrigin(dest, false, false)) {
            return false;
        }
        try {
            Card now = null;
            if (dest == ZoneType.Stack) {
                for (Card c : g.getStackZone().getCards()) {
                    if (c.getId() == e.card().getId()) {
                        now = c;
                    }
                }
            } else if (e.to().player() != null) {
                for (Player p : g.getPlayers()) {
                    if (p.getId() == e.to().player().getId()) {
                        for (Card c : p.getCardsIn(dest)) {
                            if (c.getId() == e.card().getId()) {
                                now = c;
                            }
                        }
                    }
                }
            }
            if (now == null) {
                return false;
            }
            return !now.isFaceDown() || now.getView().mayPlayerLook(viewer.getView());
        } catch (Throwable ignored) {
            return false;
        }
    }

    /** Clears all knowledge. Call between games. Diagnostics persist across games. */
    public static synchronized void reset() {
        KNOWN.clear();
        ATTACHED.clear();
        PENDING.clear();
    }

    public static synchronized void printCoverage() {
        System.out.println();
        System.out.println("=== hand reveal knowledge ===");
        System.out.printf("hand cards learned (all sources): %d%n", learned);
        System.out.printf("  of which public-origin move:    %d%n", learnedFromPublicMove);
        System.out.printf("  of which reveal-then-move:      %d%n", learnedFromRevealThenMove);
        System.out.printf("dropped on leaving hand:          %d%n", droppedByMove);
        System.out.printf("dropped by unseen hand departure: %d%n", forgotByUnseenDeparture);
        System.out.printf("ignored (viewer owns card):       %d%n", ignoredOwn);
        System.out.printf("own-cards-only reveal callbacks:  %d%n", ownOnlyCallbacks);
        System.out.printf("suppressed (no Reveal$ wish):     %d%n", suppressed);
        System.out.printf("reveals outside a hand (queued):  %d%n", ignoredNotInHand);
        System.out.printf("known-card emissions:             %d%n", emittedKnown);
    }
}
