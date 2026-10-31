package magicnet.bridge;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEvent;
import forge.game.event.GameEventShuffle;
import forge.game.event.IGameEventVisitor;
import forge.game.player.Player;

/**
 * Per-viewer knowledge of library order.
 *
 * <p>Forge does not model this, and it is worth being precise about why. {@code
 * GameAction.scry} hands the cards to {@link
 * forge.game.player.PlayerController#arrangeForScry} and then moves them; nothing
 * records that the player now knows what is on top. {@code Card.mayLook} is a
 * timestamped map for <em>continuous</em> grants ("play with the top card of your
 * library revealed"), not for momentary looks. And {@code GameEventScry} carries
 * only {@code toTop}/{@code toBottom} counts, because the <em>public</em> fact is
 * "they scried 2", not which cards they saw.
 *
 * <p>So the information exists for exactly the duration of the controller
 * callback, and nowhere else. We capture it there.
 *
 * <h2>Why we store card identities and not depths</h2>
 *
 * <p>We record <em>which cards</em> a viewer knows the position of, never the
 * position itself. Depth is recomputed from the live library at extraction time.
 * That is self-maintaining under every subsequent mutation:
 *
 * <ul>
 *   <li>scry X to top, then draw X &rarr; X leaves the library and drops out of
 *       the emitted state on its own</li>
 *   <li>scry [X, Y] to top, then draw X &rarr; Y's live index is now 0, correct
 *       with no bookkeeping</li>
 *   <li>an effect puts Z on top &rarr; X's live index becomes 1, which is what
 *       the player actually knows</li>
 * </ul>
 *
 * <p>Storing depths would require re-indexing the whole map on every zone change,
 * and would be wrong the moment we missed one.
 *
 * <h2>Scry grants exact positions, not a region</h2>
 *
 * <p>The rules let you order both piles ("put any number of them on the bottom of
 * your library in any order and the rest on top in any order"), and
 * {@code arrangeForScry} returns an ordered {@code (top, bottom)} pair. So a scry
 * produces EXACT knowledge for every card seen, in both directions. The
 * region/unordered case ({@link Constraint#GROUP_TOP}, {@link
 * Constraint#GROUP_BOTTOM}) is for "put the rest on the bottom in a random
 * order", which is a different mechanic and is not yet wired up.
 *
 * <h2>Invalidation</h2>
 *
 * <p>Shuffling destroys position knowledge, so {@link GameEventShuffle} clears
 * every viewer's knowledge of that library. Shuffles are public, so clearing for
 * all viewers at once is correct.
 *
 * <p><b>Known limitation.</b> If a card is placed into a library without the
 * viewer observing it, the viewer's known cards shift by one and our live-index
 * recomputation reports the true new depth rather than the stale one the viewer
 * believes. This understates uncertainty by one position. It does not leak a card
 * identity, and the move is public in the overwhelming majority of cases.
 */
public final class LibraryKnowledge {

    /** How precisely a viewer knows where a card sits. */
    public enum Constraint {
        /** Exact position; depth is the card's live index. */
        EXACT,
        /** Known to be within the top k, order unknown. */
        GROUP_TOP,
        /** Known to be within the bottom k, order unknown. */
        GROUP_BOTTOM
    }

    /** viewer id -> (card id -> constraint). */
    private static final Map<Integer, Map<Integer, Constraint>> KNOWN = new HashMap<>();

    /** Libraries we have attached a shuffle listener to, by game identity. */
    private static final Set<Integer> ATTACHED = new HashSet<>();

    /** Diagnostics: shuffles observed, and position facts dropped by them. */
    private static int shufflesSeen;
    private static int factsCleared;
    private static int factsLearned;

    static final boolean DEBUG = "1".equals(System.getProperty("magicnet.libknow.debug"));

    private LibraryKnowledge() {
    }

    /**
     * Records that {@code viewer} saw {@code card} at a known library position.
     * Safe to call repeatedly; the most precise constraint wins.
     */
    public static synchronized void learn(Player viewer, Card card, Constraint how) {
        if (viewer == null || card == null) {
            return;
        }
        Map<Integer, Constraint> m =
                KNOWN.computeIfAbsent(viewer.getId(), k -> new HashMap<>());
        Constraint prev = m.get(card.getId());
        if (prev == null || how == Constraint.EXACT) {
            if (prev == null) {
                factsLearned++;
            }
            m.put(card.getId(), how);
        }
    }

    /** Convenience for the scry/surveil case, where every card seen is exact. */
    public static void learnExact(Player viewer, Iterable<Card> cards) {
        if (cards == null) {
            return;
        }
        for (Card c : cards) {
            learn(viewer, c, Constraint.EXACT);
        }
    }


    /**
     * Records what a player learned from momentarily seeing a pile of cards.
     *
     * <p>This is the compensation for a callback that <em>does not exist</em>.
     * When an effect says "put the rest on the bottom of your library in a random
     * order", {@code DigEffect} and {@code ChangeZoneEffect} both take the branch
     * {@code if (sa.hasParam("RestRandomOrder")) CardLists.shuffle(rest);} and
     * never consult the controller -- correctly, because the player does not
     * choose that order. The only signal that the player <em>saw</em> the pile is
     * {@link forge.game.player.PlayerController#tempShowCards}.
     *
     * <p>We deliberately record a group only when the seen cards now occupy a
     * contiguous block anchored at the top or the bottom of the library. That is a
     * <em>verification</em>, not an inference: the emitted bounds are derived from
     * the group size and the library size, both of which the player knows
     * (they watched k cards go to the bottom and they can count their own
     * library). If the seen cards do not form such a block, we learn nothing
     * rather than guess -- understating knowledge is safe, overstating it is a
     * leak.
     *
     * <p>Cards already known EXACTly are not downgraded; {@link #learn} keeps the
     * more precise constraint.
     *
     * <p><b>This must not run while the effect is still resolving.</b>
     * {@code endTempShowCards} fires <em>before</em> the remaining cards are moved
     * -- measured: a Commune with Spirits showed 4 cards still sitting at library
     * indices 0-3 of 46, and only afterwards were they shuffled to the bottom.
     * Classifying there recorded GROUP_TOP for cards that ended up on the bottom,
     * which is not merely imprecise but false. Hence {@link #observeShown} queues
     * the group and {@link #resolvePending} classifies it at the next extraction,
     * by which point the zone changes have landed.
     */
    private static synchronized void classifyShownGroup(Player viewer, Iterable<Card> shown) {
        if (viewer == null || shown == null) {
            return;
        }
        Map<Player, java.util.List<Card>> byOwner = new HashMap<>();
        for (Card c : shown) {
            if (c == null) {
                continue;
            }
            try {
                if (c.getZone() == null
                        || c.getZone().getZoneType() != forge.game.zone.ZoneType.Library) {
                    continue;
                }
                byOwner.computeIfAbsent(c.getOwner(), k -> new java.util.ArrayList<>()).add(c);
            } catch (Throwable ignored) {
                // skip this card
            }
        }

        for (Map.Entry<Player, java.util.List<Card>> e : byOwner.entrySet()) {
            Player owner = e.getKey();
            if (owner == null) {
                continue;
            }
            java.util.List<Card> lib = new java.util.ArrayList<>();
            try {
                for (Card c : owner.getCardsIn(forge.game.zone.ZoneType.Library)) {
                    lib.add(c);
                }
            } catch (Throwable ignored) {
                continue;
            }
            int n = lib.size();
            Set<Integer> idx = new HashSet<>();
            for (Card c : e.getValue()) {
                int i = lib.indexOf(c);
                if (i >= 0) {
                    idx.add(i);
                }
            }
            int k = idx.size();
            if (k == 0 || k > n) {
                continue;
            }

            boolean topBlock = true;
            for (int i = 0; i < k; i++) {
                if (!idx.contains(i)) {
                    topBlock = false;
                    break;
                }
            }
            boolean bottomBlock = true;
            for (int i = n - k; i < n; i++) {
                if (!idx.contains(i)) {
                    bottomBlock = false;
                    break;
                }
            }

            // k == n makes both true and the distinction meaningless; prefer TOP.
            Constraint how = topBlock ? Constraint.GROUP_TOP
                    : bottomBlock ? Constraint.GROUP_BOTTOM : null;
            if (how == null) {
                continue;
            }
            // A single card in an anchored block of one has an exact position.
            if (k == 1) {
                how = Constraint.EXACT;
            }
            if (DEBUG) {
                StringBuilder sb = new StringBuilder();
                for (Card c : e.getValue()) {
                    sb.append(c.getName()).append(" ");
                }
                System.err.printf("[libknow] viewer=%s owner=%s k=%d n=%d idx=%s -> %s : %s%n",
                        viewer, owner, k, n, idx, how, sb);
            }
            for (Card c : e.getValue()) {
                learn(viewer, c, how);
            }
        }
    }


    /** A pile a player was shown, awaiting classification once the effect resolves. */
    private static final class Pending {
        final Player viewer;
        final java.util.List<Card> cards;

        Pending(Player viewer, java.util.List<Card> cards) {
            this.viewer = viewer;
            this.cards = cards;
        }
    }

    private static final java.util.List<Pending> PENDING = new java.util.ArrayList<>();

    /**
     * Queues a pile the player was momentarily shown. Classification is deferred;
     * see {@link #classifyShownGroup} for why it cannot happen at reveal time.
     */
    public static synchronized void observeShown(Player viewer, Iterable<Card> cards) {
        if (viewer == null || cards == null) {
            return;
        }
        java.util.List<Card> snapshot = new java.util.ArrayList<>();
        for (Card c : cards) {
            if (c != null) {
                snapshot.add(c);
            }
        }
        if (!snapshot.isEmpty()) {
            PENDING.add(new Pending(viewer, snapshot));
        }
    }

    /**
     * Classifies every queued pile and clears the queue. Called once per state
     * extraction, which is the first moment after resolution at which positions
     * are settled.
     *
     * <p>Groups are dropped whether or not they classified. Retaining them would
     * let an unrelated later state coincidentally satisfy the block check.
     */
    public static synchronized void resolvePending() {
        if (PENDING.isEmpty()) {
            return;
        }
        java.util.List<Pending> work = new java.util.ArrayList<>(PENDING);
        PENDING.clear();
        for (Pending p : work) {
            try {
                classifyShownGroup(p.viewer, p.cards);
            } catch (Throwable ignored) {
                // never let instrumentation break the game
            }
        }
    }

    /** What {@code viewer} knows about {@code card}, or null for nothing. */
    public static synchronized Constraint constraintFor(Player viewer, Card card) {
        if (viewer == null || card == null) {
            return null;
        }
        Map<Integer, Constraint> m = KNOWN.get(viewer.getId());
        return m == null ? null : m.get(card.getId());
    }

    /**
     * Drops all position knowledge for {@code owner}'s library, for every viewer.
     * Called on shuffle.
     */
    public static synchronized void onShuffle(Player owner) {
        if (owner == null) {
            return;
        }
        shufflesSeen++;
        Set<Integer> libraryCards = new HashSet<>();
        try {
            for (Card c : owner.getCardsIn(forge.game.zone.ZoneType.Library)) {
                libraryCards.add(c.getId());
            }
        } catch (Throwable ignored) {
            // If we cannot enumerate, clear everything rather than keep stale data.
            factsCleared += size();
            KNOWN.clear();
            return;
        }
        for (Map<Integer, Constraint> m : KNOWN.values()) {
            int before = m.size();
            m.keySet().removeAll(libraryCards);
            factsCleared += before - m.size();
        }
    }

    /** Drops what {@code viewer} knows about {@code cards} (e.g. surveilled into the graveyard). */
    public static synchronized void forget(Player viewer, Iterable<Card> cards) {
        if (viewer == null || cards == null) {
            return;
        }
        Map<Integer, Constraint> m = KNOWN.get(viewer.getId());
        if (m == null) {
            return;
        }
        for (Card c : cards) {
            m.remove(c.getId());
        }
    }

    // ---- In-progress scry/surveil arrangement -------------------------------
    //
    // A scry/surveil is decomposed into sequential pick-one decisions, but the
    // cards only move once the whole arrangement is returned. Without this, a
    // later decision could not see what earlier ones chose (combat gets that
    // for free because each declared attacker mutates the Combat). Per viewer,
    // and only ever attached to cards that viewer already knows.

    /** Pile a card has been assigned to so far, and its place in that pile (-1 = not yet ordered). */
    public static final class Arranging {
        public final String pile;
        public final int rank;

        Arranging(String pile, int rank) {
            this.pile = pile;
            this.rank = rank;
        }
    }

    private static final Map<Integer, Map<Integer, Arranging>> ARRANGING = new HashMap<>();

    public static synchronized void arranging(Player viewer, Card card, String pile, int rank) {
        if (viewer == null || card == null) {
            return;
        }
        ARRANGING.computeIfAbsent(viewer.getId(), k -> new HashMap<>())
                .put(card.getId(), new Arranging(pile, rank));
    }

    public static synchronized Arranging arrangingFor(Player viewer, Card card) {
        if (viewer == null || card == null) {
            return null;
        }
        Map<Integer, Arranging> m = ARRANGING.get(viewer.getId());
        return m == null ? null : m.get(card.getId());
    }

    public static synchronized void clearArranging(Player viewer) {
        if (viewer != null) {
            ARRANGING.remove(viewer.getId());
        }
    }

    /** Clears all knowledge. Call between games. Diagnostics persist across games. */
    public static synchronized void reset() {
        KNOWN.clear();
        ATTACHED.clear();
        PENDING.clear();
        ARRANGING.clear();
    }

    /**
     * Prints the shuffle-invalidation accounting.
     *
     * <p>This is the assertion that matters for correctness: a library shuffle
     * destroys position knowledge, and if we failed to clear it we would emit
     * confident exact depths for cards whose position the player can no longer
     * know -- a hidden-information leak that would look like a very lucky agent.
     */
    public static synchronized void printCoverage() {
        System.out.println();
        System.out.println("=== library position knowledge ===");
        System.out.printf("position facts learned:   %d%n", factsLearned);
        System.out.printf("shuffles observed:        %d%n", shufflesSeen);
        System.out.printf("facts cleared by shuffle: %d%n", factsCleared);
        if (shufflesSeen == 0) {
            System.out.println("WARN: no shuffles observed; invalidation path untested");
        } else if (factsLearned > 0 && factsCleared == 0) {
            System.out.println("WARN: shuffles fired but cleared nothing -- "
                    + "check that the listener resolves the owning player");
        } else {
            System.out.println("ok: shuffle invalidation is live");
        }
    }

    /** Number of (viewer, card) position facts currently held; for diagnostics. */
    public static synchronized int size() {
        int n = 0;
        for (Map<Integer, Constraint> m : KNOWN.values()) {
            n += m.size();
        }
        return n;
    }

    /**
     * Subscribes a shuffle listener to {@code game}, once per game.
     *
     * <p>Shuffle is not part of {@link GameHistory}'s KEEP set and is viewer
     * independent, so it gets its own listener rather than being threaded through
     * the per-viewer histories.
     */
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
                    Boolean isShuffle = ev.visit(new IGameEventVisitor.Base<Boolean>() {
                        @Override
                        public Boolean visit(GameEventShuffle e) {
                            return Boolean.TRUE;
                        }
                    });
                    if (!Boolean.TRUE.equals(isShuffle)) {
                        return;
                    }
                    forge.game.player.PlayerView pv = ((GameEventShuffle) ev).player();
                    for (Player p : g.getPlayers()) {
                        if (pv != null && pv.getId() == p.getId()) {
                            onShuffle(p);
                        }
                    }
                } catch (Throwable ignored) {
                    // never let instrumentation break the game
                }
            }
        });
    }

    /** Unmodifiable view of what a viewer knows; for tests. */
    public static synchronized Map<Integer, Constraint> snapshot(Player viewer) {
        Map<Integer, Constraint> m = KNOWN.get(viewer.getId());
        return m == null ? Collections.emptyMap() : new HashMap<>(m);
    }
}
