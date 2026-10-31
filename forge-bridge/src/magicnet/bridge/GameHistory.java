package magicnet.bridge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.google.common.eventbus.Subscribe;

import forge.game.Game;
import forge.game.GameEntity;
import forge.game.event.GameEvent;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;

/**
 * Subscribes to Forge's structured event bus to build the public game history.
 *
 * This replaces scraping history out of controller callbacks, which was both an
 * undercount and conceptually wrong. Controller callbacks are *choices*; the
 * history we want is *what happened*, and most of what happens is not a choice
 * anyone made (a creature died, a spell resolved, a card was drawn). Forge fires
 * all of it on a Guava EventBus that `Game.subscribeToEvents` exposes, so the
 * engine's own event stream is the authoritative source.
 *
 * Note this deliberately does not use `Game.getGameLog()`, which carries the same
 * events but rendered as localized prose. Strings would break under locale change
 * and would have to be re-parsed into structure we already have here.
 */
public final class GameHistory {

    /**
     * Events worth keeping as history. The bus fires a lot of redraw/bookkeeping
     * noise (priority passes, combat updates, phase changes) that is either already
     * in the board state or is pure UI signalling.
     *
     * Kept: zone changes, resolution, damage, destruction, sacrifice, combat
     * declarations, tokens, counters, scry/surveil, mulligans, land drops.
     *
     * Two deliberate exclusions beyond the noise:
     *   CardTapped (465/game) is redundant with the `tapped` property already on
     *   every battlefield token, and is overwhelmingly mana payment.
     *   SpellRemovedFromStack (182/game) fires alongside SpellResolved for the
     *   same event, so keeping both double-counts every spell.
     */
    private static final java.util.Set<String> KEEP = new java.util.HashSet<>(java.util.Arrays.asList(
            "GameEventCardChangeZone",
            "GameEventSpellAbilityCast",
            "GameEventSpellResolved",
            "GameEventCardDestroyed",
            "GameEventCardSacrificed",
            "GameEventCardDamaged",
            "GameEventPlayerDamaged",
            "GameEventCardCounters",
            "GameEventPlayerCounters",
            "GameEventAttackersDeclared",
            "GameEventBlockersDeclared",
            "GameEventTokenCreated",
            "GameEventLandPlayed",
            "GameEventCardAttachment",
            "GameEventCardPhased",
            "GameEventCardModeChosen",
            "GameEventScry",
            "GameEventSurveil",
            "GameEventMulligan",
            "GameEventCardRegenerated",
            "GameEventCardForetold",
            "GameEventFlipCoin",
            "GameEventRollDie",
            "GameEventPlayerLivesChanged",
            "GameEventPlayerPoisoned"));

    /** Every event type seen, for coverage reporting. */
    private static final Map<String, Integer> SEEN =
            Collections.synchronizedMap(new TreeMap<>());

    private static final Map<String, Integer> KEPT =
            Collections.synchronizedMap(new TreeMap<>());

    /**
     * One proposed cast or activation, from announcement until Forge either
     * puts it on the stack or refuses it.
     *
     * <p>Every history event fired while an attempt is open is tagged with its
     * id, and the attempt is closed by an {@code AttemptEnd} token carrying the
     * outcome and the proposal. History stays append-only (the delta encoding
     * depends on it), so nothing is retracted: a consumer drops the events of a
     * {@code rolled_back} attempt itself. The transcript keeps them, because a
     * refused proposal is a candidate negative example for the action decoder.
     *
     * <p>Outcomes: {@code committed}; {@code rolled_back} (refused and rewound
     * by PlaySpellAbility per CR 728, so its events never happened);
     * {@code failed_unrolled} (refused on Forge's AI path, which does not rewind,
     * so its events DID happen); {@code unknown}.
     */
    public static final class Attempt {
        final int id;
        final SpellAbility sa;
        final List<GameEntity> targets = new ArrayList<>();
        final Map<String, Integer> announced = new LinkedHashMap<>();
        /** Furthest casting stage a controller callback was seen for. */
        String reached = "proposed";

        Attempt(int id, SpellAbility sa) {
            this.id = id;
            this.sa = sa;
        }

        void reach(String stage, SpellAbility forSa) {
            if (owns(forSa)) {
                reached = stage;
            }
        }

        /** The proposal itself or one of its sub-abilities. */
        boolean owns(SpellAbility s) {
            try {
                return s != null && (s == sa || s.getRootAbility() == sa);
            } catch (Throwable ignored) {
                return false;
            }
        }

        void announce(String what, Integer value) {
            if (what != null && value != null) {
                announced.put(what, value);
            }
        }

        /**
         * Snapshots bound targets. Only replaces on a non-empty read, because a
         * rollback may clear them before the attempt is closed.
         */
        void captureTargets() {
            if (sa == null) {
                return;
            }
            List<GameEntity> seen = new ArrayList<>();
            try {
                for (TargetChoices tc : sa.getAllTargetChoices()) {
                    if (tc == null) {
                        continue;
                    }
                    for (GameEntity e : tc.getTargetEntities()) {
                        seen.add(e);
                    }
                    for (SpellAbility s : tc.getTargetSpells()) {
                        if (s != null && s.getHostCard() != null) {
                            seen.add(s.getHostCard());
                        }
                    }
                }
            } catch (Throwable ignored) {
                return;
            }
            if (!seen.isEmpty()) {
                targets.clear();
                targets.addAll(seen);
            }
        }
    }

    /** Events are dispatched synchronously on the single game thread. */
    private static Attempt open;
    private static int nextAttempt;

    /** Opens an attempt, or returns null if one is already open (nested cast). */
    public static Attempt beginAttempt(SpellAbility sa) {
        if (open != null) {
            return null;
        }
        open = new Attempt(++nextAttempt, sa);
        open.captureTargets();
        return open;
    }

    public static Attempt currentAttempt() {
        return open;
    }

    public static void closeAttempt(Attempt a) {
        if (open == a) {
            open = null;
        }
    }

    /** Attempt ids are per game. */
    public static void resetAttempts() {
        open = null;
        nextAttempt = 0;
    }

    private final List<StateExtractor.Tok> history =
            Collections.synchronizedList(new ArrayList<>());

    private final Game game;

    private final EventTokenizer tokenizer;

    /**
     * One history per viewer. Both instances subscribe to the same bus and see
     * the same events, but each redacts for its own player, so the two logs are
     * deliberately not identical. The duplication is cheap and makes a
     * cross-agent information leak structurally impossible rather than merely
     * guarded against -- which matters because in self-play both agents live in
     * the same process.
     */
    public GameHistory(Game game, forge.game.player.Player viewer) {
        this.game = game;
        this.tokenizer = new EventTokenizer(viewer == null ? null : viewer.getView());
    }

    /**
     * Guava dispatches on the declared parameter type, so subscribing to the
     * GameEvent interface catches every event rather than needing 58 methods.
     */
    @Subscribe
    public void onEvent(GameEvent ev) {
        try {
            String name = ev.getClass().getSimpleName();
            SEEN.merge(name, 1, Integer::sum);
            if (!KEEP.contains(name)) {
                return;
            }
            KEPT.merge(name, 1, Integer::sum);

            int turn = 0;
            try {
                turn = game.getPhaseHandler().getTurn();
            } catch (Throwable ignored) {
                // pre-game events have no turn
            }

            // Detail is read through Forge's own typed visitor and redacted for
            // this history's viewer at the moment the event fires.
            StateExtractor.Tok tok = tokenizer.tokenize(ev, turn, shortName(name));
            Attempt a = open;
            if (a != null) {
                tok.p("attempt", a.id);
            }
            history.add(tok);
        } catch (Throwable ignored) {
            // instrumentation must never fail a game
        }
    }

    /** Closes an attempt in this viewer's history. */
    public void appendAttemptEnd(Attempt a, String outcome) {
        try {
            history.add(tokenizer.attemptEnd(currentTurn(), a.id, outcome, a.reached,
                    a.sa, a.targets, a.announced));
        } catch (Throwable ignored) {
            // instrumentation must never fail a game
        }
    }

    /** Appends one revealed card, redacted for this viewer. See EventTokenizer#reveal. */
    public void appendReveal(forge.game.card.Card c, forge.game.zone.ZoneType zone,
            forge.game.player.Player owner, forge.game.player.Player shownTo, int group, int size) {
        try {
            StateExtractor.Tok tok = tokenizer.reveal(currentTurn(), c, zone, owner, shownTo, group, size);
            Attempt a = open;
            if (a != null) {
                tok.p("attempt", a.id);
            }
            history.add(tok);
        } catch (Throwable ignored) {
            // instrumentation must never fail a game
        }
    }

    private int currentTurn() {
        try {
            return game.getPhaseHandler().getTurn();
        } catch (Throwable ignored) {
            return 0;
        }
    }

    public int redactions() {
        return tokenizer.redactions();
    }

    private static String shortName(String cls) {
        return cls.startsWith("GameEvent") ? cls.substring("GameEvent".length()) : cls;
    }

    public List<StateExtractor.Tok> tokens() {
        return history;
    }

    public int size() {
        return history.size();
    }

    public static void printCoverage() {
        if (SEEN.isEmpty()) {
            return;
        }
        System.out.println("\n=== game event bus coverage ===");
        int totalSeen = 0;
        int totalKept = 0;
        synchronized (SEEN) {
            for (Map.Entry<String, Integer> e : SEEN.entrySet()) {
                totalSeen += e.getValue();
                int kept = KEPT.getOrDefault(e.getKey(), 0);
                totalKept += kept;
                System.out.printf("  %-38s seen=%-8d %s%n",
                        shortName(e.getKey()), e.getValue(),
                        kept > 0 ? "KEPT" : "dropped");
            }
        }
        System.out.printf("  %-38s seen=%-8d kept=%d (%.1f%%)%n",
                "TOTAL", totalSeen, totalKept,
                totalSeen == 0 ? 0.0 : 100.0 * totalKept / totalSeen);
    }
}
