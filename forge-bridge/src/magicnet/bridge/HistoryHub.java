package magicnet.bridge;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.google.common.eventbus.Subscribe;

import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameStage;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEvent;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.spellability.TargetChoices;
import forge.game.zone.ZoneType;
import forge.game.zone.ZoneView;

/**
 * Builds the game history once per game, from Forge's event bus, and hands each
 * event to every viewer's {@link GameHistory} for redaction.
 *
 * <p>Forge's events say what happened but never why. The cause is recovered at
 * the moment the event fires, from engine state, in this precedence:
 * <ol>
 *   <li><b>Resolution scope.</b> The stack ability resolving when the first
 *       event of a resolution fires (snapshotted: after {@code remove(si)}
 *       {@code peekAbility} returns the next item, and a cast-from-effect pushes
 *       a new one mid-resolution). Closed by the
 *       {@code SpellRemovedFromStack} of that same ability, so the spell's own
 *       move to the graveyard is inside it. A static trigger
 *       ({@code playTrigger}, no stack) opens the same kind of scope.</li>
 *   <li><b>Casting attempt / cost payment.</b> Events while a cast or
 *       activation is being proposed are held back and emitted after the
 *       {@code cast}/{@code activate} event they belong to, pointing at it.</li>
 *   <li><b>Replacement.</b> {@code ReplacementHandler.hasRun} (private, read by
 *       reflection) is non-empty while a replacement executes; its host becomes a
 *       {@code replaced_by} edge in addition to the cause.</li>
 *   <li><b>Triggers.</b> A trigger's triggering objects are matched per mode to
 *       the history events they came from.</li>
 *   <li><b>Rule category</b> ({@code cause_kind}) otherwise: pregame, combat,
 *       and -- by a stack walk, only for events nothing cheaper classified --
 *       state-based or turn-based.</li>
 * </ol>
 * Causes are records here; a viewer's token gets an edge to the cause's index
 * in that viewer's history ({@code e<N>}), never anything about the cause's
 * participants -- triggering objects include the opponent's hidden draw.
 */
public final class HistoryHub {

    // ---- registry ----------------------------------------------------------

    private static final Map<Game, HistoryHub> HUBS = new IdentityHashMap<>();

    /**
     * The hub for {@code game}, subscribing it on first use. The knowledge
     * stores subscribe first so that, for every event, they have already
     * learned or forgotten what the event teaches by the time it is tokenized.
     */
    public static synchronized HistoryHub attach(Game game) {
        HistoryHub h = HUBS.get(game);
        if (h == null) {
            LibraryKnowledge.attach(game);
            RevealKnowledge.attach(game);
            h = new HistoryHub(game);
            HUBS.put(game, h);
            game.subscribeToEvents(h);
        }
        return h;
    }

    public static synchronized HistoryHub of(Game game) {
        return game == null ? null : HUBS.get(game);
    }

    /** Between games: folds per-game tracking diagnostics into the totals and forgets every hub. */
    public static synchronized void reset() {
        for (HistoryHub h : HUBS.values()) {
            h.closed = true;
            for (GameHistory v : h.viewers.values()) {
                int[] s = v.trackingStats();
                for (int i = 0; i < s.length; i++) {
                    TRACKING[i] += s[i];
                }
            }
        }
        HUBS.clear();
    }

    final Game game;
    private final Map<Integer, GameHistory> viewers = new LinkedHashMap<>();
    private boolean closed;

    HistoryHub(Game game) {
        this.game = game;
    }

    /** This player's history, created on first use. Register every player before the game starts. */
    public synchronized GameHistory viewer(Player p) {
        int id = p == null ? -1 : p.getId();
        return viewers.computeIfAbsent(id, k -> new GameHistory(game, p));
    }

    Collection<GameHistory> viewers() {
        return viewers.values();
    }

    // ---- records -----------------------------------------------------------

    /** One participant, already resolved to each viewer's reference. */
    static final class Part {
        final String role;
        /** A card participant: an unresolved ref is a redaction, not an absence. */
        final boolean card;
        private final Map<Integer, String> refs = new HashMap<>(4);
        private String all;

        Part(String role, boolean card) {
            this.role = role;
            this.card = card;
        }

        String ref(int vid) {
            return all != null ? all : refs.get(vid);
        }
    }

    /** One event, viewer-independent. Rendered per viewer by {@link GameHistory#append}. */
    static final class Rec {
        final int seq;
        String verb;
        final Map<String, Object> mods = new LinkedHashMap<>();
        final List<Part> parts = new ArrayList<>();
        final List<Rec> causes = new ArrayList<>();
        String causeKind;
        /** Caused by the open attempt's cast, which has not been emitted yet. */
        boolean causeFromAttempt;
        int turn;
        String phase;
        long step;
        Integer attempt;
        ZoneType fromZone;
        ZoneType toZone;
        int fromOwner = -1;
        int toOwner = -1;
        /** A reorder within one hidden zone: only shown to viewers who can tell which card. */
        boolean privateMove;
        int count = 1;
        boolean alwaysCount;
        boolean dispatched;
        Rec mergedInto;

        // matching
        final List<Integer> cards = new ArrayList<>(2);
        int player = -1;
        int source = -1;
        int targetCard = -1;
        int targetPlayer = -1;
        int replacedBy = -1;
        String kind;
        // draw detection
        Player drawer;
        int drawnBefore = -1;
        boolean pregame;

        Rec(int seq) {
            this.seq = seq;
        }

        Rec canonical() {
            Rec r = this;
            while (r.mergedInto != null) {
                r = r.mergedInto;
            }
            return r;
        }
    }

    private int nextSeq;

    private Rec rec(String verb) {
        Rec r = new Rec(nextSeq++);
        r.verb = verb;
        try {
            r.turn = game == null ? 0 : game.getPhaseHandler().getTurn();
            r.phase = Schema.phaseName(game == null ? null : game.getPhaseHandler().getPhase());
        } catch (Throwable ignored) {
            r.phase = Schema.NO_PHASE;
        }
        GameHistory.Attempt a = GameHistory.currentAttempt();
        if (a != null) {
            r.attempt = a.id;
        }
        return r;
    }

    private Part playerPart(String role, int pid) {
        Part p = new Part(role, false);
        if (pid >= 0) {
            p.all = StateExtractor.playerRef(pid);
        }
        return p;
    }

    private Part playerPart(String role, PlayerView pv) {
        return playerPart(role, pv == null ? -1 : pv.getId());
    }

    private Part cardPart(String role, Card c) {
        Part p = new Part(role, true);
        if (c != null && game != null) {
            // Forge replaces the Card object on every zone change; a held
            // reference (an ability's host) may be the stale pre-move one.
            try {
                Card live = game.getCardState(c, null);
                if (live != null) {
                    c = live;
                }
                // An "X's Effect" card stands for the card that created it.
                if (c.isImmutable() && !c.isEmblem() && c.getEffectSource() != null) {
                    Card src = game.getCardState(c.getEffectSource(), null);
                    c = src != null ? src : c.getEffectSource();
                }
            } catch (Throwable ignored) {
                // use as given
            }
        }
        if (c != null) {
            for (GameHistory v : viewers.values()) {
                p.refs.put(v.vid, v.objRef(c));
            }
        }
        return p;
    }

    private Part abilityPart(String role, int siId, SpellAbility sa, Card host) {
        Part p = new Part(role, false);
        String text = abilityText.get(siId);
        for (GameHistory v : viewers.values()) {
            p.refs.put(v.vid, v.abilityRef(siId, sa, host, text));
        }
        return p;
    }

    /** Ability text by stack-instance id, from the cast event's view (the rendering history uses). */
    private final Map<Integer, String> abilityText = new HashMap<>();

    /** A game entity as a participant: a card (handle) or a player. */
    private Part entityPart(String role, Object e) {
        if (e instanceof Player) {
            return playerPart(role, ((Player) e).getId());
        }
        if (e instanceof PlayerView) {
            return playerPart(role, (PlayerView) e);
        }
        if (e instanceof Card) {
            return cardPart(role, (Card) e);
        }
        if (e instanceof CardView) {
            return cardPart(role, cardOf((CardView) e));
        }
        return null;
    }

    private static void add(Rec r, Part p) {
        if (p != null) {
            r.parts.add(p);
        }
    }

    // ---- scopes --------------------------------------------------------------

    /** A resolution or static-trigger scope: the event every consequence points at. */
    static final class Scope {
        final int serial;
        final int saId;
        final Rec rec;
        final boolean staticTrigger;
        final long tsFrom;
        long tsTo = Long.MAX_VALUE;
        SpellAbility sa;

        Scope(int serial, int saId, Rec rec, boolean staticTrigger, long tsFrom) {
            this.serial = serial;
            this.saId = saId;
            this.rec = rec;
            this.staticTrigger = staticTrigger;
            this.tsFrom = tsFrom;
        }
    }

    private final Deque<Scope> scopes = new ArrayDeque<>();
    private Scope resScope;
    private Scope lastClosedRes;
    private int scopeSerial;
    /** Every scope, in order of opening, for timestamp lookup. */
    private final List<Scope> allScopes = new ArrayList<>();

    private long timestamp() {
        try {
            return game == null ? 0 : game.getTimestamp();
        } catch (Throwable ignored) {
            return 0;
        }
    }

    /**
     * Continuous effects by timestamp -> the resolve (or static trigger) event
     * that created them. Built by scanning the battlefield's non-static P/T,
     * keyword and trait tables when a scope opens (anything already there is
     * not this scope's) and when it closes (anything new is).
     */
    private final Map<Long, Rec> effectStamps = new HashMap<>();
    private static final Rec NO_SCOPE = new Rec(-1);

    synchronized Rec scopeForTimestamp(long ts) {
        Rec r = effectStamps.get(ts);
        return r == null || r == NO_SCOPE || !r.dispatched ? null : r;
    }

    private void attributeEffects(Rec owner) {
        if (game == null) {
            return;
        }
        try {
            for (Player p : game.getPlayers()) {
                for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
                    stamp(c.getPTBoostTable(), owner);
                    stamp(c.getSetPTTable(), owner);
                    stamp(c.getChangedCardKeywords(), owner);
                    stamp(c.getChangedCardTraits(), owner);
                }
            }
        } catch (Throwable ignored) {
            // partial
        }
    }

    private void stamp(com.google.common.collect.Table<Long, Long, ?> table, Rec owner) {
        if (table == null) {
            return;
        }
        for (com.google.common.collect.Table.Cell<Long, Long, ?> cell : table.cellSet()) {
            Long ts = cell.getRowKey();
            if (ts != null && cell.getColumnKey() != null && cell.getColumnKey() == 0L) {
                effectStamps.putIfAbsent(ts, owner);
            }
        }
    }

    /** Opens the resolution scope lazily, at the first sign of a resolution. */
    private void syncResolution() {
        if (game == null) {
            return;
        }
        boolean resolving;
        try {
            resolving = game.getStack().isResolving();
        } catch (Throwable ignored) {
            return;
        }
        if (!resolving) {
            if (resScope != null) {
                closeResolution();
            }
            return;
        }
        if (resScope != null) {
            return;
        }
        SpellAbility sa;
        try {
            // Still "resolving" after the last item left the stack (a copy that
            // ceased to exist): nothing to open.
            sa = game.getStack().isEmpty() ? null : game.getStack().peekAbility();
        } catch (Throwable ignored) {
            sa = null;
        }
        if (sa != null) {
            openResolution(sa, sa.getId(), false);
        }
    }

    private void openResolution(SpellAbility sa, int saId, boolean fizzled) {
        flushPending();
        Rec r = rec(fizzled ? "does_not_resolve" : "resolve");
        if (fizzled) {
            r.mods.put("reason", "fizzled");
        }
        stackParts(r, sa, saId, false);
        Rec cast = castRecs.get(saId);
        if (cast != null) {
            r.causes.add(cast);
            r.causeKind = "pointer";
        } else {
            r.causeKind = "unknown";
        }
        Scope s = new Scope(++scopeSerial, saId, r, false, timestamp());
        s.sa = sa;
        scopes.push(s);
        allScopes.add(s);
        resScope = s;
        r.step = stepFor(r);
        lastEmitted = r;
        index(r);
        deliver(r);
    }

    private void closeResolution() {
        Scope s = resScope;
        if (s == null) {
            return;
        }
        flushPending();
        attributeEffects(s.rec);
        s.tsTo = timestamp();
        scopes.remove(s);
        resScope = null;
        lastClosedRes = s;
    }

    private Scope topScope() {
        return scopes.peek();
    }

    // ---- static triggers (no stack) ------------------------------------------

    /** Opens a static trigger's scope. Its event is emitted with its first consequence, or at the end if used. */
    public synchronized void beginStaticTrigger(Card host, SpellAbility wa) {
        if (closed) {
            return;
        }
        try {
            syncResolution();
            flushPending();
            Rec r = rec("trigger");
            if (wa.getActivatingPlayer() != null) {
                add(r, playerPart(Schema.ACTOR, wa.getActivatingPlayer().getId()));
            }
            add(r, cardPart(Schema.SOURCE_IS, host));
            r.mods.put("ability", EventTokenizer.abilityText(wa, game));
            matchTrigger(r, wa);
            attributeEffects(NO_SCOPE);
            Scope s = new Scope(++scopeSerial, -1, r, true, timestamp());
            s.sa = wa;
            scopes.push(s);
            allScopes.add(s);
        } catch (Throwable t) {
            error(t);
        }
    }

    public synchronized void endStaticTrigger(boolean used) {
        Scope s = scopes.peek();
        if (s == null || !s.staticTrigger) {
            return;
        }
        try {
            flushPending();
            if (used && !s.rec.dispatched) {
                emitStaticRec(s);
            }
        } catch (Throwable t) {
            error(t);
        } finally {
            attributeEffects(s.rec);
            s.tsTo = timestamp();
            scopes.remove(s);
        }
    }

    private void emitStaticRec(Scope s) {
        Rec r = s.rec;
        if (r.dispatched) {
            return;
        }
        try {
            targetParts(r, s.sa);
        } catch (Throwable ignored) {
            // no targets
        }
        r.step = stepFor(r);
        lastEmitted = r;
        index(r);
        deliver(r);
    }

    // ---- routing -------------------------------------------------------------

    /** Recent records, for trigger matching and same-step lookups. */
    private final Map<Integer, List<Rec>> byCard = new HashMap<>();
    private final List<Rec> damageLog = new ArrayList<>();
    private final Map<Integer, Rec> castRecs = new HashMap<>();
    private Rec lastEmitted;

    /** Move records held back to be batched, classified (draw?) or superseded (by cast / land play). */
    private List<Rec> pending = new ArrayList<>();
    /** Records of an attempt whose cast/activate has not been emitted yet. */
    private final List<Rec> held = new ArrayList<>();
    private GameHistory.Attempt heldFor;
    private final Map<Integer, Rec> attemptCast = new HashMap<>();

    private long curStep = -1;
    private String lastKey = "";
    private int keySerial;

    private void index(Rec r) {
        for (int id : r.cards) {
            byCard.computeIfAbsent(id, k -> new ArrayList<>()).add(r);
        }
        if (r.source >= 0) {
            byCard.computeIfAbsent(r.source, k -> new ArrayList<>()).add(r);
        }
        if (r.targetCard >= 0) {
            byCard.computeIfAbsent(r.targetCard, k -> new ArrayList<>()).add(r);
        }
        if ("deal_damage".equals(r.verb)) {
            damageLog.add(r);
        }
        VERBS.merge(r.verb, 1, Integer::sum);
        recent.add(r);
        if (recent.size() > 400) {
            recent.subList(0, 200).clear();
        }
    }

    private static boolean pushesStack(String verb) {
        return "cast".equals(verb) || "copy_spell".equals(verb) || "activate".equals(verb) || "trigger".equals(verb);
    }

    private void emit(Rec r) {
        syncResolution();
        Scope top = topScope();
        if (top != null && top.staticTrigger && !top.rec.dispatched) {
            emitStaticRec(top);
        }
        if (r.causes.isEmpty() && r.causeKind == null && !r.causeFromAttempt) {
            contextCause(r);
        }
        if (!pushesStack(r.verb)) {
            lastClosedRes = null;
        }
        r.step = stepFor(r);
        lastEmitted = r;
        index(r);
        route(r);
    }

    private static boolean holdable(Rec r) {
        return "move_zone".equals(r.verb) || "discard".equals(r.verb) || "draw".equals(r.verb)
                || "create".equals(r.verb);
    }

    private static String batchKey(Rec r) {
        StringBuilder sb = new StringBuilder(r.verb).append('|').append(r.kind)
                .append('|').append(r.fromZone).append(r.fromOwner).append('|').append(r.toZone).append(r.toOwner)
                .append('|').append(r.step).append('|').append(r.attempt).append('|').append(r.causeKind)
                .append('|').append(r.causeFromAttempt).append('|').append(r.replacedBy).append('|').append(r.privateMove)
                .append('|').append(r.drawer == null ? -1 : r.drawer.getId())
                .append('|').append(r.mods.get("face_down"));
        for (Rec c : r.causes) {
            sb.append(',').append(c.canonical().seq);
        }
        return sb.toString();
    }

    private void route(Rec r) {
        if (!pending.isEmpty()) {
            if (holdable(r) && batchKey(pending.get(pending.size() - 1)).equals(batchKey(r))) {
                pending.add(r);
                return;
            }
            flushPending();
        }
        if (holdable(r)) {
            pending.add(r);
            return;
        }
        deliver(r);
    }

    /** Emits held-back moves: classifies draws, then merges each run of identical moves into one event. */
    private void flushPending() {
        if (pending.isEmpty()) {
            return;
        }
        List<Rec> list = pending;
        pending = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            Rec r = list.get(i);
            if (r.drawer == null) {
                continue;
            }
            boolean draw = r.pregame;
            if (!draw) {
                int after;
                try {
                    after = i + 1 < list.size() && list.get(i + 1).drawer == r.drawer
                            ? list.get(i + 1).drawnBefore : r.drawer.getNumDrawnThisTurn();
                } catch (Throwable ignored) {
                    after = r.drawnBefore;
                }
                draw = after > r.drawnBefore;
            }
            if (!draw) {
                r.verb = "move_zone";
                r.mods.put("move_kind", r.kind);
            }
        }
        Rec head = null;
        String headKey = null;
        for (Rec r : list) {
            String k = batchKey(r);
            if (head != null && k.equals(headKey)) {
                for (Part p : r.parts) {
                    if (Schema.OBJ.equals(p.role)) {
                        head.parts.add(p);
                    }
                }
                head.cards.addAll(r.cards);
                head.count += r.count;
                r.mergedInto = head;
                continue;
            }
            if (head != null) {
                deliver(head);
            }
            head = r;
            headKey = k;
        }
        deliver(head);
    }

    /**
     * Drops a held-back move of {@code cardId} into {@code zone}: a cast, copy
     * or land play covers it. Returns the dropped record, which the caller
     * aliases to the covering event (a trigger may already point at it).
     */
    private Rec supersede(int cardId, ZoneType zone) {
        for (List<Rec> list : java.util.Arrays.asList(pending, held)) {
            for (int i = list.size() - 1; i >= 0; i--) {
                Rec r = list.get(i);
                if (r.toZone == zone && r.cards.size() == 1 && r.cards.get(0) == cardId
                        && ("move_zone".equals(r.verb) || "create".equals(r.verb))) {
                    list.remove(i);
                    SUPERSEDED.merge(String.valueOf(zone), 1, Integer::sum);
                    return r;
                }
            }
        }
        return null;
    }

    /** Attempt-aware delivery: an attempt's events wait for its cast. */
    private void deliver(Rec r) {
        if (r == null) {
            return;
        }
        GameHistory.Attempt a = GameHistory.currentAttempt();
        if (a != null && heldFor == a && !attemptCast.containsKey(a.id)) {
            held.add(r);
            return;
        }
        dispatch(r);
    }

    private void dispatch(Rec r) {
        if (r.dispatched) {
            return;
        }
        if (r.causeFromAttempt && r.causes.isEmpty()) {
            r.causeKind = "player_action";
        }
        r.dispatched = true;
        for (GameHistory v : viewers.values()) {
            v.append(r);
        }
        CAUSE_KIND.merge(r.causeKind == null ? "unknown" : r.causeKind, 1, Integer::sum);
    }

    // ---- steps and causes --------------------------------------------------

    private long stepFor(Rec r) {
        String key;
        Scope top = topScope();
        GameHistory.Attempt a = GameHistory.currentAttempt();
        if (top != null) {
            key = "S" + top.serial;
        } else if (a != null) {
            key = "A" + a.id;
        } else if ("pointer".equals(r.causeKind) && !r.causes.isEmpty()
                && r.causes.get(0).canonical().step == curStep && curStep >= 0) {
            return curStep;
        } else if ("trigger".equals(r.verb)) {
            key = lastKey.startsWith("TRG") ? lastKey : "TRG" + (++keySerial);
        } else {
            key = "K" + r.causeKind + "|" + r.phase + "|" + r.turn;
        }
        if (!key.equals(lastKey)) {
            curStep++;
            lastKey = key;
        }
        return curStep;
    }

    private boolean pregame() {
        try {
            return game != null && game.getAge() != GameStage.Play;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private void contextCause(Rec r) {
        Scope top = topScope();
        if (top != null) {
            r.causes.add(top.rec);
            r.causeKind = "pointer";
            return;
        }
        GameHistory.Attempt a = GameHistory.currentAttempt();
        if (a != null) {
            Rec c = attemptCast.get(a.id);
            if (c != null) {
                r.causes.add(c);
                r.causeKind = "pointer";
            } else {
                r.causeFromAttempt = true;
                r.causeKind = "player_action";
            }
            return;
        }
        try {
            forge.game.zone.CostPaymentStack.Entry e = game == null ? null : game.costPaymentStack.peek();
            if (e != null && e.payment() != null && e.payment().getAbility() != null) {
                SpellAbility s = e.payment().getAbility();
                Rec c = castRecs.get(s.getId());
                if (c == null && s.getRootAbility() != null) {
                    c = castRecs.get(s.getRootAbility().getId());
                }
                if (c != null) {
                    r.causes.add(c);
                    r.causeKind = "pointer";
                } else {
                    r.causeKind = "player_action";
                }
                COST_SCOPE.incrementAndGet();
                return;
            }
        } catch (Throwable ignored) {
            // fall through
        }
        r.causeKind = residual(r);
    }

    private String residual(Rec r) {
        if (game == null) {
            return "unknown";
        }
        if (pregame() || "mulligan".equals(r.verb)) {
            return "pregame";
        }
        if ("declare_attacker".equals(r.verb) || "declare_blockers".equals(r.verb)) {
            return "player_action";
        }
        long t0 = System.nanoTime();
        String k = STACK_WALKER.walk(s -> s.map(f -> classifyFrame(f.getClassName(), f.getMethodName()))
                .filter(java.util.Objects::nonNull).findFirst()).orElse("unknown");
        WALK_NS.addAndGet(System.nanoTime() - t0);
        WALKS.incrementAndGet();
        if (WALK_DEBUG) {
            System.err.println("[walk] " + r.verb + " -> " + k + " turn=" + r.turn + " phase=" + r.phase);
            STACK_WALKER.forEach(f -> {
                if (f.getClassName().startsWith("forge.")) {
                    System.err.println("    " + f.getClassName() + "#" + f.getMethodName());
                }
            });
        }
        return k;
    }

    private static final StackWalker STACK_WALKER = StackWalker.getInstance();
    private static final boolean WALK_DEBUG = Boolean.getBoolean("magicnet.history.walkdebug");

    /** Innermost engine frame that names a rule category; null to keep walking outward. */
    static String classifyFrame(String cls, String m) {
        if (!cls.startsWith("forge.")) {
            return null;
        }
        if (cls.equals("forge.game.GameAction")) {
            if (m.startsWith("checkStateEffects") || m.equals("checkGameOverCondition")) {
                return "state_based";
            }
            if (m.contains("ulligan")) {
                return "pregame";
            }
            return null;
        }
        if (cls.contains("ulligan")) {
            return "pregame";
        }
        if (cls.startsWith("forge.game.combat.")) {
            return "combat";
        }
        if (cls.equals("forge.game.phase.PhaseHandler")) {
            switch (m) {
                case "checkStateBasedEffects":
                    return "state_based";
                case "onPhaseBegin":
                case "onPhaseEnd":
                case "handleNextTurn":
                case "advanceToNextPhase":
                case "endCombat":
                case "setupFirstTurn":
                case "declareAttackersTurnBasedAction":
                case "declareBlockersTurnBasedAction":
                    return "turn_based";
                default:
                    return null;
            }
        }
        if (cls.startsWith("forge.game.phase.")) {
            return "turn_based";
        }
        return null;
    }

    // ---- the event bus -----------------------------------------------------

    private final Visitor visitor = new Visitor();

    @Subscribe
    public void onEvent(GameEvent ev) {
        if (closed) {
            return;
        }
        String name = ev.getClass().getSimpleName();
        if (name.startsWith("GameEvent")) {
            name = name.substring("GameEvent".length());
        }
        SEEN.merge(name, 1, Integer::sum);
        try {
            synchronized (this) {
                ev.visit(visitor);
            }
        } catch (Throwable t) {
            error(t);
        }
    }

    private static int errors;

    private static void error(Throwable t) {
        errors++;
        if (errors <= 5) {
            System.err.println("[history] " + t);
            t.printStackTrace();
        }
    }

    /** Called before every kept event: detects changes Forge fires no event for. */
    private void before(String name) {
        KEPT.merge(name, 1, Integer::sum);
        // Open a resolution that has had no event yet before the handler
        // classifies anything against the scope stack -- except at
        // SpellResolved, which opens it itself, knowing whether it fizzled.
        if (!"SpellResolved".equals(name)) {
            syncResolution();
        }
        diffDesignations();
        diffControllers(null);
        quietEffects();
    }

    /**
     * Outside any resolution or static trigger, continuous effects that already
     * exist were not made by a scope that has yet to close. (A resolution's
     * first event can come after its effect -- Giant Growth fires nothing
     * before SpellResolved -- so they are claimed only at its close.)
     */
    private void quietEffects() {
        try {
            if (topScope() == null && game != null && !game.getStack().isResolving()) {
                attributeEffects(NO_SCOPE);
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    private final class Visitor extends forge.game.event.IGameEventVisitor.Base<Void> {
        @Override
        public Void visit(forge.game.event.GameEventCardChangeZone e) {
            before("CardChangeZone");
            onMove(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventSpellAbilityCast e) {
            before("SpellAbilityCast");
            onCast(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventSpellResolved e) {
            before("SpellResolved");
            onResolved(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventSpellRemovedFromStack e) {
            before("SpellRemovedFromStack");
            onRemoved(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventLandPlayed e) {
            before("LandPlayed");
            onLandPlayed(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardDestroyed e) {
            before("CardDestroyed");
            pendingDestroy = true;
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardSacrificed e) {
            before("CardSacrificed");
            if (e.card() != null) {
                pendingSacrifice.add(e.card().getId());
            }
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardDamaged e) {
            before("CardDamaged");
            onDamage(e.source(), e.card(), null, e.amount(), false);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventPlayerDamaged e) {
            before("PlayerDamaged");
            onDamage(e.source(), null, e.target(), e.amount(), e.combat());
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardCounters e) {
            before("CardCounters");
            onCounters(e.card(), null, e.type(), e.oldValue(), e.newValue());
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventPlayerCounters e) {
            before("PlayerCounters");
            if (e.type() != null) {
                // Forge passes the new count in the field it names "amount".
                onCounters(null, e.receiver(), e.type(), e.oldValue(), e.amount());
            }
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventPlayerLivesChanged e) {
            before("PlayerLivesChanged");
            onLife(e.player(), e.oldLives(), e.newLives());
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventAttackersDeclared e) {
            before("AttackersDeclared");
            onAttackers(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventBlockersDeclared e) {
            before("BlockersDeclared");
            onBlockers(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardAttachment e) {
            before("CardAttachment");
            onAttach(e);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardModeChosen e) {
            before("CardModeChosen");
            if (e.mode() != null) {
                modes.add(EventTokenizer.normalize(e.mode(), game));
            }
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventScry e) {
            before("Scry");
            Rec r = rec("scry");
            r.mods.put("n_top", e.toTop());
            r.mods.put("n_bottom", e.toBottom());
            add(r, playerPart(Schema.PLAYER, e.player()));
            emit(r);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventSurveil e) {
            before("Surveil");
            Rec r = rec("surveil");
            r.mods.put("n_top", e.toLibrary());
            r.mods.put("n_graveyard", e.toGraveyard());
            add(r, playerPart(Schema.PLAYER, e.player()));
            emit(r);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventMulligan e) {
            before("Mulligan");
            Rec r = rec("mulligan");
            Player p = player(e.player());
            if (p != null) {
                r.mods.put("hand_size", p.getCardsIn(ZoneType.Hand).size());
            }
            add(r, playerPart(Schema.PLAYER, e.player()));
            r.causeKind = "pregame";
            emit(r);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventShuffle e) {
            before("Shuffle");
            Rec r = rec("shuffle");
            add(r, playerPart(Schema.PLAYER, e.player()));
            if (e.player() != null) {
                r.player = e.player().getId();
            }
            emit(r);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventSpeedChanged e) {
            before("SpeedChanged");
            designation("speed", e.player() == null ? -1 : e.player().getId(), e.oldValue(), e.newValue());
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventDayTimeChanged e) {
            before("DayTimeChanged");
            designation(e.daytime() ? "day" : "night", -1, 0, 1);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardStatsChanged e) {
            if (e.cards() != null && !e.cards().isEmpty()) {
                List<Card> cs = new ArrayList<>();
                for (CardView v : e.cards()) {
                    Card c = cardOf(v);
                    if (c != null) {
                        cs.add(c);
                    }
                }
                diffControllers(cs);
            }
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventTurnBegan e) {
            diffDesignations();
            diffControllers(null);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventTurnPhase e) {
            // "Until end of turn" control effects end in cleanup without any
            // event of their own; the next phase change is the first chance.
            diffDesignations();
            diffControllers(null);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardForetold e) {
            before("CardForetold");
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventTokenCreated e) {
            // Carries no fields; the token's own zone change is the create event.
            before("TokenCreated");
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventPlayerPoisoned e) {
            // Duplicate of the PlayerCounters event Forge fires just before it.
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardPhased e) {
            UNMAPPED.merge("CardPhased", 1, Integer::sum);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventCardRegenerated e) {
            UNMAPPED.merge("CardRegenerated", 1, Integer::sum);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventFlipCoin e) {
            UNMAPPED.merge("FlipCoin", 1, Integer::sum);
            return null;
        }

        @Override
        public Void visit(forge.game.event.GameEventRollDie e) {
            UNMAPPED.merge("RollDie", 1, Integer::sum);
            return null;
        }
    }

    // ---- lookups -------------------------------------------------------------

    private Player player(PlayerView pv) {
        if (pv == null || game == null) {
            return null;
        }
        for (Player p : game.getPlayers()) {
            if (p.getId() == pv.getId()) {
                return p;
            }
        }
        return null;
    }

    private static ZoneType zt(ZoneView z) {
        return z == null ? null : z.zoneType();
    }

    private static int owner(ZoneView z) {
        return z == null || z.player() == null ? -1 : z.player().getId();
    }

    /** The live card for a view: its own zone first, then everywhere. */
    Card cardOf(CardView v) {
        if (v == null || game == null) {
            return null;
        }
        try {
            ZoneType z = v.getZone();
            if (z == ZoneType.Stack) {
                for (Card c : game.getStackZone().getCards()) {
                    if (c.getId() == v.getId()) {
                        return c;
                    }
                }
            } else if (z != null) {
                for (Player p : game.getPlayers()) {
                    for (Card c : p.getCardsIn(z)) {
                        if (c.getId() == v.getId()) {
                            return c;
                        }
                    }
                }
            }
            return game.findById(v.getId());
        } catch (Throwable ignored) {
            return null;
        }
    }

    /** The card that just arrived in {@code z}, or null (ceased to exist / not found). */
    private Card cardIn(ZoneView z, int id) {
        if (z == null || game == null) {
            return null;
        }
        try {
            if (z.zoneType() == ZoneType.Stack) {
                for (Card c : game.getStackZone().getCards()) {
                    if (c.getId() == id) {
                        return c;
                    }
                }
                return null;
            }
            for (Player p : game.getPlayers()) {
                if (z.player() != null && p.getId() != z.player().getId()) {
                    continue;
                }
                for (Card c : p.getCardsIn(z.zoneType())) {
                    if (c.getId() == id) {
                        return c;
                    }
                }
            }
        } catch (Throwable ignored) {
            // not found
        }
        return null;
    }

    private SpellAbilityStackInstance stackInstance(int siId) {
        try {
            for (SpellAbilityStackInstance si : game.getStack()) {
                if (si.getId() == siId) {
                    return si;
                }
            }
        } catch (Throwable ignored) {
            // not found
        }
        return null;
    }

    private static final java.lang.reflect.Field HAS_RUN = hasRunField();

    private static java.lang.reflect.Field hasRunField() {
        try {
            java.lang.reflect.Field f = forge.game.replacement.ReplacementHandler.class.getDeclaredField("hasRun");
            f.setAccessible(true);
            return f;
        } catch (Throwable t) {
            System.err.println("[history] ReplacementHandler.hasRun not readable: " + t);
            return null;
        }
    }

    /** Host of the replacement effect executing right now, mapped from an effect card to its source. */
    private Card replacingHost() {
        if (HAS_RUN == null || game == null) {
            return null;
        }
        try {
            Object v = HAS_RUN.get(game.getReplacementHandler());
            if (!(v instanceof Collection) || ((Collection<?>) v).isEmpty()) {
                return null;
            }
            Card host = null;
            for (Object o : (Collection<?>) v) {
                if (o instanceof forge.game.replacement.ReplacementEffect) {
                    host = ((forge.game.replacement.ReplacementEffect) o).getHostCard();
                }
            }
            if (host != null && host.getEffectSource() != null) {
                host = host.getEffectSource();
            }
            if (host != null) {
                Card live = game.getCardState(host, null);
                return live != null ? live : host;
            }
        } catch (Throwable ignored) {
            // none
        }
        return null;
    }

    // ---- zone changes --------------------------------------------------------

    private boolean pendingDestroy;
    private final Set<Integer> pendingSacrifice = new HashSet<>();

    private void onMove(forge.game.event.GameEventCardChangeZone e) {
        CardView cv = e.card();
        if (cv == null) {
            return;
        }
        ZoneType from = zt(e.from());
        ZoneType to = zt(e.to());
        int fo = owner(e.from());
        int too = owner(e.to());
        if (from == null && (to == null || to == ZoneType.Library || to == ZoneType.Sideboard
                || to == ZoneType.None)) {
            return; // deck setup
        }
        try {
            // Forge's "X's Effect" cards are bookkeeping, not game objects.
            if (cv.isImmutable() && !cv.isEmblem()) {
                return;
            }
        } catch (Throwable ignored) {
            // keep
        }
        Card now = cardIn(e.to(), cv.getId());
        Part obj = new Part(Schema.OBJ, true);
        for (GameHistory v : viewers.values()) {
            obj.refs.put(v.vid, v.moveRef(cv, now, from, fo, to, too));
        }
        Rec r = rec(null);
        r.fromZone = from == ZoneType.None ? null : from;
        r.fromOwner = fo;
        r.toZone = to == ZoneType.None ? null : to;
        r.toOwner = too;
        r.cards.add(cv.getId());
        r.parts.add(obj);
        boolean token = false;
        try {
            token = cv.isToken();
        } catch (Throwable ignored) {
            // treat as card
        }
        String kind = moveKind(cv.getId(), from, to);
        r.kind = kind;
        if ((from == null || from == ZoneType.None) && to == ZoneType.Battlefield && token) {
            r.verb = "create";
            int ctrl = now != null && now.getController() != null ? now.getController().getId() : too;
            add(r, playerPart(Schema.ACTOR, ctrl));
        } else if (from == ZoneType.Hand && to == ZoneType.Graveyard) {
            r.verb = "discard";
            add(r, playerPart(Schema.PLAYER, fo));
        } else if (from == ZoneType.Library && to == ZoneType.Hand && fo == too) {
            // Draw or not is only knowable after Player.doDraw bumps its counter,
            // which it does after the move: classified when the batch is flushed.
            r.verb = "draw";
            r.drawer = player(e.to().player());
            r.drawnBefore = r.drawer == null ? 0 : r.drawer.getNumDrawnThisTurn();
            r.pregame = pregame();
            add(r, playerPart(Schema.PLAYER, too));
        } else {
            r.verb = "move_zone";
            r.mods.put("move_kind", kind);
            add(r, playerPart(Schema.PLAYER, too >= 0 ? too : fo));
        }
        if (from == ZoneType.Library && to == ZoneType.Library && fo == too) {
            r.privateMove = true;
        }
        try {
            // Public: everyone sees that a face-down object moved (and nothing more).
            if (cv.isFaceDown()) {
                r.mods.put("face_down", true);
            }
        } catch (Throwable ignored) {
            // omit
        }
        Card rep = replacingHost();
        if (rep != null) {
            r.replacedBy = rep.getId();
            add(r, cardPart(Schema.REPLACED_BY, rep));
        }
        if (from == ZoneType.Battlefield && "destroy".equals(kind) && topScope() == null
                && GameHistory.currentAttempt() == null) {
            // A creature destroyed outside any resolution or cast: state-based
            // (lethal damage). Keep the rule category, and point at the damage.
            r.causeKind = residual(r);
            if ("state_based".equals(r.causeKind)) {
                for (int i = damageLog.size() - 1; i >= 0 && i >= damageLog.size() - 40; i--) {
                    Rec d = damageLog.get(i);
                    if (d.turn != r.turn) {
                        break;
                    }
                    if (d.targetCard == cv.getId()) {
                        r.causes.add(d);
                    }
                }
            }
        }
        emit(r);
    }

    private String moveKind(int cardId, ZoneType from, ZoneType to) {
        if (pendingSacrifice.remove(cardId)) {
            return "sacrifice";
        }
        if (pendingDestroy && from == ZoneType.Battlefield) {
            pendingDestroy = false;
            return "destroy";
        }
        if (to == ZoneType.Exile) {
            return "exile";
        }
        if (from == ZoneType.Library && to == ZoneType.Graveyard) {
            return "mill";
        }
        if (from == ZoneType.Battlefield && to == ZoneType.Hand) {
            return "bounce";
        }
        if (from == ZoneType.Library && to != ZoneType.Library) {
            return "search";
        }
        if ((from == ZoneType.Graveyard || from == ZoneType.Exile || from == ZoneType.Command)
                && (to == ZoneType.Hand || to == ZoneType.Battlefield)) {
            return "return";
        }
        if (to == ZoneType.Library || to == ZoneType.Battlefield || to == ZoneType.Hand) {
            return "put";
        }
        return "other";
    }

    // ---- the stack -----------------------------------------------------------

    /** Live stack abilities by SpellAbility id: [stack-instance id, host card id, is spell]. */
    private final Map<Integer, int[]> live = new HashMap<>();
    private final List<String> modes = new ArrayList<>();

    /** stack_object and source participants, plus actor for non-attempt events. */
    private void stackParts(Rec r, SpellAbility sa, int saId, boolean actor) {
        int[] info = live.get(saId);
        Card host = null;
        try {
            host = sa == null ? null : sa.getHostCard();
            if (host != null) {
                Card cur = game == null ? null : game.getCardState(host, null);
                host = cur != null ? cur : host;
            }
        } catch (Throwable ignored) {
            // no host
        }
        if (sa != null && sa.getActivatingPlayer() != null) {
            add(r, playerPart(Schema.ACTOR, sa.getActivatingPlayer().getId()));
        }
        if (info != null && info[2] == 0) {
            add(r, abilityPart(Schema.STACK_OBJECT, info[0], sa, host));
        } else if (host != null) {
            add(r, cardPart(Schema.STACK_OBJECT, host));
        }
        if (host != null) {
            add(r, cardPart(Schema.SOURCE_IS, host));
            r.source = host.getId();
        }
    }

    /** Ordered target slots: every target of the ability and its sub-abilities, deduplicated. */
    private void targetParts(Rec r, SpellAbility sa) {
        if (sa == null) {
            return;
        }
        List<Object> seen = new ArrayList<>();
        for (TargetChoices tc : sa.getAllTargetChoices()) {
            if (tc == null) {
                continue;
            }
            for (GameEntity e : tc.getTargetEntities()) {
                if (e != null && !seen.contains(e)) {
                    seen.add(e);
                }
            }
            for (SpellAbility ts : tc.getTargetSpells()) {
                if (ts != null && !seen.contains(ts)) {
                    seen.add(ts);
                }
            }
        }
        targetParts(r, seen);
    }

    private void targetParts(Rec r, List<?> targets) {
        int i = 0;
        for (Object o : targets) {
            String slot = Schema.targetSlot(i++);
            if (o instanceof SpellAbility) {
                SpellAbility ts = (SpellAbility) o;
                int[] info = live.get(ts.getId());
                if (info != null && info[2] == 0) {
                    add(r, abilityPart(slot, info[0], ts, ts.getHostCard()));
                } else {
                    add(r, cardPart(slot, ts.getHostCard()));
                }
            } else {
                add(r, entityPart(slot, o));
                if (o instanceof Card && r.targetCard < 0) {
                    r.targetCard = ((Card) o).getId();
                } else if (o instanceof Player && r.targetPlayer < 0) {
                    r.targetPlayer = ((Player) o).getId();
                }
            }
        }
    }

    private static final byte[] MANA_COLORS = {
        forge.card.MagicColor.WHITE, forge.card.MagicColor.BLUE, forge.card.MagicColor.BLACK,
        forge.card.MagicColor.RED, forge.card.MagicColor.GREEN};

    private static void manaSpent(Rec r, SpellAbility sa) {
        try {
            List<forge.game.mana.Mana> paid = sa.getPayingMana();
            if (paid == null || paid.isEmpty()) {
                return;
            }
            int[] n = new int[6];
            for (forge.game.mana.Mana m : paid) {
                byte c = m.getColor();
                int slot = 5;
                for (int i = 0; i < MANA_COLORS.length; i++) {
                    if (c == MANA_COLORS[i]) {
                        slot = i;
                    }
                }
                n[slot]++;
            }
            if (n[0] > 0) {
                r.mods.put("mana_spent_W", n[0]);
            }
            if (n[1] > 0) {
                r.mods.put("mana_spent_U", n[1]);
            }
            if (n[2] > 0) {
                r.mods.put("mana_spent_B", n[2]);
            }
            if (n[3] > 0) {
                r.mods.put("mana_spent_R", n[3]);
            }
            if (n[4] > 0) {
                r.mods.put("mana_spent_G", n[4]);
            }
            if (n[5] > 0) {
                r.mods.put("mana_spent_C", n[5]);
            }
        } catch (Throwable ignored) {
            // payment not readable
        }
    }

    private void onCast(forge.game.event.GameEventSpellAbilityCast e) {
        if (e.si() == null || e.sa() == null) {
            return;
        }
        SpellAbilityStackInstance si = stackInstance(e.si().getId());
        if (si == null) {
            return;
        }
        SpellAbility sa = si.getSpellAbility();
        int saId = e.sa().getId();
        Card host = si.getSourceCard();
        boolean spell = sa.isSpell();
        live.put(saId, new int[] {si.getId(), host == null ? -1 : host.getId(), spell ? 1 : 0});
        abilityText.put(si.getId(), EventTokenizer.abilityText(e.sa().getDescription(), sa, game));
        Rec covered = host != null && spell ? supersede(host.getId(), ZoneType.Stack) : null;
        boolean copy = sa.isCopied() || (host != null && host.isCopiedSpell());
        String verb = sa.isTrigger() ? "trigger" : spell ? (copy ? "copy_spell" : "cast") : "activate";
        Rec r = rec(verb);
        stackParts(r, sa, saId, true);
        targetParts(r, sa);
        if (!modes.isEmpty()) {
            if (!"activate".equals(verb)) {
                r.mods.put("modes", new ArrayList<>(modes));
            }
            modes.clear();
        }
        if (spell) {
            try {
                Integer x = sa.getXManaCostPaid();
                if (x != null) {
                    r.mods.put("x_paid", x);
                }
                if (sa.isKicked()) {
                    r.mods.put("kicked", true);
                }
            } catch (Throwable ignored) {
                // optional
            }
        } else {
            r.mods.put("ability", EventTokenizer.abilityText(e.sa().getDescription(), sa, game));
        }
        if (!"trigger".equals(verb)) {
            manaSpent(r, sa);
        }
        castRecs.put(saId, r);
        if (covered != null) {
            covered.mergedInto = r;
        }
        GameHistory.Attempt a = GameHistory.currentAttempt();
        // Forge puts a copy of an activated ability on the stack, so identity
        // (Attempt.owns) fails for abilities; nothing else is pushed while a
        // proposal is still being cast, so the first non-trigger push is it.
        boolean owned = a != null && !"trigger".equals(verb)
                && (a.owns(sa) || heldFor == a && !attemptCast.containsKey(a.id));
        if ("trigger".equals(verb)) {
            matchTrigger(r, sa);
        } else if (topScope() != null || (game != null && game.getStack().isResolving())) {
            syncResolution();
            contextCause(r);
        } else if (!owned && lastClosedRes != null) {
            r.causes.add(lastClosedRes.rec);
            r.causeKind = "pointer";
        } else {
            r.causeKind = "player_action";
        }
        if (owned && heldFor == a && !attemptCast.containsKey(a.id)) {
            attemptCommit(a, r);
        } else {
            emit(r);
        }
    }

    /** Emits an attempt's cast (or land play) and then the events held back for it. */
    private void attemptCommit(GameHistory.Attempt a, Rec r) {
        flushPending();
        r.step = stepFor(r);
        lastEmitted = r;
        index(r);
        dispatch(r);
        attemptCast.put(a.id, r);
        for (Rec h : held) {
            if (h.causeFromAttempt && h.causes.isEmpty()) {
                h.causes.add(r);
                h.causeKind = "pointer";
            }
            dispatch(h);
        }
        held.clear();
    }

    private void onResolved(forge.game.event.GameEventSpellResolved e) {
        if (e.spell() == null) {
            return;
        }
        int id = e.spell().getId();
        if (resScope != null && resScope.saId == id) {
            return;
        }
        SpellAbility sa = null;
        try {
            for (SpellAbilityStackInstance si : game.getStack()) {
                if (si.getSpellAbility() != null && si.getSpellAbility().getId() == id) {
                    sa = si.getSpellAbility();
                }
            }
        } catch (Throwable ignored) {
            // not on the stack
        }
        if (resScope != null) {
            closeResolution();
        }
        openResolution(sa, id, e.hasFizzled());
    }

    private static boolean isCounter(SpellAbility sa) {
        for (SpellAbility s = sa; s != null; s = s.getSubAbility()) {
            if (s.getApi() == ApiType.Counter) {
                return true;
            }
        }
        return false;
    }

    private void onRemoved(forge.game.event.GameEventSpellRemovedFromStack e) {
        if (e.sa() == null) {
            return;
        }
        int id = e.sa().getId();
        int[] info = live.remove(id);
        if (info == null) {
            return; // already removed: Forge fires this twice for a copy that ceases to exist
        }
        if (resScope != null && resScope.saId == id) {
            closeResolution();
            return;
        }
        syncResolution();
        Rec r = rec("does_not_resolve");
        r.mods.put("reason", resScope != null && isCounter(resScope.sa) ? "countered" : "removed");
        Rec cast = castRecs.get(id);
        if (cast != null && cast.parts.size() > 0) {
            for (Part p : cast.parts) {
                if (Schema.STACK_OBJECT.equals(p.role)) {
                    r.parts.add(p);
                }
            }
        }
        emit(r);
    }

    private void onLandPlayed(forge.game.event.GameEventLandPlayed e) {
        Card land = cardOf(e.land());
        Rec covered = land == null ? null : supersede(land.getId(), ZoneType.Battlefield);
        Rec r = rec("play_land");
        if (covered != null) {
            covered.mergedInto = r;
        }
        add(r, playerPart(Schema.ACTOR, e.player()));
        add(r, cardPart(Schema.OBJ, land));
        if (land != null) {
            r.cards.add(land.getId());
        }
        r.causeKind = "player_action";
        GameHistory.Attempt a = GameHistory.currentAttempt();
        if (a != null && heldFor == a && !attemptCast.containsKey(a.id)) {
            attemptCommit(a, r);
        } else {
            emit(r);
        }
    }

    // ---- damage, life, counters ------------------------------------------------

    private void onDamage(CardView src, CardView tgtCard, PlayerView tgtPlayer, int amount, boolean combat) {
        Rec r = rec("deal_damage");
        r.mods.put("amount", amount);
        Card s = cardOf(src);
        add(r, cardPart(Schema.SOURCE_IS, s));
        if (s != null) {
            r.source = s.getId();
        }
        if (tgtCard != null) {
            Card t = cardOf(tgtCard);
            add(r, cardPart(Schema.TARGET_0, t));
            r.targetCard = tgtCard.getId();
        } else if (tgtPlayer != null) {
            add(r, playerPart(Schema.TARGET_0, tgtPlayer));
            r.targetPlayer = tgtPlayer.getId();
        }
        if (!combat && tgtCard != null && topScope() == null) {
            try {
                PhaseType ph = game.getPhaseHandler().getPhase();
                combat = (ph == PhaseType.COMBAT_DAMAGE || ph == PhaseType.COMBAT_FIRST_STRIKE_DAMAGE)
                        && !game.getStack().isResolving();
            } catch (Throwable ignored) {
                // not combat
            }
        }
        if (combat) {
            r.mods.put("combat", true);
            if (topScope() == null) {
                r.causeKind = "combat";
            }
        }
        emit(r);
    }

    /** The damage event just emitted to this entity, if this change is its result (life loss, -1/-1, poison). */
    private Rec damageResult(int card, int player) {
        Rec d = lastEmitted;
        if (d == null || !"deal_damage".equals(d.verb)) {
            return null;
        }
        if (card >= 0 ? d.targetCard == card : d.targetPlayer == player) {
            return d;
        }
        return null;
    }

    private void onLife(PlayerView pv, int before, int after) {
        Rec r = rec("life_change");
        r.mods.put("before", before);
        r.mods.put("after", after);
        add(r, playerPart(Schema.PLAYER, pv));
        r.player = pv == null ? -1 : pv.getId();
        // Combat damage to a player is applied as one life change for the
        // whole batch: point at every damage event to them in this step.
        if (damageResult(-1, r.player) != null) {
            long step = lastEmitted.step;
            for (int i = recent.size() - 1; i >= 0; i--) {
                Rec x = recent.get(i);
                if (x.step != step || "life_change".equals(x.verb) && x.player == r.player) {
                    break;
                }
                if ("deal_damage".equals(x.verb) && x.targetPlayer == r.player) {
                    r.causes.add(0, x);
                }
            }
            r.causeKind = "pointer";
        }
        emit(r);
    }

    private void onCounters(CardView cv, PlayerView pv, forge.game.card.CounterType type, int before, int after) {
        if (before == after) {
            return;
        }
        Rec r = rec("counters_change");
        String name;
        try {
            name = type.getName().toLowerCase();
        } catch (Throwable ignored) {
            name = String.valueOf(type).toLowerCase();
        }
        r.mods.put("counter_type", name);
        r.mods.put("before", before);
        r.mods.put("after", after);
        if (cv != null) {
            Card c = cardOf(cv);
            add(r, cardPart(Schema.OBJ, c));
            r.cards.add(cv.getId());
        } else {
            add(r, playerPart(Schema.PLAYER, pv));
            r.player = pv == null ? -1 : pv.getId();
        }
        Rec d = damageResult(cv == null ? -1 : cv.getId(), r.player);
        if (d != null) {
            r.causes.add(d);
            r.causeKind = "pointer";
        }
        emit(r);
    }

    // ---- combat --------------------------------------------------------------

    private void onAttackers(forge.game.event.GameEventAttackersDeclared e) {
        if (e.attackersMap() == null) {
            return;
        }
        for (Map.Entry<forge.game.GameEntityView, CardView> en : e.attackersMap().entries()) {
            Rec r = rec("declare_attacker");
            Card atk = cardOf(en.getValue());
            add(r, cardPart(Schema.ATTACKS, atk));
            if (atk != null) {
                r.cards.add(atk.getId());
            }
            add(r, entityPart(Schema.DEFENDER, en.getKey()));
            r.causeKind = "player_action";
            emit(r);
        }
    }

    /**
     * One event per blocked attacker. Forge stores an unblocked attacker as a
     * self-pair sentinel; those get no event (the state's {@code blocked} flag
     * and the absence of a block carry it).
     */
    private void onBlockers(forge.game.event.GameEventBlockersDeclared e) {
        if (e.blockers() == null) {
            return;
        }
        for (com.google.common.collect.Multimap<CardView, CardView> byAttacker : e.blockers().values()) {
            for (Map.Entry<CardView, Collection<CardView>> en : byAttacker.asMap().entrySet()) {
                List<Card> blockers = new ArrayList<>();
                for (CardView b : en.getValue()) {
                    if (b.getId() != en.getKey().getId()) {
                        blockers.add(cardOf(b));
                    }
                }
                if (blockers.isEmpty()) {
                    continue;
                }
                Rec r = rec("declare_blockers");
                Card atk = cardOf(en.getKey());
                add(r, cardPart(Schema.ATTACKS, atk));
                if (atk != null) {
                    r.cards.add(atk.getId());
                }
                for (Card b : blockers) {
                    add(r, cardPart(Schema.BLOCKS, b));
                    if (b != null) {
                        r.cards.add(b.getId());
                    }
                }
                r.causeKind = "player_action";
                emit(r);
            }
        }
    }

    private void onAttach(forge.game.event.GameEventCardAttachment e) {
        Rec r = rec("attach");
        Card eq = cardOf(e.equipment());
        add(r, cardPart(Schema.OBJ, eq));
        if (eq != null) {
            r.cards.add(eq.getId());
        }
        if (e.newTarget() != null) {
            add(r, entityPart(Schema.ATTACHED_TO, e.newTarget()));
        }
        emit(r);
    }

    // ---- state diffs: designations and control -------------------------------

    private final Map<String, Integer> designations = new HashMap<>();

    private void designation(String what, int pid, int before, int after) {
        Rec r = rec("designation_change");
        r.mods.put("designation", what);
        r.mods.put("before", before);
        r.mods.put("after", after);
        if (pid >= 0) {
            add(r, playerPart(Schema.PLAYER, pid));
            r.player = pid;
        }
        emit(r);
    }

    /** Monarch, initiative and the city's blessing have no Forge event: diffed instead. */
    private void diffDesignations() {
        if (game == null) {
            return;
        }
        try {
            Player monarch = game.getMonarch();
            Player initiative = game.getHasInitiative();
            for (Player p : game.getPlayers()) {
                flip("monarch", p, p.equals(monarch));
                flip("initiative", p, p.equals(initiative));
                flip("city's blessing", p, p.hasBlessing());
            }
        } catch (Throwable ignored) {
            // not available
        }
    }

    private void flip(String what, Player p, boolean has) {
        String key = what + "|" + p.getId();
        int now = has ? 1 : 0;
        Integer was = designations.put(key, now);
        if (was != null && was != now) {
            designation(what, p.getId(), was, now);
        } else if (was == null && now == 1) {
            designation(what, p.getId(), 0, 1);
        }
    }

    private final Map<Integer, Integer> controllers = new HashMap<>();

    /** Control changes move no card between zones and fire no event of their own: diffed. */
    private void diffControllers(Collection<Card> only) {
        if (game == null) {
            return;
        }
        try {
            Collection<Card> cards = only;
            if (cards == null) {
                cards = new ArrayList<>();
                for (Player p : game.getPlayers()) {
                    for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
                        cards.add(c);
                    }
                }
            }
            for (Card c : cards) {
                if (c == null || c.getController() == null || !c.isInZone(ZoneType.Battlefield)) {
                    continue;
                }
                int now = c.getController().getId();
                Integer was = controllers.put(c.getId(), now);
                if (was != null && was != now) {
                    Rec r = rec("change_control");
                    add(r, cardPart(Schema.OBJ, c));
                    r.cards.add(c.getId());
                    add(r, playerPart(Schema.OLD_CONTROLLER, was));
                    add(r, playerPart(Schema.NEW_CONTROLLER, now));
                    emit(r);
                }
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    // ---- controller-callback events ------------------------------------------

    /**
     * One reveal of {@code cards} to {@code shownTo}, posted to every viewer.
     * The recipient's history gets the cards as edges (the callback was
     * delivered to them); any other viewer only if it can see them anyway.
     */
    public synchronized void reveal(List<Card> cards, ZoneType zone, Player owner, Player shownTo) {
        if (closed) {
            return;
        }
        try {
            Rec r = rec("reveal");
            r.fromZone = zone;
            r.fromOwner = owner == null ? -1 : owner.getId();
            r.count = cards.size();
            r.alwaysCount = true;
            for (Card c : cards) {
                Part p = new Part(Schema.OBJ, true);
                for (GameHistory v : viewers.values()) {
                    boolean recipient = shownTo != null && v.vid == shownTo.getId();
                    p.refs.put(v.vid, recipient ? v.revealRef(c) : v.objRef(c));
                }
                r.parts.add(p);
                r.cards.add(c.getId());
            }
            add(r, playerPart(Schema.PLAYER, shownTo == null ? -1 : shownTo.getId()));
            emit(r);
        } catch (Throwable t) {
            error(t);
        }
    }

    /** A cast or activation is being proposed: its events wait for the cast. */
    public synchronized void attemptBegun(GameHistory.Attempt a) {
        if (closed || a == null) {
            return;
        }
        try {
            syncResolution();
            flushPending();
        } catch (Throwable t) {
            error(t);
        }
        heldFor = a;
    }

    /** Closes an attempt: its held events, a special action if nothing hit the stack, and attempt_end. */
    public synchronized void attemptEnd(GameHistory.Attempt a, String outcome) {
        if (closed || a == null) {
            return;
        }
        try {
            flushPending();
            Rec cast = attemptCast.remove(a.id);
            for (Rec h : held) {
                dispatch(h);
            }
            held.clear();
            heldFor = null;
            SpellAbility sa = a.sa;
            if (cast == null && "committed".equals(outcome) && sa != null && !sa.isManaAbility()
                    && !sa.isLandAbility() && !sa.isSpell()) {
                Rec s = rec("special_action");
                s.mods.put("action_kind", specialKind(sa));
                if (sa.getActivatingPlayer() != null) {
                    add(s, playerPart(Schema.ACTOR, sa.getActivatingPlayer().getId()));
                }
                add(s, cardPart(Schema.SOURCE_IS, sa.getHostCard()));
                s.causeKind = "player_action";
                s.step = stepFor(s);
                index(s);
                dispatch(s);
                cast = s;
            }
            Rec r = rec("attempt_end");
            r.mods.put("outcome", outcome);
            r.mods.put("reached", a.reached);
            for (Map.Entry<String, Integer> e : a.announced.entrySet()) {
                r.mods.put("announce_" + e.getKey(), e.getValue());
            }
            if (sa != null) {
                if (sa.getActivatingPlayer() != null) {
                    add(r, playerPart(Schema.ACTOR, sa.getActivatingPlayer().getId()));
                }
                if (cast != null) {
                    for (Part p : cast.parts) {
                        if (Schema.STACK_OBJECT.equals(p.role)) {
                            r.parts.add(p);
                        }
                    }
                }
                add(r, cardPart(Schema.SOURCE_IS, sa.getHostCard()));
            }
            targetParts(r, a.targets);
            r.causeKind = "player_action";
            r.step = stepFor(r);
            index(r);
            dispatch(r);
        } catch (Throwable t) {
            error(t);
        }
    }

    private static String specialKind(SpellAbility sa) {
        try {
            if (sa.isTurnFaceUp() || sa.isMorphUp() || sa.isManifestUp()) {
                return "turn_face_up";
            }
            if (sa.isForetelling()) {
                return "foretell";
            }
            if (sa.isPlotting()) {
                return "plot";
            }
            if (sa.isUnlock()) {
                return "unlock";
            }
            if (sa.getApi() != null) {
                return sa.getApi().name().toLowerCase();
            }
        } catch (Throwable ignored) {
            // fall through
        }
        return "other";
    }

    /**
     * Brings history up to date before a decision is extracted: opens a
     * resolution that has had no event yet, applies state diffs, and emits
     * held-back moves (draws are classifiable by now). Events of an open
     * attempt stay held until its cast.
     */
    public synchronized void sync() {
        if (closed) {
            return;
        }
        try {
            syncResolution();
            Scope top = topScope();
            if (top != null && top.staticTrigger && !top.rec.dispatched) {
                emitStaticRec(top);
            }
            diffDesignations();
            diffControllers(null);
            flushPending();
            Scope open = topScope();
            if (open != null) {
                attributeEffects(open.rec);
            } else {
                quietEffects();
            }
        } catch (Throwable t) {
            error(t);
        }
    }

    // ---- trigger matching ----------------------------------------------------

    /** Collects card ids out of a triggering object (a card, a collection, a zone table). */
    private static void cardIds(Object o, List<Integer> out, int depth) {
        if (o == null || depth > 3) {
            return;
        }
        if (o instanceof Card) {
            out.add(((Card) o).getId());
        } else if (o instanceof CardView) {
            out.add(((CardView) o).getId());
        } else if (o instanceof com.google.common.collect.Table) {
            for (Object v : ((com.google.common.collect.Table<?, ?, ?>) o).values()) {
                cardIds(v, out, depth + 1);
            }
        } else if (o instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) o).entrySet()) {
                cardIds(e.getKey(), out, depth + 1);
                cardIds(e.getValue(), out, depth + 1);
            }
        } else if (o instanceof Iterable) {
            for (Object v : (Iterable<?>) o) {
                cardIds(v, out, depth + 1);
            }
        }
    }

    private static int playerId(Object o) {
        if (o instanceof Player) {
            return ((Player) o).getId();
        }
        if (o instanceof PlayerView) {
            return ((PlayerView) o).getId();
        }
        return -1;
    }

    /** Latest dispatched-or-pending record involving {@code cardId}, with one of {@code verbs} (null: any). */
    private Rec latest(int cardId, Set<String> verbs) {
        List<Rec> l = byCard.get(cardId);
        if (l == null) {
            return null;
        }
        for (int i = l.size() - 1; i >= 0 && i >= l.size() - 40; i--) {
            Rec r = l.get(i).canonical();
            if (verbs == null || verbs.contains(r.verb)) {
                return r;
            }
        }
        return null;
    }

    private Rec latestBy(java.util.function.Predicate<Rec> p) {
        int seen = 0;
        for (int i = recent.size() - 1; i >= 0 && seen < 200; i--, seen++) {
            Rec r = recent.get(i).canonical();
            if (p.test(r)) {
                return r;
            }
        }
        return null;
    }

    private final List<Rec> recent = new ArrayList<>();

    private static Set<String> set(String... s) {
        return new HashSet<>(java.util.Arrays.asList(s));
    }

    private static final Set<String> MOVES = set("move_zone", "draw", "discard", "create", "play_land", "cast",
            "copy_spell");

    /**
     * Points a trigger at the event(s) that triggered it, per trigger mode. The
     * trigger always goes on the stack after its triggering event, so only
     * already-recorded events are candidates.
     */
    private void matchTrigger(Rec r, SpellAbility sa) {
        String mode = "?";
        List<Rec> found = new ArrayList<>();
        String kind = null;
        try {
            forge.game.trigger.Trigger tr = sa.getTrigger();
            mode = tr == null || tr.getMode() == null ? "?" : tr.getMode().name();
            Map<AbilityKey, Object> objs = sa.getTriggeringObjects();
            if (objs == null) {
                objs = Collections.emptyMap();
            }
            List<Integer> cards = new ArrayList<>();
            switch (mode) {
                case "Phase":
                case "TurnBegin":
                case "NewTurn":
                case "DayTimeChanges":
                    kind = "turn_based";
                    break;
                case "Drawn":
                    cardIds(objs.get(AbilityKey.Card), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, set("draw")));
                    }
                    break;
                case "ChangesZone":
                case "Exiled":
                case "Destroyed":
                case "Sacrificed":
                case "Discarded":
                case "Milled":
                case "TokenCreated":
                case "LandPlayed":
                case "ChangesZoneAll":
                case "DiscardedAll":
                case "MilledAll":
                case "MilledOnce":
                case "SacrificedOnce":
                case "TokenCreatedOnce":
                    cardIds(objs.get(AbilityKey.Card), cards, 0);
                    cardIds(objs.get(AbilityKey.CardLKI), cards, 0);
                    cardIds(objs.get(AbilityKey.Cards), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, MOVES));
                    }
                    break;
                case "SpellCast":
                case "SpellCastOrCopy":
                case "SpellAbilityCast":
                case "AbilityCast":
                case "SpellCopy":
                case "SpellAbilityCopy":
                case "BecomesTarget":
                case "BecomesTargetOnce":
                    for (AbilityKey k : new AbilityKey[] {AbilityKey.SpellAbility, AbilityKey.StackSa,
                            AbilityKey.SourceSA}) {
                        Object o = objs.get(k);
                        if (o instanceof SpellAbility) {
                            addFound(found, castRecs.get(((SpellAbility) o).getId()));
                            if (((SpellAbility) o).getRootAbility() != null) {
                                addFound(found, castRecs.get(((SpellAbility) o).getRootAbility().getId()));
                            }
                        }
                    }
                    break;
                case "CounterAdded":
                case "CounterAddedOnce":
                case "CounterRemoved":
                case "CounterRemovedOnce":
                case "CounterAddedAll":
                case "CounterTypeAddedAll":
                case "CounterPlayerAddedAll": {
                    cardIds(objs.get(AbilityKey.Card), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, set("counters_change")));
                    }
                    int pid = playerId(objs.get(AbilityKey.Player));
                    if (found.isEmpty() && pid >= 0) {
                        addFound(found, latestBy(x -> "counters_change".equals(x.verb) && x.player == pid));
                    }
                    break;
                }
                case "DamageDone":
                case "DamageDoneOnce":
                case "DamageAll":
                case "DamageDealtOnce":
                case "DamageDoneOnceByController":
                case "ExcessDamage":
                case "ExcessDamageAll":
                    matchDamage(objs, found);
                    break;
                case "Attacks":
                case "AttackerUnblocked":
                case "AttackerUnblockedOnce":
                case "AttackersDeclared":
                case "AttackersDeclaredOneTarget":
                    cardIds(objs.get(AbilityKey.Attacker), cards, 0);
                    cardIds(objs.get(AbilityKey.Attackers), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, set("declare_attacker")));
                    }
                    break;
                case "Blocks":
                case "AttackerBlocked":
                case "AttackerBlockedOnce":
                case "AttackerBlockedByCreature":
                case "BlockersDeclared":
                    cardIds(objs.get(AbilityKey.Attacker), cards, 0);
                    cardIds(objs.get(AbilityKey.Blocker), cards, 0);
                    cardIds(objs.get(AbilityKey.Blockers), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, set("declare_blockers")));
                    }
                    break;
                case "LifeLost":
                case "LifeGained":
                case "LifeLostAll":
                case "PayLife": {
                    int pid = playerId(objs.get(AbilityKey.Player));
                    addFound(found, latestBy(x -> "life_change".equals(x.verb) && x.player == pid));
                    break;
                }
                case "BecomeMonarch":
                case "TakesInitiative": {
                    int pid = playerId(objs.get(AbilityKey.Player));
                    addFound(found, latestBy(x -> "designation_change".equals(x.verb) && x.player == pid));
                    break;
                }
                case "Shuffled": {
                    int pid = playerId(objs.get(AbilityKey.Player));
                    addFound(found, latestBy(x -> "shuffle".equals(x.verb) && x.player == pid));
                    break;
                }
                case "ChangesController":
                    cardIds(objs.get(AbilityKey.Card), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, set("change_control")));
                    }
                    break;
                case "Attached":
                case "Unattached":
                    cardIds(objs.get(AbilityKey.AttachSource), cards, 0);
                    cardIds(objs.get(AbilityKey.Card), cards, 0);
                    for (int c : cards) {
                        addFound(found, latest(c, set("attach")));
                    }
                    break;
                default:
                    break;
            }
            if (found.isEmpty() && kind == null) {
                // Generic: the latest event of any triggering card, or the cast of any triggering ability.
                List<Integer> any = new ArrayList<>();
                for (Map.Entry<AbilityKey, Object> e : objs.entrySet()) {
                    if (e.getValue() instanceof SpellAbility) {
                        addFound(found, castRecs.get(((SpellAbility) e.getValue()).getId()));
                    } else if (e.getKey() != AbilityKey.LastStateBattlefield
                            && e.getKey() != AbilityKey.LastStateGraveyard) {
                        cardIds(e.getValue(), any, 0);
                    }
                }
                for (int c : any) {
                    addFound(found, latest(c, null));
                }
                if (!found.isEmpty()) {
                    GENERIC.merge(mode, 1, Integer::sum);
                }
            }
        } catch (Throwable t) {
            error(t);
        }
        int[] stat = TRIGGERS.computeIfAbsent(mode, k -> new int[2]);
        stat[0]++;
        if (!found.isEmpty()) {
            stat[1]++;
            for (Rec f : found) {
                if (!r.causes.contains(f)) {
                    r.causes.add(f);
                }
            }
            r.causeKind = "pointer";
        } else {
            r.causeKind = kind != null ? kind : "unknown";
            if (kind != null) {
                stat[1]++;
            }
        }
    }

    private static void addFound(List<Rec> found, Rec r) {
        if (r != null) {
            Rec c = r.canonical();
            if (!found.contains(c)) {
                found.add(c);
            }
        }
    }

    /** Damage triggers: every damage event from the source(s) to the target(s) in the latest damage step. */
    private void matchDamage(Map<AbilityKey, Object> objs, List<Rec> found) {
        List<Integer> sources = new ArrayList<>();
        cardIds(objs.get(AbilityKey.DamageSource), sources, 0);
        cardIds(objs.get(AbilityKey.DamageSources), sources, 0);
        List<Integer> tcards = new ArrayList<>();
        cardIds(objs.get(AbilityKey.DamageTarget), tcards, 0);
        cardIds(objs.get(AbilityKey.DamageTargets), tcards, 0);
        Set<Integer> tplayers = new HashSet<>();
        for (AbilityKey k : new AbilityKey[] {AbilityKey.DamageTarget, AbilityKey.DamageTargets}) {
            Object o = objs.get(k);
            if (playerId(o) >= 0) {
                tplayers.add(playerId(o));
            } else if (o instanceof Iterable) {
                for (Object x : (Iterable<?>) o) {
                    if (playerId(x) >= 0) {
                        tplayers.add(playerId(x));
                    }
                }
            }
        }
        Object map = objs.get(AbilityKey.DamageMap);
        if (map instanceof com.google.common.collect.Table) {
            for (com.google.common.collect.Table.Cell<?, ?, ?> c : ((com.google.common.collect.Table<?, ?, ?>) map).cellSet()) {
                cardIds(c.getRowKey(), sources, 0);
                if (playerId(c.getColumnKey()) >= 0) {
                    tplayers.add(playerId(c.getColumnKey()));
                } else {
                    cardIds(c.getColumnKey(), tcards, 0);
                }
            }
        }
        long step = -1;
        for (int i = damageLog.size() - 1; i >= 0 && i >= damageLog.size() - 60; i--) {
            Rec d = damageLog.get(i);
            boolean srcOk = sources.isEmpty() || sources.contains(d.source);
            boolean tgtOk = tcards.isEmpty() && tplayers.isEmpty()
                    || tcards.contains(d.targetCard) || tplayers.contains(d.targetPlayer);
            if (!srcOk || !tgtOk) {
                continue;
            }
            if (step < 0) {
                step = d.step;
            } else if (d.step != step) {
                break;
            }
            addFound(found, d);
        }
    }

    // ---- coverage ------------------------------------------------------------

    private static final Map<String, Integer> SEEN = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, Integer> KEPT = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, Integer> UNMAPPED = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, Integer> VERBS = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, Integer> CAUSE_KIND = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, Integer> SUPERSEDED = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, Integer> GENERIC = Collections.synchronizedMap(new TreeMap<>());
    private static final Map<String, int[]> TRIGGERS = Collections.synchronizedMap(new TreeMap<>());
    private static final java.util.concurrent.atomic.AtomicLong WALKS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong WALK_NS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong COST_SCOPE = new java.util.concurrent.atomic.AtomicLong();
    /** handles allocated, lost (identity phantom), retired (untracked), inconsistent. */
    private static final int[] TRACKING = new int[4];

    public static void printCoverage() {
        if (SEEN.isEmpty()) {
            return;
        }
        System.out.println("\n=== game event bus coverage ===");
        synchronized (SEEN) {
            for (Map.Entry<String, Integer> e : SEEN.entrySet()) {
                String k = e.getKey();
                String how = KEPT.containsKey(k) ? "KEPT" : UNMAPPED.containsKey(k) ? "UNMAPPED" : "dropped";
                System.out.printf("  %-38s seen=%-8d %s%n", k, e.getValue(), how);
            }
        }
        System.out.println("\n=== history events (records, before per-viewer filtering) ===");
        System.out.println("  verbs:       " + VERBS);
        System.out.println("  cause_kind:  " + CAUSE_KIND);
        System.out.println("  superseded moves (covered by cast/land play): " + SUPERSEDED);
        System.out.printf("  stack walks: %d, mean %.1f us; cost-stack causes: %d%n", WALKS.get(),
                WALKS.get() == 0 ? 0.0 : WALK_NS.get() / 1000.0 / WALKS.get(), COST_SCOPE.get());
        System.out.println("  triggers by mode (matched/total):");
        synchronized (TRIGGERS) {
            for (Map.Entry<String, int[]> e : TRIGGERS.entrySet()) {
                System.out.printf("    %-28s %d/%d%s%n", e.getKey(), e.getValue()[1], e.getValue()[0],
                        GENERIC.containsKey(e.getKey()) ? "  (generic rule: " + GENERIC.get(e.getKey()) + ")" : "");
            }
        }
        System.out.printf("  handles: allocated=%d lost(identity phantom)=%d retired(untracked)=%d inconsistent=%d%n",
                TRACKING[0], TRACKING[1], TRACKING[2], TRACKING[3]);
        System.out.println("  history errors: " + errors);
    }
}
