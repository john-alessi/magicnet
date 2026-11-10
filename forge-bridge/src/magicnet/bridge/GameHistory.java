package magicnet.bridge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetChoices;
import forge.game.zone.ZoneType;

/**
 * One viewer's game history: the event tokens, and the tracking handles their
 * edges point at.
 *
 * <p>The events themselves are built once per game by {@link HistoryHub}, which
 * subscribes to Forge's event bus and works out what happened and why (verb,
 * modifiers, participants, causes). This class turns each record into this
 * viewer's token, redacted at the moment the event fires: observation is
 * immutable, so a card drawn unseen and revealed later never leaks backwards,
 * and a card seen on the stack that went back to hand stays seen. One
 * history per viewer makes a cross-agent leak structurally impossible rather
 * than merely guarded against -- both agents live in one process in self-play.
 *
 * <h2>Handles (schema 13)</h2>
 *
 * History is append-only and delta-encoded, so an event's edges are frozen when
 * written. They must still follow the object as a player would: the creature
 * we bounced is the card now in the opponent's hand. So history edges point at
 * per-viewer <em>tracking handles</em> ({@code h<N>}), one per physical object
 * the viewer can follow, and every state assigns each handle referenced so far
 * to exactly one current token ({@link #decorate}), via that token's top-level
 * {@code handles} list:
 * <ol>
 *   <li>the object's own token when the viewer can see it;</li>
 *   <li>the aggregate token of the hidden zone it is known to be in (opponent
 *       hand, a library), never an individual hidden card;</li>
 *   <li>the owner's {@code untracked} token once the viewer can no longer tell
 *       which zone it is in (an unidentified card left the zone it was in);</li>
 *   <li>a {@code phantom} token (last-known characteristics) once it ceased
 *       to exist.</li>
 * </ol>
 * A participant the viewer cannot identify gets no edge at all.
 */
public final class GameHistory {

    /**
     * One proposed cast or activation, from announcement until Forge either
     * puts it on the stack or refuses it.
     *
     * <p>Every history event fired while an attempt is open is tagged with its
     * id, and the attempt is closed by an {@code attempt_end} token carrying the
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

    // ---- per viewer ----------------------------------------------------

    private final List<StateExtractor.Tok> history =
            Collections.synchronizedList(new ArrayList<>());

    final Game game;
    final Player viewer;
    /** Player id of the viewer, or -1 for a viewerless (test) history. */
    final int vid;

    /** Record -> this viewer's history index; also the cause-edge target. */
    private final Map<HistoryHub.Rec, Integer> index = new IdentityHashMap<>();

    private long lastGlobalStep = Long.MIN_VALUE;
    private int step = -1;
    private int stepOrder;

    private int redactions;

    public GameHistory(Game game, Player viewer) {
        this.game = game;
        this.viewer = viewer;
        this.vid = viewer == null ? -1 : viewer.getId();
    }

    public List<StateExtractor.Tok> tokens() {
        return history;
    }

    public int size() {
        return history.size();
    }

    public int redactions() {
        return redactions;
    }

    /** This viewer's history index of a record, or null if it was not shown to them. */
    Integer indexOf(HistoryHub.Rec r) {
        return r == null ? null : index.get(r.canonical());
    }

    /** Renders one record as this viewer's next history token. */
    void append(HistoryHub.Rec r) {
        List<String[]> edges = new ArrayList<>();
        int objs = 0;
        for (HistoryHub.Part pt : r.parts) {
            String ref = pt.ref(vid);
            if (ref == null) {
                if (pt.card) {
                    redactions++;
                }
                continue;
            }
            if (Schema.OBJ.equals(pt.role)) {
                objs++;
            }
            edges.add(new String[] {pt.role, ref});
        }
        if (r.privateMove && objs == 0) {
            // A reorder inside a hidden zone nobody but the arranging player can
            // follow (scry to the bottom): the public summary event covers it.
            return;
        }
        int n = history.size();
        if (r.step != lastGlobalStep) {
            lastGlobalStep = r.step;
            step++;
            stepOrder = 0;
        }
        StateExtractor.Tok t = new StateExtractor.Tok(Schema.HISTORY, Schema.HISTORY)
                .p("id", "e" + n)
                .p("verb", r.verb)
                .p("turn", r.turn)
                .p("phase", r.phase)
                .p("step", step)
                .p("step_order", stepOrder++);
        if (r.fromZone != null) {
            t.p("from", zoneName(r.fromZone, r.fromOwner));
        }
        if (r.toZone != null) {
            t.p("to", zoneName(r.toZone, r.toOwner));
        }
        for (Map.Entry<String, Object> m : r.mods.entrySet()) {
            t.p(m.getKey(), m.getValue());
        }
        if (r.count > 1 || r.alwaysCount) {
            t.p("count", r.count);
        }
        if (r.attempt != null) {
            t.p("attempt", r.attempt);
        }
        for (String[] e : edges) {
            String role = e[0];
            t.edge(role, e[1]);
            if (e[1].startsWith("h")) {
                referenced.add(e[1]);
            }
        }
        int causes = 0;
        for (HistoryHub.Rec c : r.causes) {
            Integer i = indexOf(c);
            if (i != null && i < n) {
                t.edge(Schema.CAUSE, "e" + i);
                causes++;
            }
        }
        String kind = r.causeKind == null ? "unknown" : r.causeKind;
        if ("pointer".equals(kind) && causes == 0) {
            kind = "unknown";
        }
        t.p("cause_kind", kind);
        index.put(r, n);
        history.add(t);
    }

    /**
     * Viewer-relative zone name for history {@code from}/{@code to}, or null for
     * a zone outside the vocabulary (omitted: absent means unknown).
     */
    String zoneName(ZoneType z, int owner) {
        if (z == ZoneType.Stack) {
            return Schema.STACK;
        }
        if (z == ZoneType.Merged) {
            return Schema.mergedUnder(owner == vid ? "self" : "opp");
        }
        for (ZoneType pz : Schema.PLAYER_ZONES) {
            if (pz == z) {
                return Schema.zoneName(owner == vid ? "self" : "opp", z);
            }
        }
        return null;
    }

    // ---- tracking handles ---------------------------------------------

    /** What this viewer last knew about one physical object. */
    static final class Track {
        String handle;
        final int cardId;
        /** Stack-instance id for an ability; -1 for a card. */
        final int abilityId;
        int ownerId = -1;
        /** Whether the viewer knows the object's name (not face-down / unseen). */
        boolean nameKnown;
        /**
         * The viewer knew its identity but can no longer follow this object:
         * the handle now stands for "a copy of X" (an identity phantom). Final:
         * a lost handle never returns to a live token (schema 14).
         */
        boolean lost;
        /** Lost, and the viewer never knew its identity: the owner's untracked token. Final. */
        boolean retired;
        /** Seen to have ceased to exist (a phantom with {@code ceased}). Final, even if Forge reuses the object. */
        boolean ceased;
        ZoneType zone;
        int zoneOwner = -1;
        Snap snap;

        Track(int cardId, int abilityId) {
            this.cardId = cardId;
            this.abilityId = abilityId;
        }
    }

    /** Last-known characteristics, for a phantom. Only taken while the name was visible. */
    static final class Snap {
        String name;
        String oracleId;
        String types;
        Integer power;
        Integer toughness;
        List<String> colors;
        boolean token;
        String ability;
        boolean trigger;
        int controllerId = -1;

        String key() {
            String id = ability != null ? "a|" + ability + "|" + name
                    : token ? "t|" + name + "|" + types + "|" + power + "/" + toughness + "|" + colors
                    : "c|" + (oracleId != null ? oracleId : name);
            return id + "|" + controllerId;
        }
    }

    private int nextHandle;
    private final Map<Integer, Track> tracks = new HashMap<>();
    private final Map<Integer, Track> abilities = new HashMap<>();
    private final Map<String, Track> byHandle = new LinkedHashMap<>();
    /** Handles that appear on at least one emitted history token. */
    private final Set<String> referenced = new HashSet<>();

    /** Tracking diagnostics, per viewer. */
    int widened;
    int retiredCount;
    int inconsistent;

    private String handle(Track t) {
        if (t.handle == null) {
            t.handle = "h" + (nextHandle++);
            byHandle.put(t.handle, t);
        }
        return t.handle;
    }

    private Track fresh(Card c, CardView v) {
        int id = c != null ? c.getId() : v.getId();
        Track t = new Track(id, -1);
        tracks.put(id, t);
        return t;
    }

    private void retire(Track t) {
        if (t != null && !t.retired) {
            t.retired = true;
            retiredCount++;
            if (tracks.get(t.cardId) == t) {
                tracks.remove(t.cardId);
            }
        }
    }

    /**
     * This viewer can no longer follow {@code t}. Schema 14: never re-linked to
     * a card seen later -- that would assert an identity the viewer cannot
     * know. With a known identity the handle becomes an identity phantom
     * ("a copy of X"); without one, the owner's untracked token.
     */
    private void lose(Track t) {
        if (t == null || t.lost || t.retired) {
            return;
        }
        if (t.nameKnown && t.snap != null && t.snap.name != null) {
            t.lost = true;
            widened++;
            if (tracks.get(t.cardId) == t) {
                tracks.remove(t.cardId);
            }
        } else {
            retire(t);
        }
    }

    /**
     * Cards a reveal callback showed this viewer outside a visible zone, with
     * the ability resolving at the time: the card's next move, from that zone
     * while the same ability resolves, is one the viewer can follow (a dig).
     */
    private final Map<Integer, Object[]> shown = new HashMap<>();

    private Object resolving() {
        try {
            return game == null || game.getStack().isEmpty() ? null : game.getStack().peekAbility();
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static boolean isPublic(ZoneType z) {
        if (z == null) {
            return false;
        }
        switch (z) {
            case Battlefield:
            case Graveyard:
            case Exile:
            case Stack:
            case Command:
            case Merged:
                return true;
            default:
                return false;
        }
    }

    private static int ownerOf(forge.game.zone.Zone z) {
        return z == null || z.getPlayer() == null ? -1 : z.getPlayer().getId();
    }

    /** Whether this viewer can see (identify) {@code c}, which is in {@code z}. */
    private boolean canSee(Card c, ZoneType z, int zoneOwner) {
        if (isPublic(z)) {
            return true;
        }
        if (z == ZoneType.Hand || z == ZoneType.Sideboard) {
            if (zoneOwner == vid && vid >= 0) {
                return true;
            }
            return z == ZoneType.Hand && vid >= 0 && c != null
                    && RevealKnowledge.knows(vid, c.getId(), c.getGameTimestamp());
        }
        if (z == ZoneType.Library) {
            return viewer != null && c != null && LibraryKnowledge.constraintFor(viewer, c) != null;
        }
        return false;
    }

    private boolean nameVisible(Card c, CardView v) {
        try {
            boolean fd = c != null ? c.isFaceDown() : v != null && v.isFaceDown();
            if (!fd) {
                return true;
            }
            CardView cv = c != null ? c.getView() : v;
            return viewer != null && cv != null
                    && (viewer.getView().equals(cv.getController()) || cv.mayPlayerLook(viewer.getView()));
        } catch (Throwable ignored) {
            return false;
        }
    }

    /**
     * A card left {@code (z, owner)} and the viewer could not tell which: every
     * track believed there that it could have been is lost. If the departing
     * card was seen by name ({@code departed}), only tracks of that name (and
     * anonymous ones) could have been it; otherwise all of them.
     */
    private void widen(ZoneType z, int owner, Track except, String departed) {
        for (Track t : new ArrayList<>(tracks.values())) {
            if (t == except || t.retired || t.lost || t.zone != z || t.zoneOwner != owner) {
                continue;
            }
            boolean named = t.nameKnown && t.snap != null && t.snap.name != null;
            if (!named || departed == null || departed.equals(t.snap.name)) {
                lose(t);
            }
        }
    }

    /**
     * Updates tracking for a zone change and returns the moved object's handle,
     * or null when this viewer cannot tell which object moved.
     *
     * @param old the pre-move card view (Forge fires the event with it)
     * @param now the card at its destination, or null if it ceased to exist
     */
    String moveRef(CardView old, Card now, ZoneType from, int fromOwner, ZoneType to, int toOwner) {
        try {
            int id = old != null ? old.getId() : now.getId();
            boolean atOrigin = from == null || isPublic(from)
                    || ((from == ZoneType.Hand || from == ZoneType.Sideboard) && fromOwner == vid && vid >= 0)
                    || (from == ZoneType.Library && viewer != null && now != null
                        && LibraryKnowledge.constraintFor(viewer, now) == LibraryKnowledge.Constraint.EXACT);
            boolean atDest = to != null && to != ZoneType.None && canSee(now, to, toOwner);
            boolean namedAtDest = atDest && nameVisible(now, null);
            Object[] seen = shown.remove(id);
            if (seen != null && seen[0] == from && seen[1] == resolving()) {
                // Revealed to this viewer by the effect now moving it.
                atOrigin = true;
            }
            Track t = tracks.get(id);
            boolean moved = from != to || fromOwner != toOwner;
            if (atOrigin) {
                if (t == null) {
                    t = fresh(now, old);
                    t.nameKnown = nameVisible(null, old);
                    snapshot(t, old);
                }
                if (namedAtDest) {
                    t.nameKnown = true;
                }
            } else if (atDest) {
                // Seen arriving but not leaving: the viewer cannot tell it is the
                // card it was tracking, even if the name matches (no re-link).
                lose(t);
                t = fresh(now, old);
                t.nameKnown = namedAtDest;
                if (moved && from != null) {
                    widen(from, fromOwner, t, namedAtDest && now != null ? now.getName() : null);
                }
            } else {
                if (!moved) {
                    // A reorder inside one hidden zone: still there, position unknown.
                    return null;
                }
                if (from != null) {
                    widen(from, fromOwner, null, null);
                }
                lose(t);
                return null;
            }
            t.zone = to;
            t.zoneOwner = toOwner;
            if (now != null) {
                t.ownerId = now.getOwner() == null ? t.ownerId : now.getOwner().getId();
                if (t.nameKnown && atDest && namedAtDest) {
                    snapshot(t, now.getView());
                }
            } else if (old != null && old.getOwner() != null) {
                t.ownerId = old.getOwner().getId();
            }
            return handle(t);
        } catch (Throwable ignored) {
            return null;
        }
    }

    /**
     * Handle for a card that is not moving (a damage source, a target, an
     * attacker), or null if this viewer cannot see it where it is.
     */
    String objRef(Card c) {
        return objRef(c, false);
    }

    /**
     * Handle for a card this viewer was just shown by a reveal callback
     * delivered to them: identified wherever it is, even in a library.
     */
    String revealRef(Card c) {
        return objRef(c, true);
    }

    private String objRef(Card c, boolean shown) {
        if (c == null) {
            return null;
        }
        try {
            forge.game.zone.Zone zone = c.getZone();
            ZoneType z = zone == null ? null : zone.getZoneType();
            int owner = ownerOf(zone);
            if (!shown && !canSee(c, z, owner)) {
                return null;
            }
            boolean named = shown || nameVisible(c, null);
            Track t = tracks.get(c.getId());
            if (t == null) {
                t = fresh(c, null);
                t.nameKnown = named;
            } else if (!t.nameKnown && named) {
                t.nameKnown = true;
            }
            if (shown && !canSee(c, z, owner)) {
                this.shown.put(c.getId(), new Object[] {z, resolving()});
            }
            t.zone = z;
            t.zoneOwner = owner;
            t.ownerId = c.getOwner() == null ? -1 : c.getOwner().getId();
            if (named) {
                snapshot(t, c.getView());
            }
            return handle(t);
        } catch (Throwable ignored) {
            return null;
        }
    }

    /** Handle for an ability on the stack (abilities are public). */
    String abilityRef(int siId, SpellAbility sa, Card host, String text) {
        Track t = abilities.get(siId);
        if (t == null) {
            t = new Track(-1, siId);
            t.nameKnown = true;
            t.zone = ZoneType.Stack;
            abilities.put(siId, t);
            Snap s = new Snap();
            try {
                s.ability = text != null ? text : EventTokenizer.abilityText(sa, game);
                s.trigger = sa.isTrigger();
                if (host != null && nameVisible(host, null)) {
                    s.name = host.getName();
                    s.oracleId = OracleIds.of(host);
                }
                if (sa.getActivatingPlayer() != null) {
                    s.controllerId = sa.getActivatingPlayer().getId();
                }
            } catch (Throwable ignored) {
                // partial snapshot
            }
            t.snap = s;
            t.ownerId = s.controllerId;
        }
        return handle(t);
    }


    private void snapshot(Track t, CardView v) {
        if (v == null) {
            return;
        }
        try {
            if (!nameVisible(null, v) && !(t.nameKnown && t.snap != null)) {
                return;
            }
            if (v.isFaceDown()) {
                return;
            }
            Snap s = new Snap();
            s.name = EventTokenizer.normalize(v.getName(), game);
            s.oracleId = OracleIds.of(v);
            s.token = v.isToken();
            CardView.CardStateView st = v.getCurrentState();
            if (st != null) {
                s.types = st.getType() == null ? null : st.getType().toString();
                if (s.types != null && s.types.isBlank()) {
                    // An effect card ("The Monarch") has an empty type line:
                    // absent, not an empty string component.
                    s.types = null;
                }
                if (st.getType() != null && st.getType().isCreature()) {
                    s.power = st.getPower();
                    s.toughness = st.getToughness();
                }
                if (st.getColors() != null) {
                    List<String> cs = new ArrayList<>();
                    for (forge.card.MagicColor.Color col : st.getColors()) {
                        cs.add(col.getName().toLowerCase());
                    }
                    s.colors = cs;
                }
            }
            if (v.getController() != null) {
                s.controllerId = v.getController().getId();
            }
            t.snap = s;
        } catch (Throwable ignored) {
            // keep the previous snapshot
        }
    }

    // ---- state side: handle assignment ----------------------------------

    /**
     * Assigns every handle referenced by this viewer's history to exactly one
     * state token, adding {@code phantom} and {@code untracked} tokens as
     * needed. Called once per extraction, after all state tokens exist.
     */
    void decorate(List<StateExtractor.Tok> out, int stateFrom) {
        if (referenced.isEmpty()) {
            return;
        }
        Map<String, StateExtractor.Tok> byId = new HashMap<>();
        Map<Integer, StateExtractor.Tok> spellOnStack = new HashMap<>();
        Map<String, StateExtractor.Tok> aggregate = new HashMap<>();
        Map<String, StateExtractor.Tok> selfLibByName = new HashMap<>();
        for (int i = stateFrom; i < out.size(); i++) {
            StateExtractor.Tok t = out.get(i);
            Object id = t.props.get("id");
            if ("stack_item".equals(t.type)) {
                if (id != null) {
                    byId.put(String.valueOf(id), t);
                }
                if (t.spellCard >= 0) {
                    spellOnStack.put(t.spellCard, t);
                }
                continue;
            }
            if (!"game_object".equals(t.type) && !"library_card".equals(t.type)) {
                continue;
            }
            // Schema 15: a hidden-region aggregate carries an "r:<zone>" id, but
            // it is still the aggregate, not an object a handle can resolve to by id.
            if (id != null && !StateExtractor.isRegionRef(id) && !StateExtractor.isNameAggregateRef(id)) {
                if (!isIndividualHidden(t)) {
                    byId.put(String.valueOf(id), t);
                }
            } else if (t.zone != null && t.props.get("name") == null) {
                aggregate.put(t.zone, t);
            } else if ("self_library".equals(t.zone)) {
                selfLibByName.put(String.valueOf(t.props.get("name")), t);
            }
        }
        Map<String, StateExtractor.Tok> phantoms = new LinkedHashMap<>();
        Map<String, StateExtractor.Tok> untracked = new LinkedHashMap<>();
        Map<StateExtractor.Tok, List<String>> assigned = new IdentityHashMap<>();
        for (String h : referenced) {
            Track t = byHandle.get(h);
            if (t == null) {
                continue;
            }
            StateExtractor.Tok target = locate(t, byId, spellOnStack, aggregate, selfLibByName);
            if (target == null) {
                // Ceasing to exist is public (a token off the battlefield, a copy
                // or ability off the stack), even for an object whose zone the
                // viewer had lost: a bounced token is gone, not "somewhere".
                if (t.ceased || isGone(t)) {
                    if (!t.ceased) {
                        t.ceased = true;
                        if (t.abilityId < 0 && tracks.get(t.cardId) == t) {
                            tracks.remove(t.cardId);
                        }
                    }
                    target = phantomTok(phantoms, t, true);
                } else if (t.lost) {
                    target = phantomTok(phantoms, t, false);
                } else {
                    target = untrackedTok(untracked, t);
                }
            }
            assigned.computeIfAbsent(target, k -> new ArrayList<>()).add(h);
        }
        for (StateExtractor.Tok t : untracked.values()) {
            out.add(t);
        }
        for (StateExtractor.Tok t : phantoms.values()) {
            out.add(t);
        }
        for (Map.Entry<StateExtractor.Tok, List<String>> e : assigned.entrySet()) {
            e.getKey().p("handles", e.getValue());
        }
    }

    /** A nameless card token in a hidden zone: never a handle target (that would say which card). */
    static boolean isIndividualHidden(StateExtractor.Tok t) {
        if (!"game_object".equals(t.type) && !"library_card".equals(t.type)) {
            return false;
        }
        if (t.props.get("name") != null || t.zone == null
                || StateExtractor.isRegionRef(t.props.get("id"))) {
            return false;
        }
        return t.zone.endsWith("_hand") || t.zone.endsWith("_library") || t.zone.endsWith("_sideboard");
    }

    private boolean isGone(Track t) {
        if (t.abilityId >= 0) {
            return true;
        }
        return physical(t.cardId) == null;
    }

    private Card physical(int cardId) {
        try {
            return game == null ? null : game.findById(cardId);
        } catch (Throwable ignored) {
            return null;
        }
    }

    private StateExtractor.Tok locate(Track t, Map<String, StateExtractor.Tok> byId,
            Map<Integer, StateExtractor.Tok> spellOnStack, Map<String, StateExtractor.Tok> aggregate,
            Map<String, StateExtractor.Tok> selfLibByName) {
        if (t.retired || t.lost || t.ceased) {
            return null;
        }
        if (t.abilityId >= 0) {
            return byId.get("s" + t.abilityId);
        }
        StateExtractor.Tok direct = byId.get(StateExtractor.cardRef(t.cardId));
        if (direct != null && t.zone != null && direct.zone != null
                && (direct.zone.endsWith("merged_under") || t.zone == ZoneType.Merged)) {
            return direct;
        }
        Card c = physical(t.cardId);
        if (c == null) {
            return null;
        }
        forge.game.zone.Zone zone = c.getZone();
        ZoneType z = zone == null ? null : zone.getZoneType();
        int owner = ownerOf(zone);
        if (z != t.zone || owner != t.zoneOwner) {
            // The object moved without this viewer's tracking seeing it: never
            // reveal where to. Should not happen; counted so it is visible.
            if (!(z == ZoneType.Battlefield && t.zone == ZoneType.Battlefield)) {
                inconsistent++;
                lose(t);
                return null;
            }
            t.zoneOwner = owner;
        }
        StateExtractor.Tok own = byId.get(StateExtractor.cardRef(t.cardId));
        if (own != null) {
            return own;
        }
        if (z == ZoneType.Stack) {
            return spellOnStack.get(t.cardId);
        }
        String side = owner == vid ? "self" : "opp";
        String zn = zoneName(z, owner);
        if (z == ZoneType.Library && "self".equals(side)) {
            StateExtractor.Tok byName = selfLibByName.get(c.getName());
            if (byName != null) {
                return byName;
            }
        }
        if (z == ZoneType.Hand || z == ZoneType.Library || z == ZoneType.Sideboard) {
            return aggregate.get(zn);
        }
        return null;
    }

    private StateExtractor.Tok untrackedTok(Map<String, StateExtractor.Tok> untracked, Track t) {
        int owner = t.ownerId >= 0 ? t.ownerId : t.zoneOwner;
        String zone = Schema.untracked(owner == vid ? "self" : "opp");
        return untracked.computeIfAbsent(zone, z -> new StateExtractor.Tok("library_card", z));
    }

    /**
     * A phantom: last-known characteristics of an object this viewer can no
     * longer point at. {@code ceased}: it ceased to exist (a token, a copy, an
     * ability), deduplicated by (identity, controller). Otherwise an identity
     * phantom -- the viewer lost track of a card whose identity it knew, which
     * still exists somewhere it cannot tell -- deduplicated by (identity, owner).
     */
    private StateExtractor.Tok phantomTok(Map<String, StateExtractor.Tok> phantoms, Track t, boolean ceased) {
        Snap s = t.snap;
        String key = s == null ? "anon|" + t.ownerId
                : ceased ? s.key() : "i|" + (s.oracleId != null ? s.oracleId : s.name) + "|" + t.ownerId;
        return phantoms.computeIfAbsent(key, k -> {
            StateExtractor.Tok p = new StateExtractor.Tok("game_object", Schema.PHANTOM).flag("ceased", ceased);
            if (s == null) {
                return p.p("side", t.ownerId == vid ? "self" : "opp");
            }
            p.p("name", s.name)
             .p("oracle_id", s.oracleId)
             .p("types", s.types)
             .p("power", s.power)
             .p("toughness", s.toughness)
             .p("colors", s.colors)
             .flag("token", s.token)
             .p("ability", s.ability)
             .flag("is_trigger", s.trigger)
             .p("side", (ceased ? s.controllerId : t.ownerId) == vid ? "self" : "opp");
            return p;
        });
    }

    /**
     * {@code affected_by} edges from a battlefield object to the resolve events
     * whose continuous effects still apply to it: every P/T boost, set P/T,
     * keyword change or trait change with static-ability id 0 (created by a
     * resolving spell or ability rather than a static) is matched by its
     * timestamp to the resolution or static-trigger scope that created it.
     */
    List<String> effectEvents(Card c, HistoryHub hub) {
        List<String> out = new ArrayList<>(2);
        if (hub == null || c == null) {
            return out;
        }
        Set<Long> stamps = new java.util.TreeSet<>();
        try {
            collectStamps(stamps, c.getPTBoostTable());
            collectStamps(stamps, c.getSetPTTable());
            collectStamps(stamps, c.getChangedCardKeywords());
            collectStamps(stamps, c.getChangedCardTraits());
        } catch (Throwable ignored) {
            // partial
        }
        for (long ts : stamps) {
            HistoryHub.Rec r = hub.scopeForTimestamp(ts);
            Integer i = indexOf(r);
            if (i != null) {
                String ref = "e" + i;
                if (!out.contains(ref)) {
                    out.add(ref);
                }
            }
        }
        return out;
    }

    private static void collectStamps(Set<Long> out, com.google.common.collect.Table<Long, Long, ?> table) {
        if (table == null) {
            return;
        }
        for (com.google.common.collect.Table.Cell<Long, Long, ?> cell : table.cellSet()) {
            if (cell.getRowKey() != null && cell.getColumnKey() != null && cell.getColumnKey() == 0L) {
                out.add(cell.getRowKey());
            }
        }
    }

    /** Tracking diagnostics for the coverage report. */
    int[] trackingStats() {
        return new int[] {nextHandle, widened, retiredCount, inconsistent};
    }
}
