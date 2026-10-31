package magicnet.bridge;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.TreeMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Multiset;

import forge.game.Game;
import forge.game.GameEntity;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.card.CounterType;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.spellability.TargetChoices;
import forge.game.zone.ZoneType;

/**
 * Builds the intermediate token representation of a game state.
 *
 * This is deliberately a flat list of self-describing JSON-ish records rather than
 * tensors. The embedding layer is a separate, later step. Keeping an inspectable
 * intermediate buys three things:
 *
 *   - states can be diffed and eyeballed when a policy does something strange
 *   - the hidden-information boundary is enforced in exactly one place (here),
 *     so there is no path by which a leak reaches the model without passing
 *     through code that is explicitly about redaction
 *   - the schema can change without touching the encoder
 *
 * Ids in these records are join keys for resolving edges to token indices. They are
 * NOT content features and must not reach the model: Forge assigns ids at card
 * creation, so they correlate with library order and would leak the decklist.
 */
public final class StateExtractor {

    /**
     * Canonical zone order. Fixed because the encoder's segment embeddings are
     * positional with respect to this list; changing it is a schema change.
     *
     * @see Schema#PLAYER_ZONES the single source of truth, shared with the
     *      vocabulary export that the Python side checks itself against
     */
    private static final ZoneType[] PLAYER_ZONES = Schema.PLAYER_ZONES;

    /**
     * Namespaced object references.
     *
     * Card ids and player ids come from independent counters that overlap:
     * {@code Game.nextCardId()} pre-increments and so returns 1 for the first
     * card, while players are 0 and 1. A bare id in an edge is therefore
     * ambiguous, and the relation name does not disambiguate it -- an attack or a
     * target can legitimately point at a player, a planeswalker, or a battle, and
     * the latter two are cards. Prefixing makes the referent kind explicit and
     * keeps edge targets disjoint by construction.
     */
    static String cardRef(Object id) {
        return id == null ? null : "c" + id;
    }

    static String playerRef(Object id) {
        return id == null ? null : "p" + id;
    }

    /**
     * Ref for an entity that may be either. A combat defender is a player, a
     * planeswalker, or a battle; the latter two are cards.
     */
    static String entityRef(GameEntity e) {
        if (e == null) {
            return null;
        }
        if (e instanceof Player) {
            return playerRef(((Player) e).getId());
        }
        if (e instanceof Card) {
            return cardRef(((Card) e).getId());
        }
        return null;
    }

    /** A single token in the intermediate representation. */
    public static final class Tok {
        final String type;
        final String zone;
        final Map<String, Object> props = new LinkedHashMap<>();
        final List<String[]> edges = new ArrayList<>(2);

        Tok(String type, String zone) {
            this.type = type;
            this.zone = zone;
        }

        Tok p(String k, Object v) {
            if (v != null) {
                props.put(k, v);
            }
            return this;
        }

        /**
         * A flag component, emitted only when true: absence means false, so a
         * false value would be a component that carries nothing.
         */
        Tok flag(String k, boolean v) {
            if (v) {
                props.put(k, Boolean.TRUE);
            }
            return this;
        }

        /** Edge to another token, by namespaced ref. Resolved to a token index downstream. */
        Tok edge(String rel, Object targetRef) {
            if (targetRef != null) {
                edges.add(new String[] { rel, String.valueOf(targetRef) });
            }
            return this;
        }

        public String zone() {
            return zone;
        }

        public String type() {
            return type;
        }

        public int edgeCount() {
            return edges.size();
        }

        /**
         * Signature used to test whether two tokens are interchangeable.
         * Deliberately excludes the id (a pure join key) and includes everything
         * the model can actually see.
         */
        String signature() {
            StringBuilder sb = new StringBuilder(type).append('|').append(zone);
            for (Map.Entry<String, Object> e : props.entrySet()) {
                if ("id".equals(e.getKey())) {
                    continue;
                }
                sb.append('|').append(e.getKey()).append('=').append(e.getValue());
            }
            return sb.toString();
        }

        /**
         * Schema 9 layout: {@code type}, {@code zone}, the identity/ref keys
         * ({@link Schema#isTopLevel}), {@code edges}, then every model input
         * under {@code components} -- omitted when there are none.
         */
        public String toJson() {
            StringBuilder sb = new StringBuilder("{\"type\":\"").append(type).append('"');
            if (zone != null) {
                sb.append(",\"zone\":\"").append(zone).append('"');
            }
            boolean anyComponent = false;
            for (Map.Entry<String, Object> e : props.entrySet()) {
                if (Schema.isTopLevel(e.getKey())) {
                    appendField(sb.append(','), e.getKey(), e.getValue());
                } else {
                    anyComponent = true;
                }
            }
            if (!edges.isEmpty()) {
                sb.append(",\"edges\":[");
                for (int i = 0; i < edges.size(); i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append("{\"rel\":\"").append(edges.get(i)[0])
                      .append("\",\"to\":\"").append(edges.get(i)[1]).append("\"}");
                }
                sb.append(']');
            }
            if (anyComponent) {
                sb.append(",\"").append(Schema.COMPONENTS).append("\":{");
                boolean first = true;
                for (Map.Entry<String, Object> e : props.entrySet()) {
                    if (!Schema.isTopLevel(e.getKey())) {
                        if (!first) {
                            sb.append(',');
                        }
                        first = false;
                        appendField(sb, e.getKey(), e.getValue());
                    }
                }
                sb.append('}');
            }
            return sb.append('}').toString();
        }

        private static void appendField(StringBuilder sb, String k, Object v) {
            sb.append('"').append(esc(k)).append("\":");
            if (v instanceof List) {
                // Variable-length string sets (keywords, colors).
                sb.append('[');
                List<?> l = (List<?>) v;
                for (int i = 0; i < l.size(); i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append('"').append(esc(String.valueOf(l.get(i)))).append('"');
                }
                sb.append(']');
            } else if (v instanceof Number || v instanceof Boolean) {
                // NaN/Infinity are not valid JSON; emit them as strings rather
                // than silently producing a line Python cannot parse.
                if (v instanceof Double && !Double.isFinite((Double) v)
                        || v instanceof Float && !Float.isFinite((Float) v)) {
                    sb.append('"').append(v).append('"');
                } else {
                    sb.append(v);
                }
            } else {
                sb.append('"').append(esc(String.valueOf(v))).append('"');
            }
        }

        /**
         * JSON string escape. Must cover all control characters: card rules text
         * contains tabs and other C0 characters, and a raw one makes the emitted
         * line unparseable.
         */
        private static String esc(String s) {
            StringBuilder sb = new StringBuilder(s.length() + 16);
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '\\': sb.append("\\\\"); break;
                    case '"': sb.append("\\\""); break;
                    case '\n': sb.append("\\n"); break;
                    case '\r': sb.append("\\r"); break;
                    case '\t': sb.append("\\t"); break;
                    case '\b': sb.append("\\b"); break;
                    case '\f': sb.append("\\f"); break;
                    default:
                        if (c < 0x20 || c == 0x7f) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                }
            }
            return sb.toString();
        }
    }

    private StateExtractor() {
    }

    /**
     * Extracts the full token list from the viewer's perspective.
     *
     * @param game    live game
     * @param viewer  the player to whom information is visible
     * @param history accumulated action-component history tokens for this game
     */
    public static List<Tok> extract(Game game, Player viewer, List<Tok> history) {
        // Piles the players were shown are classified here rather than at reveal
        // time, because endTempShowCards fires before the cards have moved.
        LibraryKnowledge.resolvePending();

        List<Tok> out = new ArrayList<>(128);

        // 1. History. Unbounded by construction; measuring it is the point.
        if (history != null) {
            out.addAll(history);
        }

        // 2. Global state.
        try {
            out.add(new Tok("global", null)
                    .p("turn", game.getPhaseHandler().getTurn())
                    .p("phase", String.valueOf(game.getPhaseHandler().getPhase()))
                    .flag("active_is_viewer", game.getPhaseHandler().getPlayerTurn() == viewer)
                    .p("stack_size", game.getStack().size()));
        } catch (Throwable ignored) {
            // Instrumentation must never fail a game.
        }

        // 3/4. Players, viewer first so "self" is always at a fixed offset.
        List<Player> ordered = new ArrayList<>();
        ordered.add(viewer);
        for (Player p : game.getPlayers()) {
            if (p != viewer) {
                ordered.add(p);
            }
        }

        for (Player p : ordered) {
            boolean self = (p == viewer);
            String side = self ? "self" : "opp";
            try {
                Tok pt = new Tok("player", null)
                        .p("id", playerRef(p.getId()))
                        .p("side", side)
                        .p("life", p.getLife())
                        .p("lands_played", p.getLandsPlayedThisTurn());
                addCounters(pt, p.getCounters());
                out.add(pt);
            } catch (Throwable ignored) {
                // skip
            }

            for (ZoneType z : PLAYER_ZONES) {
                emitZone(out, game, p, z, side, self, viewer);
            }
        }

        // 5. Stack.
        emitStack(out, game);

        // 6. Delayed triggers (reflection; no public getter).
        emitDelayedTriggers(out, game);

        // 7. Combat edges live on the battlefield tokens; emitted inline above is
        //    not possible because blockers are declared after attackers, so do it here.
        //    (Edges are attached to already-emitted tokens by id.)
        emitCombat(out, game);

        // 8. Continuous-effect edges, once every visible token exists.
        emitAffects(out, game);

        // Legal actions are deliberately not tokens here: they live only in the
        // record's `actions` array (Serializer), the single copy the policy
        // indexes into. The encoder builds candidate tokens from it.

        return out;
    }

    private static void emitZone(List<Tok> out, Game game, Player p, ZoneType z,
                                 String side, boolean self, Player viewer) {
        try {
            Iterable<Card> cards = p.getCardsIn(z);
            if (cards == null) {
                return;
            }
            String zoneName = Schema.zoneName(side, z);

            if (z == ZoneType.Library) {
                emitLibrary(out, p, zoneName, self, viewer);
                return;
            }

            int hiddenSideboard = 0;
            for (Card c : cards) {
                boolean visible = self;
                boolean knownByReveal = false;
                if (!self && z == ZoneType.Sideboard) {
                    // Forge's own rule for the sideboard: hidden from an opponent
                    // unless they may look at the card (e.g. it was revealed).
                    visible = canBeShownTo(c, viewer);
                    if (!visible) {
                        hiddenSideboard++;
                        continue;
                    }
                } else if (!self && z == ZoneType.Hand) {
                    // Only a reveal callback delivered to this viewer creates
                    // this, and it is checked against the live card here.
                    knownByReveal = RevealKnowledge.knows(viewer, c);
                    visible = knownByReveal;
                }
                Tok t = cardToken(c, zoneName, visible, z);
                if (knownByReveal) {
                    // Lets the leak check tell a legitimate reveal from a leak.
                    t.p("known_by_reveal", true);
                } else if (visible && !self) {
                    t.p("revealed", true);
                }
                out.add(t);
                // Mutate/meld components are not zone members in Forge; they hang off
                // the host. They get tokens in a pseudo-zone so that "if this dies,
                // what hits the graveyard" is directly readable, without polluting
                // the battlefield's entry-ordered run (components have no entry
                // timestamp of their own).
                if (z == ZoneType.Battlefield) {
                    emitMergeComponents(out, c, side);
                }
            }
            if (hiddenSideboard > 0) {
                // Hidden opponent sideboard cards collapse to one count, exactly
                // like the hidden remainder of an opponent library and for the
                // same reason: unseen copies share no distinguishing fact, so N
                // identical HIDDEN tokens carry what one count=N token does.
                // Typed library_card rather than game_object because the type
                // denotes the schema, not the zone: library_card is the
                // (identity, count[, position]) aggregate, while a game_object
                // is one object with instance state and an id that edges and
                // options resolve to -- a count=15 game_object would break that.
                // No position: the sideboard is unordered.
                out.add(new Tok("library_card", zoneName)
                        .p("name", "HIDDEN")
                        .p("count", hiddenSideboard));
            }
        } catch (Throwable ignored) {
            // skip zone
        }
    }

    private static void emitMergeComponents(List<Tok> out, Card host, String side) {
        try {
            if (!host.hasMergedCard()) {
                return;
            }
            for (Card comp : host.getMergedCards()) {
                if (comp == host) {
                    continue;
                }
                out.add(new Tok("game_object", Schema.mergedUnder(side))
                        .p("id", cardRef(comp.getId()))
                        .p("name", comp.getName())
                        .p("oracle_id", OracleIds.of(comp))
                        .edge(Schema.MERGED_UNDER, cardRef(host.getId())));
            }
        } catch (Throwable ignored) {
            // skip
        }
    }


    /**
     * Library emission: one token type, {@code (identity, position constraint, count)}.
     *
     * <p>A card whose position the viewer knows is just a count-1 entry, so there is
     * no separate "known position" token type and no special case. Cards whose
     * position is unknown are collapsed by name into a count, which is lossless
     * rather than lossy: "unknown position" means there is no fact distinguishing
     * one copy from another, so N identical tokens carry exactly the information of
     * one token with {@code count=N}. This is unlike the battlefield, where two
     * copies can diverge through damage, auras or summoning sickness.
     *
     * <p>Collapsing also matches the action space. Library choices ("search your
     * library for a card") are over card <em>names</em>; "which of the four Defiles"
     * is not a real choice and surfacing it would add spurious branching.
     *
     * <p>Depth is emitted from both ends. "Put the rest on the bottom in a random
     * order" is routine wording and bottom-referencing effects exist, so
     * {@code depth_from_bottom} saves the model from having to learn
     * {@code library_size - depth} as a subtraction. No bucketing: the range is
     * bounded, so every depth gets its own embedding and the model learns which
     * ones matter rather than us asserting it up front.
     *
     * <p><b>Own vs opponent library.</b> The viewer knows their own decklist, so the
     * remaining multiset of their library is derivable from public information and
     * the name-level histogram is legitimate. They do <em>not</em> know the
     * opponent's decklist, so an opponent library collapses to a bare count plus
     * whatever specific cards the viewer has separately learned.
     */
    // Position knowledge is carried by which depth fields are present:
    // depth_from_top/bottom (exact), depth_min/max (range), none (unknown).
    private static void emitLibrary(List<Tok> out, Player owner, String zoneName,
                                    boolean self, Player viewer) {
        List<Card> lib = new ArrayList<>();
        for (Card c : owner.getCardsIn(ZoneType.Library)) {
            lib.add(c);
        }
        final int n = lib.size();
        StateStats.recordUncompressedLibrary(n);

        // Group sizes are needed before emitting, because an unordered group's
        // bounds are derived from the group's size and the library size -- never
        // from the true index, which the viewer does not know.
        int nTop = 0;
        int nBottom = 0;
        for (Card c : lib) {
            LibraryKnowledge.Constraint k = LibraryKnowledge.constraintFor(viewer, c);
            if (k == LibraryKnowledge.Constraint.GROUP_TOP) {
                nTop++;
            } else if (k == LibraryKnowledge.Constraint.GROUP_BOTTOM) {
                nBottom++;
            }
        }

        int known = 0;
        for (int i = 0; i < n; i++) {
            Card c = lib.get(i);
            LibraryKnowledge.Constraint k = LibraryKnowledge.constraintFor(viewer, c);
            if (k == null) {
                continue;
            }
            known++;
            Tok t = new Tok("library_card", zoneName)
                    .p("id", cardRef(c.getId()))
                    .p("name", c.getName())
                    .p("oracle_id", OracleIds.of(c))
                    .p("count", 1);
            switch (k) {
                case EXACT:
                    // Index 0 is the top: Player.doDraw takes get(0), and the
                    // "draw from the bottom instead" keyword takes get(size()-1).
                    t.p("depth_from_top", i)
                     .p("depth_from_bottom", n - 1 - i);
                    break;
                case GROUP_TOP:
                    t.p("depth_min", 0)
                     .p("depth_max", Math.max(0, nTop - 1));
                    break;
                case GROUP_BOTTOM:
                    t.p("depth_min", Math.max(0, n - nBottom))
                     .p("depth_max", Math.max(0, n - 1));
                    break;
                default:
                    break;
            }
            LibraryKnowledge.Arranging arr = LibraryKnowledge.arrangingFor(viewer, c);
            if (arr != null) {
                t.p("arranged_pile", arr.pile);
                if (arr.rank >= 0) {
                    t.p("arranged_rank", arr.rank);
                }
            }
            out.add(t);
        }

        int unknown = n - known;
        if (unknown <= 0) {
            return;
        }

        if (!self) {
            out.add(new Tok("library_card", zoneName)
                    .p("name", "HIDDEN")
                    .p("count", unknown));
            return;
        }

        // TreeMap so the histogram order is deterministic across extractions;
        // a reordering would otherwise show up as spurious state churn in diffs.
        Map<String, Integer> hist = new TreeMap<>();
        Map<String, Card> example = new java.util.HashMap<>();
        for (Card c : lib) {
            if (LibraryKnowledge.constraintFor(viewer, c) != null) {
                continue;
            }
            hist.merge(c.getName(), 1, Integer::sum);
            example.putIfAbsent(c.getName(), c);
        }
        for (Map.Entry<String, Integer> e : hist.entrySet()) {
            out.add(new Tok("library_card", zoneName)
                    .p("name", e.getKey())
                    .p("oracle_id", OracleIds.of(example.get(e.getKey())))
                    .p("count", e.getValue()));
        }
    }

    private static boolean canBeShownTo(Card c, Player viewer) {
        try {
            // Forge decides sideboard visibility from PlayerView's opponent list,
            // which only HostedMatch (or Harness) populates. Fail closed if not.
            if (c.getController() != null && c.getController().isOpponentOf(viewer)
                    && !c.getController().getView().isOpponentOf(viewer.getView())) {
                return false;
            }
            return c.getView().canBeShownTo(viewer.getView());
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static Tok cardToken(Card c, String zoneName, boolean visible, ZoneType z) {
        Tok t = new Tok("game_object", zoneName);
        try {
            t.p("id", cardRef(c.getId()));

            // Redaction happens HERE, at extraction, not at embedding. A hidden card
            // must never have its name written into the intermediate record, or the
            // log itself becomes a leak path.
            boolean hidden = !visible && (z == ZoneType.Hand || z == ZoneType.Sideboard);
            if (hidden || (c.isFaceDown() && !visible)) {
                t.p("name", "HIDDEN");
            } else {
                t.p("name", c.getName());
                t.p("oracle_id", OracleIds.of(c));
                t.p("state", String.valueOf(c.getCurrentStateName()));
            }

            if (z == ZoneType.Battlefield) {
                t.flag("tapped", c.isTapped());
                t.flag("sick", c.isSick());
                t.flag("face_down", c.isFaceDown());
                // Phased-out permanents are still on the battlefield per the rules and
                // Forge only filters them on request, so this must be explicit.
                t.flag("phased_out", c.isPhasedOut());
                if (c.isToken()) {
                    t.p("token", true);
                }
                if (c.getType() != null && c.getType().isCreature()) {
                    t.p("power", c.getNetPower());
                    t.p("toughness", c.getNetToughness());
                    // After copy (layer 1) and characteristic-defining / setting
                    // effects (7a-b), before pumps and counters (7c-d).
                    t.p("base_power", c.getCurrentPower());
                    t.p("base_toughness", c.getCurrentToughness());
                    t.p("damage", c.getDamage());
                }
                addCounters(t, c.getCounters());
                Card att = c.getAttachedTo();
                if (att != null) {
                    t.edge(Schema.ATTACHED_TO, cardRef(att.getId()));
                }
                // Battlefield zone buckets are keyed by CONTROLLER, so a stolen
                // permanent is indistinguishable from a native one -- yet when it
                // dies it goes to its OWNER's graveyard (CR 404.1), handing the
                // recursion target to the other player.
                //
                // Emitted only on divergence. Absence is unambiguous rather than
                // merely terse: the zone bucket already states the controller, so
                // "no owner_is" can only mean owner == controller. Emitting it
                // unconditionally would add one edge per game object (~28 per
                // state) against a current mean of 6, inflating the O(E)
                // attention-bias pathway ~5x to encode the default case.
                //
                // Only the battlefield can diverge: by CR 404.1 and friends, cards
                // in a graveyard/hand/library are already in their owner's zone.
                Player owner = c.getOwner();
                Player ctrl = c.getController();
                if (owner != null && ctrl != null && !owner.equals(ctrl)) {
                    t.edge(Schema.OWNER_IS, playerRef(owner.getId()));
                }
                addCharacteristics(t, c);
                addTempPT(t, c);
                if (!c.isFaceDown()) {
                    addChoices(t, c);
                }
            }
        } catch (Throwable ignored) {
            // partial token is still useful
        }
        return t;
    }

    /**
     * Layer-resolved characteristics of a battlefield object (schema 7).
     *
     * <p>{@code types} and {@code colors} are the CURRENT values after continuous
     * effects; the printed values are recoverable from {@code oracle_id}, so they
     * are not duplicated. Likewise the printed abilities: {@code gained} and
     * {@code lost} (schema 10) carry only the engine-resolved difference from
     * them, as normalized rules text -- see {@link AbilityText}.
     *
     * <p>Face-down objects are safe by construction: Forge's current state for
     * them is the face-down characteristics (nameless colorless Creature), which
     * every player can see. The original face is never read.
     */
    private static void addCharacteristics(Tok t, Card c) {
        try {
            if (c.getType() != null) {
                t.p("types", c.getType().toString());
            }
        } catch (Throwable ignored) {
            // skip
        }
        try {
            List<String> colors = new ArrayList<>();
            for (forge.card.MagicColor.Color col : c.getColor()) {
                colors.add(col.getName().toLowerCase());
            }
            t.p("colors", colors);
        } catch (Throwable ignored) {
            // skip
        }
        try {
            AbilityText.Diff d = AbilityText.of(c);
            if (!d.gained.isEmpty()) {
                t.p("gained", d.gained);
            }
            if (!d.lost.isEmpty()) {
                t.p("lost", d.lost);
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    /**
     * {@code temp_power}/{@code temp_toughness}: the summed P/T boosts whose
     * static-ability id is 0. HEURISTIC: Forge files boosts from statics (lords,
     * anthems, equipment) under the static's id and boosts applied by resolving
     * spells/abilities (Giant Growth) under 0 -- which in practice are "until
     * end of turn". Exact durations are not read (only reachable by reflection
     * into end-of-turn command lists). Already included in power/toughness;
     * omitted when zero.
     */
    private static void addTempPT(Tok t, Card c) {
        try {
            if (c.getType() == null || !c.getType().isCreature()) {
                return;
            }
            int p = 0;
            int q = 0;
            for (com.google.common.collect.Table.Cell<Long, Long, org.apache.commons.lang3.tuple.Pair<Integer, Integer>> cell
                    : c.getPTBoostTable().cellSet()) {
                if (cell.getColumnKey() != null && cell.getColumnKey() == 0L && cell.getValue() != null) {
                    p += cell.getValue().getLeft() == null ? 0 : cell.getValue().getLeft();
                    q += cell.getValue().getRight() == null ? 0 : cell.getValue().getRight();
                }
            }
            if (p != 0) {
                t.p("temp_power", p);
            }
            if (q != 0) {
                t.p("temp_toughness", q);
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    /**
     * Persistent choices on a face-up battlefield object. Read through the
     * public {@link forge.game.card.CardView}, never the Card: Forge keeps
     * secret choices (Stalking Leonin's chosen player) on the Card only, and
     * publishes them to the view when revealed. Cast-time facts ({@code x_paid},
     * {@code kicked}) were public on the stack; absent if not cast.
     */
    private static void addChoices(Tok t, Card c) {
        CardView v;
        try {
            v = c.getView();
        } catch (Throwable ignored) {
            return;
        }
        try {
            List<String> cc = v.getChosenColors();
            if (cc != null && !cc.isEmpty()) {
                List<String> colors = new ArrayList<>();
                for (String s : cc) {
                    colors.add(s.toLowerCase());
                }
                t.p("chosen_colors", colors);
            }
        } catch (Throwable ignored) {
            // skip
        }
        try {
            String ty = v.getChosenType();
            if (ty != null && !ty.isEmpty()) {
                t.p("chosen_type", ty);
            }
        } catch (Throwable ignored) {
            // skip
        }
        try {
            List<String> named = v.getNamedCard();
            if (named != null && !named.isEmpty()) {
                String n = named.get(named.size() - 1);
                if (n != null && !n.isEmpty()) {
                    t.p("named_card", n);
                    t.p("named_card_oracle_id", OracleIds.ofShownName(n));
                }
            }
        } catch (Throwable ignored) {
            // skip
        }
        try {
            PlayerView pv = v.getChosenPlayer();
            if (pv != null) {
                t.p("chosen_player", playerRef(pv.getId()));
            }
        } catch (Throwable ignored) {
            // skip
        }
        try {
            String num = v.getChosenNumber();
            if (num != null && !num.isEmpty()) {
                t.p("chosen_number", Integer.parseInt(num.trim()));
            }
        } catch (Throwable ignored) {
            // skip
        }
        try {
            SpellAbility cast = c.getCastSA();
            if (cast != null) {
                Integer x = cast.getXManaCostPaid();
                if (x != null) {
                    t.p("x_paid", x);
                }
                if (cast.isKicked()) {
                    t.p("kicked", true);
                }
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    /**
     * {@code affects} edges (schema 7): from the source of each active
     * continuous static effect to every object/player it currently applies to.
     *
     * <p>Carried on the SOURCE token, like {@code attached_to} on the aura and
     * {@code source_is} on the stack item. Effect cards created by resolved
     * spells (Command-zone "X's Effect") are mapped to their real source; if
     * that has no visible token the edge is dropped. Both ends must be visible
     * to the viewer: Forge's affected sets include hidden cards (Painter's
     * Servant lists library cards), so only named {@code game_object} tokens
     * outside the opponent's hidden zones, and player tokens, qualify.
     *
     * <p>No self-loops: a self-affecting static's result is already in the
     * object's own props, and "this card affects itself" is a property of its
     * oracle text, not of the board. Duplicates (two statics of one source on
     * one target, Adaptive Automaton's self+others pair) collapse to one edge.
     */
    private static void emitAffects(List<Tok> out, Game game) {
        try {
            Map<String, Tok> visible = new java.util.HashMap<>();
            for (Tok t : out) {
                Object id = t.props.get("id");
                if (id == null) {
                    continue;
                }
                if ("player".equals(t.type)
                        || ("game_object".equals(t.type) && isVisibleObject(t))) {
                    visible.put(String.valueOf(id), t);
                }
            }
            Map<String, java.util.LinkedHashSet<String>> edges = new LinkedHashMap<>();
            for (StaticEffect se : game.getStaticEffects().getEffects()) {
                try {
                    Card src = se.getSource();
                    if (src == null) {
                        continue;
                    }
                    Card real = src.getEffectSource();
                    if (real == null) {
                        real = src;
                    }
                    String from = cardRef(real.getId());
                    if (!visible.containsKey(from)) {
                        continue;
                    }
                    java.util.LinkedHashSet<String> to =
                            edges.computeIfAbsent(from, k -> new java.util.LinkedHashSet<>());
                    if (se.getAffectedCards() != null) {
                        for (Card a : se.getAffectedCards()) {
                            String ref = cardRef(a.getId());
                            if (!ref.equals(from) && visible.containsKey(ref)) {
                                to.add(ref);
                            }
                        }
                    }
                    if (se.getAffectedPlayers() != null) {
                        for (Player p : se.getAffectedPlayers()) {
                            String ref = playerRef(p.getId());
                            if (visible.containsKey(ref)) {
                                to.add(ref);
                            }
                        }
                    }
                } catch (Throwable ignored) {
                    // skip effect
                }
            }
            for (Map.Entry<String, java.util.LinkedHashSet<String>> e : edges.entrySet()) {
                Tok src = visible.get(e.getKey());
                for (String ref : e.getValue()) {
                    src.edge(Schema.AFFECTS, ref);
                }
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    private static boolean isVisibleObject(Tok t) {
        Object name = t.props.get("name");
        if (name == null || "HIDDEN".equals(name)) {
            return false;
        }
        String z = t.zone;
        return z != null && !z.startsWith("opp_hand") && !z.startsWith("opp_sideboard")
                && !z.endsWith("_library");
    }

    private static void addCounters(Tok t, Multiset<CounterType> counters) {
        if (counters == null || counters.isEmpty()) {
            return;
        }
        for (Multiset.Entry<CounterType> e : counters.entrySet()) {
            try {
                t.p("ctr_" + e.getElement().getName().toLowerCase(), e.getCount());
            } catch (Throwable ignored) {
                // skip counter
            }
        }
    }

    private static void emitStack(List<Tok> out, Game game) {
        try {
            for (SpellAbilityStackInstance si : game.getStack()) {
                Tok t = new Tok("stack_item", Schema.STACK)
                        .p("id", "s" + si.getId())
                        .flag("is_spell", si.isSpell())
                        .flag("is_trigger", si.isTrigger());
                try {
                    Card src = si.getSourceCard();
                    if (src != null) {
                        t.p("source", src.getName());
                        t.p("source_oracle_id", OracleIds.of(src));
                        t.edge(Schema.SOURCE_IS, cardRef(src.getId()));
                    }
                } catch (Throwable ignored) {
                    // skip
                }
                try {
                    TargetChoices tc = si.getTargetChoices();
                    if (tc != null) {
                        for (Card tgt : tc.getTargetCards()) {
                            t.edge(Schema.TARGETING, cardRef(tgt.getId()));
                        }
                        for (Player tp : tc.getTargetPlayers()) {
                            t.edge(Schema.TARGETING, playerRef(tp.getId()));
                        }
                        for (SpellAbility ts : tc.getTargetSpells()) {
                            if (ts.getHostCard() != null) {
                                t.edge(Schema.TARGETING_SPELL, cardRef(ts.getHostCard().getId()));
                            }
                        }
                    }
                } catch (Throwable ignored) {
                    // skip
                }
                out.add(t);
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    private static void emitCombat(List<Tok> out, Game game) {
        try {
            Combat combat = game.getCombat();
            if (combat == null) {
                return;
            }
            Map<String, Tok> byId = new LinkedHashMap<>();
            for (Tok t : out) {
                Object id = t.props.get("id");
                if (id != null && "game_object".equals(t.type)) {
                    byId.put(String.valueOf(id), t);
                }
            }
            for (Card atk : combat.getAttackers()) {
                Tok at = byId.get(cardRef(atk.getId()));
                if (at == null) {
                    continue;
                }
                GameEntity def = combat.getDefenderByAttacker(atk);
                if (def != null) {
                    at.edge(Schema.ATTACKING, entityRef(def));
                }
                // CR 509.1h: "A creature remains blocked even if all the creatures
                // blocking it are removed from combat", and CR 510.1c: a blocked
                // creature with no blockers left "assigns no combat damage". So
                // blocked-ness is sticky status, NOT derivable from the live blocker
                // list -- kill the chump blocker and the attacker still deals nothing.
                // Forge agrees: Combat.isBlocked reads a stored band flag, separate
                // from getBlockers(). This is a unary fact about one creature, so it
                // is a property rather than a relation: it distinguishes the two
                // token multisets directly, which is exactly what an edge would
                // otherwise have to do indirectly.
                at.p("blocked", combat.isBlocked(atk));
                for (Card blk : combat.getBlockers(atk)) {
                    at.edge(Schema.BLOCKED_BY, cardRef(blk.getId()));
                    Tok bt = byId.get(cardRef(blk.getId()));
                    if (bt != null) {
                        bt.edge(Schema.BLOCKING, cardRef(atk.getId()));
                    }
                }
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

    /**
     * Delayed triggers are held in three private lists on TriggerHandler with no
     * public accessor, so this is reflection by necessity. They matter because
     * "exiled, returns at end of turn" is invisible from the exiled card itself.
     */
    private static void emitDelayedTriggers(List<Tok> out, Game game) {
        try {
            Object th = game.getTriggerHandler();
            for (String field : new String[] { "delayedTriggers", "thisTurnDelayedTriggers" }) {
                try {
                    java.lang.reflect.Field f = th.getClass().getDeclaredField(field);
                    f.setAccessible(true);
                    Object v = f.get(th);
                    if (!(v instanceof List)) {
                        continue;
                    }
                    for (Object trig : (List<?>) v) {
                        Tok t = new Tok("delayed_trigger", Schema.DELAYED);
                        try {
                            java.lang.reflect.Method m = trig.getClass().getMethod("getHostCard");
                            Object host = m.invoke(trig);
                            if (host instanceof Card) {
                                t.p("source", ((Card) host).getName());
                                t.p("source_oracle_id", OracleIds.of((Card) host));
                                t.edge(Schema.SOURCE_IS, cardRef(((Card) host).getId()));
                            }
                        } catch (Throwable ignored) {
                            // skip host
                        }
                        t.p("list", field);
                        out.add(t);
                    }
                } catch (NoSuchFieldException ignored) {
                    // Forge version drift; surfaced by the stats as zero delayed triggers.
                }
            }
        } catch (Throwable ignored) {
            // skip
        }
    }

}
