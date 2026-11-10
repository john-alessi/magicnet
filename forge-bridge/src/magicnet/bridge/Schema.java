package magicnet.bridge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import forge.game.zone.ZoneType;

/**
 * The zone vocabulary, as a single source of truth.
 *
 * <p>Zone names are <em>constructed</em> ({@code side + "_" + zoneType}), not
 * listed, so there is nowhere to read the full set off. That is a problem for
 * the encoder, whose segment embeddings are positional with respect to this
 * vocabulary: a Python-side list transcribed by hand would drift silently and
 * scramble zone semantics rather than failing.
 *
 * <p>Everything here is used by the emitters themselves, so {@link #zoneNames()}
 * cannot disagree with what is actually written. Run
 * {@code java magicnet.bridge.Schema} to print it as JSON; {@code tests/} asserts
 * the Python copy matches.
 *
 * <p>Order is fixed. Changing it is a schema change, and invalidates any
 * checkpoint whose segment embeddings were trained against the old order.
 */
public final class Schema {

    private Schema() {
    }

    /** Viewer-relative sides. "self" is always the viewer. */
    public static final String[] SIDES = {"self", "opp"};

    /**
     * Canonical zone order. Fixed because the encoder's segment embeddings are
     * positional with respect to this list; changing it is a schema change.
     */
    public static final ZoneType[] PLAYER_ZONES = {
            ZoneType.Battlefield,
            ZoneType.Hand,
            ZoneType.Graveyard,
            ZoneType.Exile,
            ZoneType.Command,
            ZoneType.Library,
            // Schema 4. Represented even in best-of-1: wishes and companions
            // bring cards into the game from here.
            ZoneType.Sideboard,
    };

    /**
     * Mutate/meld components are not zone members in Forge -- they hang off the
     * host -- so they get a pseudo-zone rather than polluting the battlefield.
     */
    public static final String MERGED_UNDER_SUFFIX = "_merged_under";

    public static final String STACK = "stack";
    public static final String DELAYED = "delayed";
    /** No longer emitted in state (schema 3); kept for encoder-built candidate tokens. */
    public static final String ACTIONS = "actions";
    public static final String HISTORY = "history";
    /**
     * Schema 13. Last-known characteristics of objects that ceased to exist (a
     * token that left the battlefield, a copy or an ability that left the
     * stack), kept while a history edge still refers to them.
     */
    public static final String PHANTOM = "phantom";
    /** Schema 13. Suffix of the per-side token holding handles whose object the viewer lost track of. */
    public static final String UNTRACKED_SUFFIX = "_untracked";

    public static String untracked(String side) {
        return side + UNTRACKED_SUFFIX;
    }

    /** The zone name for a player zone, viewer-relative. */
    public static String zoneName(String side, ZoneType z) {
        return side + "_" + z.toString().toLowerCase();
    }

    public static String mergedUnder(String side) {
        return side + MERGED_UNDER_SUFFIX;
    }

    /**
     * Every zone name that can appear on a token, in canonical order.
     *
     * <p>Excludes the unzoned tokens ({@code global}, {@code player}), which
     * carry {@code "zone": null} and need their own slot in the encoder -- a
     * "no zone" embedding, not a missing lookup.
     */
    public static List<String> zoneNames() {
        List<String> out = new ArrayList<>();
        for (String side : SIDES) {
            for (ZoneType z : PLAYER_ZONES) {
                out.add(zoneName(side, z));
            }
        }
        for (String side : SIDES) {
            out.add(mergedUnder(side));
        }
        out.add(STACK);
        out.add(DELAYED);
        out.add(ACTIONS);
        out.add(HISTORY);
        out.add(PHANTOM);
        for (String side : SIDES) {
            out.add(untracked(side));
        }
        return out;
    }

    /**
     * Relation types, in canonical order.
     *
     * <p>These index the relation-aware attention bias, so order is fixed for the
     * same reason {@link #PLAYER_ZONES} is: a mismatch between Java and Python
     * silently permutes relation semantics -- the model would read "blocks" as
     * "attacks" -- rather than failing.
     *
     * <p>Closed by design. The list is earned by the rule that an edge is
     * explicit only when two states with identical token multisets differ solely
     * by it; see {@code docs/representation_surface.md} section 6. Adding a
     * relation on intuition defeats that discipline.
     *
     * <p>Relations are over ordered pairs: {@code (query, key)}. Inverse
     * directions are separate ids, emitted explicitly rather than inferred.
     */
    public static final String SOURCE_IS = "source_is";
    /**
     * No longer emitted since schema 19 (stack targets are {@link #targetSlot}
     * edges); kept so indices do not shift.
     */
    public static final String TARGETING = "targeting";
    public static final String TARGETING_SPELL = "targeting_spell";
    public static final String ATTACKS = "attacks";
    public static final String ATTACKING = "attacking";
    public static final String DEFENDER = "defender";
    public static final String BLOCKS = "blocks";
    public static final String BLOCKING = "blocking";
    public static final String BLOCKED = "blocked";
    public static final String BLOCKED_BY = "blocked_by";
    public static final String UNBLOCKED = "unblocked";
    public static final String ATTACHED_TO = "attached_to";
    public static final String MERGED_UNDER = "merged_under";
    public static final String OWNER_IS = "owner_is";
    /**
     * Schema 7. Source of an active continuous static effect -> each visible
     * object or player it applies to. Carried on the source token.
     */
    public static final String AFFECTS = "affects";

    // Schema 13: history participants are edges whose relation is the role.
    // A history edge targets a per-viewer tracking handle (h<N>), an event
    // (e<N>) or a player (p<N>), never a raw object id.
    /** Player who took the action, or controls the ability. */
    public static final String ACTOR = "actor";
    /** Player affected (whose zone, who drew, whose life changed). */
    public static final String PLAYER = "player";
    /** Object moved or affected. */
    public static final String OBJ = "obj";
    // Reused for history roles, with identical semantics (schema 13):
    //   SOURCE_IS   - source card of a spell, ability or damage
    //   ATTACKS     - the attacking creature (declare_attacker, declare_blockers)
    //   DEFENDER    - the player/planeswalker/battle it attacks
    //   BLOCKS      - a blocking creature (declare_blockers)
    //   ATTACHED_TO - the object/player an attach event attached obj to
    // BLOCKED and UNBLOCKED are no longer emitted (kept so indices do not shift).
    /** The spell or ability on the stack. */
    public static final String STACK_OBJECT = "stack_object";
    /** Ordered target slots; anything past the fourth goes to target_more. */
    public static final String TARGET_0 = "target_0";
    public static final String TARGET_1 = "target_1";
    public static final String TARGET_2 = "target_2";
    public static final String TARGET_3 = "target_3";
    public static final String TARGET_MORE = "target_more";
    public static final String NEW_CONTROLLER = "new_controller";
    public static final String OLD_CONTROLLER = "old_controller";
    /** Event(s) this event is a consequence of. Multi-valued. */
    public static final String CAUSE = "cause";
    /** Source of a replacement effect that modified this event (in addition to its cause). */
    public static final String REPLACED_BY = "replaced_by";
    /**
     * State side: a board object -> whatever currently modifies it: the source
     * token of an active static effect (the reverse of {@link #AFFECTS}), or the
     * history resolve event whose continuous effect is still applied to it.
     */
    public static final String AFFECTED_BY = "affected_by";
    /**
     * Schema 16. Decision prompt -> tokens already chosen earlier in the same
     * decomposed decision: attackers/blockers declared so far, targets already
     * picked for the same ability, cards already placed in this scry/surveil.
     */
    public static final String SELECTED = "selected";

    private static final String[] TARGET_SLOTS = {TARGET_0, TARGET_1, TARGET_2, TARGET_3};

    /** The relation for the i-th target (0-based) of a spell or ability. */
    public static String targetSlot(int i) {
        return i < TARGET_SLOTS.length ? TARGET_SLOTS[i] : TARGET_MORE;
    }

    public static List<String> relations() {
        List<String> out = new ArrayList<>();
        out.add(SOURCE_IS);
        out.add(TARGETING);
        out.add(TARGETING_SPELL);
        out.add(ATTACKS);
        out.add(ATTACKING);
        out.add(DEFENDER);
        out.add(BLOCKS);
        out.add(BLOCKING);
        out.add(BLOCKED);
        out.add(BLOCKED_BY);
        out.add(UNBLOCKED);
        out.add(ATTACHED_TO);
        out.add(MERGED_UNDER);
        out.add(OWNER_IS);
        out.add(AFFECTS);
        out.add(ACTOR);
        out.add(PLAYER);
        out.add(OBJ);
        out.add(STACK_OBJECT);
        out.add(TARGET_0);
        out.add(TARGET_1);
        out.add(TARGET_2);
        out.add(TARGET_3);
        out.add(TARGET_MORE);
        out.add(NEW_CONTROLLER);
        out.add(OLD_CONTROLLER);
        out.add(CAUSE);
        out.add(REPLACED_BY);
        out.add(AFFECTED_BY);
        out.add(SELECTED);
        return out;
    }

    // ---- Schema 13 categorical vocabularies (keys.json `vocab`) ----------

    /** {@code verb} of a history token (vocab {@code verb}). */
    public static final List<String> HISTORY_VERBS = Collections.unmodifiableList(Arrays.asList(
            "move_zone", "draw", "discard", "play_land", "cast", "activate", "special_action",
            "trigger", "resolve", "does_not_resolve", "copy_spell", "declare_attacker",
            "declare_blockers", "deal_damage", "life_change", "counters_change", "create",
            "change_control", "reveal", "scry", "surveil", "shuffle", "mulligan",
            "designation_change", "attach", "attempt_end"));

    /** {@code move_kind} of a move_zone event (vocab {@code move_kind}). */
    public static final List<String> MOVE_KINDS = Collections.unmodifiableList(Arrays.asList(
            "put", "mill", "exile", "sacrifice", "destroy", "bounce", "search", "return", "other"));

    /** {@code reason} of a does_not_resolve event (vocab {@code dnr_reason}). */
    public static final List<String> DNR_REASONS = Collections.unmodifiableList(Arrays.asList(
            "countered", "fizzled", "removed"));

    /**
     * {@code cause_kind} (vocab {@code cause_kind}), present on every history
     * token: {@code pointer} when the event is a consequence of the events its
     * {@code cause} edges point at, otherwise the rule category that produced it.
     */
    public static final List<String> CAUSE_KINDS = Collections.unmodifiableList(Arrays.asList(
            "pointer", "turn_based", "state_based", "combat", "pregame", "player_action", "unknown"));

    /**
     * Schema 16. {@code verb} of a decision option token (type {@code action},
     * vocab {@code action_verb}). Closed: one per kind of option a routed
     * decision offers that is not a reference to an existing token.
     * {@code attack}, {@code block} and {@code choose_object} are fallbacks for
     * an object choice whose object has no token the viewer can see.
     */
    public static final List<String> ACTION_VERBS = Collections.unmodifiableList(Arrays.asList(
            "pass", "cast", "activate", "play_land", "attack", "no_attack", "block", "no_block",
            "choose_nothing", "choose_object", "confirm", "decline", "pay", "mode", "number",
            "scry_top", "scry_bottom", "surveil_top", "surveil_graveyard", "order_next",
            // schema 17: mana payment steps (pay_mana)
            "activate_mana", "use_pool", "pay_life", "convoke", "improvise",
            // schema 18: optional/additional costs, combination mana colours, waterbend
            "optional_cost", "choose_color", "waterbend"));

    /**
     * Schema 16. Decision kinds: the record's {@code kind} and the prompt
     * token's {@code kind} component (vocab {@code decision_kind}).
     */
    public static final List<String> DECISION_KINDS = Collections.unmodifiableList(Arrays.asList(
            "ability", "declare_attacker", "declare_blocker", "choose_entity", "choose_companion",
            "choose_card_zone_change", "choose_number", "confirm_trigger", "confirm_action",
            "choose_mode", "pay_unless", "scry", "surveil", "order_top", "order_bottom",
            "order_triggers", "choose_target",
            // schema 17: -Dmagicnet.payment
            "pay_mana", "pay_convoke", "pay_delve",
            // schema 18: non-mana cost parts, optional costs, combination mana
            "pay_sacrifice", "pay_discard", "pay_exile", "pay_tap", "pay_untap", "pay_return",
            "pay_reveal", "choose_optional_cost", "choose_additional_cost", "pay_mana_combo",
            // schema 20: announceRequirements (X)
            "announce"));

    /**
     * Schema 17. {@code mana_color} of a payment option (vocab {@code mana_color}):
     * the colour an {@code activate_mana} option adds or a {@code use_pool}
     * option spends; C is colourless.
     */
    public static final List<String> MANA_COLORS = Collections.unmodifiableList(Arrays.asList(
            "W", "U", "B", "R", "G", "C"));

    /** Value of {@code phase} when Forge has no current phase (before the first turn). */
    public static final String NO_PHASE = "NONE";

    /** {@code phase} (vocab {@code phase}): Forge PhaseType names, then {@link #NO_PHASE}. */
    public static List<String> phases() {
        // Read the enum constants by reflection without initializing PhaseType,
        // whose static initializer needs Forge's localizer (absent in a bare JVM).
        List<String> out = new ArrayList<>();
        try {
            Class<?> c = Class.forName("forge.game.phase.PhaseType", false, Schema.class.getClassLoader());
            for (java.lang.reflect.Field f : c.getDeclaredFields()) {
                if (f.isEnumConstant()) {
                    out.add(f.getName());
                }
            }
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
        out.add(NO_PHASE);
        return out;
    }

    public static String phaseName(forge.game.phase.PhaseType p) {
        return p == null ? NO_PHASE : p.name();
    }

    /** Token {@code type} values, which are a separate vocabulary from zones. */
    public static List<String> tokenTypes() {
        List<String> out = new ArrayList<>();
        out.add("global");
        out.add("player");
        out.add("game_object");
        out.add("library_card");
        out.add("stack_item");
        out.add("delayed_trigger");
        out.add("action");
        out.add(HISTORY);
        // Schema 16: the record's decision prompt token.
        out.add("decision");
        return out;
    }

    /**
     * Schema 9. Token property layout: which keys stay at the top level of a
     * token and which are nested under {@link #COMPONENTS}.
     *
     * <p>The top level holds what is <em>not</em> a model input: {@code type},
     * {@code zone}, {@code edges}, and every key {@code keys.json} classifies as
     * {@code identity} (join keys, card identity) or {@code ref} (references to
     * other objects/players, which are future edges). Everything else -- the
     * {@code number}, {@code flag}, {@code string} and {@code string_list}
     * values the encoder turns into (key, value) components -- goes under
     * {@code components}. {@code tests/test_schema.py} checks this agrees with
     * keys.json in both directions.
     */
    public static final String COMPONENTS = "components";

    /** Concrete top-level keys not already covered by a suffix or pattern rule. */
    public static final Set<String> TOP_LEVEL_KEYS = Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(
            "type", "zone", "edges",
            // identity
            "id", "name", "oracle_id", "source_oracle_id", "named_card_oracle_id", "attempt", "handles",
            // schema 16: a decision candidate referencing an existing token
            "ref",
            // ref
            "source", "named_card", "chosen_player")));

    /**
     * Suffix families. None since schema 13, which removed the history role
     * families ({@code <role>_id}, {@code <role>_oracle_id},
     * {@code <role>_player}, {@code target_<i>}, {@code card<i>}): history
     * participants are edges.
     */
    public static final List<String> TOP_LEVEL_SUFFIXES = Collections.unmodifiableList(new ArrayList<String>());

    /** Whole-key regexes. None since schema 13. */
    public static final List<String> TOP_LEVEL_PATTERNS = Collections.unmodifiableList(new ArrayList<String>());

    private static final List<Pattern> TOP_LEVEL_REGEXES = compile(TOP_LEVEL_PATTERNS);

    private static List<Pattern> compile(List<String> regexes) {
        List<Pattern> out = new ArrayList<>();
        for (String r : regexes) {
            out.add(Pattern.compile(r));
        }
        return out;
    }

    /** Whether a token property is written at the top level rather than under components. */
    public static boolean isTopLevel(String key) {
        if (TOP_LEVEL_KEYS.contains(key)) {
            return true;
        }
        for (String s : TOP_LEVEL_SUFFIXES) {
            if (key.endsWith(s)) {
                return true;
            }
        }
        for (Pattern p : TOP_LEVEL_REGEXES) {
            if (p.matcher(key).matches()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Prints the vocabulary as JSON, for the Python side to check itself against.
     * With {@code classify k1 k2 ...}, prints {@code {"k1":true,...}} instead:
     * whether each key is top-level ({@link #isTopLevel}).
     */
    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder();
        if (args.length > 0 && "classify".equals(args[0])) {
            sb.append('{');
            for (int i = 1; i < args.length; i++) {
                if (i > 1) {
                    sb.append(',');
                }
                sb.append('"').append(args[i].replace("\\", "\\\\").replace("\"", "\\\""))
                  .append("\":").append(isTopLevel(args[i]));
            }
            System.out.println(sb.append('}'));
            return;
        }
        sb.append("{\"schema\":").append(Serializer.SCHEMA);
        sb.append(",\"zones\":[");
        appendAll(sb, zoneNames());
        sb.append("],\"token_types\":[");
        appendAll(sb, tokenTypes());
        sb.append("],\"relations\":[");
        appendAll(sb, relations());
        sb.append("],\"vocabs\":{\"verb\":[");
        appendAll(sb, HISTORY_VERBS);
        sb.append("],\"move_kind\":[");
        appendAll(sb, MOVE_KINDS);
        sb.append("],\"dnr_reason\":[");
        appendAll(sb, DNR_REASONS);
        sb.append("],\"cause_kind\":[");
        appendAll(sb, CAUSE_KINDS);
        sb.append("],\"phase\":[");
        appendAll(sb, phases());
        sb.append("],\"zone\":[");
        appendAll(sb, zoneNames());
        sb.append("],\"action_verb\":[");
        appendAll(sb, ACTION_VERBS);
        sb.append("],\"decision_kind\":[");
        appendAll(sb, DECISION_KINDS);
        sb.append("],\"mana_color\":[");
        appendAll(sb, MANA_COLORS);
        sb.append("]}");
        sb.append(",\"components\":\"").append(COMPONENTS);
        sb.append("\",\"top_level\":{\"keys\":[");
        appendAll(sb, new ArrayList<>(TOP_LEVEL_KEYS));
        sb.append("],\"suffixes\":[");
        appendAll(sb, TOP_LEVEL_SUFFIXES);
        sb.append("],\"patterns\":[");
        appendAll(sb, TOP_LEVEL_PATTERNS);
        sb.append("]}}");
        System.out.println(sb);
    }

    private static void appendAll(StringBuilder sb, List<String> values) {
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append('"').append(values.get(i).replace("\\", "\\\\").replace("\"", "\\\""))
              .append('"');
        }
    }
}
