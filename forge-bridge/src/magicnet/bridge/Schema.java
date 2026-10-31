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
        return out;
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
            "id", "name", "oracle_id", "attempt", "reveal", "i",
            // ref
            "source", "card", "attachment", "target", "actor", "owner", "shown_to",
            "named_card", "chosen_player")));

    /**
     * Suffix families: {@code <role>_id} / {@code <role>_oracle_id} /
     * {@code target_<i>_id} / {@code card<i>_oracle_id} ... (identity) and
     * {@code <role>_player} (ref).
     */
    public static final List<String> TOP_LEVEL_SUFFIXES = Collections.unmodifiableList(Arrays.asList(
            "_id", "_player"));

    /** Whole-key regexes: {@code target_<i>} and {@code card<i>} (refs). */
    public static final List<String> TOP_LEVEL_PATTERNS = Collections.unmodifiableList(Arrays.asList(
            "target_\\d+", "card\\d+"));

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
        sb.append("],\"components\":\"").append(COMPONENTS);
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
