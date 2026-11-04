package magicnet.bridge;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import forge.card.CardRules;
import forge.card.ICardFace;
import forge.game.card.Card;
import forge.game.card.CardView;

/**
 * Resolves cards to Scryfall oracle ids, so downstream code can join any card
 * metadata from a local database instead of the record schema growing a field
 * per attribute.
 *
 * <p>Forge has no oracle ids; it identifies cards by name. The table is built
 * from Scryfall bulk data by {@code scripts/build_oracle_map.py} and loaded from
 * {@code -Dmagicnet.oracle=PATH} (default {@code ../data/processed/oracle_ids.tsv},
 * relative to {@code forge-bridge/}).
 *
 * <p><b>An oracle id must never carry more than the emitted name.</b> Forge's
 * {@link CardRules#getName()} is the real card even while it is face-down, so it
 * is consulted only when the name the viewer is shown is that card's full name
 * or one of its faces -- the case where it adds disambiguation (split faces such
 * as "Fire") but no hidden information. Otherwise the shown name alone is looked up.
 *
 * <p>Tokens, emblems and Forge's effect pseudo-cards get no id: none is a
 * printed card, and their names and properties already describe them.
 */
public final class OracleIds {

    private static final Map<String, String> BY_NAME = load();

    private static final Map<String, Integer> UNRESOLVED = new TreeMap<>();
    private static long resolved;

    private OracleIds() {
    }

    private static Map<String, String> load() {
        Path p = Paths.get(System.getProperty("magicnet.oracle", "../data/processed/oracle_ids.tsv"));
        Map<String, String> m = new HashMap<>();
        try (BufferedReader r = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
            String line;
            while ((line = r.readLine()) != null) {
                int tab = line.indexOf('\t');
                if (tab > 0) {
                    m.put(line.substring(0, tab), line.substring(tab + 1));
                }
            }
        } catch (IOException e) {
            System.out.println("WARN: no oracle id table at " + p.toAbsolutePath()
                    + "; records will carry no oracle_id. Build it with"
                    + " `uv run scripts/build_oracle_map.py` or set -Dmagicnet.oracle.");
        }
        return m;
    }

    /** Oracle id for a live card whose name is being emitted, or null. */
    public static String of(Card c) {
        // Immutable cards are Forge's effect bookkeeping ("X's Effect",
        // adventure-exile markers), not printed cards.
        if (c == null || c.isToken() || c.isImmutable() || c.isEmblem()) {
            return null;
        }
        String shown = c.getName();
        try {
            CardRules rules = c.getRules();
            if (rules != null && namesCard(shown, rules)) {
                String id = BY_NAME.get(rules.getName());
                if (id != null) {
                    return hit(id);
                }
            }
        } catch (Throwable ignored) {
            // fall back to the shown name
        }
        return ofName(shown);
    }

    /** Oracle id for an event's card view, from the name the viewer sees. */
    /**
     * {@code {name, oracle_id}} of a card's real identity, for training-only
     * hidden-information labels: a face-down card's original face, not the
     * nameless face-down state. A pure lookup -- unlike {@link #of(Card)} it
     * does not touch the coverage counters, so turning labels on leaves the
     * printed statistics unchanged. {@code oracle_id} is null if unresolved.
     */
    public static String[] truthOf(Card c) {
        String name = null;
        String id = null;
        try {
            name = c.isFaceDown()
                    ? c.getState(forge.card.CardStateName.Original).getName()
                    : c.getName();
            CardRules rules = c.getRules();
            if (rules != null && namesCard(name, rules)) {
                id = BY_NAME.get(rules.getName());
            }
            if (id == null && name != null) {
                id = BY_NAME.get(name);
            }
        } catch (Throwable ignored) {
            // reported by the caller as a missing oracle id
        }
        return new String[] { name, id };
    }

    public static String of(CardView c) {
        if (c == null || c.isToken() || c.isImmutable() || c.isEmblem()) {
            return null;
        }
        return ofName(c.getName());
    }

    /** Oracle id for a card name the viewer was shown (e.g. a named card choice). */
    public static String ofShownName(String shown) {
        return ofName(shown);
    }

    private static boolean namesCard(String shown, CardRules rules) {
        if (shown == null) {
            return false;
        }
        if (shown.equals(rules.getName())) {
            return true;
        }
        ICardFace main = rules.getMainPart();
        ICardFace other = rules.getOtherPart();
        return (main != null && shown.equals(main.getName()))
                || (other != null && shown.equals(other.getName()));
    }

    private static String ofName(String shown) {
        if (shown == null || shown.isEmpty() || BY_NAME.isEmpty()) {
            return null;
        }
        String id = BY_NAME.get(shown);
        if (id != null) {
            return hit(id);
        }
        synchronized (UNRESOLVED) {
            UNRESOLVED.merge(shown, 1, Integer::sum);
        }
        return null;
    }

    private static String hit(String id) {
        synchronized (UNRESOLVED) {
            resolved++;
        }
        return id;
    }

    public static void printCoverage() {
        synchronized (UNRESOLVED) {
            System.out.println("\n=== oracle ids ===");
            System.out.printf("  table names:      %d%n", BY_NAME.size());
            System.out.printf("  resolved lookups: %d%n", resolved);
            System.out.printf("  unresolved names: %d %s%n", UNRESOLVED.size(), UNRESOLVED);
        }
    }
}
