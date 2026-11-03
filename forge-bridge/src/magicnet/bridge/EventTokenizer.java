package magicnet.bridge;

/**
 * Text normalization shared by history and state tokens.
 *
 * <p>Schema 13 moved event tokenization into {@link HistoryHub} (what happened
 * and why, once per game) and {@link GameHistory} (per-viewer redaction and
 * tracking handles). What remains here is the rule for ability text.
 */
public final class EventTokenizer {

    private EventTokenizer() {
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

    /**
     * Schema 14: rules text as it reaches a component. Beyond
     * {@link #withoutObjectIds}, Forge's stack descriptions end in bracketed
     * run context -- "[Zone Changer: Palace Jailer]", "[Player: P1-&lt;deck&gt;]",
     * "[Card: X, Activator: P0-...]" -- naming objects (already edges) and
     * harness player names, which embed the deck name. Player names are
     * replaced by "a player" first (they contain brackets themselves), then
     * every trailing bracketed group is dropped.
     */
    static String normalize(String s, forge.game.Game game) {
        s = withoutObjectIds(s);
        if (s == null) {
            return null;
        }
        if (game != null) {
            try {
                for (forge.game.player.Player p : game.getPlayers()) {
                    String n = p.getName();
                    if (n != null && !n.isEmpty()) {
                        s = s.replace(n, "a player");
                    }
                }
            } catch (Throwable ignored) {
                // keep
            }
        }
        String prev;
        do {
            prev = s;
            s = TRAILING_CONTEXT.matcher(s).replaceAll("");
        } while (!s.equals(prev));
        return s.trim();
    }

    /**
     * Rules text of an ability for a component, or null when none exists.
     *
     * <p>Some abilities have no description of their own once Forge's
     * bracketed context is stripped -- Saga chapter triggers render as just
     * "[...]" -- so fall back to the stack description, then the trigger's
     * own description, then the SpellDescription params along the ability's
     * sub-ability chain (a chapter's effects). An empty string is never
     * returned: an ability with no recoverable text gets no component.
     */
    static String abilityText(forge.game.spellability.SpellAbility sa, forge.game.Game game) {
        return abilityText(null, sa, game);
    }

    /** As above, preferring an already-rendered description (e.g. the cast event's view). */
    static String abilityText(String rendered, forge.game.spellability.SpellAbility sa,
            forge.game.Game game) {
        String first = normalize(rendered, game);
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (sa == null) {
            return null;
        }
        java.util.List<java.util.function.Supplier<String>> tries = new java.util.ArrayList<>();
        tries.add(sa::getDescription);
        tries.add(sa::getStackDescription);
        tries.add(() -> sa.getTrigger() == null ? null : sa.getTrigger().toString());
        if (sa instanceof forge.game.trigger.WrappedAbility) {
            forge.game.spellability.SpellAbility inner =
                    ((forge.game.trigger.WrappedAbility) sa).getWrappedAbility();
            tries.add(() -> inner == null ? null : inner.getDescription());
            tries.add(() -> spellDescriptions(inner));
        }
        tries.add(() -> spellDescriptions(sa));
        for (java.util.function.Supplier<String> t : tries) {
            try {
                String s = normalize(t.get(), game);
                if (s != null && !s.isBlank()) {
                    return s;
                }
            } catch (Throwable ignored) {
                // next fallback
            }
        }
        return null;
    }

    private static String spellDescriptions(forge.game.spellability.SpellAbility sa) {
        StringBuilder sb = new StringBuilder();
        for (forge.game.spellability.SpellAbility a = sa; a != null; a = a.getSubAbility()) {
            String d = a.getParam("SpellDescription");
            if (d != null && !d.isBlank()) {
                if (sb.length() > 0) {
                    sb.append(' ');
                }
                sb.append(d.trim());
            }
        }
        return sb.length() == 0 ? null : sb.toString();
    }

    private static final java.util.regex.Pattern TRAILING_CONTEXT =
            java.util.regex.Pattern.compile("\\s*\\[(?:[^\\[\\]]|\\[[^\\[\\]]*\\])*\\]\\s*$");

    /** " (123)" as Card.toString() appends it, before punctuation, space or the end. */
    private static final java.util.regex.Pattern OBJECT_ID =
            java.util.regex.Pattern.compile(" \\(\\d+\\)(?=[:,\\]')\\s]|$)");
}
