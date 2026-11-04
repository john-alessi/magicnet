package magicnet.bridge;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import forge.game.Game;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

/**
 * Line-delimited JSON writer: one record per decision, one JSON object per line.
 *
 * <p>This is the Java-to-Python boundary. It is deliberately the *only* place the
 * token representation is written out, so the offline dump and the live
 * environment loop cannot drift apart -- the live loop writes the same record to
 * a socket instead of a file.
 *
 * <h2>What a record contains</h2>
 *
 * <ul>
 *   <li>{@code tokens} -- the model input, from {@link StateExtractor}. Already
 *       redacted to the viewer's information set.</li>
 *   <li>{@code actions} -- the decode table: the legal action pool, in pool
 *       order. The policy emits an index into this array. It is kept separate
 *       from {@code tokens} because they are consumed differently: tokens are
 *       embedded, actions are selected over.</li>
 *   <li>{@code chosen} -- index of the option actually taken. On ability
 *       decisions the last option is the explicit pass (api "Pass").
 *       This is the supervised label for the offline dump and the replay input
 *       for the equivalence test.</li>
 * </ul>
 *
 * <h2>Ids are join keys, not features</h2>
 *
 * Token {@code id} properties exist only so {@code edges} can be resolved to
 * token indices on the Python side. They correlate with library order and would
 * leak the decklist if embedded. Python must resolve edges and then drop every
 * id before tensorisation; {@code StateExtractor.Tok.signature()} already
 * excludes them for exactly this reason.
 *
 * <h2>Training-only labels: {@code hidden_truth}</h2>
 *
 * With {@code -Dmagicnet.labels=true} (default off) each DUMP record carries a
 * top-level {@code hidden_truth}: for every hidden-region token in
 * {@code state} -- an aggregate with an {@code r:<zone>} id or a face-down
 * object the viewer cannot see -- the true multiset of cards it stands for,
 * keyed by that token's id. It is a target for a belief model, never an
 * input: it is appended only by {@link #write}/{@link #writeChoice}, after the
 * body the agent sees was built, so {@link #asRequest} cannot carry it (and
 * refuses a body that does). Default off so an ordinary run's dump is exactly
 * what a live agent would have been sent.
 *
 * <h2>Schema stability</h2>
 *
 * {@code schema} is bumped on any change that would silently alter the meaning
 * of an existing field. Adding a new optional field does not bump it. The Python
 * loader should refuse a schema it does not recognise rather than guess.
 */
public final class Serializer {

    /**
     * Bump on any breaking change to record layout or field meaning.
     * 9: token model inputs nested under {@code components} (see
     * {@link Schema#isTopLevel}); flags emitted only when true; object ids
     * removed from history {@code ability} text; {@code target_candidates}
     * counts card candidates, not just players.
     * 10: game objects carry {@code gained}/{@code lost} (normalized rules
     * text differing from the printed card) instead of {@code kw},
     * {@code kw_granted} and {@code abilities_removed}; creatures carry
     * {@code base_power}/{@code base_toughness}.
     * 12: a card the viewer may not see carries no {@code name} or
     * {@code oracle_id} (state tokens) and history roles keep only
     * {@code <role>_id}, instead of the literal "HIDDEN" / "&lt;hidden&gt;".
     * 13: history redesign. One token per event with {@code verb} and
     * modifiers as components, participants as role edges to per-viewer
     * tracking handles ({@code h<N>}, assigned to state tokens through their
     * {@code handles} list), events ({@code e<N>}, the history token's id) or
     * players; {@code cause} edges and {@code cause_kind}; phantom and
     * untracked tokens; unseen opponent hand cards are one count aggregate;
     * {@code affected_by} edges replace {@code temp_power}/{@code temp_toughness};
     * a missing phase is "NONE".
     * 14: no re-linking of tracking handles. A handle whose object the viewer
     * can no longer follow but whose identity it knew moves for good to an
     * identity phantom ("a copy of X"; {@code ceased} absent), instead of the
     * untracked token and back to a live token when a card of that name shows
     * up again. Ability/mode text loses Forge's bracketed run context and
     * player names.
     * 15: hidden-region count aggregates (opponent hand, library remainder,
     * sideboard) carry a top-level {@code id} {@code "r:<zone>"}, so an id on a
     * {@code library_card} no longer implies one physical card. Records may
     * carry the training-only {@code hidden_truth} (see above).
     */
    public static final int SCHEMA = 15;

    /** A priority decision: which spell or ability to play. */
    public static final String KIND_ABILITY = "ability";

    /** {@code api} of the explicit pass option on ability decisions. */
    public static final String PASS_API = "Pass";

    private static final String FILE = System.getProperty("magicnet.jsonl", "");

    /** Training-only hidden-information labels in the dump (never in requests). */
    public static final boolean LABELS = Boolean.getBoolean("magicnet.labels");

    /** Record field holding the labels. */
    static final String HIDDEN_TRUTH = "hidden_truth";

    /**
     * Labels of bodies built but not yet written, keyed by the body INSTANCE.
     * The body is what the agent is sent, so the labels live beside it, never
     * in it.
     */
    private static final java.util.Map<String, String> pendingTruth =
            new java.util.IdentityHashMap<>();
    private static final AtomicLong LABELLED = new AtomicLong();
    private static final AtomicLong LABEL_REGIONS = new AtomicLong();
    private static final AtomicLong LABEL_NO_ORACLE = new AtomicLong();

    private static final AtomicInteger GAME = new AtomicInteger(-1);
    private static final AtomicLong DECISION = new AtomicLong();
    private static final AtomicLong WRITTEN = new AtomicLong();
    private static final AtomicLong PASSES = new AtomicLong();
    private static final AtomicLong UNADDRESSABLE = new AtomicLong();
    private static final AtomicLong RESYNCS = new AtomicLong();

    /** Per-viewer count of history tokens already emitted, for delta encoding. */
    private static final java.util.Map<String, Integer> emitted =
            new java.util.HashMap<>();

    private static PrintWriter out;

    private Serializer() {
    }

    public static boolean enabled() {
        return !FILE.isEmpty();
    }

    /** Starts a new game; resets the per-game decision counter. */
    public static synchronized void beginGame() {
        GAME.incrementAndGet();
        DECISION.set(0);
        emitted.clear();
        pendingTruth.clear();
    }

    /** Index of the game currently being played. */
    public static int gameIndex() {
        return Math.max(0, GAME.get());
    }

    public static synchronized void close() {
        if (out != null) {
            out.flush();
            out.close();
            out = null;
        }
    }

    public static void printSummary() {
        if (!enabled()) {
            return;
        }
        System.out.println("\n=== jsonl dump ===");
        System.out.printf("  file:            %s%n", FILE);
        System.out.printf("  records written: %d%n", WRITTEN.get());
        System.out.printf("  passes:          %d%n", PASSES.get());
        System.out.printf("  unaddressable:   %d%n", UNADDRESSABLE.get());
        System.out.printf("  history resyncs: %d%n", RESYNCS.get());
        if (LABELS) {
            System.out.printf("  hidden_truth:    %d records, %d regions, %d cards without oracle_id%n",
                    LABELLED.get(), LABEL_REGIONS.get(), LABEL_NO_ORACLE.get());
        }
        if (UNADDRESSABLE.get() > 0) {
            System.out.println("  WARN: chosen action absent from pool; those records"
                    + " cannot be replayed and must not be used as labels");
        }
    }

    /**
     * Writes one decision record.
     *
     * <p>Called after the decision is known but describing the state *before* it,
     * because the tokens are captured ahead of the choice. Never throws: a
     * serialisation failure must not change the game.
     */
    /**
     * Builds the record body: everything except the trailing chosen fields and
     * the closing brace.
     *
     * <p>Split out so the request sent to an external agent is byte-identical to
     * what lands in the dump. If the agent saw a different rendering of the
     * state than the dump records, replaying the dump would not reproduce the
     * agent's decisions and nothing downstream could be trusted.
     *
     * <p>Advances the per-viewer history delta cursor, so it must be called
     * exactly once per decision.
     */
    public static synchronized String buildBody(Game game, Player viewer,
            List<StateExtractor.Tok> tokens, int histSize, List<SpellAbility> pool) {
        StringBuilder opts = new StringBuilder();
        int n = pool == null ? 0 : pool.size();
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                opts.append(',');
            }
            opts.append(actionJson(i, pool.get(i)));
        }
        // Passing priority is an explicit option, always last, so a policy
        // never needs to know which decision kinds allow passing. Other
        // decision kinds (modes, discards, combat) get no pass option.
        if (n > 0) {
            opts.append(',');
        }
        opts.append("{\"i\":").append(n).append(",\"api\":\"").append(PASS_API)
            .append("\",\"desc\":\"Pass priority\"}");
        return buildCore(game, viewer, tokens, histSize, KIND_ABILITY, opts.toString());
    }

    /**
     * A decision that is not a priority action: a blocker assignment, a mode
     * choice, a replacement-effect ordering.
     *
     * <p>Deliberately the same record shape as {@link #buildBody}, with the
     * options in the same {@code actions} array and a {@code kind} saying what
     * is being asked. Most of Forge's forty-odd decision callbacks are
     * "pick one of N", so they need no new encoding, no new policy head and no
     * new wire format -- only a label. Giving each callback a bespoke record
     * would multiply the surface the model has to learn for no gain.
     */
    public static synchronized String buildChoiceBody(Game game, Player viewer,
            List<StateExtractor.Tok> tokens, int histSize, String kind,
            List<Opt> options) {
        StringBuilder opts = new StringBuilder();
        for (int i = 0; i < options.size(); i++) {
            if (i > 0) {
                opts.append(',');
            }
            opts.append(options.get(i).toJson(i));
        }
        return buildCore(game, viewer, tokens, histSize, kind, opts.toString());
    }

    /**
     * One option at a choice point.
     *
     * <p>Carries edges to the game objects it refers to, so a pointer-style
     * head can attend to those objects' own token representations instead of
     * parsing them back out of the description string. The description stays
     * for readability and as a fallback; the edges are what should be learned
     * from.
     */
    public static final class Opt {
        private final String desc;
        private final List<String> refs = new java.util.ArrayList<>();
        private String oracleId;

        public Opt(String desc) {
            this.desc = desc;
        }

        /**
         * Oracle id of the card this option names. Needed where the card has no
         * token of its own to resolve an edge to (cards in a searched library).
         */
        public Opt oracle(String id) {
            this.oracleId = id;
            return this;
        }

        /** Adds a reference to a game object; null refs are ignored. */
        public Opt ref(String namespacedId) {
            if (namespacedId != null) {
                refs.add(namespacedId);
            }
            return this;
        }

        String toJson(int i) {
            StringBuilder sb = new StringBuilder(96);
            sb.append("{\"i\":").append(i);
            sb.append(",\"desc\":\"").append(esc(desc)).append('"');
            if (oracleId != null) {
                sb.append(",\"oracle_id\":\"").append(oracleId).append('"');
            }
            if (!refs.isEmpty()) {
                sb.append(",\"edges\":[");
                for (int j = 0; j < refs.size(); j++) {
                    if (j > 0) {
                        sb.append(',');
                    }
                    sb.append("{\"rel\":\"refers_to\",\"to\":\"")
                      .append(esc(refs.get(j))).append("\"}");
                }
                sb.append(']');
            }
            return sb.append('}').toString();
        }
    }

    private static String buildCore(Game game, Player viewer,
            List<StateExtractor.Tok> tokens, int histSize, String kind,
            String optionsJson) {
        String vk = name(viewer);
        int already = emitted.getOrDefault(vk, 0);
        if (already > histSize) {
            // History is append-only by construction; a shrink means the
            // listener was reset underneath us and the delta stream is no
            // longer reconstructable. Resync rather than emit a bad prefix.
            already = 0;
            RESYNCS.incrementAndGet();
        }

        StringBuilder sb = new StringBuilder(4096);
        sb.append("{\"schema\":").append(SCHEMA);
        sb.append(",\"game\":").append(Math.max(0, GAME.get()));
        sb.append(",\"decision\":").append(DECISION.getAndIncrement());
        sb.append(",\"viewer\":\"").append(esc(vk)).append('"');
        sb.append(",\"kind\":\"").append(esc(kind)).append('"');
        sb.append(",\"turn\":").append(turn(game));
        sb.append(",\"phase\":\"").append(esc(phase(game))).append('"');

        // History is cumulative and append-only, so emitting it in full on
        // every record makes the dump quadratic in game length (measured:
        // 45 MB for 2 games). Only the new suffix is written; the Python
        // loader accumulates per (game, viewer) and checks its running total
        // against hist_total, which makes any desync loud rather than silent.
        sb.append(",\"hist_total\":").append(histSize);
        sb.append(",\"hist_from\":").append(already);
        sb.append(",\"hist_new\":[");
        for (int i = already; i < histSize && i < tokens.size(); i++) {
            if (i > already) {
                sb.append(',');
            }
            sb.append(tokens.get(i).toJson());
        }
        sb.append(']');
        emitted.put(vk, histSize);

        sb.append(",\"state\":[");
        for (int i = histSize; i < tokens.size(); i++) {
            if (i > histSize) {
                sb.append(',');
            }
            sb.append(tokens.get(i).toJson());
        }
        sb.append(']');

        sb.append(",\"actions\":[").append(optionsJson).append(']');
        String body = sb.toString();
        if (LABELS && enabled()) {
            pendingTruth.put(body, truthJson(tokens, histSize));
        }
        return body;
    }

    /**
     * The {@code hidden_truth} array: one label per hidden-region state token,
     * in token order. Cards are grouped by (name, oracle_id), sorted by name.
     */
    private static String truthJson(List<StateExtractor.Tok> tokens, int histSize) {
        StringBuilder sb = new StringBuilder(256).append('[');
        boolean first = true;
        for (int i = histSize; i < tokens.size(); i++) {
            StateExtractor.Tok t = tokens.get(i);
            Object region = t.props.get("id");
            if (t.truth == null || region == null) {
                continue;
            }
            java.util.TreeMap<String, String[]> keys = new java.util.TreeMap<>();
            java.util.Map<String, Integer> counts = new java.util.HashMap<>();
            for (forge.game.card.Card c : t.truth) {
                String[] nid = OracleIds.truthOf(c);
                if (nid[1] == null) {
                    LABEL_NO_ORACLE.incrementAndGet();
                }
                String k = nid[0] + "\0" + nid[1];
                keys.putIfAbsent(k, nid);
                counts.merge(k, 1, Integer::sum);
            }
            if (!first) {
                sb.append(',');
            }
            first = false;
            LABEL_REGIONS.incrementAndGet();
            sb.append("{\"region\":\"").append(esc(String.valueOf(region))).append("\",\"cards\":[");
            boolean firstCard = true;
            for (java.util.Map.Entry<String, String[]> e : keys.entrySet()) {
                if (!firstCard) {
                    sb.append(',');
                }
                firstCard = false;
                String[] nid = e.getValue();
                sb.append("{\"oracle_id\":");
                sb.append(nid[1] == null ? "null" : "\"" + esc(nid[1]) + "\"");
                sb.append(",\"name\":");
                sb.append(nid[0] == null ? "null" : "\"" + esc(nid[0]) + "\"");
                sb.append(",\"count\":").append(counts.get(e.getKey())).append('}');
            }
            sb.append("],\"total\":").append(t.truth.size()).append('}');
        }
        return sb.append(']').toString();
    }

    /** Appends the pending labels of {@code body}, if any, to a dump record. */
    private static void appendTruth(StringBuilder sb, String body) {
        String truth = pendingTruth.remove(body);
        if (truth != null) {
            sb.append(",\"").append(HIDDEN_TRUTH).append("\":").append(truth);
            LABELLED.incrementAndGet();
        }
    }

    /** Appends a plain integer outcome and writes the record. */
    public static synchronized void writeChoice(String body, int chosen) {
        if (!enabled() || body == null) {
            return;
        }
        try {
            if (out == null) {
                out = new PrintWriter(new FileWriter(FILE));
            }
            StringBuilder sb = new StringBuilder(body.length() + 48).append(body);
            if (chosen < 0) {
                PASSES.incrementAndGet();
            }
            appendTruth(sb, body);
            sb.append(",\"chosen\":").append(chosen);
            sb.append(",\"chosen_match\":\"")
              .append(chosen < 0 ? "pass" : "index").append("\"}");
            out.println(sb);
            out.flush();
            WRITTEN.incrementAndGet();
        } catch (IOException | RuntimeException ignored) {
            // Instrumentation must never change play.
        }
    }

    /** The agent-facing request: a complete JSON object with no chosen fields. */
    public static String asRequest(String body) {
        // Labels are appended only on the dump path; this is the backstop.
        if (body.contains("\"" + HIDDEN_TRUTH + "\"")) {
            throw new IllegalStateException("request body carries " + HIDDEN_TRUTH);
        }
        return body + "}";
    }

    /** Appends the outcome to a body and writes it to the dump. */
    public static synchronized void write(String body, List<SpellAbility> pool,
            List<SpellAbility> chosen) {
        if (!enabled() || body == null) {
            return;
        }
        try {
            if (out == null) {
                out = new PrintWriter(new FileWriter(FILE));
            }
            StringBuilder sb = new StringBuilder(body.length() + 48).append(body);
            appendTruth(sb, body);
            SpellAbility pick = (chosen == null || chosen.isEmpty()) ? null : chosen.get(0);
            if (pick == null) {
                PASSES.incrementAndGet();
                int passIndex = pool == null ? 0 : pool.size();
                sb.append(",\"chosen\":").append(passIndex).append(",\"chosen_match\":\"pass\"");
            } else {
                int idx = indexOf(pool, pick);
                String how = idx < 0 ? "absent" : (pool.get(idx) == pick ? "identity" : "signature");
                if (idx < 0) {
                    UNADDRESSABLE.incrementAndGet();
                }
                sb.append(",\"chosen\":").append(idx);
                sb.append(",\"chosen_match\":\"").append(how).append('"');
            }
            sb.append('}');
            out.println(sb);
            out.flush();
            WRITTEN.incrementAndGet();
        } catch (IOException | RuntimeException ignored) {
            // Instrumentation must never change play.
        }
    }

    /**
     * Resolves the chosen ability to a pool index.
     *
     * <p>Identity first, then content signature: the AI returns an
     * equivalent-but-distinct {@code SpellAbility} instance a measurable fraction
     * of the time (6.9% on pio_41), so identity alone would spuriously report the
     * action as unaddressable.
     */
    private static int indexOf(List<SpellAbility> pool, SpellAbility pick) {
        if (pool == null || pick == null) {
            return -1;
        }
        for (int i = 0; i < pool.size(); i++) {
            if (pool.get(i) == pick) {
                return i;
            }
        }
        String want = sig(pick);
        for (int i = 0; i < pool.size(); i++) {
            if (pool.get(i) != null && want.equals(sig(pool.get(i)))) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Content signature for an ability. Host card name, API and ability text --
     * never object identity or card id, both of which vary across runs or leak.
     */
    private static String sig(SpellAbility sa) {
        try {
            String host = sa.getHostCard() == null ? "?" : sa.getHostCard().getName();
            return host + "::" + sa.getApi() + "::" + sa;
        } catch (RuntimeException e) {
            return "?";
        }
    }

    private static String actionJson(int i, SpellAbility sa) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("{\"i\":").append(i);
        if (sa == null) {
            return sb.append(",\"api\":\"null\"}").toString();
        }
        try {
            sb.append(",\"api\":\"").append(esc(String.valueOf(sa.getApi()))).append('"');
            if (sa.getHostCard() != null) {
                sb.append(",\"source\":\"").append(esc(sa.getHostCard().getName())).append('"');
                String oid = OracleIds.of(sa.getHostCard());
                if (oid != null) {
                    sb.append(",\"source_oracle_id\":\"").append(oid).append('"');
                }
                // Referenced as a namespaced edge, exactly like every other
                // token. A raw integer id here resolved against nothing and
                // survived strip_ids (which matches "c<digits>" strings), so
                // it reached the model as an unstable feature.
                sb.append(",\"edges\":[{\"rel\":\"source_is\",\"to\":\"")
                  .append(esc(StateExtractor.cardRef(sa.getHostCard().getId())))
                  .append("\"}]");
            }
            sb.append(",\"mana\":").append(sa.isManaAbility());
            // Stage-two fan-out, recorded rather than expanded: enumerating
            // every target combination is what factored decoding avoids.
            int candidates = targetCandidates(sa);
            if (candidates >= 0) {
                sb.append(",\"target_candidates\":").append(candidates);
            }
            sb.append(",\"desc\":\"").append(esc(String.valueOf(sa))).append('"');
        } catch (RuntimeException e) {
            sb.append(",\"err\":true");
        }
        return sb.append('}').toString();
    }

    /**
     * Legal targets (players and cards) for an option's target restrictions, or
     * -1 when it does not target or they cannot be counted. Best-effort:
     * instrumentation must never fail a game.
     *
     * <p>{@code getAllCandidates(sa, true)} would count players only. Forge
     * reads the activating player for the game; an option whose activator is
     * not yet set borrows the host's controller for the count and is restored,
     * so live state is unchanged.
     */
    static int targetCandidates(SpellAbility sa) {
        try {
            if (sa == null || sa.getTargetRestrictions() == null) {
                return -1;
            }
            Player prev = sa.getActivatingPlayer();
            if (prev == null) {
                if (sa.getHostCard() == null || sa.getHostCard().getController() == null) {
                    return -1;
                }
                sa.setActivatingPlayer(sa.getHostCard().getController());
            }
            try {
                return sa.getTargetRestrictions().getAllCandidates(sa, false).size();
            } finally {
                if (prev == null) {
                    sa.setActivatingPlayer(null);
                }
            }
        } catch (Throwable ignored) {
            return -1;
        }
    }

    private static String name(Player p) {
        try {
            return p == null ? "?" : p.getName();
        } catch (RuntimeException e) {
            return "?";
        }
    }

    private static int turn(Game g) {
        try {
            return g == null ? -1 : g.getPhaseHandler().getTurn();
        } catch (RuntimeException e) {
            return -1;
        }
    }

    private static String phase(Game g) {
        try {
            return g == null ? "?" : Schema.phaseName(g.getPhaseHandler().getPhase());
        } catch (RuntimeException e) {
            return "?";
        }
    }

    /**
     * JSON string escape. Must cover *all* control characters, not just the
     * common ones: card rules text contains tabs and other C0 characters, and a
     * raw one makes the whole line unparseable on the Python side.
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
