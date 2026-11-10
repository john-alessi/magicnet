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
 *   <li>{@code prompt} (schema 16) -- one {@code decision} token (id
 *       {@code "q"}) saying what is asked: {@code kind}, Forge's prompt text,
 *       min/max, {@code optional}, and edges to the asking ability/card and
 *       ({@code selected}) to what earlier steps of the same decomposed
 *       decision already chose.</li>
 *   <li>{@code actions} -- the ordered candidate list the policy emits an
 *       index into. Schema 16: each candidate is either {@code {"ref": id}},
 *       a reference to the existing token (via the id/handle table edges use)
 *       of the object or player it names, or an option token ({@code type}
 *       {@code action}, id {@code a<i>}, a {@code verb} and components) for a
 *       choice with no token of its own: an ability to play, pass, a mode, a
 *       number, yes/no. Built by {@link #buildChoiceBody}.</li>
 *   <li>{@code chosen} -- index of the candidate actually taken. On ability
 *       decisions the last candidate is the explicit pass (verb "pass").
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
     * 16: decision candidates restructured. {@code actions} is a mixed list
     * of {@code {"ref": ...}} references to existing tokens and {@code action}
     * option tokens ({@code verb}, {@code ability}, ...; no {@code api},
     * {@code desc}, {@code i}, {@code refers_to}); a top-level {@code prompt}
     * decision token; own-library name aggregates carry an {@code n:<name>}
     * id; relation {@code selected}.
     * 17: mana payment decisions (-Dmagicnet.payment): decision kind
     * {@code pay_mana}, option verbs {@code activate_mana}/{@code use_pool}/
     * {@code pay_life}, vocab {@code mana_color}, prompt components
     * {@code remaining_*}/{@code spent_*}/{@code life_paid}, option components
     * {@code mana_color}/{@code mana_amount}/{@code restricted}/{@code snow}/
     * {@code rider}. Vocabularies grew; no existing field changed meaning,
     * but a schema-16 reader would misread the new kind as unknown.
     * 18: non-mana cost parts, optional/additional costs, combination mana.
     * 19: a spell on the stack (cast, copied, or still being cast) is its
     * card's {@code game_object} token in zone {@code stack} (no stack_item,
     * no {@code source_is}; {@code side}, {@code stack_depth}, {@code casting},
     * {@code copy}, {@code modes}, {@code target_i} edges); abilities stay
     * {@code stack_item} with {@code side}, {@code stack_depth}, {@code ability},
     * {@code modes} and {@code target_i} edges instead of
     * {@code targeting}/{@code targeting_spell} (no {@code is_spell}); every
     * state edge resolves to a token or tracking handle of the record or is
     * absent; prompts asked while a spell is cast point {@code stack_object}
     * at its card token.
     */
    public static final int SCHEMA = 20;

    /** A priority decision: which spell or ability to play. */
    public static final String KIND_ABILITY = "ability";

    /** Schema 16: {@code verb} of the explicit pass option token on ability decisions. */
    public static final String VERB_PASS = "pass";

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

    /** History tokens already emitted to {@code viewer}'s records (the delta cursor). */
    static synchronized int emittedFor(Player viewer) {
        return emitted.getOrDefault(name(viewer), 0);
    }

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

    /** The next game is numbered {@code first} (a harness restarted mid-list continues the count). */
    public static synchronized void setFirstGameIndex(int first) {
        GAME.set(first - 1);
    }

    /** Records written so far in this process. */
    public static long written() {
        return WRITTEN.get();
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
        System.out.printf("  merged object options: %d%n", MERGED.get());
        System.out.printf("  object options with no token (option-token fallback): %s%n",
                UNRESOLVED.isEmpty() ? "0" : UNRESOLVED.toString());
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
        List<Cand> cands = new java.util.ArrayList<>();
        int n = pool == null ? 0 : pool.size();
        for (int i = 0; i < n; i++) {
            cands.add(abilityOption(pool.get(i), game));
        }
        // Passing priority is an explicit option, always last, so a policy
        // never needs to know which decision kinds allow passing. Other
        // decision kinds (modes, discards, combat) get no pass option.
        cands.add(Cand.option(VERB_PASS));
        return buildChoiceBody(game, viewer, tokens, histSize, new Prompt(KIND_ABILITY), cands,
                false).body;
    }

    /** The option token of one ability in the priority pool. */
    private static Cand abilityOption(SpellAbility sa, Game game) {
        if (sa == null) {
            return Cand.option("activate");
        }
        String verb;
        String fallback;
        try {
            if (sa.isLandAbility()) {
                verb = "play_land";
                fallback = "Play land";
            } else if (sa.isSpell()) {
                verb = "cast";
                fallback = "Cast this spell";
            } else {
                verb = "activate";
                fallback = "Activate this ability";
            }
        } catch (RuntimeException e) {
            verb = "activate";
            fallback = "Activate this ability";
        }
        Cand c = Cand.option(verb);
        try {
            String text = optionText(String.valueOf(sa), sa, game);
            c.ability(text != null ? text : fallback)
             .targetCandidates(targetCandidates(sa))
             .manaAbility(sa.isManaAbility())
             .edge(Schema.SOURCE_IS, sa.getHostCard());
        } catch (RuntimeException e) {
            c.ability(fallback);
        }
        return c;
    }

    /**
     * Rules text of an ability for an option or prompt component: Forge's
     * rendering (or the fallbacks of {@link EventTokenizer#abilityText}),
     * without object ids, bracketed run context or player names, and with the
     * host's own name replaced by "this creature" / "this permanent" / "this
     * spell" ({@link AbilityText#normalize}). Null, never empty, when there is
     * no text.
     */
    static String optionText(String rendered, SpellAbility sa, Game game) {
        String s;
        try {
            s = EventTokenizer.abilityText(rendered, sa, game);
        } catch (RuntimeException e) {
            s = null;
        }
        forge.game.card.Card host = sa == null ? null : sa.getHostCard();
        if (s != null && host != null) {
            try {
                String self = host.isCreature() ? "this creature"
                        : host.isPermanent() ? "this permanent" : "this spell";
                String n = AbilityText.normalize(s, host, self);
                if (n != null) {
                    s = n;
                }
            } catch (RuntimeException ignored) {
                // keep the unsubstituted text
            }
        }
        return s == null || s.isBlank() ? null : s;
    }

    /** Free text (a Forge prompt or title) normalized like ability text, or null. */
    static String promptText(String raw, Game game) {
        String s = EventTokenizer.normalize(raw, game);
        return s == null || s.isBlank() ? null : s;
    }

    /**
     * A decision that is not a priority action: a blocker assignment, a mode
     * choice, a target.
     *
     * <p>Deliberately the same record shape as {@link #buildBody}: a
     * {@code prompt} token saying what is asked and an ordered {@code actions}
     * candidate list the policy returns an index into. Schema 16: a candidate
     * is either a reference {@code {"ref": ...}} to the existing token of the
     * game object or player it names, or an option token ({@code type}
     * {@code action}) for a choice that has no token of its own. Two object
     * candidates resolving to the same token (identical unknown-position
     * copies in a searched library resolve to their name aggregate) become ONE
     * candidate; {@link Built} maps between option and candidate indices.
     *
     * @param deferPrompt render the prompt only when the record is written
     *        (observe mode: the {@code selected} edges come from the expert's
     *        answer to earlier steps of the same decomposed decision); never
     *        for a body sent as a request
     */
    public static synchronized Built buildChoiceBody(Game game, Player viewer,
            List<StateExtractor.Tok> tokens, int histSize, Prompt prompt,
            List<Cand> options, boolean deferPrompt) {
        Resolver res = new Resolver(tokens, histSize, game, viewer);
        StringBuilder opts = new StringBuilder();
        int[] candOf = new int[options.size()];
        List<Integer> origOf = new java.util.ArrayList<>();
        java.util.Map<String, Integer> byRef = new java.util.HashMap<>();
        for (int i = 0; i < options.size(); i++) {
            Cand c = options.get(i);
            String json;
            if (c.entity != null) {
                String ref = res.ref(c.entity);
                if (ref != null) {
                    Integer prev = byRef.get(ref);
                    if (prev != null) {
                        candOf[i] = prev;
                        MERGED.incrementAndGet();
                        continue;
                    }
                    byRef.put(ref, origOf.size());
                    json = "{\"ref\":\"" + esc(ref) + "\"}";
                } else {
                    UNRESOLVED.computeIfAbsent(prompt.kind(), k -> new AtomicLong()).incrementAndGet();
                    json = c.render(origOf.size(), res);
                }
            } else {
                json = c.render(origOf.size(), res);
            }
            candOf[i] = origOf.size();
            origOf.add(i);
            if (opts.length() > 0) {
                opts.append(',');
            }
            opts.append(json);
        }
        String promptJson = deferPrompt ? PROMPT_PENDING : prompt.render(res);
        String body = buildCore(game, viewer, tokens, histSize, prompt.kind(), promptJson,
                opts.toString());
        int[] orig = new int[origOf.size()];
        for (int i = 0; i < orig.length; i++) {
            orig[i] = origOf.get(i);
        }
        return new Built(body, candOf, orig, deferPrompt ? prompt : null, deferPrompt ? res : null);
    }

    /** Placeholder for a deferred prompt; never valid JSON, so it cannot pass as a request. */
    private static final String PROMPT_PENDING = "\u0000prompt-pending\u0000";

    /** Object candidates with no token the viewer can see, by decision kind (emitted as option tokens). */
    private static final java.util.Map<String, AtomicLong> UNRESOLVED =
            new java.util.concurrent.ConcurrentHashMap<>();
    /** Object options merged into an earlier candidate resolving to the same token. */
    private static final AtomicLong MERGED = new AtomicLong();

    /**
     * A built record body plus the mapping between the caller's option list and
     * the candidate list actually written (they differ when options merge).
     */
    public static final class Built {
        public final String body;
        private final int[] candOf;
        private final int[] origOf;
        private final Prompt deferred;
        private final Resolver resolver;

        Built(String body, int[] candOf, int[] origOf, Prompt deferred, Resolver resolver) {
            this.body = body;
            this.candOf = candOf;
            this.origOf = origOf;
            this.deferred = deferred;
            this.resolver = resolver;
        }

        /** Number of candidates in the record's {@code actions}. */
        public int candidates() {
            return origOf.length;
        }

        /** Candidate index of option {@code orig}, or -1. */
        public int candidateOf(int orig) {
            return orig < 0 || orig >= candOf.length ? -1 : candOf[orig];
        }

        /** First option index behind candidate {@code cand}, or -1. */
        public int optionOf(int cand) {
            return cand < 0 || cand >= origOf.length ? -1 : origOf[cand];
        }

        /** The deferred prompt, still open for {@link Prompt#select}; null if already rendered. */
        public Prompt prompt() {
            return deferred;
        }

        /** The body as written: a deferred prompt rendered now. */
        String complete() {
            return deferred == null ? body : body.replace(PROMPT_PENDING, deferred.render(resolver));
        }
    }

    /**
     * One candidate at a choice point: a reference to the token of a game
     * object or player ({@link #ref}), or an option token ({@link #option}).
     * Option-token components are set through the typed methods so that every
     * key is a literal the key registry scan sees.
     */
    public static final class Cand {
        final Object entity;
        final StateExtractor.Tok tok = new StateExtractor.Tok("action", null);
        final List<Object[]> edges = new java.util.ArrayList<>(2);

        private Cand(Object entity, String verb) {
            this.entity = entity;
            tok.p("verb", verb);
        }

        /** An option token with this {@code verb} (vocab {@code action_verb}). */
        public static Cand option(String verb) {
            return new Cand(null, verb);
        }

        /**
         * A reference to the token of {@code entity} (a card or player). Falls
         * back to an option token with {@code fallbackVerb} when the viewer has
         * no token for it.
         */
        public static Cand ref(Object entity, String fallbackVerb) {
            return new Cand(entity, fallbackVerb);
        }

        public Cand ability(String text) {
            tok.p("ability", text == null || text.isBlank() ? null : text);
            return this;
        }

        public Cand targetCandidates(int n) {
            if (n >= 0) {
                tok.p("target_candidates", n);
            }
            return this;
        }

        public Cand manaAbility(boolean mana) {
            tok.flag("mana_ability", mana);
            return this;
        }

        public Cand cost(String text) {
            tok.p("cost", text == null || text.isBlank() ? null : text);
            return this;
        }

        public Cand value(int v) {
            tok.p("value", v);
            return this;
        }

        /** Schema 17: colour of the mana a payment option adds or spends (vocab {@code mana_color}). */
        public Cand manaColor(String c) {
            tok.p("mana_color", c);
            return this;
        }

        /** Schema 17: how much mana the option represents (produced, or units floating). */
        public Cand manaAmount(int n) {
            if (n > 0) {
                tok.p("mana_amount", n);
            }
            return this;
        }

        /** Schema 17: pool mana that may only be spent on some spells. */
        public Cand restricted(boolean v) {
            tok.flag("restricted", v);
            return this;
        }

        /** Schema 17: the shard a Convoke/Improvise tap pays, e.g. "{G}" or "{1}". */
        public Cand pays(String shard) {
            tok.p("pays", shard == null || shard.isBlank() ? null : shard);
            return this;
        }

        /** Schema 17: snow mana. */
        public Cand snow(boolean v) {
            tok.flag("snow", v);
            return this;
        }

        /**
         * Schema 17: what spending this pool mana does besides paying (Cavern of
         * Souls: the spell can't be countered) -- the producing ability's text.
         */
        public Cand rider(String text) {
            tok.p("rider", text == null || text.isBlank() ? null : text);
            return this;
        }

        /** Edge to a card, player or stack ability; written only if it resolves to a token. */
        public Cand edge(String rel, Object target) {
            if (target != null) {
                edges.add(new Object[] {rel, target});
            }
            return this;
        }

        String render(int index, Resolver res) {
            StateExtractor.Tok t = new StateExtractor.Tok("action", null);
            t.p("id", "a" + index);
            t.props.putAll(tok.props);
            for (Object[] e : edges) {
                String role = (String) e[0];
                t.edge(role, res.ref(e[1]));
            }
            return t.toJson();
        }
    }

    /**
     * The record's decision prompt token ({@code type} {@code decision},
     * {@code id} {@code "q"}): what is being asked. Components through typed
     * methods, as for {@link Cand}.
     */
    public static final class Prompt {
        private final String kind;
        final StateExtractor.Tok tok = new StateExtractor.Tok("decision", null);
        final List<Object[]> edges = new java.util.ArrayList<>(2);
        private final List<Object> selected = new java.util.ArrayList<>();
        private int selectedOther;
        private String ability;

        public Prompt(String kind) {
            this.kind = kind;
            tok.p("id", "q");
            tok.p("kind", kind);
        }

        String kind() {
            return kind;
        }

        /** Forge's prompt or title string, normalized; omitted when blank. */
        public Prompt text(String raw, Game game) {
            tok.p("prompt_text", raw == null ? null : promptText(raw, game));
            return this;
        }

        public Prompt min(int v) {
            tok.p("min", v);
            return this;
        }

        public Prompt max(int v) {
            tok.p("max", v);
            return this;
        }

        /** Schema 20 (announce): what is being announced, e.g. "X". */
        public Prompt announce(String what) {
            tok.p("announce", what == null || what.isBlank() ? null : what);
            return this;
        }

        public Prompt optional(boolean v) {
            tok.flag("optional", v);
            return this;
        }

        /** Schema 18: the cost part or optional cost being paid, e.g. "Sacrifice a creature". */
        public Prompt cost(String text) {
            tok.p("cost", text == null || text.isBlank() ? null : text);
            return this;
        }

        /** Schema 18 (pay_tap, crew): power still to tap; absent when 0. */
        public Prompt remainingPower(int n) {
            if (n > 0) {
                tok.p("remaining_power", n);
            }
            return this;
        }

        /** Schema 17 (payment): the unpaid cost, e.g. "{1}{U}". */
        public Prompt remainingCost(String text) {
            tok.p("remaining_cost", text == null || text.isBlank() ? null : text);
            return this;
        }

        /**
         * Schema 17 (payment): unpaid generic mana, and unpaid shards by the
         * colour (W, U, B, R, G, C) that can pay them -- a hybrid shard counts
         * under each of its colours. Zero counts are omitted.
         */
        public Prompt remaining(int generic, int[] byColour) {
            if (generic > 0) {
                tok.p("remaining_generic", generic);
            }
            if (byColour[0] > 0) {
                tok.p("remaining_W", byColour[0]);
            }
            if (byColour[1] > 0) {
                tok.p("remaining_U", byColour[1]);
            }
            if (byColour[2] > 0) {
                tok.p("remaining_B", byColour[2]);
            }
            if (byColour[3] > 0) {
                tok.p("remaining_R", byColour[3]);
            }
            if (byColour[4] > 0) {
                tok.p("remaining_G", byColour[4]);
            }
            if (byColour[5] > 0) {
                tok.p("remaining_C", byColour[5]);
            }
            return this;
        }

        /**
         * Schema 17 (payment): mana already spent on this cost by colour
         * (W, U, B, R, G, C), how much of it was snow, and life paid for
         * Phyrexian shards. Zero counts are omitted.
         */
        public Prompt spent(int[] byColour, int snow, int life) {
            if (byColour[0] > 0) {
                tok.p("spent_W", byColour[0]);
            }
            if (byColour[1] > 0) {
                tok.p("spent_U", byColour[1]);
            }
            if (byColour[2] > 0) {
                tok.p("spent_B", byColour[2]);
            }
            if (byColour[3] > 0) {
                tok.p("spent_R", byColour[3]);
            }
            if (byColour[4] > 0) {
                tok.p("spent_G", byColour[4]);
            }
            if (byColour[5] > 0) {
                tok.p("spent_C", byColour[5]);
            }
            if (snow > 0) {
                tok.p("spent_snow", snow);
            }
            if (life > 0) {
                tok.p("life_paid", life);
            }
            return this;
        }

        /**
         * Rules text of the asking ability, written only when it has no token
         * to point at (its {@code stack_object} edge does not resolve: a
         * trigger not yet on the stack, a spell being cast).
         */
        public Prompt ability(String text) {
            this.ability = text == null || text.isBlank() ? null : text;
            return this;
        }

        /** Edge to a card, player or stack ability; written only if it resolves. */
        public Prompt edge(String rel, Object target) {
            if (target != null) {
                edges.add(new Object[] {rel, target});
            }
            return this;
        }

        /**
         * Something already chosen earlier in this decomposed decision: a
         * {@code selected} edge if it has a token, and counted in
         * {@code selected_count} either way.
         */
        public Prompt select(Object target) {
            if (target != null) {
                selected.add(target);
            }
            return this;
        }

        /** Already-chosen things that have no token (triggers not yet on the stack). */
        public Prompt selectedOther(int n) {
            selectedOther = n;
            return this;
        }

        String render(Resolver res) {
            StateExtractor.Tok t = new StateExtractor.Tok("decision", null);
            t.props.putAll(tok.props);
            boolean stackResolved = false;
            for (Object[] e : edges) {
                String role = (String) e[0];
                String ref = res.ref(e[1]);
                if (ref != null && Schema.STACK_OBJECT.equals(role)) {
                    stackResolved = true;
                }
                t.edge(role, ref);
            }
            int count = selectedOther;
            for (Object s : selected) {
                t.edge(Schema.SELECTED, res.ref(s));
                count++;
            }
            if (count > 0) {
                t.p("selected_count", count);
            }
            if (!stackResolved) {
                t.p("ability", ability);
            }
            return t.toJson();
        }
    }

    /**
     * Resolves a card, player or stack ability to the token the viewer has for
     * it, through the same id/handle table as edges, preferring a tracking
     * handle ({@code h<N>}) when the token has one. A card the viewer cannot
     * see individually (an unseen hand or library card) resolves to nothing,
     * except a card in the viewer's own library at an unknown position, which
     * resolves to that name's aggregate ({@code n:<name>}).
     */
    static final class Resolver {
        private final java.util.Map<String, StateExtractor.Tok> byId = new java.util.HashMap<>();
        private final java.util.Map<String, StateExtractor.Tok> selfLib = new java.util.HashMap<>();
        private final Game game;
        private final Player viewer;

        Resolver(List<StateExtractor.Tok> tokens, int from, Game game, Player viewer) {
            this.game = game;
            this.viewer = viewer;
            for (int i = Math.max(0, from); tokens != null && i < tokens.size(); i++) {
                StateExtractor.Tok t = tokens.get(i);
                Object id = t.props.get("id");
                if (id == null) {
                    continue;
                }
                byId.put(String.valueOf(id), t);
                if (StateExtractor.isNameAggregateRef(id)) {
                    selfLib.put(String.valueOf(t.props.get("name")), t);
                }
            }
        }

        String ref(Object target) {
            StateExtractor.Tok t = token(target);
            if (t == null) {
                return null;
            }
            Object hs = t.props.get("handles");
            if (hs instanceof List && !((List<?>) hs).isEmpty()) {
                return String.valueOf(((List<?>) hs).get(0));
            }
            Object id = t.props.get("id");
            return id == null ? null : String.valueOf(id);
        }

        private StateExtractor.Tok token(Object target) {
            try {
                if (target instanceof Player) {
                    return byId.get(StateExtractor.playerRef(((Player) target).getId()));
                }
                if (target instanceof forge.game.card.Card) {
                    forge.game.card.Card c = (forge.game.card.Card) target;
                    StateExtractor.Tok own = byId.get(StateExtractor.cardRef(c.getId()));
                    if (own != null) {
                        return GameHistory.isIndividualHidden(own) ? null : own;
                    }
                    forge.game.zone.Zone z = c.getZone();
                    if (z == null) {
                        return null;
                    }
                    if (z.getZoneType() == forge.game.zone.ZoneType.Library && viewer != null
                            && viewer.equals(z.getPlayer())) {
                        return selfLib.get(c.getName());
                    }
                    return null;
                }
                if (target instanceof SpellAbility) {
                    return stackToken((SpellAbility) target);
                }
            } catch (RuntimeException ignored) {
                // unresolvable
            }
            return null;
        }

        /**
         * The token of a spell or ability being resolved or waiting on the
         * stack: an ability's {@code stack_item}, a spell's card token in zone
         * {@code stack} (schema 19), including a spell still being cast (its
         * card is in the Stack zone before its stack instance exists).
         */
        private StateExtractor.Tok stackToken(SpellAbility sa) {
            if (game == null) {
                return null;
            }
            String ref = StateExtractor.stackRef(sa, game);
            return ref == null ? null : byId.get(ref);
        }
    }

    private static String buildCore(Game game, Player viewer,
            List<StateExtractor.Tok> tokens, int histSize, String kind,
            String promptJson, String optionsJson) {
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

        sb.append(",\"prompt\":").append(promptJson);
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

    /**
     * Appends a candidate index outcome and writes the record. A deferred
     * prompt is rendered here, after the caller recorded what earlier steps
     * of the same decomposed decision selected.
     */
    public static synchronized void writeChoice(Built built, int chosen) {
        if (!enabled() || built == null || built.body == null) {
            return;
        }
        try {
            if (out == null) {
                out = new PrintWriter(new FileWriter(FILE));
            }
            String body = built.body;
            StringBuilder sb = new StringBuilder(body.length() + 48).append(built.complete());
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
        if (body.contains(PROMPT_PENDING)) {
            throw new IllegalStateException("request body has a deferred prompt");
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
                int idx = poolIndex(pool, pick);
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
    static int poolIndex(List<SpellAbility> pool, SpellAbility pick) {
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
        // Schema 18 (-Dmagicnet.payment): Forge's AI casts a variant with
        // optional or alternative additional costs added (its
        // getOriginalAndAltCostAbilities); those costs are now their own
        // choose_optional_cost decisions, so the variant is its pool spell.
        if (ManaPayment.ROUTE && pick.isSpell() && pick.getHostCard() != null) {
            for (int i = 0; i < pool.size(); i++) {
                SpellAbility s = pool.get(i);
                if (s != null && s.isSpell() && s.getHostCard() != null
                        && s.getHostCard().getId() == pick.getHostCard().getId() && s.getApi() == pick.getApi()) {
                    return i;
                }
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
