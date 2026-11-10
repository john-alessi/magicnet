package magicnet.bridge;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import forge.deck.Deck;
import forge.deck.io.DeckSerializer;
import forge.game.Game;
import forge.game.GameEndReason;
import forge.game.GameOutcome;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.model.FModel;
import forge.util.MyRandom;

/**
 * Embeds Forge as a library and runs AI-vs-AI games with instrumented controllers.
 *
 * This deliberately does not use the `sim` CLI. Going through Match directly is
 * what makes live (state, legal-action) access possible: the controller holds a
 * reference to the real Game object, so everything the engine knows is reachable
 * in-process.
 *
 * Usage:
 *   Harness --decks DIR [--games N] [--out DIR] [--trace N] [--seed S] [--timeout SEC]
 *   Harness --deck-a FILE --deck-b FILE [--games N] ...   (that pairing every game)
 *   Harness --game-list FILE [--results FILE] [--first-game N] [--exit-on-timeout] ...
 *   Harness --check-decks DIR [--format NAME]
 *
 * A game list has one game per line, {@code deckA<TAB>deckB<TAB>seed[<TAB>number]}:
 * Forge's RNG is re-seeded with that seed before the game, and its records
 * carry {@code number} as their {@code game} (default: the running count). {@code --results} appends
 * one JSON line per game (decks, seed, winner seat, turns, timeout, records
 * written). {@code --exit-on-timeout} exits with status 3 after a timed-out
 * game, because the wedged game thread cannot be stopped and would keep
 * writing into the next game.
 */
public final class Harness {

    private static String deckDir = null;
    private static int games = 20;
    private static String outDir = "out";
    private static int trace = 0;
    private static Long seed = null;
    private static int timeoutSec = 120;
    private static boolean schema = false;
    private static String deckA = null;
    private static String deckB = null;
    private static String gameList = null;
    private static String resultsFile = null;
    private static int firstGame = 0;
    private static boolean exitOnTimeout = false;
    private static String checkDecksDir = null;
    private static String formatName = "Premodern";

    /** One game to play: the two decks and, from a game list, its seed. */
    private record Planned(File a, File b, Long seed, Integer number) {
    }

    public static void main(String[] args) throws Exception {
        parseArgs(args);
        boolean pair = deckA != null || deckB != null;
        if (pair && (deckA == null || deckB == null)) {
            System.err.println("--deck-a and --deck-b go together");
            System.exit(2);
        }
        if (deckDir == null && !pair && gameList == null && checkDecksDir == null) {
            System.err.println("Usage: Harness (--decks DIR | --deck-a FILE --deck-b FILE | --game-list FILE"
                    + " | --check-decks DIR [--format NAME]) [--games N] [--out DIR] [--trace N] [--seed S]"
                    + " [--timeout SEC] [--results FILE] [--first-game N] [--exit-on-timeout]");
            System.exit(2);
        }

        System.out.println("Initializing Forge card database (this takes ~10-20s)...");
        long initStart = System.currentTimeMillis();

        // Forge's ForgeConstants static initializer resolves the assets dir through
        // GuiBase.getInterface(), so the GUI interface must be set before any Forge
        // class is touched. forge.view.Main does this too; we bypass Main entirely,
        // which also means we never run its Sentry.init -- no telemetry from the harness.
        System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
        forge.gui.GuiBase.setInterface(new forge.GuiDesktop());

        FModel.initialize(null, null);
        System.out.printf("Card DB ready in %.1fs%n", (System.currentTimeMillis() - initStart) / 1000.0);

        if (seed != null) {
            MyRandom.setRandom(new Random(seed));
            System.out.println("RNG seeded with " + seed);
        }

        if (checkDecksDir != null) {
            checkDecks(new File(checkDecksDir), formatName);
            System.exit(0);
        }

        List<Planned> plan = new ArrayList<>();
        if (gameList != null) {
            for (String ln : java.nio.file.Files.readAllLines(new File(gameList).toPath())) {
                if (ln.isBlank() || ln.startsWith("#")) {
                    continue;
                }
                String[] f = ln.split("\t");
                plan.add(new Planned(new File(f[0]), new File(f[1]),
                        f.length > 2 && !f[2].isBlank() ? Long.valueOf(f[2].trim()) : null,
                        f.length > 3 && !f[3].isBlank() ? Integer.valueOf(f[3].trim()) : null));
            }
            games = plan.size();
            System.out.println("Game list " + gameList + ": " + games + " games");
        } else if (pair) {
            for (int i = 0; i < games; i++) {
                plan.add(new Planned(new File(deckA), new File(deckB), null, null));
            }
        } else {
            List<File> deckFiles = findDecks(new File(deckDir));
            if (deckFiles.size() < 2) {
                System.err.println("Need at least 2 .dck files in " + deckDir + ", found " + deckFiles.size());
                System.exit(2);
            }
            System.out.println("Found " + deckFiles.size() + " decks in " + deckDir);
            Random pick = seed != null ? new Random(seed * 31 + 7) : new Random();
            for (int i = 0; i < games; i++) {
                File fa = deckFiles.get(pick.nextInt(deckFiles.size()));
                File fb = deckFiles.get(pick.nextInt(deckFiles.size()));
                plan.add(new Planned(fa, fb, null, null));
            }
        }
        if (firstGame > 0) {
            Serializer.setFirstGameIndex(firstGame);
        }
        PrintWriter results = resultsFile == null ? null
                : new PrintWriter(new java.io.FileWriter(resultsFile, true), true);

        System.setProperty("magicnet.outdir", outDir);
        DecisionLog.setTraceLimit(trace);
        DecisionLog.setSchemaSampling(schema);

        GameRules rules = new GameRules(GameType.Constructed);
        rules.setAppliedVariants(EnumSet.of(GameType.Constructed));
        rules.setGamesPerMatch(1);

        int completed = 0;
        int timeouts = 0;
        long wallStart = System.currentTimeMillis();

        boolean stoppedOnTimeout = false;
        for (int i = 0; i < games; i++) {
            Planned pl = plan.get(i);
            File fa = pl.a();
            File fb = pl.b();
            Deck da = fa.isFile() ? DeckSerializer.fromFile(fa) : null;
            Deck db = fb.isFile() ? DeckSerializer.fromFile(fb) : null;
            if (da == null || db == null) {
                System.out.println("  skipped unreadable deck pair");
                writeResult(results, i, -1, pl, null, -1, false, 0, 0, "unreadable deck");
                continue;
            }
            if (pl.seed() != null) {
                MyRandom.setRandom(new Random(pl.seed()));
            }
            if (pl.number() != null) {
                Serializer.setFirstGameIndex(pl.number());
            }
            long written0 = Serializer.written();

            RegisteredPlayer ra = new RegisteredPlayer(da);
            ra.setPlayer(new InstrumentedLobbyPlayer("P0-" + da.getName()));
            RegisteredPlayer rb = new RegisteredPlayer(db);
            rb.setPlayer(new InstrumentedLobbyPlayer("P1-" + db.getName()));

            Match match = new Match(rules, new ArrayList<>(Arrays.asList(ra, rb)), "instrument");
            Serializer.beginGame();
            InstrumentedController.ACTIONS.beginGame(Serializer.gameIndex());
            Game game = match.createGame();
            // Mirrors HostedMatch.startMatch. Bypassing it leaves every
            // PlayerView's opponent list empty, and CardView.canBeShownTo then
            // treats an opponent's sideboard as visible.
            for (Player p : game.getPlayers()) {
                p.updateOpponentsForView();
            }

            long t0 = System.currentTimeMillis();
            Throwable[] thrown = new Throwable[1];
            boolean timedOut = !runWithTimeout(match, game, timeoutSec, thrown);
            long ms = System.currentTimeMillis() - t0;

            if (timedOut) {
                timeouts++;
                game.setGameOver(GameEndReason.Draw);
            }
            DecisionLog.noteGameFinished();
            InstrumentedController.resetHistory();
            LibraryKnowledge.reset();
            RevealKnowledge.reset();
            completed++;

            String winner = "draw/timeout";
            int seat = -1;
            try {
                GameOutcome o = game.getOutcome();
                if (o != null && !o.isDraw() && o.getWinningLobbyPlayer() != null) {
                    winner = o.getWinningLobbyPlayer().getName();
                    seat = winner.startsWith("P0-") ? 0 : winner.startsWith("P1-") ? 1 : -1;
                }
            } catch (Exception ignored) {
                // Outcome may be unset on a hard timeout.
            }
            writeResult(results, i, Serializer.gameIndex(), pl, winner, seat, timedOut, safeTurn(game), ms,
                    thrown[0] == null ? null : String.valueOf(thrown[0]),
                    Serializer.written() - written0);

            System.out.printf("game %2d/%d  %-28s vs %-28s  %6dms  turns=%-3d  %s%n",
                    i + 1, games,
                    truncate(da.getName(), 28), truncate(db.getName(), 28),
                    ms, safeTurn(game), winner);
            if (timedOut && exitOnTimeout) {
                stoppedOnTimeout = true;
                break;
            }
        }
        if (results != null) {
            results.close();
        }
        if (stoppedOnTimeout) {
            // The wedged game thread cannot be stopped; leave before it writes more.
            System.out.println("game timed out; exiting (--exit-on-timeout)");
            Serializer.close();
            System.exit(3);
        }

        double wall = (System.currentTimeMillis() - wallStart) / 1000.0;
        System.out.printf("%n%d games in %.1fs (%.2f games/s single-threaded), %d timeouts%n",
                completed, wall, completed / wall, timeouts);

        DecisionLog.printTopToConsole(30);
        DecisionLog.printTargetingProbe();
        DecisionLog.printTwoStageProbe();
        DecisionLog.printChosenInPool();
        InstrumentedController.printCensus();
        ManaPayment.printSummary();
        PlayFilter.printSummary();
        Announce.printSummary();
        if (InstrumentedController.ENGINE_UNPLAYABLE.get() > 0) {
            System.out.println("engine path: unactivatable abilities refused: "
                    + InstrumentedController.ENGINE_UNPLAYABLE.get());
        }
        Serializer.printSummary();
        InstrumentedController.ACTIONS.printSummary();
        InstrumentedController.ACTIONS.close();
        Serializer.close();
        DecisionLog.printPoolAudit();
        StateStats.print();
        HistoryHub.printCoverage();
        LibraryKnowledge.printCoverage();
        RevealKnowledge.printCoverage();
        OracleIds.printCoverage();
        DecisionLog.printAffordability();
        DecisionLog.printDecisionBudget();
        InstrumentedController.printSimulationProbe();
        ActionCardinality.print();
        MaskCost.print();
        DecisionLog.printFilterSoundness();
        verifyNoHistoryLeak(outDir);

        new File(outDir).mkdirs();
        File summary = new File(outDir, "decision_summary.csv");
        try (PrintWriter w = new PrintWriter(summary, "UTF-8")) {
            DecisionLog.writeSummary(w);
        }
        System.out.println("\nWrote " + summary.getPath());

        File tokStats = new File(outDir, "token_stats.csv");
        try (PrintWriter w = new PrintWriter(tokStats, "UTF-8")) {
            StateStats.writeCsv(w);
        }
        System.out.println("Wrote " + tokStats.getPath());

        if (schema) {
            DecisionLog.printOpaqueParams();
            File sf = new File(outDir, "arg_schema.csv");
            try (PrintWriter w = new PrintWriter(sf, "UTF-8")) {
                DecisionLog.writeSchema(w);
            }
            System.out.println("Wrote " + sf.getPath());
        }

        if (trace > 0) {
            File tf = new File(outDir, "decision_trace.csv");
            try (PrintWriter w = new PrintWriter(tf, "UTF-8")) {
                DecisionLog.writeTrace(w);
            }
            System.out.println("Wrote " + tf.getPath());
        }

        System.out.printf("%ndistinct decision types fired: %d of 111 declared%n",
                DecisionLog.distinctFired());

        // Forge leaves non-daemon threads alive; exit explicitly.
        System.exit(0);
    }

    /**
     * Runs the match on a worker thread so a wedged game cannot hang the harness.
     * Returns false on timeout.
     */
    private static boolean runWithTimeout(Match match, Game game, int sec, Throwable[] thrown) {
        Thread t = new Thread(() -> {
            try {
                match.startGame(game);
            } catch (Throwable e) {
                thrown[0] = e;
                System.out.println("  game threw: " + e);
            }
        }, "forge-game");
        t.setDaemon(true);
        t.start();
        try {
            t.join(sec * 1000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return !t.isAlive();
    }

    private static void writeResult(PrintWriter w, int index, int game, Planned pl, String winner, int seat,
            boolean timedOut, int turns, long ms, String error) {
        writeResult(w, index, game, pl, winner, seat, timedOut, turns, ms, error, 0);
    }

    /** One JSON line per planned game (--results). */
    private static void writeResult(PrintWriter w, int index, int game, Planned pl, String winner, int seat,
            boolean timedOut, int turns, long ms, String error, long records) {
        if (w == null) {
            return;
        }
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"index\":").append(index);
        sb.append(",\"game\":").append(game);
        sb.append(",\"deck_a\":").append(jsonStr(pl.a().getPath()));
        sb.append(",\"deck_b\":").append(jsonStr(pl.b().getPath()));
        sb.append(",\"seed\":").append(pl.seed() == null ? "null" : pl.seed().toString());
        sb.append(",\"winner_seat\":").append(seat < 0 ? "null" : String.valueOf(seat));
        sb.append(",\"winner\":").append(jsonStr(winner));
        sb.append(",\"timeout\":").append(timedOut);
        sb.append(",\"turns\":").append(turns);
        sb.append(",\"ms\":").append(ms);
        sb.append(",\"records\":").append(records);
        sb.append(",\"error\":").append(jsonStr(error));
        sb.append("}");
        w.println(sb);
        w.flush();
    }

    private static String jsonStr(String s) {
        if (s == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.append('"').toString();
    }

    /**
     * --check-decks: one JSON line per .dck file -- whether Forge reads it,
     * card counts, names Forge does not know (unimplemented cards are absent
     * from its card database, and the deck loader drops them), the format's
     * conformance problem (null when legal) and cards Forge's AI is told not
     * to put in decks (AI:RemoveDeck:All; informational).
     */
    private static void checkDecks(File dir, String format) throws java.io.IOException {
        forge.game.GameFormat fmt = FModel.getFormats().getFormat(format);
        if (fmt == null) {
            System.out.println("DECKCHECK-FORMAT-MISSING " + format);
        }
        forge.card.CardDb db = FModel.getMagicDb().getCommonCards();
        for (File f : findDecks(dir)) {
            List<String> unknown = new ArrayList<>();
            int listed = 0;
            String section = "";
            for (String raw : java.nio.file.Files.readAllLines(f.toPath())) {
                String ln = raw.trim();
                if (ln.startsWith("[")) {
                    section = ln.toLowerCase();
                    continue;
                }
                if (!(section.equals("[main]") || section.equals("[sideboard]")) || ln.isEmpty()) {
                    continue;
                }
                java.util.regex.Matcher m = java.util.regex.Pattern.compile("^(\\d+)\\s+(.+)$").matcher(ln);
                if (!m.matches()) {
                    continue;
                }
                listed += Integer.parseInt(m.group(1));
                String name = m.group(2).split("\\|")[0].trim();
                if (db.getRules(name, true) == null && db.getCard(name) == null) {
                    unknown.add(name);
                }
            }
            Deck d = null;
            String err = null;
            try {
                d = DeckSerializer.fromFile(f);
            } catch (Throwable t) {
                err = String.valueOf(t);
            }
            int main = -1;
            int side = -1;
            String problem = null;
            java.util.Set<String> aiRemove = new java.util.TreeSet<>();
            if (d != null) {
                main = d.getMain().countAll();
                side = d.has(forge.deck.DeckSection.Sideboard) ? d.get(forge.deck.DeckSection.Sideboard).countAll() : 0;
                try {
                    problem = fmt == null ? "format not found" : fmt.getDeckConformanceProblem(d);
                } catch (Throwable t) {
                    problem = "conformance check threw: " + t;
                }
                for (java.util.Map.Entry<forge.item.PaperCard, Integer> e : d.getMain()) {
                    try {
                        if (e.getKey().getRules().getAiHints().getRemAIDecks()) {
                            aiRemove.add(e.getKey().getName());
                        }
                    } catch (Throwable ignored) {
                        // informational
                    }
                }
            }
            StringBuilder sb = new StringBuilder("DECKCHECK {");
            sb.append("\"file\":").append(jsonStr(f.getName()));
            sb.append(",\"name\":").append(jsonStr(d == null ? null : d.getName()));
            sb.append(",\"readable\":").append(d != null);
            sb.append(",\"error\":").append(jsonStr(err));
            sb.append(",\"listed\":").append(listed);
            sb.append(",\"main\":").append(main);
            sb.append(",\"sideboard\":").append(side);
            sb.append(",\"unknown\":[");
            for (int i = 0; i < unknown.size(); i++) {
                sb.append(i > 0 ? "," : "").append(jsonStr(unknown.get(i)));
            }
            sb.append("],\"format_problem\":").append(jsonStr(problem));
            sb.append(",\"ai_remove\":[");
            int k = 0;
            for (String n : aiRemove) {
                sb.append(k++ > 0 ? "," : "").append(jsonStr(n));
            }
            sb.append("]}");
            System.out.println(sb);
        }
    }

    private static int safeTurn(Game g) {
        try {
            return g.getPhaseHandler().getTurn();
        } catch (Exception e) {
            return -1;
        }
    }

    private static String truncate(String s, int n) {
        if (s == null) {
            return "?";
        }
        return s.length() <= n ? s : s.substring(0, n - 1) + "~";
    }

    private static List<File> findDecks(File dir) {
        List<File> out = new ArrayList<>();
        File[] kids = dir.listFiles();
        if (kids == null) {
            return out;
        }
        Arrays.sort(kids);
        for (File f : kids) {
            if (f.isDirectory()) {
                out.addAll(findDecks(f));
            } else if (f.getName().toLowerCase().endsWith(".dck")) {
                out.add(f);
            }
        }
        return out;
    }

    private static void parseArgs(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--decks"   -> deckDir = args[++i];
                case "--games"   -> games = Integer.parseInt(args[++i]);
                case "--out"     -> outDir = args[++i];
                case "--trace"   -> trace = Integer.parseInt(args[++i]);
                case "--schema"  -> schema = true;
                case "--seed"    -> seed = Long.parseLong(args[++i]);
                case "--timeout" -> timeoutSec = Integer.parseInt(args[++i]);
                case "--deck-a"  -> deckA = args[++i];
                case "--deck-b"  -> deckB = args[++i];
                case "--game-list" -> gameList = args[++i];
                case "--results" -> resultsFile = args[++i];
                case "--first-game" -> firstGame = Integer.parseInt(args[++i]);
                case "--exit-on-timeout" -> exitOnTimeout = true;
                case "--check-decks" -> checkDecksDir = args[++i];
                case "--format"  -> formatName = args[++i];
                default -> {
                    System.err.println("Unknown arg: " + args[i]);
                    System.exit(2);
                }
            }
        }
    }

    /**
     * Checks the information boundary on history tokens.
     *
     * The specific failure this guards against: history carries card identity, so
     * a library-to-hand CardChangeZone would expose the card the opponent drew.
     * Each dumped state is from one viewer's perspective and names the opposing
     * player, so any history token moving a card into that player's hand must
     * have had its name omitted.
     *
     * This is a real assertion rather than a count, because in self-play both
     * agents share a process and a leak would be silent and fatal to training.
     */
    private static void verifyNoHistoryLeak(String outDir) {
        File statesDir = new File(outDir, "states");
        File[] files = statesDir.listFiles((d, n) -> n.endsWith(".jsonl"));
        if (files == null || files.length == 0) {
            System.out.println("\n=== history redaction: no state dumps to check ===");
            return;
        }
        int checked = 0;
        int leaks = 0;
        String firstLeak = null;
        for (File f : files) {
            try {
                java.util.List<String> lines = java.nio.file.Files.readAllLines(f.toPath());
                String oppId = null;
                for (String ln : lines) {
                    if (ln.contains("\"type\":\"player\"") && ln.contains("\"side\":\"opp\"")) {
                        oppId = field(ln, "id");
                        break;
                    }
                }
                if (oppId == null) {
                    continue;
                }
                // Schema 13: history tokens carry no card identity at all --
                // participants are edges to handles -- so any name or oracle id
                // on a history line is a leak.
                for (String ln : lines) {
                    if (!ln.contains("\"type\":\"history\"")) {
                        continue;
                    }
                    checked++;
                    if (ln.contains("\"name\":") || ln.contains("oracle_id\":")
                            || ln.matches(".*\"to\":\"c\\d+\".*")) {
                        leaks++;
                        if (firstLeak == null) {
                            firstLeak = f.getName() + ": " + ln;
                        }
                    }
                }
            } catch (Exception ignored) {
                // a dump we cannot read is not evidence of a leak
            }
        }
        System.out.println("\n=== history redaction ===");
        System.out.printf("history tokens seen in dumped states:  %d%n", checked);
        System.out.printf("redactions applied across all games:   %d%n",
                StateStats.totalRedactions());
        if (leaks > 0) {
            System.out.printf("LEAK: %d history tokens carry card identity%n", leaks);
            System.out.println("  first: " + firstLeak);
        } else if (checked > 0) {
            System.out.println("ok: no history token carries card identity");
        } else {
            System.out.println("inconclusive: no opponent draws in the dumped sample");
        }
    }

    /** Minimal field read for the flat JSON the extractor emits. */
    private static String field(String line, String key) {
        String k = "\"" + key + "\":";
        int i = line.indexOf(k);
        if (i < 0) {
            return null;
        }
        int j = i + k.length();
        if (j < line.length() && line.charAt(j) == '"') {
            int end = line.indexOf('"', j + 1);
            return end < 0 ? null : line.substring(j + 1, end);
        }
        int end = j;
        while (end < line.length() && "-0123456789".indexOf(line.charAt(end)) >= 0) {
            end++;
        }
        return end > j ? line.substring(j, end) : null;
    }
}
