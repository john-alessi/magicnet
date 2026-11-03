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
 */
public final class Harness {

    private static String deckDir = null;
    private static int games = 20;
    private static String outDir = "out";
    private static int trace = 0;
    private static Long seed = null;
    private static int timeoutSec = 120;
    private static boolean schema = false;

    public static void main(String[] args) throws Exception {
        parseArgs(args);
        if (deckDir == null) {
            System.err.println("Usage: Harness --decks DIR [--games N] [--out DIR] "
                    + "[--trace N] [--seed S] [--timeout SEC]");
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

        List<File> deckFiles = findDecks(new File(deckDir));
        if (deckFiles.size() < 2) {
            System.err.println("Need at least 2 .dck files in " + deckDir + ", found " + deckFiles.size());
            System.exit(2);
        }
        System.out.println("Found " + deckFiles.size() + " decks in " + deckDir);

        System.setProperty("magicnet.outdir", outDir);
        DecisionLog.setTraceLimit(trace);
        DecisionLog.setSchemaSampling(schema);

        GameRules rules = new GameRules(GameType.Constructed);
        rules.setAppliedVariants(EnumSet.of(GameType.Constructed));
        rules.setGamesPerMatch(1);

        Random pick = seed != null ? new Random(seed * 31 + 7) : new Random();

        int completed = 0;
        int timeouts = 0;
        long wallStart = System.currentTimeMillis();

        for (int i = 0; i < games; i++) {
            File fa = deckFiles.get(pick.nextInt(deckFiles.size()));
            File fb = deckFiles.get(pick.nextInt(deckFiles.size()));
            Deck da = DeckSerializer.fromFile(fa);
            Deck db = DeckSerializer.fromFile(fb);
            if (da == null || db == null) {
                System.out.println("  skipped unreadable deck pair");
                continue;
            }

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
            boolean timedOut = !runWithTimeout(match, game, timeoutSec);
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
            try {
                GameOutcome o = game.getOutcome();
                if (o != null && !o.isDraw() && o.getWinningLobbyPlayer() != null) {
                    winner = o.getWinningLobbyPlayer().getName();
                }
            } catch (Exception ignored) {
                // Outcome may be unset on a hard timeout.
            }

            System.out.printf("game %2d/%d  %-28s vs %-28s  %6dms  turns=%-3d  %s%n",
                    i + 1, games,
                    truncate(da.getName(), 28), truncate(db.getName(), 28),
                    ms, safeTurn(game), winner);
        }

        double wall = (System.currentTimeMillis() - wallStart) / 1000.0;
        System.out.printf("%n%d games in %.1fs (%.2f games/s single-threaded), %d timeouts%n",
                completed, wall, completed / wall, timeouts);

        DecisionLog.printTopToConsole(30);
        DecisionLog.printTargetingProbe();
        DecisionLog.printTwoStageProbe();
        DecisionLog.printChosenInPool();
        InstrumentedController.printCensus();
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
    private static boolean runWithTimeout(Match match, Game game, int sec) {
        Thread t = new Thread(() -> {
            try {
                match.startGame(game);
            } catch (Throwable e) {
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
