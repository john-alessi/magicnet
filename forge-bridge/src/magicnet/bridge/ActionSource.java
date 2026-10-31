package magicnet.bridge;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Where the acting player's choice comes from.
 *
 * <p>The default is Forge's own AI. The other implementations exist to validate
 * the action path end to end before a model exists:
 *
 * <ul>
 *   <li>{@code replay} -- feeds back indices recorded in an earlier dump. If the
 *       resulting game is identical, the whole round trip (enumeration order,
 *       serialisation, index decoding, action application) is sound. This is a
 *       genuine differential test: the recording side chose through Forge's own
 *       path, the replay side chooses by index lookup, so the two do not share
 *       logic.</li>
 *   <li>{@code random} -- a seeded uniform choice over the legal pool. Reaches
 *       states the AI never visits, which is where the rewind gets stressed.</li>
 *   <li>{@code socket} -- the real boundary: one JSON request per decision, one
 *       integer reply. A Python agent on the other end.</li>
 * </ul>
 *
 * <p>Every implementation returns an index into the pool, or -1 to pass.
 */
public interface ActionSource {

    /** Sentinel meaning "let Forge's AI decide"; not a pool index. */
    int DELEGATE = Integer.MIN_VALUE;

    /**
     * @param request the decision record as JSON, identical to what the dump
     *                would contain minus the chosen fields
     * @param poolSize number of options, including the explicit pass option
     *                 when there is one
     * @param passIndex index of the "pass priority" option, or -1 when this
     *                  decision has none (modes, discards, combat): passing is
     *                  an option like any other, never a sentinel
     * @return index into the options, or {@link #DELEGATE}
     */
    int choose(String request, int poolSize, int passIndex);

    /**
     * Whether this source decides routed choices itself, as opposed to always
     * returning {@link #DELEGATE}.
     *
     * <p>Building a record commits the per-viewer history cursor, so a body
     * that is built and then discarded silently drops the history events it
     * carried. Callers must consult this <em>before</em> building.
     */
    default boolean decides() {
        return true;
    }

    /** Called once per game so stateful sources can resynchronise. */
    default void beginGame(int game) {
    }

    default void close() {
    }

    /** Diagnostic summary printed at the end of a run. */
    default void printSummary() {
    }

    /**
     * Whether this source needs the serialised decision record. Only the socket
     * source does; building it otherwise would pay the serialisation cost on
     * every decision for nothing.
     */
    default boolean needsRequest() {
        return false;
    }

    // ---- factory ----------------------------------------------------------

    /**
     * Builds the source named by {@code -Dmagicnet.actions=...}:
     * {@code ai} (default), {@code random}, {@code replay:<dump.jsonl>} or
     * {@code socket:<host>:<port>}.
     */
    static ActionSource fromProperties() {
        String spec = System.getProperty("magicnet.actions", "ai").trim();
        if (spec.isEmpty() || "ai".equals(spec)) {
            return new AiSource();
        }
        if ("random".equals(spec)) {
            return new RandomSource(Long.getLong("magicnet.actions.seed", 42L));
        }
        if (spec.startsWith("replay:")) {
            return new ReplaySource(spec.substring("replay:".length()));
        }
        if (spec.startsWith("socket:")) {
            String[] hp = spec.substring("socket:".length()).split(":");
            return new SocketSource(hp[0], Integer.parseInt(hp[1]));
        }
        throw new IllegalArgumentException("unknown magicnet.actions: " + spec);
    }

    // ---- implementations --------------------------------------------------

    /** Delegates to Forge's AI. The default, and the baseline everything else is compared against. */
    final class AiSource implements ActionSource {
        @Override
        public int choose(String request, int poolSize, int passIndex) {
            return DELEGATE;
        }

        @Override
        public boolean decides() {
            return false;
        }
    }

    /**
     * Uniform random choice over the pool, plus an explicit chance to pass.
     *
     * <p>Passing has to stay reachable: a player that never passes cannot let
     * the turn advance, and the game would not terminate.
     */
    final class RandomSource implements ActionSource {
        private final Random rng;
        private final double passRate =
                Double.parseDouble(System.getProperty("magicnet.actions.pass", "0.5"));
        private final AtomicLong chosen = new AtomicLong();
        private final AtomicLong passed = new AtomicLong();

        RandomSource(long seed) {
            this.rng = new Random(seed);
        }

        @Override
        public synchronized int choose(String request, int poolSize, int passIndex) {
            if (passIndex >= 0 && (poolSize <= 1 || rng.nextDouble() < passRate)) {
                passed.incrementAndGet();
                return passIndex;
            }
            chosen.incrementAndGet();
            int n = passIndex >= 0 ? poolSize - 1 : poolSize;
            int pick = rng.nextInt(n);
            return passIndex >= 0 && pick >= passIndex ? pick + 1 : pick;
        }

        @Override
        public void printSummary() {
            System.out.printf("  random agent: %d actions, %d passes%n",
                    chosen.get(), passed.get());
        }
    }

    /**
     * Replays the {@code chosen} indices from an earlier dump, in order.
     *
     * <p>Deliberately does *not* try to re-match by content: the point is to test
     * whether a bare index is sufficient to reproduce a game. Running out of
     * recorded decisions means the replayed game diverged and went longer than
     * the original, which is itself the test failing.
     */
    final class ReplaySource implements ActionSource {
        private final List<int[]> recorded = new ArrayList<>();
        private int cursor;
        private final AtomicLong overrun = new AtomicLong();

        ReplaySource(String path) {
            try {
                for (String line : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
                    if (line.isEmpty()) {
                        continue;
                    }
                    int g = intField(line, "\"game\":");
                    int d = intField(line, "\"decision\":");
                    int c = intField(line, "\"chosen\":");
                    recorded.add(new int[] { g, d, c });
                }
            } catch (IOException e) {
                throw new IllegalStateException("cannot read replay dump: " + path, e);
            }
        }

        private static int intField(String line, String key) {
            int i = line.indexOf(key);
            if (i < 0) {
                return Integer.MIN_VALUE;
            }
            int j = i + key.length();
            int k = j;
            while (k < line.length() && (line.charAt(k) == '-' || Character.isDigit(line.charAt(k)))) {
                k++;
            }
            return Integer.parseInt(line.substring(j, k));
        }

        @Override
        public synchronized int choose(String request, int poolSize, int passIndex) {
            if (cursor >= recorded.size()) {
                overrun.incrementAndGet();
                return passIndex >= 0 ? passIndex : 0;
            }
            int[] rec = recorded.get(cursor++);
            int idx = rec[2];
            if (idx < 0 || idx >= poolSize) {
                // The pool differs from when recorded: the games have already
                // diverged. Pass (or take the first option where passing is not
                // possible) rather than pick an unrelated action, so the
                // divergence is attributable to this decision.
                overrun.incrementAndGet();
                return passIndex >= 0 ? passIndex : 0;
            }
            return idx;
        }

        @Override
        public void printSummary() {
            System.out.printf("  replay: %d recorded, %d consumed, %d overruns%n",
                    recorded.size(), cursor, overrun.get());
        }
    }

    /**
     * One decision per line over a TCP socket: JSON request out, integer reply in.
     *
     * <p>TCP rather than stdio because Forge logs to stdout, and interleaving a
     * protocol with game logs would be unreadable and fragile. One connection per
     * environment, so a vectorised trainer runs N processes against N ports.
     */
    final class SocketSource implements ActionSource {
        private final Socket socket;
        private final PrintWriter out;
        private final BufferedReader in;
        private final AtomicLong calls = new AtomicLong();
        private final AtomicLong errors = new AtomicLong();

        SocketSource(String host, int port) {
            try {
                socket = new Socket(host, port);
                socket.setTcpNoDelay(true);
                out = new PrintWriter(new java.io.OutputStreamWriter(
                        socket.getOutputStream(), StandardCharsets.UTF_8));
                in = new BufferedReader(new InputStreamReader(
                        socket.getInputStream(), StandardCharsets.UTF_8));
            } catch (IOException e) {
                throw new IllegalStateException("cannot connect to agent at "
                        + host + ":" + port, e);
            }
        }

        @Override
        public boolean needsRequest() {
            return true;
        }

        @Override
        public synchronized int choose(String request, int poolSize, int passIndex) {
            try {
                calls.incrementAndGet();
                out.println(request);
                out.flush();
                String reply = in.readLine();
                if (reply == null) {
                    errors.incrementAndGet();
                    throw new IllegalStateException(
                            "agent closed the connection mid-game after "
                            + calls.get() + " decisions");
                }
                int pick = Integer.parseInt(reply.trim());
                if (pick < 0 || pick >= poolSize) {
                    errors.incrementAndGet();
                    throw new IllegalStateException("agent returned index " + pick
                            + " outside [0, " + poolSize + ")");
                }
                return pick;
            } catch (IOException | NumberFormatException e) {
                // Deliberately fatal. Returning -1 here would make a broken or
                // crashed agent indistinguishable from a policy that always
                // passes, and training would silently collapse to that.
                errors.incrementAndGet();
                throw new IllegalStateException("agent transport failed after "
                        + calls.get() + " decisions", e);
            }
        }

        @Override
        public synchronized void beginGame(int game) {
            out.println("{\"event\":\"begin_game\",\"game\":" + game + "}");
            out.flush();
            try {
                // Acknowledgement is required, not optional: this read blocks.
                // It is what keeps the two sides in lockstep at a game boundary.
                if (in.readLine() == null) {
                    throw new IllegalStateException(
                            "agent closed the connection at start of game " + game);
                }
            } catch (IOException e) {
                throw new IllegalStateException(
                        "agent failed to acknowledge start of game " + game, e);
            }
        }

        @Override
        public synchronized void close() {
            try {
                out.println("{\"event\":\"end\"}");
                out.flush();
                socket.close();
            } catch (IOException ignored) {
                // Nothing useful to do on a failed close.
            }
        }

        @Override
        public void printSummary() {
            System.out.printf("  socket agent: %d decisions, %d errors%n",
                    calls.get(), errors.get());
        }
    }
}
