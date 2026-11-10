"""Generate observe-mode (Forge AI vs Forge AI) imitation dumps at scale.

Each game samples both seats' decks independently by metagame share and gets
its own seed. Games are grouped into shards; a worker plays one shard in one
JVM (``Harness --game-list``), streams nothing to stdout, then gzips the
shard's JSONL into ``OUT/shard_XXXXX.jsonl.gz``, validates it with
``scripts/validate_dump.py`` and appends one manifest line per game to
``OUT/manifest.jsonl``.

Usage (from the repo root)::

    uv run scripts/generate_dumps.py --out data/dumps/premodern_v1 --games 2000 \\
        --shares my_goldfish_shares.csv [--workers 5] [--heap 1536m] [--games-per-shard 50]

Re-running the same command resumes: shards whose manifest lines and gzip are
complete are skipped; partial shards are deleted and replayed (the plan is
deterministic in ``--seed``). Ctrl-C once: finish the running shards and
stop; twice (or SIGTERM/SIGHUP): abandon them (their partial files are deleted).

Memory safety: only one generator may use an output directory at a time
(``OUT/.lock``); ``--workers`` is reduced so that workers x (heap +
``JVM_OVERHEAD_MB``) stays within half of MemAvailable; every JVM runs with
``-XX:+ExitOnOutOfMemoryError``, a high ``oom_score_adj`` and a parent-death
signal (``setpriv --pdeathsig KILL``), so no harness outlives the generator.
Per-JVM peak RSS, CPU use, startup and per-game time go to ``OUT/stats.jsonl``.

Read the output with ``magicnet.dump.load_many("OUT")``.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import fcntl
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "forge-bridge"
FORGE_JAR = Path(os.environ.get("FORGE_JAR", "/usr/share/java/forge-gui-desktop.jar"))
DEFAULT_DECKS = BRIDGE / "fixtures" / "premodern"
VALIDATOR = REPO / "scripts" / "validate_dump.py"
FLAGS = [
    "-Dmagicnet.actions=ai", "-Dmagicnet.choices=true", "-Dmagicnet.combat=true",
    "-Dmagicnet.payment=true", "-Dmagicnet.labels=true",
]
# A worker JVM dies cleanly on heap exhaustion (its game is recorded as crashed)
# instead of thrashing; metaspace and GC threads are bounded so N workers do
# not each size themselves for the whole machine.
JVM_SAFETY = [
    "-XX:+ExitOnOutOfMemoryError", "-XX:MaxMetaspaceSize=512m",
    "-XX:ParallelGCThreads=2", "-XX:ConcGCThreads=1",
]
# Measured non-heap footprint of a harness JVM (metaspace, code cache, thread
# stacks, GC structures); used only to cap --workers.
JVM_OVERHEAD_MB = 768
# Fraction of MemAvailable all workers together may claim.
MEM_FRACTION = 0.5
RECORD_PREFIX = re.compile(rb'^\{"schema":(\d+),"game":(\d+),')
DECK_PREFIX = re.compile(r"^\[[^\]]*\]\s*#\d+\s*-\s*")

sys.path.insert(0, str(REPO / "src"))
from magicnet.dump import SUPPORTED_SCHEMA  # noqa: E402


# ---- decks and shares ----------------------------------------------------------------

def archetype_of(deck_file: str) -> str:
    """Display archetype of a Forge deck file: "[Premodern] #05 - Naya Terragedon.dck" -> "Naya Terragedon"."""
    return DECK_PREFIX.sub("", Path(deck_file).stem).strip()


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", DECK_PREFIX.sub("", name).lower()).strip()


def java_cmd(args: argparse.Namespace) -> list[str]:
    cmd = _guard([args.java, f"-Xmx{args.heap}", *JVM_SAFETY, *FLAGS, *args.jvm_arg])
    if not os.environ.get("DISPLAY"):
        cmd = _guard(["xvfb-run", "-a", *cmd])
    return cmd


# ---- memory and process safety -----------------------------------------------------

def heap_mb(heap: str) -> int:
    """-Xmx value ("2048m", "2g", "1536M", bytes) in MiB."""
    m = re.fullmatch(r"(\d+)([kmgt]?)", heap.strip().lower())
    if not m:
        raise ValueError(f"unrecognised heap size {heap!r}")
    n, unit = int(m.group(1)), m.group(2)
    return int(n * {"": 2**-20, "k": 2**-10, "m": 1, "g": 2**10, "t": 2**20}[unit])


def mem_available_mb(meminfo: str = "/proc/meminfo") -> int | None:
    try:
        for ln in Path(meminfo).read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def cap_workers(requested: int, heap: int, available: int | None) -> int:
    """Largest worker count <= requested with workers x (heap + overhead) within
    MEM_FRACTION of MemAvailable (at least 1)."""
    if available is None:
        return max(1, requested)
    fit = int(available * MEM_FRACTION) // (heap + JVM_OVERHEAD_MB)
    return max(1, min(requested, fit))


def _guard(cmd: list[str]) -> list[str]:
    """Kill the process if the thread that started it dies (generator killed,
    even by SIGKILL): PR_SET_PDEATHSIG via setpriv, kept across exec."""
    return ["setpriv", "--pdeathsig", "KILL", *cmd] if HAVE_SETPRIV else cmd


HAVE_SETPRIV = shutil.which("setpriv") is not None
LIVE: set[subprocess.Popen] = set()
LIVE_LOCK = threading.Lock()


def kill_live() -> None:
    """Kills the process group of every harness still running (atexit, signals)."""
    with LIVE_LOCK:
        procs = list(LIVE)
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def acquire_lock(out: Path):
    """One generator per output directory. Returns the open lock file (keep it)."""
    fh = (out / ".lock").open("a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        sys.exit(f"another generator is using {out} (pid {fh.read().strip() or '?'}); refusing to start")
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def check_decks(args: argparse.Namespace, out: Path) -> list[dict]:
    """Harness --check-decks: readable, unknown cards, format conformance."""
    log = out / "logs" / "deck_check.log"
    cmd = java_cmd(args) + ["-cp", f"build:{FORGE_JAR}", "magicnet.bridge.Harness",
                            "--check-decks", str(args.decks.resolve()), "--format", args.format]
    proc = subprocess.Popen(_prefer_oom_kill(cmd), cwd=BRIDGE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    with LIVE_LOCK:
        LIVE.add(proc)
    try:
        fh_text, _ = proc.communicate()
    finally:
        with LIVE_LOCK:
            LIVE.discard(proc)
    log.write_text(fh_text)
    rows = [json.loads(ln[len("DECKCHECK "):]) for ln in fh_text.splitlines() if ln.startswith("DECKCHECK ")]
    if not rows:
        sys.exit(f"deck check produced no output (rc {proc.returncode}); see {log}")
    return rows


def usable_decks(rows: list[dict], keep_suspicious: bool) -> tuple[list[str], list[tuple[str, str]]]:
    ok, bad = [], []
    for r in rows:
        why = None
        if not r["readable"]:
            why = f"unreadable ({r['error']})"
        elif r["unknown"]:
            why = f"cards Forge does not implement: {', '.join(r['unknown'])}"
        elif r["format_problem"]:
            why = f"not format-legal: {r['format_problem']}"
        elif r["main"] + r["sideboard"] != r["listed"]:
            why = f"Forge loaded {r['main'] + r['sideboard']} of {r['listed']} listed cards"
        elif not keep_suspicious and r["main"] >= 70 and r["sideboard"] == 0:
            why = f"suspicious: {r['main']}-card main and no sideboard (sideboard merged into main?)"
        if why:
            bad.append((r["file"], why))
        else:
            ok.append(r["file"])
    return sorted(ok), bad


def read_shares(path: Path) -> list[tuple[str, float, str | None]]:
    rows = []
    with path.open(newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    for rec in csv.DictReader(lines):
        rec = {(k or "").strip().lower(): (v or "").strip() for k, v in rec.items()}
        if not rec.get("archetype"):
            continue
        rows.append((rec["archetype"], float(rec["share"]), rec.get("deck") or None))
    return rows


def weights(decks: list[str], shares: Path | None) -> dict[str, float]:
    """Deck file -> sampling weight (sums to 1). Archetype shares are split
    equally over the decks they match; unmatched rows and decks are printed."""
    if shares is None:
        return {d: 1.0 / len(decks) for d in decks}
    by_arch: dict[str, list[str]] = {}
    for d in decks:
        by_arch.setdefault(norm(archetype_of(d)), []).append(d)
    w: Counter = Counter()
    unmatched = []
    for arch, share, deck in read_shares(shares):
        if deck:
            hits = [d for d in decks if d == deck or norm(d) == norm(deck)]
        else:
            hits = by_arch.get(norm(arch), [])
        if not hits or share <= 0:
            unmatched.append((arch, share, deck))
            continue
        for d in hits:
            w[d] += share / len(hits)
    for arch, share, deck in unmatched:
        print(f"  unmatched archetype: {arch!r} (share {share}{', deck ' + deck if deck else ''})")
    for d in decks:
        if d not in w:
            print(f"  deck with no share (never sampled): {d}")
    total = sum(w.values())
    if total <= 0:
        sys.exit("no archetype in --shares matched a usable deck")
    print(f"  shares matched: {sum(1 for _ in w)} decks, {total:.4f} of the listed share (renormalized)")
    return {d: v / total for d, v in w.items()}


# ---- plan --------------------------------------------------------------------------

@dataclass
class Game:
    shard: int
    index: int
    deck_a: str
    deck_b: str
    seed: int


def plan(n_games: int, per_shard: int, master: int, w: dict[str, float]) -> list[list[Game]]:
    decks = sorted(w)
    p = [w[d] for d in decks]
    shards = []
    for s in range((n_games + per_shard - 1) // per_shard):
        games = []
        for g in range(min(per_shard, n_games - s * per_shard)):
            rng = random.Random(f"{master}:{s}:{g}")
            a, b = rng.choices(decks, weights=p, k=2)
            games.append(Game(s, g, a, b, rng.getrandbits(62)))
        shards.append(games)
    return shards


# ---- provenance --------------------------------------------------------------------

def bridge_hash() -> str:
    h = hashlib.sha256()
    classes = sorted((BRIDGE / "build").rglob("*.class"))
    if not classes:
        sys.exit("forge-bridge/build has no classes: run forge-bridge/build.sh first")
    for f in classes:
        h.update(str(f.relative_to(BRIDGE)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def forge_info() -> dict:
    st = FORGE_JAR.stat()
    version = None
    try:
        with zipfile.ZipFile(FORGE_JAR) as z:
            for ln in z.read("META-INF/MANIFEST.MF").decode(errors="replace").splitlines():
                if ln.startswith("Implementation-Version:"):
                    version = ln.split(":", 1)[1].strip()
    except (KeyError, zipfile.BadZipFile):
        pass
    return {"path": str(FORGE_JAR), "version": version, "mtime": int(st.st_mtime), "size": st.st_size}


# ---- shards ------------------------------------------------------------------------

def shard_name(s: int) -> str:
    return f"shard_{s:05d}"


def gzip_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


@dataclass
class Totals:
    games: int = 0
    records: int = 0
    gz_bytes: int = 0
    failed_games: int = 0
    timeouts: int = 0
    invalid_shards: int = 0
    shards: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class Stop:
    def __init__(self) -> None:
        self.soft = threading.Event()
        self.hard = threading.Event()

    def handler(self, signum, frame) -> None:  # noqa: ARG002
        if not self.soft.is_set():
            self.soft.set()
            print("\n[ctrl-c] finishing the running shards, then stopping (ctrl-c again to abandon them)",
                  flush=True)
        else:
            self.hard.set()
            print("\n[ctrl-c] abandoning the running shards", flush=True)


OOM_SCORE_ADJ = 1000


def _prefer_oom_kill(cmd: list[str]) -> list[str]:
    """If memory runs out, the kernel kills a harness (its game is recorded as
    crashed and the shard continues past it) before anything else on the
    machine. Set through a shell, not preexec_fn, which is unsafe with threads."""
    if not OOM_SCORE_ADJ:
        return cmd
    return ["sh", "-c", f'echo {OOM_SCORE_ADJ} > /proc/self/oom_score_adj 2>/dev/null; exec "$@"', "sh", *cmd]


CLK_TCK = os.sysconf("SC_CLK_TCK")


def _java_pids(pgid: int) -> list[int]:
    pids = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        if comm == "java" and int(fields[2]) == pgid:
            pids.append(int(d.name))
    return pids


def _sample(pid: int) -> tuple[int, float] | None:
    """(peak RSS MiB = VmHWM, CPU seconds = utime + stime) of a live process."""
    try:
        hwm = next(int(ln.split()[1]) for ln in Path(f"/proc/{pid}/status").read_text().splitlines()
                   if ln.startswith("VmHWM:"))
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        return hwm // 1024, (int(fields[11]) + int(fields[12])) / CLK_TCK
    except (OSError, StopIteration, ValueError, IndexError):
        return None


@dataclass
class JvmStats:
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_mb: int = 0


def run_jvm(cmd: list[str], log: Path, stop: Stop, stats: JvmStats | None = None) -> int | None:
    """Runs one harness process; None if abandoned (its process group is killed).
    Samples the JVM's peak RSS and CPU time once a second into *stats*."""
    with log.open("a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        t0 = time.time()
        proc = subprocess.Popen(_prefer_oom_kill(cmd), cwd=BRIDGE, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        with LIVE_LOCK:
            LIVE.add(proc)
        java = None
        try:
            while True:
                try:
                    rc = proc.wait(timeout=1.0)
                    # Killed by a SIGTERM/SIGHUP abandon: not a crash of its game.
                    return None if stop.hard.is_set() else rc
                except subprocess.TimeoutExpired:
                    if stop.hard.is_set():
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                        return None
                    if stats is not None:
                        if java is None:
                            found = _java_pids(proc.pid)
                            java = found[0] if found else None
                        s = _sample(java) if java else None
                        if s:
                            stats.peak_rss_mb = max(stats.peak_rss_mb, s[0])
                            stats.cpu_s = max(stats.cpu_s, s[1])
        finally:
            with LIVE_LOCK:
                LIVE.discard(proc)
            if stats is not None:
                stats.wall_s = time.time() - t0


def play_shard(games: list[Game], out: Path, args: argparse.Namespace, stop: Stop,
               jvms: list[dict] | None = None) -> dict | None:
    """Plays a shard; returns per-game results (index -> result) or None if
    abandoned. Appends one stats dict per JVM run to *jvms*."""
    name = shard_name(games[0].shard)
    work = out / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"{name}.log"
    results_path = work / "results.jsonl"
    results: dict[int, dict] = {}
    part = 0
    no_progress = 0
    while len(results) < len(games):
        if stop.hard.is_set():
            return None
        todo = [g for g in games if g.index not in results]
        first = todo[0].index
        lst = work / f"games_{part}.tsv"
        lst.write_text("".join(f"{args.decks / g.deck_a}\t{args.decks / g.deck_b}\t{g.seed}\t{g.index}\n"
                               for g in todo))
        cmd = java_cmd(args) + [
            f"-Dmagicnet.jsonl={work / f'part_{part}.jsonl'}",
            "-cp", f"build:{FORGE_JAR}", "magicnet.bridge.Harness",
            "--game-list", str(lst), "--results", str(results_path), "--first-game", str(first),
            "--exit-on-timeout", "--timeout", str(args.timeout), "--out", str(work / f"out_{part}"),
        ]
        st = JvmStats()
        rc = run_jvm(cmd, log, stop, st)
        if rc is None:
            return None
        before = len(results)
        n_before = set(results)
        if results_path.exists():
            for ln in results_path.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                gi = todo[r["index"]].index if r.get("part") is None and r["index"] < len(todo) else None
                if gi is not None and gi not in results:
                    r["part"] = part
                    results[gi] = r
            results_path.unlink()
        if jvms is not None:
            played = [results[g] for g in results if g not in n_before]
            game_s = sum((r.get("ms") or 0) for r in played) / 1000
            jvms.append({"part": part, "rc": rc, "games": len(played), "wall_s": round(st.wall_s, 1),
                         "game_s": round(game_s, 1), "startup_s": round(st.wall_s - game_s, 1),
                         "cpu_s": round(st.cpu_s, 1), "peak_rss_mb": st.peak_rss_mb,
                         "cpu_cores": round(st.cpu_s / st.wall_s, 2) if st.wall_s else None})
        if len(results) < len(games) and (rc not in (0, 3) or len(results) == before):
            # The JVM died inside a game: that game is lost, the rest are retried
            # (a game that kills two JVMs in a row, or a run that reports
            # nothing, is given up rather than retried forever).
            nxt = next(g.index for g in games if g.index not in results)
            if len(results) == before:
                no_progress += 1
            if (len(results) > before and rc not in (0, 3)) or no_progress >= 2:
                results[nxt] = {"crashed": True, "error": f"harness exited with status {rc}", "part": part,
                                "game": nxt, "records": 0}
                no_progress = 0
        part += 1
    return results


def write_shard(games: list[Game], results: dict, out: Path) -> tuple[Path, Counter, int]:
    """Concatenates the parts into the shard's gzip, keeping only records of
    games that were played to an end (or timed out) and parse as records."""
    name = shard_name(games[0].shard)
    work = out / "work" / name
    # A game's records come only from the part (JVM run) that finished it: a
    # retried game also left a partial copy in the part that died inside it.
    keep = {g: r["part"] for g, r in results.items() if not r.get("crashed") and not r.get("error")}
    per_game: Counter = Counter()
    dropped = 0
    tmp = out / f"{name}.jsonl.gz.tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as gz:
        for part_file in sorted(work.glob("part_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
            part = int(part_file.stem.split("_")[1])
            with part_file.open("rb") as fh:
                for line in fh:
                    m = RECORD_PREFIX.match(line)
                    if not m or not line.endswith(b"\n") or int(m.group(1)) != SUPPORTED_SCHEMA:
                        dropped += 1
                        continue
                    g = int(m.group(2))
                    if keep.get(g) != part:
                        dropped += 1
                        continue
                    gz.write(line)
                    per_game[g] += 1
    final = out / f"{name}.jsonl.gz"
    tmp.replace(final)
    return final, per_game, dropped


def validate(path: Path, log: Path) -> tuple[bool, list[str]]:
    proc = subprocess.run([sys.executable, str(VALIDATOR), "--dump", str(path)], capture_output=True, text=True,
                          start_new_session=True)
    with log.open("a") as fh:
        fh.write(f"\n$ validate_dump.py --dump {path}\n{proc.stdout}{proc.stderr}")
    fails = [ln.strip() for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("FAIL")]
    if proc.returncode != 0 and not fails:
        fails = [f"validator exited with status {proc.returncode}: {proc.stderr.strip()[-300:]}"]
    return not fails, fails


def manifest_line(g: Game, r: dict, n_records: int, valid: bool, fails: list[str], prov: dict) -> dict:
    crashed = bool(r.get("crashed"))
    timeout = bool(r.get("timeout"))
    seat = r.get("winner_seat")
    return {
        "shard": g.shard, "game": g.index,
        "deck_a": {"archetype": archetype_of(g.deck_a), "file": g.deck_a},
        "deck_b": {"archetype": archetype_of(g.deck_b), "file": g.deck_b},
        "seed": g.seed, "schema": SUPPORTED_SCHEMA,
        "forge": prov["forge"], "bridge_build": prov["bridge_build"],
        "wall_ms": r.get("ms"),
        "outcome": {
            "winner": None if seat is None or timeout or crashed else ("a" if seat == 0 else "b"),
            "turns": r.get("turns"),
            "timeout": timeout,
            "draw": not crashed and not timeout and seat is None and not r.get("error"),
            "crashed": crashed,
            "error": r.get("error"),
        },
        "records": n_records,
        "shard_valid": valid,
        "validation_failures": fails,
    }


def worker(queue: list[list[Game]], qlock: threading.Lock, out: Path, args: argparse.Namespace, stop: Stop,
           totals: Totals, mlock: threading.Lock, prov: dict) -> None:
    while not stop.soft.is_set():
        with qlock:
            if not queue:
                return
            games = queue.pop(0)
        name = shard_name(games[0].shard)
        t0 = time.time()
        jvms: list[dict] = []
        results = play_shard(games, out, args, stop, jvms)
        if results is None:
            cleanup(out, games[0].shard)
            print(f"  {name}: abandoned", flush=True)
            continue
        path, per_game, dropped = write_shard(games, results, out)
        valid, fails = (True, []) if args.no_validate else validate(path, out / "logs" / f"{name}.log")
        lines = []
        for g in games:
            r = results[g.index]
            # The record's game number is the shard-local index (--first-game).
            lines.append(manifest_line(g, r, per_game.get(g.index, 0), valid, fails, prov))
        with mlock:
            with (out / "stats.jsonl").open("a") as fh:
                fh.write(json.dumps({"shard": games[0].shard, "heap": args.heap, "jvms": jvms}) + "\n")
            with (out / "manifest.jsonl").open("a") as fh:
                for ln in lines:
                    fh.write(json.dumps(ln) + "\n")
        clean_work(out, games[0].shard)
        size = path.stat().st_size
        bad = [ln for ln in lines if ln["outcome"]["crashed"] or ln["outcome"]["error"]]
        tos = [ln for ln in lines if ln["outcome"]["timeout"]]
        with totals.lock:
            totals.games += len(games)
            totals.records += sum(per_game.values())
            totals.gz_bytes += size
            totals.failed_games += len(bad)
            totals.timeouts += len(tos)
            totals.invalid_shards += 0 if valid else 1
            totals.shards += 1
        wall = sum(j["wall_s"] for j in jvms) or 1e-9
        print(f"  {name}: {len(games)} games, {sum(per_game.values())} records, {size / 1e6:.1f} MB gz, "
              f"{time.time() - t0:.0f}s ({len(jvms)} JVM run(s): startup {sum(j['startup_s'] for j in jvms):.0f}s, "
              f"peak RSS {max((j['peak_rss_mb'] for j in jvms), default=0)} MB, "
              f"{sum(j['cpu_s'] for j in jvms) / wall:.2f} cores), "
              f"{'valid' if valid else 'INVALID: ' + '; '.join(fails)}"
              + (f", {len(bad)} crashed/errored" if bad else "") + (f", {len(tos)} timeouts" if tos else "")
              + (f", {dropped} partial lines dropped" if dropped else ""), flush=True)


def clean_work(out: Path, s: int) -> None:
    work = out / "work" / shard_name(s)
    if work.is_dir():
        for f in work.rglob("*"):
            if f.is_file():
                f.unlink()
        for d in sorted((d for d in work.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
            d.rmdir()
        work.rmdir()


def cleanup(out: Path, s: int) -> None:
    """Removes what this script created for an unfinished shard."""
    clean_work(out, s)
    for suffix in (".jsonl.gz", ".jsonl.gz.tmp"):
        f = out / f"{shard_name(s)}{suffix}"
        if f.exists():
            f.unlink()


def reporter(totals: Totals, t0: float, every: float, done: threading.Event, total_games: int) -> None:
    while not done.wait(every):
        report(totals, t0, total_games)


def report(totals: Totals, t0: float, total_games: int) -> None:
    hours = max(time.time() - t0, 1e-9) / 3600
    with totals.lock:
        g, r, b = totals.games, totals.records, totals.gz_bytes
    if g == 0:
        print(f"[progress] {time.time() - t0:.0f}s elapsed, no shard finished yet", flush=True)
        return
    eta = (total_games - g) / (g / hours) if g else float("nan")
    print(f"[progress] {g}/{total_games} games: {g / hours:.0f} games/h, {r / hours:,.0f} records/h, "
          f"{b / 1e6 / hours:.1f} MB/h gz; {r / g:.0f} records/game, {b / 1e6 / g:.2f} MB/game; "
          f"eta {eta:.1f}h", flush=True)


# ---- main --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, required=True, help="total games")
    ap.add_argument("--games-per-shard", type=int, default=50)
    # A harness JVM uses ~2.5 cores: throughput is CPU-bound well before memory is.
    ap.add_argument("--workers", type=int, default=min(5, max(1, (os.cpu_count() or 3) // 3)))
    ap.add_argument("--decks", type=Path, default=DEFAULT_DECKS, help="directory of .dck files")
    ap.add_argument("--shares", type=Path, help="CSV archetype,share[,deck]; uniform over decks without it")
    ap.add_argument("--seed", type=int, default=0, help="master seed of the game plan")
    ap.add_argument("--timeout", type=int, default=900, help="per-game watchdog, seconds")
    ap.add_argument("--heap", default="1536m", help="JVM -Xmx per worker (measured peak RSS at 1536m: <= 1.8 GB)")
    ap.add_argument("--format", default="Premodern", help="Forge format the decks must conform to")
    ap.add_argument("--java", default="java")
    ap.add_argument("--jvm-arg", action="append", default=[], help="extra JVM argument (repeatable)")
    ap.add_argument("--keep-suspicious", action="store_true", help="keep decks the check flags as suspicious")
    ap.add_argument("--no-validate", action="store_true", help="skip validate_dump.py on each shard")
    ap.add_argument("--progress-every", type=float, default=120.0, help="seconds between throughput lines")
    ap.add_argument("--oom-score-adj", type=int, default=1000,
                    help="oom_score_adj of the harness JVMs (1000: killed first under memory pressure; 0: off)")
    args = ap.parse_args()
    global OOM_SCORE_ADJ
    OOM_SCORE_ADJ = args.oom_score_adj

    out = args.out.resolve()
    args.decks = args.decks.resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(out)  # noqa: F841 -- held until exit
    atexit.register(kill_live)
    if not HAVE_SETPRIV:
        print("warning: setpriv not found; harness JVMs will not die with the generator if it is killed",
              flush=True)
    avail = mem_available_mb()
    workers = cap_workers(args.workers, heap_mb(args.heap), avail)
    if workers < args.workers:
        print(f"warning: --workers {args.workers} reduced to {workers}: {args.workers} x ({args.heap} heap + "
              f"{JVM_OVERHEAD_MB} MB JVM overhead) exceeds {MEM_FRACTION:.0%} of MemAvailable ({avail} MB)",
              flush=True)
    args.workers = workers
    prov = {"forge": forge_info(), "bridge_build": bridge_hash()}

    print(f"checking decks in {args.decks} ({args.format}) ...", flush=True)
    rows = check_decks(args, out)
    decks, bad = usable_decks(rows, args.keep_suspicious)
    print(f"  {len(rows)} decks, {len(decks)} usable")
    for f, why in bad:
        print(f"  EXCLUDED {f}: {why}")
    ai_remove = Counter(c for r in rows if r["file"] in decks for c in r["ai_remove"])
    if ai_remove:
        print(f"  note: {len(ai_remove)} distinct cards in usable decks are marked AI:RemoveDeck:All in Forge "
              f"(the AI plays them poorly): {', '.join(sorted(ai_remove))}")
    w = weights(decks, args.shares)

    run = {"games": args.games, "games_per_shard": args.games_per_shard, "seed": args.seed,
           "weights": w, "schema": SUPPORTED_SCHEMA, "shares": str(args.shares) if args.shares else None}
    run_file = out / "run.json"
    if run_file.exists():
        prev = json.loads(run_file.read_text())
        same = {k: prev.get(k) for k in ("games", "games_per_shard", "seed", "weights", "schema")}
        if same != {k: run[k] for k in same}:
            sys.exit(f"{run_file} describes a different plan (games/shard size/seed/decks/shares/schema); "
                     "use a new --out")
    else:
        run_file.write_text(json.dumps({**run, **prov, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))

    shards = plan(args.games, args.games_per_shard, args.seed, w)
    done_lines: dict[int, int] = Counter()
    mpath = out / "manifest.jsonl"
    if mpath.exists():
        for ln in mpath.read_text().splitlines():
            try:
                done_lines[json.loads(ln)["shard"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    queue, skipped = [], 0
    for games in shards:
        s = games[0].shard
        gz = out / f"{shard_name(s)}.jsonl.gz"
        if done_lines.get(s) == len(games) and gz.exists() and gzip_ok(gz):
            skipped += 1
            continue
        if done_lines.get(s):
            sys.exit(f"manifest has {done_lines[s]} of {len(games)} lines for {shard_name(s)} or its gzip is "
                     "missing/corrupt; inspect it before resuming")
        cleanup(out, s)
        queue.append(games)
    print(f"{len(shards)} shards ({args.games} games, {args.games_per_shard}/shard): {skipped} complete, "
          f"{len(queue)} to play with {args.workers} workers, -Xmx{args.heap}, timeout {args.timeout}s", flush=True)
    if not queue:
        return

    stop = Stop()
    signal.signal(signal.SIGINT, stop.handler)

    def abandon(signum, frame) -> None:  # noqa: ARG001
        stop.soft.set()
        stop.hard.set()
        kill_live()
        print(f"\n[signal {signum}] abandoning the running shards", flush=True)

    signal.signal(signal.SIGTERM, abandon)
    signal.signal(signal.SIGHUP, abandon)
    totals = Totals()
    t0 = time.time()
    qlock, mlock = threading.Lock(), threading.Lock()
    done = threading.Event()
    todo_games = sum(len(g) for g in queue)
    rep = threading.Thread(target=reporter, args=(totals, t0, args.progress_every, done, todo_games), daemon=True)
    rep.start()
    threads = [threading.Thread(target=worker, args=(queue, qlock, out, args, stop, totals, mlock, prov))
               for _ in range(max(1, args.workers))]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(0.5)
    done.set()
    report(totals, t0, todo_games)
    print(f"done in {time.time() - t0:.0f}s: {totals.shards} shards, {totals.games} games, "
          f"{totals.failed_games} crashed/errored, {totals.timeouts} timeouts, "
          f"{totals.invalid_shards} shards failing validation"
          + (" (stopped early)" if stop.soft.is_set() else ""), flush=True)
    work = out / "work"
    if work.is_dir() and not any(work.iterdir()):
        work.rmdir()


if __name__ == "__main__":
    main()
