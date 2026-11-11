"""PNG visualisations of decision records and belief-model samples.

Two kinds of image, written straight to disk (no TensorBoard/W&B):

* :func:`render_board` -- one decision record drawn from the viewer's seat:
  opponent on top, the stack in the middle, the viewer at the bottom, and a
  side panel with the candidate actions (expert label highlighted, optional
  policy probabilities as bars).
* :func:`render_belief` -- for each hidden region (``dump.is_region`` order,
  aligned with ``Decision.hidden_truth``), a density heatmap of the belief
  samples on a frozen 2-D UMAP of every card's whitened PCA-32 target, the
  true cards marked, and the samples decoded to their nearest card next to
  the true composition.

The 2-D map (:class:`CardMap`) is fitted once per (catalog version, PCA
basis, dims) and cached under ``~/.cache/magicnet`` so plots from different
epochs share the same axes.

matplotlib / umap-learn / scipy live in the ``notebook`` dependency group
(``uv sync --group notebook``); they are imported lazily, so importing this
module never fails.

Training-loop usage (once per epoch)::

    from magicnet import viz
    from magicnet.catalog import CatalogClient

    catalog = CatalogClient()
    card_map = viz.CardMap.load_or_fit(catalog)       # fitted once, then cached on disk

    # at the end of each epoch, for a fixed validation decision ``d``
    # (``ctx``: its region tokens' context vectors, R = number of regions):
    with torch.no_grad():
        z = model.posterior.flow(ctx).rsample((512,))  # (512, R, 32), is_region order
    samples = viz.region_samples(d, z)                 # {"r:opp_hand": (512, 32), ...}
    # (also accepts the data loader's one-row-per-unknown-card layout
    #  (n, sum(count), 32), and 3072-d samples, projected with the map's PCA basis)
    viz.render_decision(d, f"viz/epoch{epoch:03d}", samples=samples,
                        action_probs=policy_probs,     # optional, (len(d.actions),)
                        card_map=card_map)
    # -> viz/epoch007_board.png, viz/epoch007_belief.png

Command line: ``uv run scripts/render_decision.py --help``.
"""

from __future__ import annotations

import json
import math
import pickle
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from magicnet import dump
from magicnet.dump import comp

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "magicnet"
_INSTALL_HINT = "run `uv sync --group notebook` (matplotlib, umap-learn, scikit-learn live in that group)"


def _plt():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(f"magicnet.viz needs matplotlib: {_INSTALL_HINT}") from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _to_numpy(x) -> np.ndarray:
    if type(x).__module__.split(".")[0] == "torch":
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Board model: tokens -> drawable items
# --------------------------------------------------------------------------------------

# Card dimensions and spacing, in inches (the board axes use inch coordinates).
CW, CH = 1.32, 1.72
GAP = 0.12
LABEL_W = 1.15  # left column holding row labels
BOARD_W = 16.0
PANEL_W = 6.6
LINE_H = 0.165  # text line height at ~8pt
MAX_CARD_LINES = 3  # card rows per zone before summarising the rest

_COLOR_FILL = {
    "white": "#f7f3df", "blue": "#d5e6f7", "black": "#cfc8cc", "red": "#f6d3c8",
    "green": "#d3ead2", "colorless": "#e4e4e4",
}
_LAND_FILL = "#e8dcc4"
_MULTI_FILL = "#efe0a8"
_SIDE_EDGE = {"self": "#1f5fa8", "opp": "#a8321f"}


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: max(1, n - 1)] + "…"


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    lines = textwrap.wrap(str(text), width=width, break_long_words=True) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _short(lines[-1] + " …", width)
    return lines


def _is_land(tok: dict) -> bool:
    types = comp(tok, "types", "")
    return isinstance(types, str) and "Land" in types.replace("-", " ").split()


def _fill(tok: dict) -> str:
    if _is_land(tok):
        return _LAND_FILL
    colors = comp(tok, "colors") or []
    colors = [c for c in colors if c != "colorless"] if isinstance(colors, list) else []
    if len(colors) > 1:
        return _MULTI_FILL
    if len(colors) == 1:
        return _COLOR_FILL.get(colors[0], "#eeeeee")
    return _COLOR_FILL["colorless"]


class _Lookup:
    """Id/handle -> token, over the decision's tokens (state, prompt, options)."""

    def __init__(self, decision: dump.Decision):
        self.tokens = dump.decision_tokens(decision)
        self.index = dump._index_of(self.tokens)

    def get(self, ref) -> dict | None:
        i = self.index.get(str(ref))
        return None if i is None else self.tokens[i]

    def name(self, ref, default: str = "?") -> str:
        tok = self.get(ref)
        if tok is None:
            return default
        return describe_token(tok)


def describe_token(tok: dict) -> str:
    """Short human label for any token."""
    t = tok.get("type")
    if t == "player":
        side = comp(tok, "side")
        who = "you" if side == "self" else "opponent" if side == "opp" else tok.get("id", "player")
        return f"{who} ({tok.get('id', '?')})"
    if "name" in tok:
        return str(tok["name"])
    if t == "stack_item":
        return f"ability of {tok.get('source', 'hidden source')}"
    if t == "library_card" and str(tok.get("id", "")).startswith(dump.REGION_PREFIX):
        return f"{tok['id'][2:]} ({comp(tok, 'count', '?')} hidden)"
    if dump.is_hidden(tok):
        return "face-down card" if tok.get("zone", "").endswith(("battlefield", "exile", "stack")) else "hidden card"
    if t == "action":
        return str(comp(tok, "verb", "option"))
    return str(tok.get("id") or t or "token")


@dataclass
class _Item:
    """A drawable rectangle (card, card back, ability) or a text box."""

    kind: str  # card | back | ability | textbox | hidden
    title: str = ""
    lines: list[str] = field(default_factory=list)  # annotations (cards) / entries (text boxes)
    fill: str = "#eeeeee"
    edge: str = "#444444"
    tapped: bool = False
    count: int = 1
    pt: str | None = None
    pt_color: str = "black"
    subtitle: str = ""
    w: float = CW
    h: float = CH
    ncols: int = 1
    highlight: bool = False
    sig: tuple = ()


def _card_item(tok: dict, look: _Lookup) -> _Item:
    hidden = dump.is_hidden(tok)
    name = tok.get("name") or ("face-down" if hidden else describe_token(tok))
    notes: list[str] = []
    state = comp(tok, "state")
    if comp(tok, "face_down"):
        notes.append("FACE-DOWN")
    if state not in (None, "Original"):
        notes.append(f"[{state}]")
    if comp(tok, "token"):
        notes.append("token")
    if comp(tok, "copy"):
        notes.append("copy")
    if comp(tok, "sick"):
        notes.append("summoning sick")
    if comp(tok, "casting"):
        notes.append("being cast")
    if comp(tok, "revealed") or comp(tok, "known_by_reveal"):
        notes.append("revealed")
    ctrs = [f"{k[4:]}×{v}" for k, v in sorted((tok.get(dump.COMPONENTS) or {}).items())
            if k.startswith("ctr_") and v]
    if ctrs:
        notes.append("ctr " + " ".join(ctrs))
    dmg = comp(tok, "damage", 0)
    if isinstance(dmg, (int, float)) and dmg > 0:
        notes.append(f"damage {dmg}")
    for key in ("gained", "lost"):
        val = comp(tok, key)
        if val:
            vals = val if isinstance(val, list) else [val]
            notes.append(("+" if key == "gained" else "−") + ", ".join(_short(v, 18) for v in vals))
    for key in ("x_paid", "kicked", "modes"):
        val = comp(tok, key)
        if val not in (None, False, [], ""):
            notes.append(f"{key}: {_short(val if not isinstance(val, list) else ', '.join(map(str, val)), 22)}")
    rel_text = {"attacking": "ATTACKING", "blocking": "blocks", "blocked_by": "blocked by",
                "attached_to": "on", "target_0": "→", "target_1": "→", "target_2": "→", "target_3": "→",
                "owner_is": "owner"}
    edges = [e for e in tok.get("edges", ()) if e.get("rel") in rel_text]
    for e in edges:
        rel = e["rel"]
        if rel == "attacking":
            tgt = look.get(e["to"])
            notes.append("ATTACKING" + ("" if tgt is not None and tgt.get("type") == "player"
                                        else f" {_short(look.name(e['to']), 16)}"))
        else:
            notes.append(f"{rel_text[rel]} {_short(look.name(e['to']), 18)}")
    if comp(tok, "blocked") and not any(e["rel"] == "blocked_by" for e in edges):
        notes.append("blocked")
    pt = None
    pt_color = "black"
    p, t = comp(tok, "power"), comp(tok, "toughness")
    if p is not None and t is not None:
        pt = f"{p}/{t}"
        bp, bt = comp(tok, "base_power", p), comp(tok, "base_toughness", t)
        try:
            delta = (p - bp) + (t - bt)
            pt_color = "#137a13" if delta > 0 else "#b01010" if delta < 0 else "black"
        except TypeError:
            pass
    types = comp(tok, "types", "")
    fill = "#3a3f55" if hidden else _fill(tok)
    side = comp(tok, "side")
    edge = _SIDE_EDGE.get(side, "#444444")
    tapped = bool(comp(tok, "tapped"))
    # Only plain, unconnected cards are collapsed into "Name ×n".
    sig = () if edges or hidden else (name, tuple(notes), pt, pt_color, types, tapped, fill)
    return _Item("hidden" if hidden else "card", title=name, lines=notes, fill=fill, edge=edge, tapped=tapped,
                 pt=pt, pt_color=pt_color, subtitle=_short(types, 26) if isinstance(types, str) else "",
                 w=CH if tapped else CW, h=CW if tapped else CH, sig=sig)


def _collapse(items: list[_Item]) -> list[_Item]:
    out: list[_Item] = []
    by_sig: dict[tuple, _Item] = {}
    for it in items:
        if it.sig and it.sig in by_sig:
            by_sig[it.sig].count += 1
            continue
        if it.sig:
            by_sig[it.sig] = it
        out.append(it)
    return out


def _stack_item(tok: dict, look: _Lookup) -> _Item:
    if tok.get("type") == "stack_item":
        src = tok.get("source") or "hidden source"
        notes = ["trigger" if comp(tok, "is_trigger") else "activated ability"]
        ability = comp(tok, "ability")
        if ability:
            notes += _wrap(ability, 22, 4)
        for e in tok.get("edges", ()):
            if e.get("rel", "").startswith("target"):
                notes.append(f"→ {_short(look.name(e['to']), 18)}")
        side = comp(tok, "side")
        return _Item("ability", title=f"{src}", lines=notes, fill="#fbf7e9", edge=_SIDE_EDGE.get(side, "#444"),
                     subtitle="ability", w=CW * 1.25)
    it = _card_item(tok, look)
    it.subtitle = (it.subtitle + " · spell").strip(" ·")
    return it


def _textbox(title: str, entries: list[str], avail_w: float, max_rows: int = 9, colw: float = 2.3) -> _Item:
    if not entries:
        entries = ["(empty)"]
    max_cols = max(1, int((avail_w - 0.2) // colw))
    ncols = min(max_cols, max(1, math.ceil(len(entries) / max_rows)))
    cap = ncols * max_rows
    if len(entries) > cap:
        rest = len(entries) - (cap - 1)
        entries = entries[: cap - 1] + [f"… +{rest} more"]
    rows = math.ceil(len(entries) / ncols)
    return _Item("textbox", title=title, lines=entries, w=ncols * colw + 0.1, h=0.34 + rows * LINE_H,
                 ncols=ncols, fill="#f6f6f6", edge="#999999")


def _name_counts(toks: list[dict], key=None) -> list[str]:
    cnt: Counter = Counter()
    for t in toks:
        nm = key(t) if key else (t.get("name") or describe_token(t))
        cnt[nm] += int(comp(t, "count", 1) or 1) if t.get("type") == "library_card" else 1
    return [f"{_short(n, 27)} ×{c}" if c > 1 else _short(n, 30) for n, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]


@dataclass
class _Row:
    label: str
    items: list[_Item]
    kind: str = "cards"  # cards | info
    text: str = ""
    lines: list[list[tuple[_Item, float]]] = field(default_factory=list)
    height: float = 0.0


def _summary_box(items: list[_Item], width: float) -> _Item:
    """One text box standing for overflowing cards, aggregated by name."""
    agg: dict[str, list[int]] = {}
    for r in items:
        if r.kind == "textbox":
            continue
        a = agg.setdefault(r.title, [0, 0])
        a[0] += r.count
        a[1] += r.count if r.tapped else 0
    entries = [f"{_short(n, 30)} ×{c}" + (f" ({t} tapped)" if t else "")
               for n, (c, t) in sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))]
    return _textbox(f"+{sum(c for c, _ in agg.values())} more (not drawn)", entries, width, colw=3.4)


def _flow(row: _Row, width: float) -> None:
    """Place items left to right, wrapping; cap at MAX_CARD_LINES lines of cards and
    summarise the overflow in a text box."""
    lines: list[list[tuple[_Item, float]]] = [[]]
    x = 0.0
    items = list(row.items)
    i = 0
    while i < len(items):
        it = items[i]
        if x > 0 and x + it.w > width:
            if len(lines) >= MAX_CARD_LINES and it.kind != "textbox":
                rest = items[i:]
                lines.append([(_summary_box(rest, width), 0.0)])
                lines += [[(r, 0.0)] for r in rest if r.kind == "textbox"]
                break
            lines.append([])
            x = 0.0
        lines[-1].append((it, x))
        x += it.w + GAP
        i += 1
    row.lines = [ln for ln in lines if ln]
    row.height = sum(max(it.h for it, _ in ln) + GAP for ln in row.lines) + GAP if row.lines else 0.45


def _zone(state: list[dict], zone: str) -> list[dict]:
    return [t for t in state if t.get("zone") == zone]


def _player_info(state: list[dict], side: str) -> str:
    for t in state:
        if t.get("type") == "player" and comp(t, "side") == side:
            parts = []
            for k, v in sorted((t.get(dump.COMPONENTS) or {}).items()):
                if k == "side":
                    continue
                if k == "life":
                    parts.insert(0, f"LIFE {v}")
                else:
                    parts.append(f"{k.replace('ctr_', '')} {v}")
            return f"{t.get('id', '?')} · " + " · ".join(parts)
    return "(no player token)"


_KNOWN_ZONES = {f"{s}_{z}" for s in ("self", "opp") for z in
                ("battlefield", "hand", "library", "graveyard", "exile", "sideboard", "command")} | {"stack", "phantom"}


def _board_rows(d: dump.Decision, look: _Lookup) -> tuple[list[_Row], _Row, list[_Row]]:
    st = d.state
    width = BOARD_W - LABEL_W - 0.2

    def battlefield(side: str) -> tuple[list[_Item], list[_Item]]:
        # Hidden/face-down and combat/attached cards first so they survive the overflow cap.
        def prio(t):
            if dump.is_hidden(t) or comp(t, "face_down") or any(
                    e.get("rel") in ("attacking", "blocking", "blocked_by") for e in t.get("edges", ())):
                return 0
            return 2 if comp(t, "token") else 1

        toks = sorted(_zone(st, f"{side}_battlefield"), key=prio)
        # Put attachments right after their host.
        host_of = {}
        for t in toks:
            for e in t.get("edges", ()):
                if e.get("rel") == "attached_to":
                    host_of[id(t)] = look.get(e["to"])
        order = [t for t in toks if id(t) not in host_of]
        for t in toks:
            if id(t) in host_of:
                h = host_of[id(t)]
                pos = next((i for i, o in enumerate(order) if o is h), len(order) - 1)
                order.insert(pos + 1, t)
        lands = [_card_item(t, look) for t in order if _is_land(t) and comp(t, "power") is None]
        other = [_card_item(t, look) for t in order if not (_is_land(t) and comp(t, "power") is None)]
        return _collapse(lands), _collapse(other)

    def zone_boxes(side: str) -> list[_Item]:
        boxes = []
        lib = _zone(st, f"{side}_library")
        if side == "self":
            agg = [t for t in lib if not dump.is_region(t)]
            total = sum(int(comp(t, "count", 1) or 1) for t in agg)
            known = [f"#{comp(t, 'depth_from_top')} from top: {_short(t.get('name', '?'), 18)}"
                     for t in agg if comp(t, "depth_from_top") is not None]
            names = _name_counts([t for t in agg if comp(t, "depth_from_top") is None])
            boxes.append(_textbox(f"Library ({total})", known + names, width, max_rows=8, colw=2.2))
        else:
            hidden = sum(int(comp(t, "count", 0) or 0) for t in lib if dump.is_region(t))
            known = [f"#{comp(t, 'depth_from_top')} from top: {_short(t.get('name', '?'), 18)}"
                     for t in lib if not dump.is_region(t)]
            boxes.append(_textbox(f"Library ({hidden + len(known)})", [f"{hidden} hidden"] + known, width))
        for z, title in (("graveyard", "Graveyard"), ("exile", "Exile")):
            toks = _zone(st, f"{side}_{z}")
            boxes.append(_textbox(f"{title} ({len(toks)})", _name_counts(toks), width))
        sb = _zone(st, f"{side}_sideboard")
        hidden = sum(int(comp(t, "count", 0) or 0) for t in sb if t.get("type") == "library_card" and dump.is_region(t))
        named = [t for t in sb if not dump.is_region(t)]
        entries = ([f"{hidden} hidden"] if hidden else []) + _name_counts(named)
        boxes.append(_textbox(f"Sideboard ({hidden + len(named)})", entries, width, max_rows=5))
        cmd = _zone(st, f"{side}_command")
        if cmd:
            boxes.append(_textbox(f"Command ({len(cmd)})", _name_counts(cmd), width, max_rows=5, colw=3.0))
        return boxes

    # Opponent (top), outermost first.
    opp_hand = _zone(st, "opp_hand")
    n_hidden = sum(int(comp(t, "count", 0) or 0) for t in opp_hand if dump.is_region(t))
    revealed = [_card_item(t, look) for t in opp_hand if not dump.is_region(t)]
    backs = []
    if n_hidden:
        shown = min(n_hidden, 7)
        backs.append(_Item("back", title=f"{n_hidden} hidden", count=shown, fill="#2b3d6b", edge="#111",
                           w=CW + (shown - 1) * 0.22))
    opp_lands, opp_other = battlefield("opp")
    top = [
        _Row("OPPONENT", [], kind="info", text=_player_info(st, "opp")),
        _Row("opp hand", backs + revealed + ([] if (backs or revealed) else [_textbox("Hand", ["(empty)"], width)])),
        _Row("opp zones", zone_boxes("opp")),
        _Row("opp lands", opp_lands),
        _Row("opp\nnonlands", opp_other),
    ]
    stack_toks = sorted(_zone(st, "stack"), key=lambda t: (comp(t, "stack_depth", 99), str(t.get("id"))))
    stack_items = [_stack_item(t, look) for t in stack_toks]
    for i, it in enumerate(stack_items):
        it.lines = [f"depth {comp(stack_toks[i], 'stack_depth', '?')}" + (" (top)" if i == 0 else "")] + it.lines
    stack = _Row("STACK\n(top → left)", stack_items) if stack_items else _Row("STACK", [], kind="info", text="(empty)")

    own_lands, own_other = battlefield("self")
    hand = _collapse([_card_item(t, look) for t in _zone(st, "self_hand")])
    other = [t for t in st if t.get("zone") is not None and t.get("zone") not in _KNOWN_ZONES]
    other += [t for t in st if t.get("zone") is None and t.get("type") not in ("global", "player")]
    bottom = [
        _Row("your\nnonlands", own_other),
        _Row("your lands", own_lands),
        _Row("your hand", hand or [_textbox("Hand", ["(empty)"], width)]),
        _Row("your zones", zone_boxes("self")),
    ]
    if other:
        bottom.append(_Row("other", [_textbox("Other tokens", [f"{t.get('zone') or t.get('type')}: "
                                                               f"{_short(describe_token(t), 30)}" for t in other],
                                              width, colw=3.4)]))
    bottom.append(_Row("YOU", [], kind="info", text=_player_info(st, "self")))
    for r in top + [stack] + bottom:
        if r.kind == "cards":
            _flow(r, width)
        else:
            r.height = 0.42
    return top, stack, bottom


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------


def describe_action(cand: dict, look: _Lookup) -> tuple[str, str]:
    """(headline, detail) for one candidate."""
    if dump.is_ref(cand):
        tok = look.get(cand["ref"])
        if tok is None:
            return f"ref {cand['ref']}", "(unresolved)"
        zone = tok.get("zone") or tok.get("type", "")
        extra = ""
        if comp(tok, "power") is not None:
            extra = f" {comp(tok, 'power')}/{comp(tok, 'toughness')}"
        return f"pick {describe_token(tok)}{extra}", str(zone).replace("_", " ")
    verb = str(comp(cand, "verb", "?"))
    src = None
    others = []
    for e in cand.get("edges", ()):
        if e.get("rel") == "source_is" and src is None:
            src = look.name(e["to"])
        else:
            others.append(f"{e.get('rel')}: {_short(look.name(e['to']), 22)}")
    head = verb + (f"  {src}" if src else "")
    comps = cand.get(dump.COMPONENTS) or {}
    detail = []
    for k in ("value", "mana_color", "mana_amount", "cost", "pays", "optional_cost"):
        if k in comps:
            detail.append(f"{k}={comps[k]}")
    if "target_candidates" in comps:
        detail.append(f"{comps['target_candidates']} targets")
    if comps.get("mana_ability"):
        detail.append("mana")
    if "ability" in comps and verb not in ("cast",):
        detail.append(str(comps["ability"]))
    detail += others
    return head, " · ".join(map(str, detail))


def _probs(action_probs, n: int) -> np.ndarray | None:
    if action_probs is None:
        return None
    p = _to_numpy(action_probs).reshape(-1)
    if p.shape[0] != n:
        raise ValueError(f"action_probs has {p.shape[0]} entries, decision has {n} candidates")
    return p


# --------------------------------------------------------------------------------------
# render_board
# --------------------------------------------------------------------------------------


def _draw_item(ax, it: _Item, x: float, y: float, row_h: float):
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # Cards sit on the row's bottom edge so tapped (landscape) ones line up;
    # text boxes hang from the top.
    y0 = y if it.kind == "textbox" else y + (row_h - it.h)
    if it.kind == "textbox":
        ax.add_patch(FancyBboxPatch((x, y0), it.w, it.h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=it.fill, ec=it.edge, lw=0.8))
        ax.text(x + 0.07, y0 + 0.07, it.title, fontsize=8.5, weight="bold", va="top", ha="left")
        rows = math.ceil(len(it.lines) / it.ncols)
        colw = (it.w - 0.1) / it.ncols
        for k, line in enumerate(it.lines):
            c, r = divmod(k, rows)
            ax.text(x + 0.08 + c * colw, y0 + 0.30 + r * LINE_H, line, fontsize=7.5, va="top", ha="left")
        return
    if it.kind == "back":
        for k in range(it.count):
            ax.add_patch(FancyBboxPatch((x + k * 0.22, y0), CW, CH, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=it.fill, ec="#c8c8e8", lw=1.0))
        ax.text(x + (it.count - 1) * 0.22 + CW / 2, y0 + CH / 2, f"HAND\n{it.title}", fontsize=9, weight="bold", color="white",
                ha="center", va="center")
        return
    style = "round,pad=0,rounding_size=0.08"
    ls = "--" if it.kind == "ability" else "-"
    ax.add_patch(FancyBboxPatch((x, y0), it.w, it.h, boxstyle=style, fc=it.fill, ec=it.edge,
                                lw=2.0 if it.edge in _SIDE_EDGE.values() else 1.0, ls=ls))
    txt_color = "white" if it.kind == "hidden" else "black"
    wrap_w = int((it.w - 0.1) / 0.074)
    name_lines = _wrap(it.title, wrap_w, 3)
    ax.text(x + 0.06, y0 + 0.06, "\n".join(name_lines), fontsize=8.6, weight="bold", va="top", ha="left",
            color=txt_color, linespacing=1.05)
    cy = y0 + 0.09 + len(name_lines) * 0.155
    if it.subtitle:
        ax.text(x + 0.06, cy, _short(it.subtitle, wrap_w + 6), fontsize=6.6, style="italic", va="top",
                ha="left", color="#dddddd" if it.kind == "hidden" else "#555555")
        cy += 0.14
    ann_h = 0.135
    max_ann = max(0, int((y0 + it.h - 0.25 - cy) / ann_h))
    ann = []
    for line in it.lines:
        color = "#a00000" if line.startswith(("ATTACKING", "blocks", "blocked", "damage")) else txt_color
        ann += [(part, color) for part in _wrap(line, wrap_w + 4, 2)]
    if len(ann) > max_ann:
        keep = max(0, max_ann - 1)
        ann = ann[:keep] + [(f"+{len(ann) - keep} more lines", txt_color)]
    for k, (line, color) in enumerate(ann):
        ax.text(x + 0.06, cy + k * ann_h, line, fontsize=6.8, va="top", ha="left", color=color)
    if it.pt:
        ax.text(x + it.w - 0.06, y0 + it.h - 0.05, it.pt, fontsize=10, weight="bold", ha="right", va="bottom",
                color=it.pt_color if it.kind != "hidden" else "white",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#888", lw=0.6) if it.kind != "hidden" else None)
    if it.tapped:
        ax.add_patch(Rectangle((x, y0), it.w, it.h, fc="none", ec="#777", hatch="//", lw=0, alpha=0.35))
        ax.text(x + 0.06, y0 + it.h - 0.05, "TAPPED", fontsize=6.5, weight="bold", color="#555", va="bottom")
    if it.count > 1:
        ax.text(x + it.w - 0.04, y0 + 0.04, f"×{it.count}", fontsize=11, weight="bold", ha="right", va="top",
                color="white", bbox=dict(boxstyle="round,pad=0.15", fc="#333", ec="none"))


def render_board(decision: dump.Decision, path: str | Path | None = None, action_probs=None, *,
                 dpi: int = 100, title: str | None = None, max_actions: int = 60) -> Path:
    """Draw one decision from the viewer's seat and save it as a PNG.

    *action_probs*: optional per-candidate probabilities (array or tensor of
    length ``len(decision.actions)``), shown as bars in the action panel.
    Returns the written path (default ``board_g<game>_d<decision>.png``).
    """
    plt = _plt()
    d = decision
    path = Path(path) if path is not None else Path(f"board_g{d.game}_d{d.decision}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    look = _Lookup(d)
    top, stack, bottom = _board_rows(d, look)
    probs = _probs(action_probs, len(d.actions))

    glob_tok = next((t for t in d.state if t.get("type") == "global"), {})
    prompt = d.prompt or {}
    pcomps = {k: v for k, v in (prompt.get(dump.COMPONENTS) or {}).items() if k != "kind"}
    header = [title or f"game {d.game} · decision {d.decision} · viewer {d.viewer}",
              f"Turn {d.turn} · {d.phase} · decision: {d.kind}"
              f" · {'your' if comp(glob_tok, 'active_is_viewer') else 'opponent'} turn"
              f" · stack {comp(glob_tok, 'stack_size', 0)} · label: {d.chosen} ({d.chosen_match})"]
    ptxt = pcomps.pop("prompt_text", None)
    pedges = [f"{e['rel']}: {_short(look.name(e['to']), 24)}" for e in prompt.get("edges", ())]
    extra = " · ".join([f"{k}={_short(v, 30)}" for k, v in pcomps.items()] + pedges)
    if ptxt or extra:
        header.append(_short(f"Prompt: {ptxt or ''}  {extra}".strip(), 150))
    header_h = 0.3 + 0.27 * len(header)

    board_h = header_h + sum(r.height for r in top + [stack] + bottom) + 0.3

    # Action panel geometry.
    order = list(range(len(d.actions)))
    hidden_actions = 0
    if len(order) > max_actions:
        if probs is not None:
            order = sorted(order, key=lambda i: -probs[i])
        keep = order[:max_actions]
        if 0 <= d.chosen < len(d.actions) and d.chosen not in keep:
            keep[-1] = d.chosen
        hidden_actions = len(order) - len(keep)
        order = sorted(keep)
    entries = []
    for i in order:
        head, detail = describe_action(d.actions[i], look)
        entries.append((i, _short(head, 46), _wrap(detail, 62, 2) if detail else []))
    panel_h = 0.9 + sum(0.2 + LINE_H * len(e[2]) + 0.08 for e in entries) + (0.3 if hidden_actions else 0)
    H = max(board_h, panel_h)
    W = BOARD_W + PANEL_W

    fig = plt.figure(figsize=(W, H), dpi=dpi)
    ax = fig.add_axes([0, 0, BOARD_W / W, 1])
    ax.set_xlim(0, BOARD_W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    y = 0.2
    for k, line in enumerate(header):
        ax.text(0.2, y, line, fontsize=13 if k == 0 else 10.5, weight="bold" if k < 2 else "normal",
                va="top", ha="left")
        y += 0.27 if k else 0.32
    y = header_h

    def draw_rows(rows, y, bg):
        y_start = y
        for r in rows:
            if r.kind == "info":
                ax.text(LABEL_W + 0.1, y + 0.1, r.text, fontsize=11, weight="bold", va="top", ha="left")
            else:
                ly = y + GAP
                for line in r.lines:
                    lh = max(it.h for it, _ in line)
                    for it, x in line:
                        _draw_item(ax, it, LABEL_W + 0.1 + x, ly, lh)
                    ly += lh + GAP
                if not r.lines:
                    ax.text(LABEL_W + 0.1, y + 0.12, "(none)", fontsize=8, color="#888", va="top")
            ax.text(0.12, y + 0.1, r.label, fontsize=8.5, weight="bold", color="#333", va="top", ha="left")
            y += r.height
            ax.plot([0.1, BOARD_W - 0.1], [y, y], color="#dddddd", lw=0.6)
        if bg:
            ax.add_patch(plt.Rectangle((0.05, y_start), BOARD_W - 0.1, y - y_start, fc=bg, ec="none", zorder=-5))
        return y

    y = draw_rows(top, y, "#fbeeee")
    y = draw_rows([stack], y, "#fff8e0")
    draw_rows(bottom, y, "#edf3fb")

    # Action panel.
    pax = fig.add_axes([BOARD_W / W, 0, PANEL_W / W, 1])
    pax.set_xlim(0, PANEL_W)
    pax.set_ylim(H, 0)
    pax.axis("off")
    pax.add_patch(plt.Rectangle((0.05, 0.05), PANEL_W - 0.15, H - 0.1, fc="#fafafa", ec="#cccccc"))
    pax.text(0.2, 0.25, f"Candidates ({len(d.actions)})", fontsize=12, weight="bold", va="top")
    legend = "green = label (chosen)" + (" · bars = action_probs" if probs is not None else "")
    pax.text(0.2, 0.55, legend, fontsize=8, color="#555", va="top")
    y = 0.9
    bar_x, bar_w = 0.55, 1.2
    text_x = bar_x + (bar_w + 0.12 if probs is not None else 0)
    pmax = float(probs.max()) if probs is not None and len(probs) else 1.0
    for i, head, detail in entries:
        h = 0.2 + LINE_H * len(detail)
        if i == d.chosen:
            pax.add_patch(plt.Rectangle((0.1, y - 0.04), PANEL_W - 0.25, h + 0.04, fc="#c9f0c4", ec="#2e8b2e", lw=1.2))
        pax.text(0.18, y, f"{i}", fontsize=9, weight="bold", va="top")
        if probs is not None:
            pax.add_patch(plt.Rectangle((bar_x, y), bar_w, 0.15, fc="#eeeeee", ec="#bbbbbb", lw=0.5))
            frac = max(0.0, float(probs[i])) / pmax if pmax > 0 else 0.0
            is_top = float(probs[i]) == pmax
            pax.add_patch(plt.Rectangle((bar_x, y), bar_w * frac, 0.15, fc="#d9822b" if is_top else "#4a7fc1",
                                        ec="none"))
            pax.text(bar_x + bar_w / 2, y + 0.075, f"{float(probs[i]):.3f}", fontsize=7, ha="center", va="center")
        pax.text(text_x, y, head, fontsize=9, weight="bold" if i == d.chosen else "normal", va="top")
        for k, dl in enumerate(detail):
            pax.text(text_x + 0.1, y + 0.2 + k * LINE_H, dl, fontsize=7.3, color="#444", va="top")
        y += h + 0.08
    if hidden_actions:
        pax.text(0.2, y, f"… {hidden_actions} more candidates not shown", fontsize=8, color="#888", va="top")

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Belief samples
# --------------------------------------------------------------------------------------


def regions(decision: dump.Decision) -> list[dict]:
    """The decision's hidden-region tokens, in ``is_region`` (= hidden_truth) order."""
    return [t for t in decision.state if dump.is_region(t)]


def _region_count(tok: dict) -> int:
    return int(comp(tok, "count", 1) or 1) if str(tok.get("id", "")).startswith(dump.REGION_PREFIX) else 1


def region_samples(decision: dump.Decision, samples) -> dict[str, np.ndarray]:
    """Split a flow's samples into ``{region id: (m, D)}``.

    *samples* is ``(n, R, D)`` (one context per region token, ``is_region``
    order) or ``(n, U, D)`` with ``U = sum of region counts`` (the data
    loader's one-row-per-unknown-card layout, rows grouped by region); with
    the latter, a region's rows are pooled into ``(n * count, D)``.
    """
    x = _to_numpy(samples)
    regs = regions(decision)
    if x.ndim != 3:
        raise ValueError(f"expected (n, R, D) samples, got shape {x.shape}")
    counts = [_region_count(t) for t in regs]
    if x.shape[1] == len(regs):
        return {t["id"]: x[:, i] for i, t in enumerate(regs)}
    if x.shape[1] == sum(counts):
        out, s = {}, 0
        for t, c in zip(regs, counts):
            out[t["id"]] = x[:, s:s + c].reshape(-1, x.shape[-1])
            s += c
        return out
    raise ValueError(f"samples have {x.shape[1]} rows per draw; decision has {len(regs)} regions "
                     f"({sum(counts)} unknown cards)")


class CardMap:
    """A frozen 2-D UMAP of every card's whitened PCA target.

    ``targets`` (N, dims) whitened PCA coordinates, ``coords`` (N, 2) the UMAP
    embedding, ``oracle_ids``/``names`` aligned with them. :meth:`transform`
    places new (n, dims) points on the same map, :meth:`nearest` decodes them
    to the nearest card (exact Euclidean NN in the dims-d target space).
    """

    def __init__(self, oracle_ids: list[str], names: list[str], targets, coords, reducer=None, pca=None,
                 meta: dict | None = None):
        self.oracle_ids = list(oracle_ids)
        self.names = list(names)
        self.targets = _to_numpy(targets)
        self.coords = _to_numpy(coords)
        self._reducer = reducer
        self._reducer_file: Path | None = None
        self.pca = pca
        self.meta = dict(meta or {})
        self._by_oid = {o: i for i, o in enumerate(self.oracle_ids)}
        self._by_name: dict[str, int] = {}
        for i, n in enumerate(self.names):
            self._by_name.setdefault(n, i)
        self._torch_table = None

    @property
    def dims(self) -> int:
        return self.targets.shape[1]

    def index(self, oracle_id: str | None = None, name: str | None = None) -> int | None:
        if oracle_id is not None and oracle_id in self._by_oid:
            return self._by_oid[oracle_id]
        if name is not None:
            return self._by_name.get(name)
        return None

    @staticmethod
    def cache_path(catalog_version: int, basis_id: int, dims: int = 32, cache_dir: str | Path | None = None) -> Path:
        return Path(cache_dir or DEFAULT_CACHE_DIR) / f"cardmap_v{catalog_version}_b{basis_id}_d{dims}.npz"

    @classmethod
    def fit(cls, oracle_ids, names, targets, *, random_state: int = 42, n_neighbors: int = 30,
            min_dist: float = 0.1, pca=None, meta: dict | None = None) -> "CardMap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(f"CardMap.fit needs umap-learn: {_INSTALL_HINT}") from exc
        targets = _to_numpy(targets)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric="euclidean", random_state=random_state)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value.*")
            coords = reducer.fit_transform(targets)
        meta = {**(meta or {}), "random_state": random_state, "n_neighbors": n_neighbors, "min_dist": min_dist}
        return cls(oracle_ids, names, targets, coords, reducer=reducer, pca=pca, meta=meta)

    @classmethod
    def load_or_fit(cls, catalog=None, dims: int = 32, cache_dir: str | Path | None = None, *,
                    random_state: int = 42, verbose: bool = True) -> "CardMap":
        """The map for *catalog*'s pinned version and current PCA basis: loaded
        from ``<cache_dir>/cardmap_v{version}_b{basis}_d{dims}.npz`` (+ ``.reducer.pkl``) or, the
        first time, built from every card in the catalog and saved there."""
        if catalog is None:
            from magicnet.catalog import CatalogClient

            catalog = CatalogClient()
        pca = catalog.pca(dims)
        path = cls.cache_path(catalog.version, pca.basis_id, dims, cache_dir)
        if path.exists():
            t = time.perf_counter()
            obj = cls.load(path)
            if verbose:
                print(f"[viz] CardMap loaded from {path} in {time.perf_counter() - t:.2f}s ({len(obj.names)} cards)")
            return obj
        t = time.perf_counter()
        oids = sorted(set(catalog.names().values()))
        found_ids, found_names, vecs = [], [], []
        for s in range(0, len(oids), 2000):
            cards = catalog.lookup(oids[s:s + 2000])
            for o in oids[s:s + 2000]:
                c = cards.get(o)
                if c is not None and c.vector is not None:
                    found_ids.append(o)
                    found_names.append(c.name)
                    vecs.append(c.vector)
        targets = pca.project(np.stack(vecs)).astype(np.float32)
        t_fetch = time.perf_counter() - t
        obj = cls.fit(found_ids, found_names, targets, random_state=random_state, pca=pca,
                      meta={"catalog_version": catalog.version, "basis_id": pca.basis_id, "dims": dims,
                            "skipped": len(oids) - len(found_ids)})
        obj.meta["fit_seconds"] = time.perf_counter() - t - t_fetch
        obj.save(path)
        if verbose:
            print(f"[viz] CardMap fitted on {len(found_ids)} cards ({len(oids) - len(found_ids)} without vectors "
                  f"skipped): fetch {t_fetch:.1f}s, UMAP fit {obj.meta['fit_seconds']:.1f}s -> {path}")
        return obj

    def save(self, path: str | Path) -> None:
        """Arrays to ``<path>`` (.npz, loads in well under a second) and the fitted
        reducer to ``<path minus .npz>.reducer.pkl`` (unpickling it imports umap/numba,
        so it is only loaded when ``transform(method="umap")`` needs it)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(oracle_ids=np.array(self.oracle_ids), names=np.array(self.names), targets=self.targets,
                      coords=self.coords, meta=np.array(json.dumps(self.meta, default=str)))
        if self.pca is not None:
            p = self.pca
            arrays.update(pca_mean=p.mean, pca_components=p.components, pca_std=p.std,
                          pca_evr=np.asarray(p.explained_variance_ratio),
                          pca_info=np.array(json.dumps({"basis_id": p.basis_id, "dims": p.dims, "meta": p.meta},
                                                       default=str)))
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        tmp.replace(path)
        if self._reducer is not None:
            rpath = self._reducer_path(path)
            with rpath.with_name(rpath.name + ".tmp").open("wb") as fh:
                pickle.dump(self._reducer, fh, protocol=pickle.HIGHEST_PROTOCOL)
            rpath.with_name(rpath.name + ".tmp").replace(rpath)

    @staticmethod
    def _reducer_path(path: Path) -> Path:
        return path.with_name(path.name.removesuffix(".npz") + ".reducer.pkl")

    @classmethod
    def load(cls, path: str | Path) -> "CardMap":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            pca = None
            if "pca_info" in z:
                from magicnet.catalog.pca import PCABasis

                info = json.loads(str(z["pca_info"]))
                pca = PCABasis(basis_id=info["basis_id"], dims=info["dims"], mean=z["pca_mean"],
                               components=z["pca_components"], std=z["pca_std"],
                               explained_variance_ratio=z["pca_evr"], meta=info["meta"])
            obj = cls(z["oracle_ids"].tolist(), z["names"].tolist(), z["targets"], z["coords"], pca=pca,
                      meta=json.loads(str(z["meta"])))
        rpath = cls._reducer_path(path)
        obj._reducer_file = rpath if rpath.exists() else None
        return obj

    @property
    def reducer(self):
        """The fitted UMAP reducer (unpickled on first use), or None."""
        if self._reducer is None and self._reducer_file is not None:
            try:
                import umap  # noqa: F401
            except ImportError as exc:
                raise ImportError(f"the pickled UMAP reducer needs umap-learn: {_INSTALL_HINT}") from exc
            with self._reducer_file.open("rb") as fh:
                self._reducer = pickle.load(fh)
        return self._reducer

    def project(self, x) -> np.ndarray:
        """(n, dims) whitened coordinates from (n, dims) or (n, embed_dims) input."""
        x = _to_numpy(x)
        if x.shape[-1] == self.dims:
            return x
        if self.pca is None or x.shape[-1] != self.pca.embed_dims:
            raise ValueError(f"cannot project samples of width {x.shape[-1]} (map dims {self.dims})")
        return np.asarray(self.pca.project(x), dtype=np.float32)

    def nearest(self, z, k: int = 1, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        """Exact k nearest cards of each (n, dims) point: (indices (n, k), distances (n, k))."""
        z = self.project(z)
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        idx = np.empty((len(z), k), dtype=np.int64)
        dist = np.empty((len(z), k), dtype=np.float32)
        if torch is not None:
            if self._torch_table is None:
                self._torch_table = torch.from_numpy(self.targets)
            table = self._torch_table
            for s in range(0, len(z), chunk):
                dd = torch.cdist(torch.from_numpy(np.ascontiguousarray(z[s:s + chunk])), table)
                v, i = torch.topk(dd, k, dim=1, largest=False)
                idx[s:s + chunk], dist[s:s + chunk] = i.numpy(), v.numpy()
            return idx, dist
        sq = (self.targets ** 2).sum(1)
        for s in range(0, len(z), chunk):
            q = z[s:s + chunk]
            dd = np.maximum((q ** 2).sum(1)[:, None] - 2 * q @ self.targets.T + sq[None], 0)
            i = np.argpartition(dd, k - 1, axis=1)[:, :k] if k < dd.shape[1] else np.argsort(dd, 1)[:, :k]
            o = np.take_along_axis(dd, i, 1).argsort(1)
            idx[s:s + chunk] = np.take_along_axis(i, o, 1)
            dist[s:s + chunk] = np.sqrt(np.take_along_axis(dd, idx[s:s + chunk], 1))
        return idx, dist

    def transform(self, z, method: str = "knn", k: int = 10) -> np.ndarray:
        """2-D map coordinates of (n, dims) points.

        ``method="knn"`` (default, fast): inverse-distance-weighted mean of the
        map positions of the k nearest cards in target space. ``"umap"``: the
        fitted reducer's ``transform`` (slower; needs the pickled reducer).
        """
        z = self.project(z)
        if len(z) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if method == "umap":
            if self.reducer is None:
                raise ValueError("this CardMap has no fitted reducer; use method='knn'")
            return np.asarray(self.reducer.transform(z), dtype=np.float32)
        k = min(k, len(self.targets))
        idx, dist = self.nearest(z, k=k)
        w = 1.0 / (dist + 1e-3) ** 2
        w /= w.sum(1, keepdims=True)
        return (self.coords[idx] * w[..., None]).sum(1).astype(np.float32)


def _norm_samples(decision: dump.Decision, samples, card_map: CardMap) -> dict[str, np.ndarray]:
    regs = regions(decision)
    if not isinstance(samples, dict):
        samples = region_samples(decision, samples)
    out = {}
    for key, val in samples.items():
        rid = regs[key]["id"] if isinstance(key, (int, np.integer)) else str(key)
        x = _to_numpy(val)
        x = x.reshape(-1, x.shape[-1])
        out[rid] = card_map.project(x)
    return out


def _density(xy: np.ndarray, extent: tuple[float, float, float, float], bins: int = 220, sigma: float = 2.2):
    x0, x1, y0, y1 = extent
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    try:
        from scipy.ndimage import gaussian_filter

        H = gaussian_filter(H, sigma=sigma)
    except ImportError:  # pragma: no cover
        pass
    H = H.T
    if H.max() > 0:
        H = np.sqrt(H / H.max())  # sqrt so diffuse tails stay visible next to sharp peaks
    return np.ma.masked_less(H, 0.06)


def _place_labels(ax, items: list[tuple[np.ndarray, str]], fontsize: float = 7.5) -> None:
    """Greedy non-overlapping placement of point labels (offsets tried in
    rings around each point, in display space)."""
    fig = ax.figure
    fig.canvas.draw_idle()
    to_disp = ax.transData.transform
    ax_box = ax.get_window_extent()
    scale = fig.dpi / 72.0
    placed: list[tuple[float, float, float, float]] = []
    pts_disp = [to_disp(p) for p, _ in items]
    for (p, text), (px, py) in zip(items, pts_disp):
        w = (0.6 * fontsize * len(text) + 8) * scale
        h = (fontsize + 6) * scale
        best = None
        for radius in (18, 34, 52, 72, 95):
            for k in range(12):
                ang = 2 * math.pi * k / 12 + 0.3
                cx, cy = px + radius * scale * math.cos(ang), py + radius * scale * 0.8 * math.sin(ang)
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if box[0] < ax_box.x0 or box[2] > ax_box.x1 or box[1] < ax_box.y0 or box[3] > ax_box.y1:
                    continue
                hits_label = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
                hits_point = any(box[0] - 3 < qx < box[2] + 3 and box[1] - 3 < qy < box[3] + 3 for qx, qy in pts_disp)
                if not hits_label and not hits_point:
                    best = (cx, cy, box)
                    break
            if best:
                break
        if best is None:
            cx, cy = px + 20 * scale, py + 14 * scale
            best = (cx, cy, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        placed.append(best[2])
        ax.annotate(text, p, xytext=((best[0] - px) / scale, (best[1] - py) / scale), textcoords="offset points",
                    fontsize=fontsize, color="#00405a", ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#0090c0", lw=0.5, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="#0090c0", lw=0.7))


def decode_counts(card_map: CardMap, z) -> tuple[Counter, float]:
    """Nearest-card decode of (n, dims) samples: (Counter of card index, mean NN distance)."""
    z = card_map.project(z)
    if len(z) == 0:
        return Counter(), float("nan")
    idx, dist = card_map.nearest(z, k=1)
    return Counter(idx[:, 0].tolist()), float(dist.mean())


def render_belief(decision: dump.Decision, samples, card_map: CardMap, path: str | Path | None = None, *,
                  top_k: int = 12, max_true_labels: int = 10, dpi: int = 100, method: str = "knn",
                  title: str | None = None) -> Path:
    """One row per hidden region: density of the belief samples on the frozen
    card map with the true cards marked (size ∝ count), and the samples
    decoded to their nearest card (top-k frequencies) beside the true fractions.

    *samples*: ``{region id or is_region index: (n, D)}`` (arrays or tensors,
    any device), or a ``(n, R, D)`` / ``(n, U, D)`` array (see
    :func:`region_samples`). D is the map's dims (whitened PCA) or the raw
    embedding width (projected with the map's PCA basis).
    """
    plt = _plt()
    d = decision
    path = Path(path) if path is not None else Path(f"belief_g{d.game}_d{d.decision}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    smp = _norm_samples(d, samples, card_map)
    truth = {t.get("region"): t for t in (d.hidden_truth or [])}
    regs = [t for t in regions(d) if t["id"] in smp or t["id"] in truth] or [{"id": k} for k in smp]
    regs = [t for t in regs if t["id"] in smp] or regs

    cx, cy = card_map.coords[:, 0], card_map.coords[:, 1]
    pad_x, pad_y = (cx.max() - cx.min()) * 0.03, (cy.max() - cy.min()) * 0.03
    extent = (float(cx.min() - pad_x), float(cx.max() + pad_x), float(cy.min() - pad_y), float(cy.max() + pad_y))

    nrow = max(1, len(regs))
    fig_h = 6.4 * nrow + 1.1
    fig = plt.figure(figsize=(17, fig_h), dpi=dpi)
    fig.suptitle(title or f"Belief samples · game {d.game} · decision {d.decision} · turn {d.turn} · {d.viewer}",
                 fontsize=14, weight="bold", y=1 - 0.2 / fig_h)
    fig.text(0.5, 1 - 0.5 / fig_h, "map: frozen UMAP of all cards' whitened PCA targets (grey) · heatmap/dots: "
             "belief samples · blue rings: ground truth", ha="center", va="top", fontsize=9, color="#555")
    gs = fig.add_gridspec(nrow, 2, width_ratios=[1.0, 1.05], left=0.04, right=0.98, top=1 - 1.15 / fig_h,
                          bottom=0.4 / fig_h, hspace=0.3, wspace=0.32)
    for r, tok in enumerate(regs):
        rid = tok["id"]
        z = smp.get(rid, np.zeros((0, card_map.dims), np.float32))
        lab = truth.get(rid, {"cards": [], "total": _region_count(tok)})
        true_cards = lab.get("cards") or []
        total = lab.get("total") or sum(c.get("count", 1) for c in true_cards) or 0

        ax = fig.add_subplot(gs[r, 0])
        ax.scatter(cx, cy, s=0.6, c="#c8c8c8", lw=0, rasterized=True)
        if len(z):
            xy = card_map.transform(z, method=method)
            ax.imshow(_density(xy, extent), extent=extent, origin="lower", cmap="inferno_r", alpha=0.75,
                      aspect="auto", interpolation="bilinear", zorder=2)
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#5a1a00", alpha=min(0.5, 30 / len(xy) + 0.08), lw=0,
                       zorder=3, rasterized=True)
        tc = []
        for c in true_cards:
            i = card_map.index(c.get("oracle_id"), c.get("name"))
            if i is not None:
                tc.append((c, i))
        if tc:
            pts = card_map.coords[[i for _, i in tc]]
            sizes = np.array([c.get("count", 1) for c, _ in tc], float)
            ax.scatter(pts[:, 0], pts[:, 1], s=30 + 45 * np.sqrt(sizes), facecolors="none", edgecolors="#0090c0",
                       lw=1.8, zorder=5)
            ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#0090c0", zorder=6)
            ranked = sorted(range(len(tc)), key=lambda k: -tc[k][0].get("count", 1))[:max_true_labels]
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            _place_labels(ax, [(pts[k], f"{_short(tc[k][0].get('name', '?'), 24)} ×{tc[k][0].get('count', 1)}")
                               for k in ranked])
            from matplotlib.lines import Line2D

            extra = f" · {len(tc) - len(ranked)} more unlabelled" if len(tc) > len(ranked) else ""
            ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec="#0090c0", mew=1.8, ms=9,
                                      label=f"true card (size ∝ count){extra}")],
                      loc="lower right", fontsize=7.5, framealpha=0.85)
        missing = len(true_cards) - len(tc)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{rid}  ·  {total} true card(s), {len(true_cards)} distinct"
                     f"{f' ({missing} not on map)' if missing else ''}  ·  {len(z)} samples", fontsize=10.5,
                     weight="bold", loc="left")

        bx = fig.add_subplot(gs[r, 1])
        counts, mean_d = decode_counts(card_map, z)
        n = max(1, len(z))
        true_frac = {}
        for c, i in tc:
            true_frac[i] = true_frac.get(i, 0) + c.get("count", 1) / max(1, total)
        pred_frac = {i: v / n for i, v in counts.items()}
        top_pred = [i for i, _ in counts.most_common(top_k)]
        top_true = sorted(true_frac, key=lambda i: -true_frac[i])[: max(3, top_k // 2)]
        cards = list(dict.fromkeys(top_pred + [i for i in top_true if i not in top_pred]))
        cards.sort(key=lambda i: -max(pred_frac.get(i, 0), true_frac.get(i, 0)))
        cards = cards[: top_k + 4]
        hit = sum(v for i, v in pred_frac.items() if i in true_frac)
        tv = (0.5 * sum(abs(pred_frac.get(i, 0) - true_frac.get(i, 0)) for i in set(pred_frac) | set(true_frac))
              if true_frac else float("nan"))
        if cards:
            yy = np.arange(len(cards))
            bx.barh(yy - 0.2, [pred_frac.get(i, 0) for i in cards], height=0.38, color="#d9822b",
                    label="decoded samples (nearest card)")
            bx.barh(yy + 0.2, [true_frac.get(i, 0) for i in cards], height=0.38, color="#00a0d0",
                    label="ground truth (count / total)")
            bx.set_yticks(yy, [_short(card_map.names[i], 30) for i in cards], fontsize=8.5)
            for lbl, i in zip(bx.get_yticklabels(), cards):
                if i in true_frac:
                    lbl.set_weight("bold")
            bx.invert_yaxis()
            bx.legend(loc="lower right", fontsize=8)
            bx.grid(axis="x", alpha=0.3)
            bx.set_xlim(0, max(0.05, max(max(pred_frac.get(i, 0), true_frac.get(i, 0)) for i in cards) * 1.12))
        else:
            bx.text(0.5, 0.5, "no samples and no ground truth", ha="center", va="center", transform=bx.transAxes)
            bx.set_xticks([])
            bx.set_yticks([])
        if not true_cards:
            bx.text(0.98, 0.5, "empty region (no true cards)", ha="right", va="center", transform=bx.transAxes,
                    fontsize=10, color="#888")
        stats = (f"mass on true cards {hit:.2f} · TV distance {'n/a' if tv != tv else f'{tv:.2f}'} · distinct decoded {len(counts)}"
                 f" · mean NN dist {mean_d:.2f}" if len(z) else "no samples")
        bx.set_title(f"decoded top-{top_k} vs truth (bold = true card)\n{stats}", fontsize=9.5, loc="left")

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def render_decision(decision: dump.Decision, path_prefix: str | Path, samples=None, action_probs=None,
                    card_map: CardMap | None = None, **kwargs) -> dict[str, Path]:
    """Write ``<prefix>_board.png`` and, given *samples*, ``<prefix>_belief.png``
    (loading/fitting the default :class:`CardMap` if none is passed)."""
    prefix = str(path_prefix)
    out = {"board": render_board(decision, prefix + "_board.png", action_probs=action_probs)}
    if samples is not None:
        if card_map is None:
            card_map = CardMap.load_or_fit()
        out["belief"] = render_belief(decision, samples, card_map, prefix + "_belief.png", **kwargs)
    return out


def fake_belief(decision: dump.Decision, card_map: CardMap, sigma: float, n: int = 500,
                seed: int = 0) -> dict[str, np.ndarray]:
    """Demo samples: per region, draw true cards ∝ count, take their targets
    and add N(0, sigma²) noise (empty regions draw uniform random cards)."""
    rng = np.random.default_rng(seed)
    out = {}
    for lab in decision.hidden_truth or []:
        idx, w = [], []
        for c in lab.get("cards") or []:
            i = card_map.index(c.get("oracle_id"), c.get("name"))
            if i is not None:
                idx.append(i)
                w.append(c.get("count", 1))
        if idx:
            pick = rng.choice(idx, size=n, p=np.asarray(w, float) / sum(w))
        else:
            pick = rng.integers(0, len(card_map.targets), size=n)
        out[lab["region"]] = (card_map.targets[pick] + sigma * rng.standard_normal((n, card_map.dims))).astype(np.float32)
    return out
