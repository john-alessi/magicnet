import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_dumps", Path(__file__).resolve().parents[1] / "scripts" / "generate_dumps.py")
gen = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gen  # dataclasses resolve their module by name
_SPEC.loader.exec_module(gen)

DECKS = [
    "[Premodern] #01 - Selesnya Enchantress.dck",
    "[Premodern] #24 - 4c Terragedon.dck",
    "[Premodern] #30 - 4c Terragedon.dck",
    "[Premodern] #08 - Psychatog.dck",
]


def test_shares_split_over_matching_decks_and_renormalize(tmp_path: Path) -> None:
    csv = tmp_path / "shares.csv"
    csv.write_text("# comment\nArchetype,Share,Deck\nSelesnya enchantress,0.2,\n4c Terragedon,0.4,\n"
                   "Unknown Thing,0.1,\nPsychatog,0.1,[Premodern] #08 - Psychatog.dck\n")
    w = gen.weights(DECKS, csv)
    assert w == pytest.approx({DECKS[0]: 0.2 / 0.7, DECKS[1]: 0.2 / 0.7, DECKS[2]: 0.2 / 0.7,
                               DECKS[3]: 0.1 / 0.7})


def test_worker_cap_keeps_half_of_available_memory() -> None:
    assert gen.heap_mb("2g") == gen.heap_mb("2048m") == 2048
    per = 2048 + gen.JVM_OVERHEAD_MB
    assert gen.cap_workers(8, 2048, 4 * per) == 2
    assert gen.cap_workers(1, 2048, 100 * per) == 1
    assert gen.cap_workers(4, 2048, 100) == 1
    assert gen.cap_workers(4, 2048, None) == 4
