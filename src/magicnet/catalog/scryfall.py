"""Scryfall bulk-data access. Injectable so tests never touch the network."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Protocol

import httpx

# Scryfall rejects API requests (HTTP 400) without a custom User-Agent and an
# explicit Accept header.
SCRYFALL_HEADERS = {
    "User-Agent": "magicnet-catalog/0.1 (research; card catalog service)",
    "Accept": "application/json",
}
BULK_METADATA_URL = "https://api.scryfall.com/bulk-data/default-cards"


class ScryfallSource(Protocol):
    def bulk_metadata(self) -> dict:
        """The bulk-data object for default-cards (has ``updated_at``)."""
        ...

    def download(self, metadata: dict, dest: Path) -> None:
        """Write the dump described by ``metadata`` to ``dest`` as ``.jsonl.gz``."""
        ...


class HttpScryfall:
    def __init__(self, *, timeout: float = 30.0):
        self.timeout = timeout

    def bulk_metadata(self) -> dict:
        r = httpx.get(BULK_METADATA_URL, headers=SCRYFALL_HEADERS, timeout=self.timeout)
        r.raise_for_status()
        meta = r.json()
        if "updated_at" not in meta:
            raise ValueError(f"unexpected bulk-data response: {sorted(meta)}")
        return meta

    def download(self, metadata: dict, dest: Path) -> None:
        # Scryfall currently publishes `jsonl_download_uri` (.jsonl.gz, the
        # format cards.py reads); older responses only had `download_uri`
        # (a JSON array), which we convert.
        uri = metadata.get("jsonl_download_uri") or metadata["download_uri"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with httpx.stream(
            "GET", uri, headers={"User-Agent": SCRYFALL_HEADERS["User-Agent"]},
            timeout=self.timeout, follow_redirects=True,
        ) as r:
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_bytes():  # undoes transfer encoding only
                    f.write(chunk)
        if uri.endswith(".jsonl.gz"):
            part.rename(dest)
            return
        raw = part.with_name(part.name + ".json")
        part.rename(raw)
        _json_array_to_jsonl_gz(raw, dest)
        raw.unlink()


def _json_array_to_jsonl_gz(src: Path, dest: Path) -> None:
    with src.open("rb") as f:
        cards = json.load(f)
    part = dest.with_name(dest.name + ".part")
    with gzip.open(part, "wt", encoding="utf-8") as out:
        for card in cards:
            out.write(json.dumps(card, ensure_ascii=False) + "\n")
    shutil.move(part, dest)
