"""IO helpers for game logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_game_jsonl(path: str | Path, game_record_dict: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(game_record_dict) + "\n")


def load_games_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    games: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            games.append(json.loads(line))
    return games


def _summary_row(game: dict[str, Any]) -> dict[str, Any]:
    meta = game.get("meta", {})
    result = game.get("result", {})
    return {
        "game_id": meta.get("game_id"),
        "seed": meta.get("seed"),
        "agents": json.dumps(meta.get("agents", {}), ensure_ascii=False),
        "winner": result.get("winner"),
        "score_diff": result.get("score_diff"),
        "n_moves": result.get("n_moves"),
        "duration_ms": result.get("duration_ms"),
    }


def write_summary_csv(path: str | Path, games: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(game) for game in games]
    fieldnames = ["game_id", "seed", "agents", "winner", "score_diff", "n_moves", "duration_ms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
