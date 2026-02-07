from __future__ import annotations

import csv

from carc_rl.logging.io import append_game_jsonl, load_games_jsonl, write_summary_csv
from carc_rl.logging.recorder import GameRecorder


def test_logging_roundtrip(tmp_path):
    recorder = GameRecorder()
    recorder.start_game(
        {
            "engine_name": "test-engine",
            "ruleset": "test",
            "seed": 123,
            "agents": {"A": "Random", "B": "Random"},
        }
    )
    recorder.record_step(
        t=0,
        player=0,
        action_key="action-0",
        action_repr="Action()",
        reward=0.0,
        scores_after=[0, 0],
        extra={"tile": "T1"},
    )
    recorder.end_game(
        {
            "final_scores": [1, -1],
            "winner": 0,
            "score_diff": 2.0,
            "n_moves": 1,
            "duration_ms": 10,
        }
    )
    record = recorder.to_dict()

    jsonl_path = tmp_path / "games.jsonl"
    append_game_jsonl(jsonl_path, record)
    loaded = load_games_jsonl(jsonl_path)
    assert loaded[0]["meta"]["seed"] == 123
    assert loaded[0]["steps"][0]["action_key"] == "action-0"

    csv_path = tmp_path / "summary.csv"
    write_summary_csv(csv_path, loaded)
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows[0]["game_id"]
    assert rows[0]["seed"] == "123"
    assert rows[0]["winner"] == "0"
