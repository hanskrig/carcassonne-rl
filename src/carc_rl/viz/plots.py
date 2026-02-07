"""Matplotlib plots for logged games."""

from __future__ import annotations

from collections import Counter
from typing import Any

import matplotlib.pyplot as plt


def _score_diff(game: dict[str, Any]) -> float:
    return float(game.get("result", {}).get("score_diff", 0.0))


def plot_score_diff_hist(games: list[dict[str, Any]]):
    diffs = [_score_diff(game) for game in games]
    fig, ax = plt.subplots()
    ax.hist(diffs, bins=20, color="#4C78A8", edgecolor="black")
    ax.set_title("Score Difference Histogram (A - B)")
    ax.set_xlabel("Score Difference")
    ax.set_ylabel("Games")
    fig.tight_layout()
    return fig


def plot_score_diff_over_time(games: list[dict[str, Any]]):
    diffs = [_score_diff(game) for game in games]
    fig, ax = plt.subplots()
    ax.plot(range(len(diffs)), diffs, marker="o", linewidth=1.5)
    ax.set_title("Score Difference Over Time")
    ax.set_xlabel("Game Index")
    ax.set_ylabel("Score Difference (A - B)")
    fig.tight_layout()
    return fig


def plot_game_length_hist(games: list[dict[str, Any]]):
    lengths = [game.get("result", {}).get("n_moves", 0) for game in games]
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=20, color="#F58518", edgecolor="black")
    ax.set_title("Game Length Distribution")
    ax.set_xlabel("Moves")
    ax.set_ylabel("Games")
    fig.tight_layout()
    return fig


def plot_winrate_by_agentpair(games: list[dict[str, Any]]):
    counts = Counter()
    wins = Counter()
    for game in games:
        agents = game.get("meta", {}).get("agents", {})
        label = f"{agents.get('A', 'A')} vs {agents.get('B', 'B')}"
        counts[label] += 1
        winner = game.get("result", {}).get("winner")
        if winner == 0:
            wins[label] += 1
    labels = list(counts.keys())
    rates = [wins[label] / counts[label] if counts[label] else 0.0 for label in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, rates, color="#54A24B")
    ax.set_ylim(0, 1)
    ax.set_title("Win Rate by Agent Pair (Agent A)")
    ax.set_ylabel("Win Rate")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig
