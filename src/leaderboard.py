"""Persistent leaderboard for manually-built Playing XIs (append-only CSV, no DB dependency)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEADERBOARD_PATH = Path(__file__).resolve().parent.parent / "data" / "leaderboard.csv"

COLUMNS = [
    "timestamp",
    "team_name",
    "entrant",
    "squad_label",
    "xi",
    "batting_order",
    "lambda_used",
    "batting_score",
    "bowling_score",
    "total_score",
]


def load_leaderboard(path: Path = LEADERBOARD_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def record_entry(
    team_name: str,
    entrant: str,
    squad_label: str,
    batting_order: dict[int, str],
    lambda_used: float,
    batting_score: float,
    bowling_score: float,
    path: Path = LEADERBOARD_PATH,
) -> None:
    xi = [batting_order[k] for k in sorted(batting_order)]
    row = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "team_name": team_name,
        "entrant": entrant,
        "squad_label": squad_label,
        "xi": "; ".join(xi),
        "batting_order": "; ".join(f"{k}:{p}" for k, p in sorted(batting_order.items())),
        "lambda_used": lambda_used,
        "batting_score": round(batting_score, 2),
        "bowling_score": round(bowling_score, 2),
        "total_score": round(batting_score + lambda_used * bowling_score, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_leaderboard(path)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(path, index=False)
