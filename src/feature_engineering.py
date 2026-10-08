"""Turn cleaned ball-by-ball deliveries into per-player statistics used by the scoring layer.

All functions operate on the output of `data_cleaning.build_clean_dataset` and, where a
`players` list is given, restrict output to those players (the candidate squad `P`).
"""
from __future__ import annotations

import pandas as pd

TOP_POSITIONS = (1, 2, 3)


def compute_batting_positions(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Derive each player's batting position (1..11+) per innings from entry order.

    The dataset has no explicit position column, so position is inferred from the order in
    which players first appear (as striker or non-striker) within a match+inning, sorted by
    (over, ball). The two openers are assigned positions 1 and 2 from the first ball; every
    later new name is the next batter in.
    """
    cols = ["match_id", "inning", "over", "ball", "batter", "non_striker"]
    df = deliveries[cols].sort_values(["match_id", "inning", "over", "ball"])

    records = []
    for (match_id, inning), g in df.groupby(["match_id", "inning"], sort=False):
        seen: dict[str, int] = {}
        pos = 0
        for i, row in enumerate(g.itertuples(index=False)):
            if i == 0:
                pos += 1
                seen[row.batter] = pos
                pos += 1
                seen[row.non_striker] = pos
                continue
            if row.batter not in seen:
                pos += 1
                seen[row.batter] = pos
            if row.non_striker not in seen:
                pos += 1
                seen[row.non_striker] = pos
        for player, p in seen.items():
            records.append(
                {"match_id": match_id, "inning": inning, "player": player, "batting_position": min(p, 11)}
            )
    return pd.DataFrame.from_records(records)


def _batting_rates(g: pd.DataFrame) -> pd.Series:
    runs = g["batsman_runs"].sum()
    balls_faced = g.loc[g["is_ball_faced"], "batsman_runs"].count()
    dismissals = g["is_batting_dismissal"].sum()
    boundaries = g["batsman_runs"].isin([4, 6]).sum()
    dot_balls = (g.loc[g["is_ball_faced"], "batsman_runs"] == 0).sum()
    average = runs / dismissals if dismissals > 0 else float(runs)
    strike_rate = (runs / balls_faced) * 100 if balls_faced > 0 else 0.0
    boundary_pct = (boundaries / balls_faced) * 100 if balls_faced > 0 else 0.0
    dot_pct = (dot_balls / balls_faced) * 100 if balls_faced > 0 else 0.0
    return pd.Series(
        {
            "runs": runs,
            "balls_faced": balls_faced,
            "dismissals": dismissals,
            "average": average,
            "strike_rate": strike_rate,
            "boundary_pct": boundary_pct,
            "dot_pct": dot_pct,
        }
    )


def batting_overall(deliveries: pd.DataFrame, players: list[str] | None = None) -> pd.DataFrame:
    d = deliveries if players is None else deliveries[deliveries["batter"].isin(players)]
    out = d.groupby("batter").apply(_batting_rates, include_groups=False)
    out.index.name = "player"
    return out.reset_index()


def batting_by_position(
    deliveries: pd.DataFrame, positions: pd.DataFrame, players: list[str] | None = None
) -> pd.DataFrame:
    d = deliveries.merge(
        positions, left_on=["match_id", "inning", "batter"], right_on=["match_id", "inning", "player"], how="inner"
    )
    if players is not None:
        d = d[d["batter"].isin(players)]
    out = d.groupby(["batter", "batting_position"]).apply(_batting_rates, include_groups=False)
    out.index.names = ["player", "batting_position"]
    return out.reset_index()


def batting_by_phase(deliveries: pd.DataFrame, players: list[str] | None = None) -> pd.DataFrame:
    d = deliveries if players is None else deliveries[deliveries["batter"].isin(players)]
    out = d.groupby(["batter", "phase"]).apply(_batting_rates, include_groups=False)
    out.index.names = ["player", "phase"]
    return out.reset_index()


def _bowling_rates(g: pd.DataFrame) -> pd.Series:
    legal_balls = g["is_legal_delivery"].sum()
    runs_conceded = g.loc[g["is_legal_delivery"] | g["is_noball"], "total_runs"].sum()
    wickets = g["is_bowler_wicket"].sum()
    dot_balls = (g.loc[g["is_legal_delivery"], "total_runs"] == 0).sum()
    overs = legal_balls / 6.0
    economy = runs_conceded / overs if overs > 0 else 0.0
    bowling_sr = legal_balls / wickets if wickets > 0 else float(legal_balls)
    dot_pct = (dot_balls / legal_balls) * 100 if legal_balls > 0 else 0.0
    return pd.Series(
        {
            "legal_balls": legal_balls,
            "runs_conceded": runs_conceded,
            "wickets": wickets,
            "economy": economy,
            "bowling_strike_rate": bowling_sr,
            "dot_pct": dot_pct,
        }
    )


def bowling_overall(deliveries: pd.DataFrame, players: list[str] | None = None) -> pd.DataFrame:
    d = deliveries if players is None else deliveries[deliveries["bowler"].isin(players)]
    out = d.groupby("bowler").apply(_bowling_rates, include_groups=False)
    out.index.name = "player"
    return out.reset_index()


def bowling_by_phase(deliveries: pd.DataFrame, players: list[str] | None = None) -> pd.DataFrame:
    d = deliveries if players is None else deliveries[deliveries["bowler"].isin(players)]
    out = d.groupby(["bowler", "phase"]).apply(_bowling_rates, include_groups=False)
    out.index.names = ["player", "phase"]
    return out.reset_index()


def recent_form(
    deliveries: pd.DataFrame, players: list[str] | None = None, n_recent_matches: int = 10
) -> pd.DataFrame:
    """Batting+bowling recent-form raw inputs: stats over each player's last N matches played."""
    d = deliveries if players is None else deliveries[
        deliveries["batter"].isin(players) | deliveries["bowler"].isin(players)
    ]

    records = []
    target_players = players if players is not None else sorted(
        set(d["batter"]).union(d["bowler"])
    )
    for player in target_players:
        bat_matches = d.loc[d["batter"] == player, ["match_id", "date"]].drop_duplicates()
        bowl_matches = d.loc[d["bowler"] == player, ["match_id", "date"]].drop_duplicates()
        recent_bat_ids = (
            bat_matches.sort_values("date", ascending=False).head(n_recent_matches)["match_id"]
        )
        recent_bowl_ids = (
            bowl_matches.sort_values("date", ascending=False).head(n_recent_matches)["match_id"]
        )
        bat_rows = d[(d["batter"] == player) & (d["match_id"].isin(recent_bat_ids))]
        bowl_rows = d[(d["bowler"] == player) & (d["match_id"].isin(recent_bowl_ids))]
        bat_stats = _batting_rates(bat_rows) if len(bat_rows) else pd.Series(
            {"runs": 0, "balls_faced": 0, "dismissals": 0, "average": 0.0, "strike_rate": 0.0, "boundary_pct": 0.0, "dot_pct": 0.0}
        )
        bowl_stats = _bowling_rates(bowl_rows) if len(bowl_rows) else pd.Series(
            {"legal_balls": 0, "runs_conceded": 0, "wickets": 0, "economy": 0.0, "bowling_strike_rate": 0.0, "dot_pct": 0.0}
        )
        records.append(
            {
                "player": player,
                "recent_matches_batted": len(recent_bat_ids),
                "recent_strike_rate": bat_stats["strike_rate"],
                "recent_average": bat_stats["average"],
                "recent_matches_bowled": len(recent_bowl_ids),
                "recent_economy": bowl_stats["economy"],
                "recent_wickets": bowl_stats["wickets"],
            }
        )
    return pd.DataFrame.from_records(records)


def matchup_score(
    deliveries: pd.DataFrame, opponent: str, players: list[str] | None = None
) -> pd.DataFrame:
    """Batting average/strike-rate for each player specifically against `opponent`."""
    d = deliveries[deliveries["bowling_team"] == opponent]
    if players is not None:
        d = d[d["batter"].isin(players)]
    out = d.groupby("batter").apply(_batting_rates, include_groups=False)
    out.index.name = "player"
    return out.reset_index().rename(columns={"strike_rate": "matchup_strike_rate", "average": "matchup_average"})
