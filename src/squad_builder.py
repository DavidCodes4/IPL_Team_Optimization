"""Build a candidate squad for any team+season from the raw data, and auto-infer a role table
(wicketkeeper/batter/all-rounder/bowler, can-bowl, top-order-eligible) since the dataset has no
role column. Heuristics use each player's full career stats, not just the selected season, to
avoid misclassifying someone from a handful of matches.
"""
from __future__ import annotations

import pandas as pd

from src import feature_engineering as fe

MIN_STUMPINGS_FOR_KEEPER = 3
MIN_BOWL_BALLS_CAN_BOWL = 60  # ~10 overs career-wide
MIN_BOWL_BALLS_FOR_BOWLER_ROLE = 150
MIN_BAT_BALLS_FOR_ALLROUNDER = 200
TOP_ORDER_POSITION_THRESHOLD = 4.0

# "All-time" pool thresholds: a career-wide minimum sample so the all-time XI is drawn from
# players with a real body of work, not one-off appearances that would just collapse to the
# league-baseline score under R_ik/O_i shrinkage and clutter the candidate pool for no benefit.
ALLTIME_MIN_BALLS_FACED = 300
ALLTIME_MIN_LEGAL_BALLS_BOWLED = 180


def list_teams(matches: pd.DataFrame) -> list[str]:
    return sorted(set(matches["team1"]).union(matches["team2"]))


def list_seasons_for_team(matches: pd.DataFrame, team: str) -> list[str]:
    seasons = matches.loc[(matches["team1"] == team) | (matches["team2"] == team), "season"]
    return sorted(seasons.unique(), reverse=True)


def build_squad(matches: pd.DataFrame, deliveries: pd.DataFrame, team: str, season: str) -> list[str]:
    """Every player who batted or bowled for `team` during `season` (played-squad approximation:
    a player who featured in the squad but never faced or bowled a ball can't be recovered from
    ball-by-ball data alone).
    """
    match_ids = matches.loc[
        (matches["season"] == season) & ((matches["team1"] == team) | (matches["team2"] == team)),
        "id",
    ]
    d = deliveries[deliveries["match_id"].isin(match_ids)]
    batters = d.loc[d["batting_team"] == team, "batter"]
    bowlers = d.loc[d["bowling_team"] == team, "bowler"]
    return sorted(set(batters).union(bowlers))


def _detect_wicketkeepers(deliveries: pd.DataFrame, squad: list[str]) -> set[str]:
    stumpings = deliveries[deliveries["dismissal_kind"] == "stumped"]
    counts = stumpings[stumpings["fielder"].isin(squad)]["fielder"].value_counts()
    return set(counts[counts >= MIN_STUMPINGS_FOR_KEEPER].index)


def _top_order_flags(deliveries: pd.DataFrame, squad: list[str]) -> dict[str, bool]:
    positions = fe.compute_batting_positions(deliveries)
    bat_pos = fe.batting_by_position(deliveries, positions, squad)
    flags = {}
    for player in squad:
        rows = bat_pos[bat_pos["player"] == player]
        total_balls = rows["balls_faced"].sum()
        if total_balls == 0:
            flags[player] = False
            continue
        weighted_pos = (rows["batting_position"] * rows["balls_faced"]).sum() / total_balls
        flags[player] = weighted_pos <= TOP_ORDER_POSITION_THRESHOLD
    return flags


def infer_roles(deliveries: pd.DataFrame, squad: list[str]) -> pd.DataFrame:
    """Auto-generate a role table matching role_table.csv's schema, using career-wide stats."""
    if not squad:
        return pd.DataFrame(columns=["player", "role", "can_bowl", "top_order_eligible"])
    bat_overall = fe.batting_overall(deliveries, squad).set_index("player")
    bowl_overall = fe.bowling_overall(deliveries, squad).set_index("player")
    keepers = _detect_wicketkeepers(deliveries, squad)
    top_order = _top_order_flags(deliveries, squad)

    rows = []
    for player in squad:
        balls_faced = bat_overall["balls_faced"].get(player, 0)
        legal_balls = bowl_overall["legal_balls"].get(player, 0)
        can_bowl = int(legal_balls >= MIN_BOWL_BALLS_CAN_BOWL)

        if player in keepers:
            role = "Wicketkeeper"
        elif balls_faced >= MIN_BAT_BALLS_FOR_ALLROUNDER and legal_balls >= MIN_BOWL_BALLS_FOR_BOWLER_ROLE:
            role = "All-rounder"
        elif legal_balls >= MIN_BOWL_BALLS_FOR_BOWLER_ROLE:
            role = "Bowler"
        else:
            role = "Batter"

        rows.append(
            {
                "player": player,
                "role": role,
                "can_bowl": can_bowl,
                "top_order_eligible": int(top_order.get(player, False)),
            }
        )

    return pd.DataFrame(rows)


def build_alltime_pool(
    deliveries: pd.DataFrame,
    min_balls_faced: int = ALLTIME_MIN_BALLS_FACED,
    min_legal_balls_bowled: int = ALLTIME_MIN_LEGAL_BALLS_BOWLED,
) -> list[str]:
    """Every player across all teams/seasons with a meaningful career batting or bowling
    sample — the pool the "all-time, any team/season" best XI is chosen from.
    """
    bat = fe.batting_overall(deliveries, None)
    bowl = fe.bowling_overall(deliveries, None)
    qualifiers = set(bat.loc[bat["balls_faced"] >= min_balls_faced, "player"]) | set(
        bowl.loc[bowl["legal_balls"] >= min_legal_balls_bowled, "player"]
    )
    return sorted(qualifiers)


def validate_squad_roles(role_table: pd.DataFrame) -> list[str]:
    """Return a list of human-readable warnings about a role table (e.g. no wicketkeeper)."""
    warnings = []
    if (role_table["role"] == "Wicketkeeper").sum() == 0:
        warnings.append(
            "No wicketkeeper was auto-detected in this squad (no one had 3+ career stumpings "
            "credited). Manually set one player's role to 'Wicketkeeper' below, or the hard "
            "constraint will make both models infeasible."
        )
    if (role_table["role"] == "Batter").sum() < 4:
        warnings.append(
            "Fewer than 4 players were classified as specialist 'Batter'. The optimisers "
            "default to requiring at least 4 — lower that in Optimisation Settings or re-tag "
            "a couple of players here."
        )
    if role_table["can_bowl"].sum() < 5:
        warnings.append(
            "Fewer than 5 players were flagged as able to bowl. The optimisers default to "
            "requiring at least 5 bowling options — lower that in Optimisation Settings or "
            "re-tag a couple of players here."
        )
    return warnings
