"""Normalise raw features (spec §5) and build the composite scores the optimisation models
consume: R_ik (position-specific batting score), O_i (bowling score), F_i (recent-form score),
and optionally M_i (opponent match-up score).

Sparse-data fallback for R_ik follows the spec's hierarchy (§11 Step 4) via a credibility
(shrinkage) blend of a player's own position-specific stats and a league-wide position baseline,
instead of using the player's raw position stats directly when the sample is thin.
"""
from __future__ import annotations

import pandas as pd

POSITIONS = list(range(1, 12))
SHRINKAGE_K = 30  # balls-faced "prior weight" given to the league baseline when data is sparse
BOWLING_SHRINKAGE_K = 60  # legal-balls-bowled prior weight (overs are a coarser sample unit)


def normalise(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Min-max scale a series to 0-100, per spec §5. Constant series map to 50 (neutral)."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(50.0, index=series.index)
    if higher_is_better:
        return 100 * (series - lo) / (hi - lo)
    return 100 * (hi - series) / (hi - lo)


def build_league_position_baseline(bat_by_position_all: pd.DataFrame) -> pd.DataFrame:
    """League-wide average/strike-rate per batting position, used as the shrinkage prior."""
    qualifying = bat_by_position_all[bat_by_position_all["balls_faced"] >= 10]
    baseline = (
        qualifying.groupby("batting_position")
        .apply(
            lambda g: pd.Series(
                {
                    "league_average": (g["average"] * g["balls_faced"]).sum() / g["balls_faced"].sum(),
                    "league_strike_rate": (g["strike_rate"] * g["balls_faced"]).sum() / g["balls_faced"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return baseline


def build_R_ik(
    squad: list[str],
    bat_by_position_squad: pd.DataFrame,
    league_baseline: pd.DataFrame,
    recent_form_normalised: pd.Series,
    weights: tuple[float, float, float] = (0.40, 0.35, 0.15),
) -> pd.DataFrame:
    """Build R_ik for every (player, position) pair in the squad x {1..11} grid."""
    grid = pd.MultiIndex.from_product([squad, POSITIONS], names=["player", "batting_position"]).to_frame(
        index=False
    )
    grid = grid.merge(bat_by_position_squad, on=["player", "batting_position"], how="left")
    grid = grid.merge(league_baseline, on="batting_position", how="left")

    grid["balls_faced"] = grid["balls_faced"].fillna(0)
    grid["average"] = grid["average"].fillna(0.0)
    grid["strike_rate"] = grid["strike_rate"].fillna(0.0)

    credibility = grid["balls_faced"] / (grid["balls_faced"] + SHRINKAGE_K)
    grid["shrunk_average"] = credibility * grid["average"] + (1 - credibility) * grid["league_average"]
    grid["shrunk_strike_rate"] = (
        credibility * grid["strike_rate"] + (1 - credibility) * grid["league_strike_rate"]
    )

    # Normalise within the candidate squad, position by position (the pool the model chooses from).
    grid["norm_average"] = grid.groupby("batting_position")["shrunk_average"].transform(normalise)
    grid["norm_strike_rate"] = grid.groupby("batting_position")["shrunk_strike_rate"].transform(normalise)

    grid = grid.merge(
        recent_form_normalised.rename("norm_recent_form"), left_on="player", right_index=True, how="left"
    )
    grid["norm_recent_form"] = grid["norm_recent_form"].fillna(0.0)

    w_avg, w_sr, w_form = weights
    grid["R_ik"] = w_avg * grid["norm_average"] + w_sr * grid["norm_strike_rate"] + w_form * grid["norm_recent_form"]

    return grid[["player", "batting_position", "R_ik", "norm_average", "norm_strike_rate", "balls_faced"]]


def build_bowling_league_baseline(bowl_overall_all: pd.DataFrame, min_balls: int = 60) -> pd.Series:
    """League-wide economy/strike-rate baseline, used as the shrinkage prior for O_i."""
    qualifying = bowl_overall_all[bowl_overall_all["legal_balls"] >= min_balls]
    total_balls = qualifying["legal_balls"].sum()
    return pd.Series(
        {
            "league_economy": (qualifying["economy"] * qualifying["legal_balls"]).sum() / total_balls,
            "league_bowling_sr": (qualifying["bowling_strike_rate"] * qualifying["legal_balls"]).sum()
            / total_balls,
        }
    )


def build_O_i(
    bowl_overall_squad: pd.DataFrame, role_table: pd.DataFrame, league_baseline: pd.Series
) -> pd.Series:
    """Bowling-capability score O_i, 0 for players who don't bowl (q_i = 0).

    Small-sample bowlers (e.g. part-timers with a handful of overs) are shrunk toward a
    league-wide baseline so a few noisy deliveries can't produce an extreme score, mirroring
    the credibility blend used for R_ik.
    """
    d = bowl_overall_squad.copy()
    credibility = d["legal_balls"] / (d["legal_balls"] + BOWLING_SHRINKAGE_K)
    d["shrunk_economy"] = credibility * d["economy"] + (1 - credibility) * league_baseline["league_economy"]
    d["shrunk_bowling_sr"] = (
        credibility * d["bowling_strike_rate"] + (1 - credibility) * league_baseline["league_bowling_sr"]
    )
    d["norm_economy"] = normalise(d["shrunk_economy"], higher_is_better=False)
    d["norm_bowling_sr"] = normalise(d["shrunk_bowling_sr"], higher_is_better=False)
    d["O_i"] = 0.5 * d["norm_economy"] + 0.5 * d["norm_bowling_sr"]
    o = d.set_index("player")["O_i"]

    can_bowl = role_table.set_index("player")["can_bowl"]
    o = o.reindex(can_bowl.index).fillna(0.0)
    o = o.where(can_bowl == 1, 0.0)
    return o


def build_F_i(recent_form_df: pd.DataFrame, role_table: pd.DataFrame) -> pd.Series:
    """Recent-form score F_i: batting form for everyone, blended with bowling form for bowlers."""
    d = recent_form_df.set_index("player")
    norm_avg = normalise(d["recent_average"])
    norm_sr = normalise(d["recent_strike_rate"])
    bat_form = 0.5 * norm_avg + 0.5 * norm_sr

    norm_econ_inv = normalise(d["recent_economy"], higher_is_better=False)
    norm_wkts = normalise(d["recent_wickets"])
    bowl_form = 0.5 * norm_econ_inv + 0.5 * norm_wkts

    can_bowl = role_table.set_index("player")["can_bowl"].reindex(d.index).fillna(0)
    f_i = bat_form.where(can_bowl == 0, 0.6 * bat_form + 0.4 * bowl_form)
    return f_i.rename("F_i")


def build_all_scores(
    deliveries: pd.DataFrame,
    role_table: pd.DataFrame,
    opponent: str | None = None,
    n_recent_matches: int = 10,
) -> dict:
    """Run the full scoring pipeline for the squad listed in `role_table`.

    Returns a dict with:
      - R_ik: DataFrame with columns [player, batting_position, R_ik]
      - O_i, F_i: pd.Series indexed by player
      - M_i: pd.Series indexed by player (neutral 50 for every player if `opponent` is None)
    """
    # Import locally to avoid a hard dependency from scoring.py on feature_engineering.py at
    # module-load time (keeps scoring.py usable/testable standalone).
    from src import feature_engineering as fe

    squad = role_table["player"].tolist()

    positions = fe.compute_batting_positions(deliveries)
    bat_pos_squad = fe.batting_by_position(deliveries, positions, squad)
    bat_pos_all = fe.batting_by_position(deliveries, positions, None)
    league_pos_baseline = build_league_position_baseline(bat_pos_all)

    form = fe.recent_form(deliveries, squad, n_recent_matches=n_recent_matches)
    recent_form_norm = normalise(
        0.5 * normalise(form.set_index("player")["recent_average"])
        + 0.5 * normalise(form.set_index("player")["recent_strike_rate"])
    )

    R_ik = build_R_ik(squad, bat_pos_squad, league_pos_baseline, recent_form_norm)

    bowl_overall_squad = fe.bowling_overall(deliveries, squad)
    bowl_overall_all = fe.bowling_overall(deliveries, None)
    bowl_baseline = build_bowling_league_baseline(bowl_overall_all)
    O_i = build_O_i(bowl_overall_squad, role_table, bowl_baseline)

    F_i = build_F_i(form, role_table)

    if opponent is not None:
        matchup = fe.matchup_score(deliveries, opponent, squad)
        M_i = build_M_i(matchup, squad)
    else:
        M_i = pd.Series(50.0, index=pd.Index(squad, name="player"), name="M_i")

    return {"R_ik": R_ik, "O_i": O_i, "F_i": F_i, "M_i": M_i}


def build_M_i(matchup_df: pd.DataFrame, squad: list[str]) -> pd.Series:
    """Opponent match-up score M_i (optional); 50 (neutral) where data is insufficient."""
    d = matchup_df.set_index("player").reindex(squad)
    qualifies = d["balls_faced"].fillna(0) >= 10
    norm_avg = normalise(d["matchup_average"].fillna(0))
    norm_sr = normalise(d["matchup_strike_rate"].fillna(0))
    m_i = (0.5 * norm_avg + 0.5 * norm_sr).where(qualifies, 50.0)
    m_i.index.name = "player"
    return m_i.rename("M_i")
