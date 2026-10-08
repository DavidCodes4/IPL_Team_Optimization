"""Goal Programming model (spec §9): select the feasible Playing XI and batting order that
minimises weighted shortfall against four targets (batting, bowling, top-order, recent form),
subject to the same hard constraints as the IP model.
"""
from __future__ import annotations

import pandas as pd
import pulp

from src.ip_model import POSITIONS, TOP_ORDER_POSITIONS


def solve_gp(
    role_table: pd.DataFrame,
    R_ik: pd.DataFrame,
    O_i: pd.Series,
    F_i: pd.Series,
    targets: dict,
    priorities: dict,
    min_batters: int = 4,
    min_allrounders: int = 1,
    min_bowling_options: int = 5,
    enforce_top_order_eligibility: bool = True,
) -> dict:
    """Solve the weighted GP model.

    `targets` keys: T_B, T_O, T_T, T_F.
    `priorities` keys: p_B, p_O, p_T, p_F (weights on the under-achievement deviations).
    """
    players = role_table["player"].tolist()
    role = role_table.set_index("player")["role"]
    can_bowl = role_table.set_index("player")["can_bowl"]
    top_order_eligible = role_table.set_index("player")["top_order_eligible"]

    WK = [p for p in players if role[p] == "Wicketkeeper"]
    BAT = [p for p in players if role[p] == "Batter"]
    AR = [p for p in players if role[p] == "All-rounder"]
    TOP = [p for p in players if top_order_eligible[p] == 1]

    r = R_ik.set_index(["player", "batting_position"])["R_ik"]

    prob = pulp.LpProblem("IPL_XI_Selection_GP", pulp.LpMinimize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in players}
    y = {
        (i, k): pulp.LpVariable(f"y_{i}_{k}", cat="Binary")
        for i in players
        for k in POSITIONS
    }

    d_minus = {g: pulp.LpVariable(f"d_minus_{g}", lowBound=0) for g in ("B", "O", "T", "F")}
    d_plus = {g: pulp.LpVariable(f"d_plus_{g}", lowBound=0) for g in ("B", "O", "T", "F")}

    # Hard constraints (identical to IP, spec §9.5)
    prob += pulp.lpSum(x[i] for i in players) == 11, "select_exactly_11"
    for i in players:
        prob += pulp.lpSum(y[i, k] for k in POSITIONS) == x[i], f"one_position_if_selected_{i}"
    for k in POSITIONS:
        prob += pulp.lpSum(y[i, k] for i in players) == 1, f"fill_position_{k}"
    prob += pulp.lpSum(x[i] for i in WK) >= 1, "min_wicketkeeper"
    prob += pulp.lpSum(x[i] for i in BAT) >= min_batters, "min_batters"
    prob += pulp.lpSum(x[i] for i in AR) >= min_allrounders, "min_allrounders"
    prob += (
        pulp.lpSum(can_bowl.get(i, 0) * x[i] for i in players) >= min_bowling_options,
        "min_bowling_options",
    )
    if enforce_top_order_eligibility:
        for k in TOP_ORDER_POSITIONS:
            prob += pulp.lpSum(y[i, k] for i in TOP) == 1, f"top_order_eligibility_pos_{k}"

    # Goal equations: f_g(x,y) + d_g^- - d_g^+ = T_g
    batting_score = pulp.lpSum(r.get((i, k), 0.0) * y[i, k] for i in players for k in POSITIONS)
    bowling_score = pulp.lpSum(O_i.get(i, 0.0) * x[i] for i in players)
    top_order_score = pulp.lpSum(
        r.get((i, k), 0.0) * y[i, k] for i in players for k in TOP_ORDER_POSITIONS
    )
    form_score = pulp.lpSum(F_i.get(i, 0.0) * x[i] for i in players)

    prob += batting_score + d_minus["B"] - d_plus["B"] == targets["T_B"], "goal_batting"
    prob += bowling_score + d_minus["O"] - d_plus["O"] == targets["T_O"], "goal_bowling"
    prob += top_order_score + d_minus["T"] - d_plus["T"] == targets["T_T"], "goal_top_order"
    prob += form_score + d_minus["F"] - d_plus["F"] == targets["T_F"], "goal_recent_form"

    prob += (
        priorities["p_B"] * d_minus["B"]
        + priorities["p_O"] * d_minus["O"]
        + priorities["p_T"] * d_minus["T"]
        + priorities["p_F"] * d_minus["F"]
    )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    batting_order = {}
    for k in POSITIONS:
        for i in players:
            if pulp.value(y[i, k]) and pulp.value(y[i, k]) > 0.5:
                batting_order[k] = i

    selected_xi = sorted(i for i in players if pulp.value(x[i]) and pulp.value(x[i]) > 0.5)

    deviations = {
        g: {"d_minus": pulp.value(d_minus[g]), "d_plus": pulp.value(d_plus[g])} for g in ("B", "O", "T", "F")
    }
    achieved = {
        "batting_score": pulp.value(batting_score),
        "bowling_score": pulp.value(bowling_score),
        "top_order_score": pulp.value(top_order_score),
        "form_score": pulp.value(form_score),
    }

    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective),
        "selected_xi": selected_xi,
        "batting_order": batting_order,
        "deviations": deviations,
        "achieved": achieved,
        "targets": targets,
        "priorities": priorities,
    }
