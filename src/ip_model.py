"""Integer Programming model (spec §8): select the feasible Playing XI and batting order
that maximises max Z1 = sum_i sum_k R_ik*y_ik + lambda * sum_i O_i*x_i.
"""
from __future__ import annotations

import pandas as pd
import pulp

POSITIONS = list(range(1, 12))
TOP_ORDER_POSITIONS = (1, 2, 3)


def solve_ip(
    role_table: pd.DataFrame,
    R_ik: pd.DataFrame,
    O_i: pd.Series,
    lam: float = 1.0,
    min_batters: int = 4,
    min_allrounders: int = 1,
    min_bowling_options: int = 5,
    enforce_top_order_eligibility: bool = True,
) -> dict:
    """Solve the IP model. Returns dict with status, objective, selected XI, batting order."""
    players = role_table["player"].tolist()
    role = role_table.set_index("player")["role"]
    can_bowl = role_table.set_index("player")["can_bowl"]
    top_order_eligible = role_table.set_index("player")["top_order_eligible"]

    WK = [p for p in players if role[p] == "Wicketkeeper"]
    BAT = [p for p in players if role[p] == "Batter"]
    AR = [p for p in players if role[p] == "All-rounder"]
    TOP = [p for p in players if top_order_eligible[p] == 1]

    r = R_ik.set_index(["player", "batting_position"])["R_ik"]

    prob = pulp.LpProblem("IPL_XI_Selection_IP", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in players}
    y = {
        (i, k): pulp.LpVariable(f"y_{i}_{k}", cat="Binary")
        for i in players
        for k in POSITIONS
    }

    prob += (
        pulp.lpSum(r.get((i, k), 0.0) * y[i, k] for i in players for k in POSITIONS)
        + lam * pulp.lpSum(O_i.get(i, 0.0) * x[i] for i in players)
    )

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

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    batting_order = {}
    for k in POSITIONS:
        for i in players:
            if pulp.value(y[i, k]) and pulp.value(y[i, k]) > 0.5:
                batting_order[k] = i

    selected_xi = sorted(i for i in players if pulp.value(x[i]) and pulp.value(x[i]) > 0.5)

    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective),
        "selected_xi": selected_xi,
        "batting_order": batting_order,
        "lambda": lam,
    }
