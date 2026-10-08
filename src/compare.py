"""Side-by-side IP vs GP comparison and sensitivity-analysis sweeps (spec §11 Steps 8-9)."""
from __future__ import annotations

import pandas as pd

from src import ip_model as ipm
from src import gp_model as gpm


def compare_results(ip_result: dict, gp_result: dict) -> dict:
    ip_xi = set(ip_result["selected_xi"])
    gp_xi = set(gp_result["selected_xi"])

    order_diff = []
    for k in range(1, 12):
        ip_p = ip_result["batting_order"].get(k)
        gp_p = gp_result["batting_order"].get(k)
        if ip_p != gp_p:
            order_diff.append({"position": k, "ip_player": ip_p, "gp_player": gp_p})

    return {
        "common_players": sorted(ip_xi & gp_xi),
        "ip_only": sorted(ip_xi - gp_xi),
        "gp_only": sorted(gp_xi - ip_xi),
        "batting_order_differences": order_diff,
        "ip_objective": ip_result["objective"],
        "gp_objective": gp_result["objective"],
        "gp_achieved": gp_result["achieved"],
        "gp_targets": gp_result["targets"],
        "gp_deviations": gp_result["deviations"],
    }


def sensitivity_lambda(
    role_table: pd.DataFrame, R_ik: pd.DataFrame, O_i: pd.Series, lambdas: list[float]
) -> pd.DataFrame:
    """Re-solve the IP model across a range of lambda and report how the XI changes."""
    rows = []
    for lam in lambdas:
        result = ipm.solve_ip(role_table, R_ik, O_i, lam=lam)
        rows.append(
            {
                "lambda": lam,
                "objective": result["objective"],
                "xi": ", ".join(result["selected_xi"]),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_gp_weights(
    role_table: pd.DataFrame,
    R_ik: pd.DataFrame,
    O_i: pd.Series,
    F_i: pd.Series,
    targets: dict,
    priority_sweeps: list[dict],
) -> pd.DataFrame:
    """Re-solve the GP model across a list of priority-weight dicts {p_B, p_O, p_T, p_F}."""
    rows = []
    for priorities in priority_sweeps:
        result = gpm.solve_gp(role_table, R_ik, O_i, F_i, targets, priorities)
        rows.append(
            {
                **{f"priority_{k}": v for k, v in priorities.items()},
                "objective": result["objective"],
                "xi": ", ".join(result["selected_xi"]),
                **{f"deviation_{g}_minus": v["d_minus"] for g, v in result["deviations"].items()},
            }
        )
    return pd.DataFrame(rows)


def sensitivity_min_bowlers(
    role_table: pd.DataFrame, R_ik: pd.DataFrame, O_i: pd.Series, min_values: list[int]
) -> pd.DataFrame:
    rows = []
    for min_bowlers in min_values:
        result = ipm.solve_ip(role_table, R_ik, O_i, min_bowling_options=min_bowlers)
        rows.append(
            {
                "min_bowling_options": min_bowlers,
                "status": result["status"],
                "objective": result["objective"],
                "xi": ", ".join(result["selected_xi"]) if result["status"] == "Optimal" else "-",
            }
        )
    return pd.DataFrame(rows)
