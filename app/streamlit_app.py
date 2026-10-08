"""IPL Playing XI Optimisation Dashboard (spec §13).

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_cleaning as dc
from src import feature_engineering as fe
from src import scoring as sc
from src import ip_model as ipm
from src import gp_model as gpm
from src import compare as cmp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="IPL Team Optimisation Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning IPL dataset...")
def load_clean_data():
    matches, deliveries = dc.build_clean_dataset(PROJECT_ROOT / "archive")
    return matches, deliveries


@st.cache_data(show_spinner=False)
def load_role_table():
    return pd.read_csv(PROJECT_ROOT / "role_table.csv")


@st.cache_data(show_spinner="Computing player scores (R_ik, O_i, F_i, M_i)...")
def compute_scores(_deliveries: pd.DataFrame, role_table: pd.DataFrame, opponent: str | None):
    return sc.build_all_scores(_deliveries, role_table, opponent=opponent)


def main():
    st.title("IPL Team Optimisation Dashboard")
    st.caption("Optimal Playing XI selection and batting order — Integer Programming vs Goal Programming")

    matches, deliveries = load_clean_data()
    role_table = load_role_table()
    squad = role_table["player"].tolist()

    page = st.sidebar.radio(
        "Navigate",
        ["Load Data", "Statistics", "Optimisation Settings", "Run Optimisation", "Results"],
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Squad size:** {len(squad)}")
    st.sidebar.write(f"**Matches in dataset:** {len(matches)}")
    st.sidebar.write(f"**Seasons:** {matches['season'].nunique()}")

    if page == "Load Data":
        render_load_data(matches, deliveries, role_table)
    elif page == "Statistics":
        render_statistics(deliveries, role_table, squad)
    elif page == "Optimisation Settings":
        render_optimisation_settings(role_table)
    elif page == "Run Optimisation":
        render_run_optimisation(deliveries, role_table)
    elif page == "Results":
        render_results()


def render_load_data(matches, deliveries, role_table):
    st.header("A. Load Data")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Matches", len(matches))
    col2.metric("Deliveries", len(deliveries))
    col3.metric("Seasons", matches["season"].nunique())
    col4.metric("Teams", pd.concat([matches["team1"], matches["team2"]]).nunique())
    col5.metric("Squad players", len(role_table))

    st.subheader("Candidate squad & role table")
    st.dataframe(role_table, width="stretch")

    st.subheader("Sample cleaned deliveries")
    st.dataframe(deliveries.head(20), width="stretch")


def render_statistics(deliveries, role_table, squad):
    st.header("B. Statistics")

    positions = fe.compute_batting_positions(deliveries)
    bat_overall = fe.batting_overall(deliveries, squad).merge(
        role_table[["player", "role"]], on="player", how="left"
    )
    bowl_overall = fe.bowling_overall(deliveries, squad).merge(
        role_table[["player", "role"]], on="player", how="left"
    )

    st.subheader("Batting leaderboard")
    st.dataframe(
        bat_overall.sort_values("runs", ascending=False).round(1),
        width="stretch",
    )

    st.subheader("Bowling leaderboard")
    st.dataframe(
        bowl_overall.sort_values("wickets", ascending=False).round(1),
        width="stretch",
    )

    st.subheader("Role distribution in candidate squad")
    st.bar_chart(role_table["role"].value_counts())

    st.subheader("Player detail")
    player = st.selectbox("Select a player", squad)
    bat_pos = fe.batting_by_position(deliveries, positions, [player])
    bat_phase = fe.batting_by_phase(deliveries, [player])
    bowl_phase = fe.bowling_by_phase(deliveries, [player])

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Batting by position**")
        st.dataframe(bat_pos.round(1), width="stretch")
    with c2:
        st.write("**Batting by phase**")
        st.dataframe(bat_phase.round(1), width="stretch")
        st.write("**Bowling by phase**")
        st.dataframe(bowl_phase.round(1), width="stretch")


def render_optimisation_settings(role_table):
    st.header("C. Optimisation Settings")

    opponents = [
        "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
        "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings", "Rajasthan Royals",
        "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants",
    ]
    opponent = st.selectbox("Opponent (optional, for match-up score M_i)", ["None"] + opponents)
    st.session_state["opponent"] = None if opponent == "None" else opponent

    st.subheader("Role minimums (hard constraints)")
    c1, c2, c3 = st.columns(3)
    st.session_state["min_batters"] = c1.number_input("Minimum specialist batters", min_value=1, max_value=8, value=4)
    st.session_state["min_allrounders"] = c2.number_input("Minimum all-rounders", min_value=0, max_value=5, value=1)
    st.session_state["min_bowling_options"] = c3.number_input("Minimum bowling options", min_value=3, max_value=8, value=5)

    st.subheader("Integer Programming: lambda")
    st.session_state["lam"] = st.slider(
        "lambda (weight on bowling strength in IP objective)", 0.0, 5.0, 1.0, 0.1
    )

    st.subheader("Goal Programming: targets and priority weights")
    c1, c2, c3, c4 = st.columns(4)
    T_B = c1.number_input("Target batting score (T_B)", value=650)
    T_O = c2.number_input("Target bowling score (T_O)", value=350)
    T_T = c3.number_input("Target top-order score (T_T)", value=200)
    T_F = c4.number_input("Target recent-form score (T_F)", value=500)
    st.session_state["targets"] = {"T_B": T_B, "T_O": T_O, "T_T": T_T, "T_F": T_F}

    c1, c2, c3, c4 = st.columns(4)
    p_B = c1.number_input("Priority: batting (p_B)", value=5, min_value=0)
    p_O = c2.number_input("Priority: bowling (p_O)", value=4, min_value=0)
    p_T = c3.number_input("Priority: top-order (p_T)", value=3, min_value=0)
    p_F = c4.number_input("Priority: recent form (p_F)", value=2, min_value=0)
    st.session_state["priorities"] = {"p_B": p_B, "p_O": p_O, "p_T": p_T, "p_F": p_F}

    st.success("Settings saved. Go to **Run Optimisation** next.")


def render_run_optimisation(deliveries, role_table):
    st.header("D. Run Optimisation")

    opponent = st.session_state.get("opponent")
    scores = compute_scores(deliveries, role_table, opponent)
    st.session_state["scores"] = scores

    min_batters = st.session_state.get("min_batters", 4)
    min_allrounders = st.session_state.get("min_allrounders", 1)
    min_bowling_options = st.session_state.get("min_bowling_options", 5)
    lam = st.session_state.get("lam", 1.0)
    targets = st.session_state.get("targets", {"T_B": 650, "T_O": 350, "T_T": 200, "T_F": 500})
    priorities = st.session_state.get("priorities", {"p_B": 5, "p_O": 4, "p_T": 3, "p_F": 2})

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run Integer Programming", width="stretch"):
            with st.spinner("Solving IP model..."):
                result = ipm.solve_ip(
                    role_table, scores["R_ik"], scores["O_i"], lam=lam,
                    min_batters=min_batters, min_allrounders=min_allrounders,
                    min_bowling_options=min_bowling_options,
                )
            st.session_state["ip_result"] = result
            st.success(f"IP solved: {result['status']}, objective = {result['objective']:.2f}")

    with c2:
        if st.button("Run Goal Programming", width="stretch"):
            with st.spinner("Solving GP model..."):
                result = gpm.solve_gp(
                    role_table, scores["R_ik"], scores["O_i"], scores["F_i"], targets, priorities,
                    min_batters=min_batters, min_allrounders=min_allrounders,
                    min_bowling_options=min_bowling_options,
                )
            st.session_state["gp_result"] = result
            st.success(f"GP solved: {result['status']}, objective = {result['objective']:.2f}")

    if "ip_result" in st.session_state and "gp_result" in st.session_state:
        st.info("Both models solved. Go to **Results** to see the comparison.")
    elif "ip_result" in st.session_state or "gp_result" in st.session_state:
        st.info("Run the other model too, then go to **Results** to compare.")


def _render_xi_table(result: dict, role_table: pd.DataFrame):
    role = role_table.set_index("player")["role"]
    rows = [
        {"Position": k, "Player": p, "Role": role.get(p, "-")}
        for k, p in sorted(result["batting_order"].items())
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_results():
    st.header("E. Results")

    ip_result = st.session_state.get("ip_result")
    gp_result = st.session_state.get("gp_result")
    role_table = load_role_table()

    if ip_result is None and gp_result is None:
        st.warning("Run at least one model first (see **Run Optimisation**).")
        return

    c1, c2 = st.columns(2)
    if ip_result:
        with c1:
            st.subheader("Integer Programming")
            st.metric("Status", ip_result["status"])
            st.metric("Objective (Z1)", f"{ip_result['objective']:.2f}")
            if ip_result["status"] == "Optimal":
                _render_xi_table(ip_result, role_table)
            else:
                st.error(
                    "No feasible Playing XI under the current constraints "
                    "(e.g. a role minimum exceeds how many eligible players are in the squad). "
                    "Relax a constraint in **Optimisation Settings** and re-run."
                )
    if gp_result:
        with c2:
            st.subheader("Goal Programming")
            st.metric("Status", gp_result["status"])
            st.metric("Objective (Z2, shortfall)", f"{gp_result['objective']:.2f}")
            if gp_result["status"] != "Optimal":
                st.error(
                    "No feasible Playing XI under the current constraints "
                    "(e.g. a role minimum exceeds how many eligible players are in the squad). "
                    "Relax a constraint in **Optimisation Settings** and re-run."
                )
                return
            _render_xi_table(gp_result, role_table)

            st.write("**Goal achievement vs targets**")
            dev_rows = []
            labels = {"B": "Batting", "O": "Bowling", "T": "Top-order", "F": "Recent form"}
            target_keys = {"B": "T_B", "O": "T_O", "T": "T_T", "F": "T_F"}
            achieved_keys = {"B": "batting_score", "O": "bowling_score", "T": "top_order_score", "F": "form_score"}
            for g, label in labels.items():
                dev_rows.append(
                    {
                        "Goal": label,
                        "Target": gp_result["targets"][target_keys[g]],
                        "Achieved": round(gp_result["achieved"][achieved_keys[g]], 1),
                        "Shortfall (d-)": round(gp_result["deviations"][g]["d_minus"], 1),
                        "Surplus (d+)": round(gp_result["deviations"][g]["d_plus"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(dev_rows), width="stretch", hide_index=True)

    if ip_result and gp_result and ip_result["status"] == "Optimal" and gp_result["status"] == "Optimal":
        st.subheader("Model comparison")
        summary = cmp.compare_results(ip_result, gp_result)

        def _bullet_list(players: list[str]) -> str:
            return "\n".join(f"- {p}" for p in players) if players else "_none_"

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Common to both**\n\n{_bullet_list(summary['common_players'])}")
        c2.markdown(f"**IP only**\n\n{_bullet_list(summary['ip_only'])}")
        c3.markdown(f"**GP only**\n\n{_bullet_list(summary['gp_only'])}")

        st.write("**Batting-order differences**")
        if summary["batting_order_differences"]:
            st.dataframe(pd.DataFrame(summary["batting_order_differences"]), width="stretch", hide_index=True)
        else:
            st.write("No differences — both models produced the same batting order.")


if __name__ == "__main__":
    main()
