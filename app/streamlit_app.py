"""IPL Playing XI Optimisation Dashboard.

A "Compare"-first dashboard: pick a squad, then immediately see three directly-comparable
scores — the all-time (any team/season) best XI, the best XI from your chosen squad, and the
XI you build yourself — side by side, with the full optimisation settings tucked into a
collapsed expander so they don't get in the way of the comparison.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_cleaning as dc
from src import feature_engineering as fe
from src import scoring as sc
from src import ip_model as ipm
from src import gp_model as gpm
from src import compare as cmp
from src import squad_builder as sqb
from src import leaderboard as lb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEAM = "Chennai Super Kings"
DEFAULT_SEASON = "2024"
DEFAULT_TARGETS = {"T_B": 170, "T_O": 150, "T_T": 160, "T_F": 140}
DEFAULT_PRIORITIES = {"p_B": 5, "p_O": 4, "p_T": 3, "p_F": 2}

st.set_page_config(page_title="IPL Team Optimisation Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning IPL dataset...")
def load_clean_data():
    matches, deliveries = dc.build_clean_dataset(PROJECT_ROOT / "archive")
    return matches, deliveries


@st.cache_data(show_spinner="Building squad...")
def build_squad_and_roles(_deliveries: pd.DataFrame, matches: pd.DataFrame, team: str, season: str):
    squad = sqb.build_squad(matches, _deliveries, team, season)
    roles = sqb.infer_roles(_deliveries, squad)
    return roles


@st.cache_data(show_spinner="Building the all-time candidate pool (one-time, ~20s)...")
def build_alltime_role_table(_deliveries: pd.DataFrame) -> pd.DataFrame:
    pool = sqb.build_alltime_pool(_deliveries)
    return sqb.infer_roles(_deliveries, pool)


@st.cache_data(show_spinner="Computing player scores (R_ik, O_i, F_i, M_i)...")
def compute_scores(_deliveries: pd.DataFrame, role_table: pd.DataFrame, opponent: str | None):
    return sc.build_all_scores(_deliveries, role_table, opponent=opponent)


def ensure_default_squad(matches, deliveries):
    if "role_table" not in st.session_state:
        st.session_state["squad_team"] = DEFAULT_TEAM
        st.session_state["squad_season"] = DEFAULT_SEASON
        st.session_state["role_table"] = build_squad_and_roles(
            deliveries, matches, DEFAULT_TEAM, DEFAULT_SEASON
        )


def current_settings() -> dict:
    return {
        "opponent": st.session_state.get("opponent"),
        "min_batters": st.session_state.get("min_batters", 4),
        "min_allrounders": st.session_state.get("min_allrounders", 1),
        "min_bowling_options": st.session_state.get("min_bowling_options", 5),
        "lam": st.session_state.get("lam", 1.0),
        "targets": st.session_state.get("targets", DEFAULT_TARGETS),
        "priorities": st.session_state.get("priorities", DEFAULT_PRIORITIES),
    }


def main():
    st.title("IPL Team Optimisation Dashboard")
    st.caption("Optimal Playing XI selection and batting order — Integer Programming vs Goal Programming")

    matches, deliveries = load_clean_data()
    ensure_default_squad(matches, deliveries)

    page = st.sidebar.radio(
        "Navigate",
        ["Compare", "Squad Selector", "Build Your XI", "Statistics", "Leaderboard"],
    )

    role_table = st.session_state["role_table"]
    squad_label = f"{st.session_state['squad_team']} {st.session_state['squad_season']}"

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Current squad:** {squad_label}")
    st.sidebar.write(f"**Squad size:** {len(role_table)}")

    if page == "Compare":
        render_compare(matches, deliveries, role_table, squad_label)
    elif page == "Squad Selector":
        render_squad_selector(matches, deliveries)
    elif page == "Build Your XI":
        render_build_your_xi(deliveries, role_table, squad_label)
    elif page == "Statistics":
        render_statistics(deliveries, role_table)
    elif page == "Leaderboard":
        render_leaderboard(squad_label)


def render_settings_expander(role_table: pd.DataFrame):
    with st.expander("⚙️ Settings — role minimums, lambda, Goal Programming targets", expanded=False):
        opponents = [
            "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
            "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings", "Rajasthan Royals",
            "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants",
        ]
        opponent = st.selectbox("Opponent (optional, for match-up score M_i)", ["None"] + opponents)
        st.session_state["opponent"] = None if opponent == "None" else opponent

        st.markdown("**Role minimums (hard constraints)**")
        c1, c2, c3 = st.columns(3)
        st.session_state["min_batters"] = c1.number_input("Minimum specialist batters", min_value=1, max_value=8, value=4)
        st.session_state["min_allrounders"] = c2.number_input("Minimum all-rounders", min_value=0, max_value=5, value=1)
        st.session_state["min_bowling_options"] = c3.number_input("Minimum bowling options", min_value=3, max_value=8, value=5)

        st.markdown("**Integer Programming: lambda** — weight on bowling strength (0-3; higher starts to swamp batting).")
        st.session_state["lam"] = st.slider("lambda", 0.0, 3.0, 1.0, 0.1)

        st.markdown(
            "**Goal Programming targets** — each is a 0-200 quality index (XI's average "
            "position-quality score, doubled) so it lands in the ballpark of a real IPL total "
            "(e.g. ~170) instead of an opaque summed score. Same normalised-score model, just "
            "rescaled for readability — not a predicted run count."
        )
        c1, c2, c3, c4 = st.columns(4)
        T_B = c1.number_input("Target batting (T_B)", value=170, min_value=0, max_value=200)
        T_O = c2.number_input("Target bowling (T_O)", value=150, min_value=0, max_value=200)
        T_T = c3.number_input("Target top-order (T_T)", value=160, min_value=0, max_value=200)
        T_F = c4.number_input("Target recent-form (T_F)", value=140, min_value=0, max_value=200)
        st.session_state["targets"] = {"T_B": T_B, "T_O": T_O, "T_T": T_T, "T_F": T_F}

        c1, c2, c3, c4 = st.columns(4)
        p_B = c1.number_input("Priority: batting", value=5, min_value=0, max_value=10)
        p_O = c2.number_input("Priority: bowling", value=4, min_value=0, max_value=10)
        p_T = c3.number_input("Priority: top-order", value=3, min_value=0, max_value=10)
        p_F = c4.number_input("Priority: recent form", value=2, min_value=0, max_value=10)
        st.session_state["priorities"] = {"p_B": p_B, "p_O": p_O, "p_T": p_T, "p_F": p_F}


def _render_xi_table(result: dict, role_table: pd.DataFrame):
    role = role_table.set_index("player")["role"]
    rows = [
        {"Position": k, "Player": p, "Role": role.get(p, "-")}
        for k, p in sorted(result["batting_order"].items())
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_compare(matches, deliveries, role_table: pd.DataFrame, squad_label: str):
    st.header("Compare")
    st.caption(
        "Three XIs, scored on the same scale (batting quality + lambda x bowling quality), "
        "so you can see exactly how your pick stacks up."
    )

    render_settings_expander(role_table)
    s = current_settings()

    alltime_role = build_alltime_role_table(deliveries)
    alltime_scores = compute_scores(deliveries, alltime_role, None)
    alltime_ip = ipm.solve_ip(
        alltime_role, alltime_scores["R_ik"], alltime_scores["O_i"], lam=s["lam"],
        min_batters=s["min_batters"], min_allrounders=s["min_allrounders"],
        min_bowling_options=s["min_bowling_options"],
    )

    squad_scores = compute_scores(deliveries, role_table, s["opponent"])
    squad_ip = ipm.solve_ip(
        role_table, squad_scores["R_ik"], squad_scores["O_i"], lam=s["lam"],
        min_batters=s["min_batters"], min_allrounders=s["min_allrounders"],
        min_bowling_options=s["min_bowling_options"],
    )
    squad_gp = gpm.solve_gp(
        role_table, squad_scores["R_ik"], squad_scores["O_i"], squad_scores["F_i"],
        s["targets"], s["priorities"], min_batters=s["min_batters"],
        min_allrounders=s["min_allrounders"], min_bowling_options=s["min_bowling_options"],
    )
    squad_gp_score = (
        cmp.ip_style_score(squad_gp, squad_scores["R_ik"], squad_scores["O_i"], s["lam"])
        if squad_gp["status"] == "Optimal" else None
    )

    my_xi = st.session_state.get("my_xi_result")
    my_xi_matches_squad = my_xi is not None and my_xi["squad_label"] == squad_label

    st.subheader("Score comparison")
    scores_row = {}
    cols = st.columns(4)
    with cols[0]:
        if alltime_ip["status"] == "Optimal":
            st.metric("🌍 All-Time Best XI", f"{alltime_ip['objective']:.1f}")
            scores_row["All-Time Best"] = alltime_ip["objective"]
        else:
            st.metric("🌍 All-Time Best XI", "infeasible")
    with cols[1]:
        if squad_ip["status"] == "Optimal":
            st.metric("🏏 Squad Best (IP)", f"{squad_ip['objective']:.1f}", help=f"Best XI from {squad_label}")
            scores_row["Squad Best (IP)"] = squad_ip["objective"]
        else:
            st.metric("🏏 Squad Best (IP)", "infeasible", help=f"Best XI from {squad_label}")
    with cols[2]:
        if squad_gp_score is not None:
            st.metric(
                "🏏 Squad Best (GP)", f"{squad_gp_score:.1f}",
                help=f"Goal-Programming XI from {squad_label}, scored on the IP objective for a fair comparison.",
            )
            scores_row["Squad Best (GP)"] = squad_gp_score
        else:
            st.metric("🏏 Squad Best (GP)", "infeasible", help=f"Best XI from {squad_label}")
    with cols[3]:
        if my_xi_matches_squad:
            st.metric(
                "🙋 Your XI", f"{my_xi['total_score']:.1f}",
                delta=f"{my_xi['total_score'] - squad_ip['objective']:.1f} vs squad-best (IP)",
            )
            scores_row["Your XI"] = my_xi["total_score"]
        elif my_xi is not None:
            st.info(f"Your saved XI was built for **{my_xi['squad_label']}**, not {squad_label}.")
        else:
            st.info("Not built yet — go to **Build Your XI**.")

    if len(scores_row) >= 2:
        chart_df = pd.DataFrame({"Score": scores_row})
        st.bar_chart(chart_df)

    st.subheader("XI details")
    with st.expander("🌍 All-Time Best XI"):
        if alltime_ip["status"] == "Optimal":
            _render_xi_table(alltime_ip, alltime_role)
        else:
            st.error("No feasible all-time XI under the current role minimums — relax them in Settings.")

    with st.expander(f"🏏 {squad_label} — Best XI (IP)"):
        if squad_ip["status"] == "Optimal":
            _render_xi_table(squad_ip, role_table)
        else:
            st.error(
                "No feasible XI under the current constraints — a role minimum likely exceeds "
                "how many eligible players are in this squad. Relax it in Settings, or fix roles "
                "on **Squad Selector**."
            )

    with st.expander(f"🏏 {squad_label} — Best XI (GP) + goal achievement"):
        if squad_gp["status"] == "Optimal":
            _render_xi_table(squad_gp, role_table)
            dev_rows = []
            labels = {"B": "Batting", "O": "Bowling", "T": "Top-order", "F": "Recent form"}
            target_keys = {"B": "T_B", "O": "T_O", "T": "T_T", "F": "T_F"}
            achieved_keys = {"B": "batting_score", "O": "bowling_score", "T": "top_order_score", "F": "form_score"}
            for g, label in labels.items():
                dev_rows.append(
                    {
                        "Goal": label,
                        "Target": squad_gp["targets"][target_keys[g]],
                        "Achieved": round(squad_gp["achieved"][achieved_keys[g]], 1),
                        "Shortfall (d-)": round(squad_gp["deviations"][g]["d_minus"], 1),
                        "Surplus (d+)": round(squad_gp["deviations"][g]["d_plus"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(dev_rows), width="stretch", hide_index=True)
        else:
            st.error(
                "No feasible XI under the current constraints — a role minimum likely exceeds "
                "how many eligible players are in this squad. Relax it in Settings, or fix roles "
                "on **Squad Selector**."
            )

    if my_xi_matches_squad:
        with st.expander("🙋 Your XI"):
            order_df = pd.DataFrame(
                {
                    "Position": range(1, 12),
                    "Player": my_xi["xi"],
                    "Role": [role_table.set_index("player")["role"].get(p, "-") for p in my_xi["xi"]],
                }
            )
            st.dataframe(order_df, width="stretch", hide_index=True)

    if squad_ip["status"] == "Optimal" and squad_gp["status"] == "Optimal":
        with st.expander("IP vs GP — what's different for this squad"):
            summary = cmp.compare_results(squad_ip, squad_gp)

            def _bullet_list(players: list[str]) -> str:
                return "\n".join(f"- {p}" for p in players) if players else "_none_"

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Common to both**\n\n{_bullet_list(summary['common_players'])}")
            c2.markdown(f"**IP only**\n\n{_bullet_list(summary['ip_only'])}")
            c3.markdown(f"**GP only**\n\n{_bullet_list(summary['gp_only'])}")
            if summary["batting_order_differences"]:
                st.write("**Batting-order differences**")
                st.dataframe(pd.DataFrame(summary["batting_order_differences"]), width="stretch", hide_index=True)


def render_squad_selector(matches, deliveries):
    st.header("Squad Selector")
    st.caption(
        "Pick a team and a season; the squad is built from everyone who batted or bowled for "
        "that team that season. Roles (wicketkeeper / batter / all-rounder / bowler) are "
        "auto-inferred from career stats since the raw data has no role column — review and "
        "fix the table below before relying on it, since a wrong role call (e.g. no "
        "wicketkeeper detected) will make the optimisers infeasible."
    )

    with st.expander("Dataset overview"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Matches", len(matches))
        col2.metric("Deliveries", len(deliveries))
        col3.metric("Seasons", matches["season"].nunique())
        col4.metric("Teams", pd.concat([matches["team1"], matches["team2"]]).nunique())

    teams = sqb.list_teams(matches)
    c1, c2 = st.columns(2)
    team = c1.selectbox("Team", teams, index=teams.index(st.session_state["squad_team"]))
    seasons = sqb.list_seasons_for_team(matches, team)
    if st.session_state["squad_season"] in seasons:
        season_index = seasons.index(st.session_state["squad_season"])
    else:
        season_index = 0
    season = c2.selectbox("Season", seasons, index=season_index)

    if st.button("Build squad", type="primary"):
        st.session_state["squad_team"] = team
        st.session_state["squad_season"] = season
        st.session_state["role_table"] = build_squad_and_roles(deliveries, matches, team, season)
        st.session_state.pop("my_xi_result", None)
        st.rerun()

    role_table = st.session_state["role_table"]
    st.subheader(f"{st.session_state['squad_team']} — {st.session_state['squad_season']} squad ({len(role_table)} players)")

    warnings = sqb.validate_squad_roles(role_table)
    for w in warnings:
        st.warning(w)

    st.caption("Edit any cell below to correct an auto-inferred role, bowling flag, or top-order flag.")
    edited = st.data_editor(
        role_table,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "role": st.column_config.SelectboxColumn(
                options=["Wicketkeeper", "Batter", "All-rounder", "Bowler"]
            ),
            "can_bowl": st.column_config.CheckboxColumn(),
            "top_order_eligible": st.column_config.CheckboxColumn(),
        },
        key="role_table_editor",
    )
    if not edited.equals(role_table):
        st.session_state["role_table"] = edited
        st.session_state.pop("my_xi_result", None)
        st.rerun()


def render_statistics(deliveries, role_table):
    st.header("Statistics")
    squad = role_table["player"].tolist()

    positions = fe.compute_batting_positions(deliveries)
    bat_overall = fe.batting_overall(deliveries, squad).merge(
        role_table[["player", "role"]], on="player", how="left"
    )
    bowl_overall = fe.bowling_overall(deliveries, squad).merge(
        role_table[["player", "role"]], on="player", how="left"
    )

    st.subheader("Batting leaderboard")
    st.dataframe(bat_overall.sort_values("runs", ascending=False).round(1), width="stretch")

    st.subheader("Bowling leaderboard")
    st.dataframe(bowl_overall.sort_values("wickets", ascending=False).round(1), width="stretch")

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


def render_build_your_xi(deliveries: pd.DataFrame, role_table: pd.DataFrame, squad_label: str):
    st.header("Build Your XI")
    st.caption(
        "Pick 11 players, drag them into your batting order, and see how your team scores "
        "against the same objective the Integer Programming model maximises. Beat the machine!"
    )

    s = current_settings()
    scores = compute_scores(deliveries, role_table, s["opponent"])
    lam = s["lam"]

    squad = role_table["player"].tolist()
    selected = st.multiselect("Pick exactly 11 players", squad, max_selections=11)

    if len(selected) != 11:
        st.info(f"Selected {len(selected)}/11. Pick {11 - len(selected)} more." if len(selected) < 11 else "")
        return

    st.subheader("Drag to set your batting order (top = position 1)")
    ordered = sort_items(selected, direction="vertical", key=f"sortable_{'_'.join(sorted(selected))}")

    role = role_table.set_index("player")["role"]
    can_bowl = role_table.set_index("player")["can_bowl"]
    top_order_eligible = role_table.set_index("player")["top_order_eligible"]

    n_wk = sum(role.get(p) == "Wicketkeeper" for p in ordered)
    n_bat = sum(role.get(p) == "Batter" for p in ordered)
    n_ar = sum(role.get(p) == "All-rounder" for p in ordered)
    n_bowl_options = sum(can_bowl.get(p, 0) == 1 for p in ordered)
    top3 = ordered[:3]
    top3_ok = all(top_order_eligible.get(p, 0) == 1 for p in top3)

    checks = [
        (n_wk >= 1, f"Wicketkeeper: {n_wk} (need >=1)"),
        (n_bat >= s["min_batters"], f"Specialist batters: {n_bat} (need >={s['min_batters']})"),
        (n_ar >= s["min_allrounders"], f"All-rounders: {n_ar} (need >={s['min_allrounders']})"),
        (n_bowl_options >= s["min_bowling_options"], f"Bowling options: {n_bowl_options} (need >={s['min_bowling_options']})"),
        (top3_ok, f"Positions 1-3 are top-order eligible: {', '.join(top3)}"),
    ]

    st.subheader("Constraint check")
    for ok, label in checks:
        (st.success if ok else st.error)(label)
    all_ok = all(ok for ok, _ in checks)

    r = scores["R_ik"].set_index(["player", "batting_position"])["R_ik"]
    batting_score = sum(r.get((p, k + 1), 0.0) for k, p in enumerate(ordered))
    bowling_score = sum(scores["O_i"].get(p, 0.0) for p in ordered)
    total_score = batting_score + lam * bowling_score

    st.subheader("Your team's batting order and score")
    order_df = pd.DataFrame(
        {"Position": range(1, 12), "Player": ordered, "Role": [role.get(p, "-") for p in ordered]}
    )
    st.dataframe(order_df, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Batting score", f"{batting_score:.1f}")
    c2.metric("Bowling score (x lambda)", f"{bowling_score:.1f}")
    c3.metric("Total score", f"{total_score:.1f}")

    if all_ok:
        batting_order = {k + 1: p for k, p in enumerate(ordered)}
        st.session_state["my_xi_result"] = {
            "squad_label": squad_label,
            "xi": ordered,
            "batting_order": batting_order,
            "batting_score": batting_score,
            "bowling_score": bowling_score,
            "total_score": total_score,
            "lambda_used": lam,
        }
        st.success("Valid XI — this is now shown on the **Compare** page. Submit below to also add it to the leaderboard.")
    else:
        st.warning("Fix the failing constraints above to use this XI in the Compare page or submit it.")

    st.subheader("Submit to leaderboard")
    team_name = st.text_input("Team name", max_chars=40)
    entrant = st.text_input("Your name (optional)", max_chars=40)
    if st.button("Submit to Leaderboard", disabled=not all_ok or not team_name):
        batting_order = {k + 1: p for k, p in enumerate(ordered)}
        lb.record_entry(
            team_name=team_name,
            entrant=entrant,
            squad_label=squad_label,
            batting_order=batting_order,
            lambda_used=lam,
            batting_score=batting_score,
            bowling_score=bowling_score,
        )
        st.success(f"Submitted! Total score: {total_score:.1f}. Check the **Leaderboard** page.")


def render_leaderboard(squad_label: str):
    st.header("Leaderboard")

    board = lb.load_leaderboard()
    if board.empty:
        st.info("No entries yet — build a team on **Build Your XI** and submit it.")
        return

    squad_options = ["All squads"] + sorted(board["squad_label"].unique())
    default_index = squad_options.index(squad_label) if squad_label in squad_options else 0
    chosen = st.selectbox("Filter by squad", squad_options, index=default_index)
    view = board if chosen == "All squads" else board[board["squad_label"] == chosen]
    view = view.sort_values("total_score", ascending=False).reset_index(drop=True)
    view.insert(0, "Rank", view.index + 1)

    st.dataframe(
        view[["Rank", "team_name", "entrant", "squad_label", "total_score", "batting_score", "bowling_score", "lambda_used", "timestamp"]],
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
