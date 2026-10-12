# Optimal IPL Playing XI Selection and Batting Order Optimisation

Operations Research course project (Team A4, School of AI, Amrita Vishwa Vidyapeetham). Selects
a Playing XI and batting order from a candidate squad using two methods:

- **Integer Programming (IP)** — maximises a weighted performance score.
- **Goal Programming (GP)** — minimises shortfall against targets for batting strength, bowling
  depth, top-order quality, and recent form.

Full mathematical formulation: `OR (1).pdf` (first-review slides) and
`ipl_optimisation_project_explanation.pdf` (detailed spec).

## Setup

```bash
pip install -r requirements.txt
```

Raw data (`archive/matches.csv`, `archive/deliveries.csv`) is the Cricsheet/Kaggle-style IPL
ball-by-ball dataset — already included.

## Project layout

- `src/data_cleaning.py` — load + clean the raw dataset (legal deliveries, dismissals, phases).
- `src/feature_engineering.py` — batting/bowling metrics, batting position inference, phases,
  recent form, opponent match-up.
- `src/scoring.py` — 0–100 normalisation and the composite scores `R_ik`, `O_i`, `F_i`, `M_i`
  the models consume, with credibility shrinkage toward a league baseline for sparse samples.
- `src/squad_builder.py` — builds a squad for any team+season from the raw data and
  auto-infers each player's role (wicketkeeper/batter/all-rounder/bowler), bowling capability,
  and top-order eligibility from career-wide stats (the dataset has no role column). Review the
  result in the app's **Squad Selector** page before relying on it — the heuristic can be wrong
  (e.g. a backup/emergency wicketkeeper getting flagged from a handful of career stumpings).
- `src/ip_model.py` — PuLP Integer Programming model (spec §8).
- `src/gp_model.py` — PuLP Goal Programming model (spec §9). The four goal targets are rescaled
  to a 0–200 "quality index" (roughly: average per-goal 0–100 score × 2) so they read like a
  plausible IPL score (e.g. target ≈170) instead of an opaque summed total — a presentation
  rescaling of the same normalised-score model, not a predicted run count.
- `src/compare.py` — IP vs GP comparison, sensitivity-analysis sweeps, and `ip_style_score`
  (evaluates any XI — IP's, GP's, or a manually-built one — on the IP objective so they're all
  directly comparable regardless of which method, if any, produced them).
- `src/leaderboard.py` — persistent (CSV-backed) leaderboard for manually-built XIs.
- `app/streamlit_app.py` — interactive dashboard (spec §13, extended — see below).
- `role_table.csv` — a worked example of a manually curated role table (Chennai Super Kings,
  2024 season, 19 players) in the schema `squad_builder.infer_roles` also produces. The app
  itself builds squads live via the Squad Selector page rather than reading this file.

## Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

Pages, in sidebar order:

1. **Compare** — the main page. Shows three (or four) XIs on one directly-comparable score
   (batting quality + λ × bowling quality), with no "Run" button to click first:
   - 🌍 **All-Time Best XI** — the best XI from *any* team/season, drawn from every player with
     a meaningful career sample (≥300 balls faced or ≥180 legal balls bowled, ~370 players
     league-wide). Built once and cached (~20s the first time this page is opened, instant after).
   - 🏏 **Squad Best (IP)** / **Squad Best (GP)** — the best XI from the currently selected squad
     under each method, auto-solved fresh (sub-second) whenever you open the page or change a
     setting.
   - 🙋 **Your XI** — appears once you've built a valid one on **Build Your XI**, with a delta
     against the squad's best.
   Settings (role minimums, λ, GP targets/priorities) live in a collapsed expander at the top of
   this page rather than a separate page. Each score has an expandable "XI details" section below
   with the full batting order (and, for GP, the goal-achievement breakdown).
2. **Squad Selector** — pick any team + season; the squad and its auto-inferred role table are
   built live (dataset overview is a small expander here too). Edit the table to fix a
   misclassified role before using it anywhere else.
3. **Build Your XI** — pick 11 players, drag them into a batting order, see a live constraint
   check and score (same objective IP maximises) — feeds straight into the Compare page without
   needing to submit anywhere — and optionally submit to the leaderboard.
4. **Statistics** — leaderboards and per-player batting/bowling/position/phase breakdowns.
5. **Leaderboard** — manually-submitted XIs ranked by score, filterable by squad.

## Running the pipeline from Python

```python
from src import data_cleaning as dc, squad_builder as sb, scoring as sc, ip_model as ipm, gp_model as gpm

matches, deliveries = dc.build_clean_dataset()
squad = sb.build_squad(matches, deliveries, "Chennai Super Kings", "2024")
role = sb.infer_roles(deliveries, squad)  # or pd.read_csv("role_table.csv") for the worked example
scores = sc.build_all_scores(deliveries, role)

ip_result = ipm.solve_ip(role, scores["R_ik"], scores["O_i"], lam=1.0)

targets = {"T_B": 170, "T_O": 150, "T_T": 160, "T_F": 140}  # 0-200 quality index, see src/gp_model.py
priorities = {"p_B": 5, "p_O": 4, "p_T": 3, "p_F": 2}
gp_result = gpm.solve_gp(role, scores["R_ik"], scores["O_i"], scores["F_i"], targets, priorities)
```

## Notes

- `pulp` is pinned to `2.9.0`: the current PyPI `pulp` release (4.x) is an unrelated rewrite with
  a different, Rust-backed API that breaks the classic `LpVariable(cat=...)` / `lpSum` /
  `PULP_CBC_CMD` interface this project (and most PuLP documentation/tutorials) relies on.
- `data/processed/` (cleaned CSVs) is regenerated by `python src/data_cleaning.py` and is
  gitignored — ~47 MB of derived data, not worth versioning.
- `data/leaderboard.csv` is created on first submission from the **Build Your XI** page and is
  *not* gitignored — delete it (or just don't commit it) to reset the leaderboard for a demo.
- Role inference is a heuristic over ball-by-ball data (no official source for wicketkeeper /
  all-rounder status exists in this dataset) — always sanity-check the auto-built table on
  **Squad Selector** before trusting a squad's optimisation result.
- The all-time pool (`squad_builder.build_alltime_pool`) deliberately excludes players below the
  career-sample thresholds (§`src/squad_builder.py`) — without that filter, one-ball careers would
  collapse to the league-baseline score under shrinkage and clutter the candidate pool with
  meaningless near-ties, not add real options.
