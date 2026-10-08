"""Load and clean the raw IPL ball-by-ball dataset (archive/matches.csv, archive/deliveries.csv).

Conventions (confirmed against the dataset, not assumed):
- `over` is 0-indexed (0..19); `ball` can exceed 6 because wides/no-balls add extra balls.
- `extras_type` in {wides, noballs} marks illegal deliveries; blank/NaN, legbyes, byes, penalty are legal.
- A ball counts toward "balls faced" (strike-rate denominator) unless it was a wide.
- A ball counts toward "legal deliveries" (over/economy-rate denominator) unless it was a wide or no-ball.
- Bowler wickets exclude run out, retired hurt, retired out, obstructing the field.
- Batting dismissals (average denominator) exclude retired hurt (not out).
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "archive"

NOT_BOWLER_WICKET = {"run out", "retired hurt", "retired out", "obstructing the field"}
NOT_BATTING_DISMISSAL = {"retired hurt"}


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(raw_dir / "matches.csv")
    deliveries = pd.read_csv(raw_dir / "deliveries.csv")
    return matches, deliveries


def _standardise_names(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    d = deliveries.drop_duplicates().copy()

    d = _standardise_names(d, ["batter", "bowler", "non_striker", "player_dismissed"])

    d["extras_type"] = d["extras_type"].fillna("")
    d["is_wide"] = d["extras_type"] == "wides"
    d["is_noball"] = d["extras_type"] == "noballs"
    d["is_legal_delivery"] = ~(d["is_wide"] | d["is_noball"])
    d["is_ball_faced"] = ~d["is_wide"]

    d["dismissal_kind"] = d["dismissal_kind"].fillna("")
    d["is_batting_dismissal"] = (d["is_wicket"] == 1) & (
        ~d["dismissal_kind"].isin(NOT_BATTING_DISMISSAL)
    )
    d["is_bowler_wicket"] = (d["is_wicket"] == 1) & (
        ~d["dismissal_kind"].isin(NOT_BOWLER_WICKET)
    )

    # Phase per over: Powerplay 0-5, Middle 6-14, Death 15-19 (0-indexed overs).
    def phase(over: int) -> str:
        if over <= 5:
            return "powerplay"
        if over <= 14:
            return "middle"
        return "death"

    d["phase"] = d["over"].apply(phase)
    return d


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.drop_duplicates().copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.sort_values("date").reset_index(drop=True)
    return m


def build_clean_dataset(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches, deliveries = load_raw(raw_dir)
    matches = clean_matches(matches)
    deliveries = clean_deliveries(deliveries)
    # Attach match context (season, venue, date) needed for recent-form / venue filtering.
    deliveries = deliveries.merge(
        matches[["id", "season", "date", "venue"]],
        left_on="match_id",
        right_on="id",
        how="left",
    ).drop(columns="id")
    return matches, deliveries


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_clean, deliveries_clean = build_clean_dataset()
    deliveries_clean.to_csv(out_dir / "deliveries_clean.csv", index=False)
    matches_clean.to_csv(out_dir / "matches_clean.csv", index=False)
    print(f"matches: {len(matches_clean)} rows, deliveries: {len(deliveries_clean)} rows")
    print(f"written to {out_dir}")
