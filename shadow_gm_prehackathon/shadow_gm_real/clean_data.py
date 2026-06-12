"""
clean_data.py
-------------
Step 1 of the Shadow GM pre-hackathon pipeline.

Loads the auction CSV (real or sample), standardizes it, and enriches it with the
derived fields the scoring engine needs:
  - base_price_crore / winning_price_crore as clean floats
  - overseas (bool)        : country != India
  - capped (bool)          : from data if present, else heuristic
  - role (str)             : from data if present, else heuristic ("Unknown" allowed)
  - actual_band (str|None) : which price band the player actually sold into
                             (None when unsold) -- this is the ANSWER KEY for the
                             prediction game.

The price bands are the four the UI shows and the brief asked us to keep coarse:
  "Under 5", "5-10", "10-15", "15+"   (crore)

On hackathon day: set CSV_PATH to the real file. If the real file uses slightly
different column names, adjust COLUMN_ALIASES below — nothing else needs to change.
"""
import numpy as np
import pandas as pd

# ----- price bands (single source of truth, imported by other modules) --------
BAND_EDGES = [0, 5, 10, 15, np.inf]
BAND_LABELS = ["Under 5", "5-10", "10-15", "15+"]


def price_to_band(price):
    """Map a crore price to one of the four UI bands. Returns None for NaN/unsold."""
    if price is None or (isinstance(price, float) and np.isnan(price)):
        return None
    for i in range(len(BAND_LABELS)):
        if BAND_EDGES[i] <= price < BAND_EDGES[i + 1]:
            return BAND_LABELS[i]
    return BAND_LABELS[-1]


# Map real-file column names -> our canonical names. Extend if the real CSV differs.
COLUMN_ALIASES = {
    "lot_no": "lot_no",
    "player_name": "player_name",
    "country": "country",
    "winning_team": "winning_team",
    "previous_team": "previous_team",
    "source_team_raw": "source_team_raw",
    "base_price_crore": "base_price_crore",
    "base_price_label": "base_price_label",
    "winning_price_crore": "winning_price_crore",
    "winning_price_label": "winning_price_label",
    "auction_result": "auction_result",
    "source_status": "source_status",
}

ALLROUNDER_HINTS = ("all", "rounder", "ar")
KEEPER_HINTS = ("keeper", "wk", "wicket")


def _coerce_price(series):
    """Strip currency symbols/labels and coerce to float crore."""
    if series.dtype.kind in "if":
        return series.astype(float)
    cleaned = (series.astype(str)
               .str.replace("₹", "", regex=False)
               .str.replace("Cr", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def _derive_capped(df):
    """Use a real 'capped' column if present; else heuristic.

    Heuristic: a player who attracted a winning price comfortably above base, or
    who has a prior IPL team, is more likely capped. This is a stand-in ONLY for
    when the real dataset lacks the field. Clearly approximate; documented as such.
    """
    for col in df.columns:
        if col.lower() in ("capped", "is_capped", "_sample_capped"):
            return df[col].astype(bool)
    has_prior = df["previous_team"].astype(str).str.len() > 0
    price_signal = (df["winning_price_crore"].fillna(0) >= 2.0)
    return (has_prior & price_signal)


def _derive_role(df):
    """Use a real role column if present; else 'Unknown'.

    We do NOT fabricate roles for the real dataset — an honest 'Unknown' is better
    than a wrong guess, and the fair-value model degrades gracefully without it.
    """
    for col in df.columns:
        if col.lower() in ("role", "player_role", "_sample_role"):
            return df[col].astype(str)
    return pd.Series(["Unknown"] * len(df), index=df.index)


def load_and_clean(csv_path):
    raw = pd.read_csv(csv_path)
    raw = raw.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in raw.columns})

    df = pd.DataFrame()
    df["lot_no"] = pd.to_numeric(raw["lot_no"], errors="coerce").astype("Int64")
    df["player_name"] = raw["player_name"].astype(str).str.strip()
    df["country"] = raw["country"].astype(str).str.strip()
    df["winning_team"] = raw.get("winning_team", "").astype(str).str.strip()
    df["previous_team"] = raw.get("previous_team", "").astype(str).str.strip()
    df["base_price_crore"] = _coerce_price(raw["base_price_crore"])
    df["winning_price_crore"] = _coerce_price(raw["winning_price_crore"])
    df["source_status"] = raw["source_status"].astype(str).str.strip().str.title()

    # carry helper cols so derivation can see them
    for helper in ("_sample_role", "_sample_capped", "role", "capped"):
        if helper in raw.columns:
            df[helper] = raw[helper]

    # Real IPL file: overseas = blank/NaN country. Sample uses named countries.
    overseas_via_nan = df["country"].isna() | (df["country"].str.strip().str.lower().isin(["", "nan"]))
    if overseas_via_nan.sum() > 5:
        df["overseas"] = overseas_via_nan
    else:
        df["overseas"] = df["country"].str.lower() != "india"
    df["capped"] = _derive_capped(df)
    df["role"] = _derive_role(df)
    df["sold"] = df["source_status"].str.lower().eq("sold")

    # the answer key: the band the player actually sold into
    df["actual_band"] = df["winning_price_crore"].apply(price_to_band)

    # drop helper cols from the clean frame
    df = df.drop(columns=[c for c in ("_sample_role", "_sample_capped") if c in df.columns])

    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "shadow_gm/data/ipl_2025_auction_players_SAMPLE.csv"
    df = load_and_clean(path)
    print(f"Loaded & cleaned {len(df)} rows from {path}\n")
    print("Dtypes:")
    print(df.dtypes.to_string())
    print(f"\nSold {df.sold.sum()} | Unsold {(~df.sold).sum()} | "
          f"Overseas {df.overseas.sum()} | Capped {df.capped.sum()}")
    print("\nActual band distribution (sold players):")
    print(df[df.sold]["actual_band"].value_counts().reindex(BAND_LABELS).to_string())
    print("\nSample:")
    print(df[["lot_no","player_name","country","overseas","capped","role",
              "base_price_crore","winning_price_crore","actual_band"]].head(8).to_string(index=False))
