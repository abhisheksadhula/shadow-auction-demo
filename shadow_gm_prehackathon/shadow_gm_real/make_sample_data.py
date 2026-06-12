"""
make_sample_data.py
-------------------
Generates a realistic SAMPLE auction CSV matching the documented 12-field schema
of `ipl_2025_auction_players_dummy_dataset.csv` (577 rows, 12 fields).

This exists ONLY so the rest of the pipeline runs end-to-end before you have the
real file. On hackathon day, drop the real CSV into shadow_gm/data/ and point
clean_data.py at it — DO NOT run this script.

Documented schema (from DataSets.txt):
  lot_no, player_name, country, winning_team, previous_team, source_team_raw,
  base_price_crore, base_price_label, winning_price_crore, winning_price_label,
  auction_result (Won/Lost), source_status (Sold/Unsold)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

TEAMS = ["CSK", "MI", "RCB", "KKR", "SRH", "DC", "RR", "PBKS", "GT", "LSG"]
COUNTRIES = ["India", "Australia", "England", "South Africa", "New Zealand",
             "West Indies", "Sri Lanka", "Afghanistan", "Bangladesh"]
# weight India heavily, like a real IPL pool
COUNTRY_W = np.array([0.62, 0.07, 0.07, 0.06, 0.05, 0.05, 0.03, 0.03, 0.02])

ROLES = ["Batter", "Bowler", "All-Rounder", "Wicketkeeper"]
ROLE_W = np.array([0.34, 0.34, 0.22, 0.10])

# base price tiers used in real IPL auctions (in crore)
BASE_TIERS = [0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
BASE_TIER_W = np.array([0.30, 0.22, 0.16, 0.14, 0.10, 0.08])

FIRST = ["Aarav","Rohit","Virat","Hardik","Rishabh","Shubman","Ishan","Suryakumar",
         "Yuzvendra","Jasprit","Mohammed","Ravindra","Axar","Kuldeep","Shreyas",
         "Tilak","Sanju","Arshdeep","Mukesh","Avesh","Glenn","Travis","Jos","Mitchell",
         "Pat","Kagiso","Rashid","Trent","Marcus","David","Faf","Quinton","Wanindu",
         "Nicholas","Liam","Sam","Jofra","Gerald","Heinrich","Matheesha"]
LAST = ["Sharma","Kohli","Pandya","Pant","Gill","Kishan","Yadav","Chahal","Bumrah",
         "Siraj","Jadeja","Patel","Iyer","Varma","Samson","Singh","Kumar","Maxwell",
         "Head","Buttler","Starc","Cummins","Rabada","Boult","Stoinis","Warner",
         "Plessis","Klaasen","Hasaranga","Pooran","Livingstone","Curran","Archer",
         "Coetzee","Brevis","Green","Conway","Ferguson","Theekshana","Pathirana"]


def make_name(used):
    while True:
        n = f"{RNG.choice(FIRST)} {RNG.choice(LAST)}"
        if n not in used:
            used.add(n)
            return n


def generate(n_rows=577, seed=42):
    rng = np.random.default_rng(seed)
    globals()["RNG"] = rng
    used_names = set()
    rows = []
    for i in range(1, n_rows + 1):
        country = rng.choice(COUNTRIES, p=COUNTRY_W)
        overseas = country != "India"
        role = rng.choice(ROLES, p=ROLE_W)
        capped = rng.random() < 0.45  # ~45% capped
        base = float(rng.choice(BASE_TIERS, p=BASE_TIER_W))

        # ---- realistic "true value" generative model (hidden ground truth) ----
        # capped, overseas, all-rounders and keepers command premiums; noise added.
        mult = 1.0
        mult *= 4.2 if capped else 1.3
        mult *= 1.45 if overseas else 1.0
        mult *= {"All-Rounder": 1.5, "Wicketkeeper": 1.35,
                 "Batter": 1.15, "Bowler": 1.1}[role]
        # star factor: a few players blow up
        star = rng.random()
        if star > 0.93:
            mult *= rng.uniform(3.0, 6.0)
        elif star > 0.75:
            mult *= rng.uniform(1.5, 2.5)
        mult *= rng.lognormal(0, 0.35)  # noise

        winning = round(base * mult, 2)
        winning = float(np.clip(winning, base, 27.0))

        # ~22% go unsold
        unsold = rng.random() < 0.22
        prev = rng.choice(TEAMS)
        if unsold:
            winning_team = ""
            winning_price = np.nan
            source_status = "Unsold"
            auction_result = "Lost"
            winning_price_label = ""
        else:
            winning_team = rng.choice(TEAMS)
            winning_price = winning
            source_status = "Sold"
            auction_result = "Won"
            winning_price_label = f"₹{winning_price} Cr"

        rows.append({
            "lot_no": i,
            "player_name": make_name(used_names),
            "country": country,
            "winning_team": winning_team,
            "previous_team": prev,
            "source_team_raw": prev,
            "base_price_crore": base,
            "base_price_label": f"₹{base} Cr",
            "winning_price_crore": winning_price,
            "winning_price_label": winning_price_label,
            "auction_result": auction_result,
            "source_status": source_status,
            # NOTE: role/capped are NOT in the documented schema. We store them in
            # hidden helper columns the cleaner will re-derive heuristically when the
            # real file lacks them. Prefixed with _sample_ so it's obvious they are
            # synthetic and must be ignored / re-derived for the real dataset.
            "_sample_role": role,
            "_sample_capped": capped,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    out = "data/ipl_2025_auction_players_SAMPLE.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    print(f"Sold: {(df.source_status=='Sold').sum()}  Unsold: {(df.source_status=='Unsold').sum()}")
    print(f"Overseas: {(df.country!='India').sum()}  Indian: {(df.country=='India').sum()}")
    print("\nSample rows:")
    print(df.drop(columns=['_sample_role','_sample_capped']).head(6).to_string(index=False))
