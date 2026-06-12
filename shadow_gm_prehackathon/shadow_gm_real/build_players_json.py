"""
build_players_json.py
---------------------
Step 7: the master pre-hackathon script. Run this once; hand output/players.json to
the backend engineer at hour zero. It is the single artifact the whole demo runs on.

Pipeline:
  clean_data -> fair_value -> per-lot steal truth + Auction IQ baseline -> JSON

Each lot in players.json carries everything the backend needs to score a fan live
WITHOUT recomputing any model: the answer-key band, the fair value, the steal truth,
and a compact Auction IQ baseline (the crowd score distribution, summarized).

Run:  python3 shadow_gm/build_players_json.py
"""
import json
import os

import numpy as np

from config import CSV_PATH, PLAYERS_JSON, OUTPUT_DIR
from clean_data import load_and_clean, BAND_LABELS
from fair_value import fit_fair_value, explain
from scoring import classify_steal, AuctionIQ


def _iq_baseline_for_lot(fair_band, actual_band):
    """Pre-compute a compact summary of the simulated-crowd score distribution.

    Storing all 2000 sim scores per lot would bloat the JSON. Instead we store the
    crowd's mean score and the three score buckets (0 / 40 / 100) as probabilities,
    which is enough for the backend to compute a percentile cheaply at runtime.
    """
    iq = AuctionIQ.from_simulated_crowd(fair_band, actual_band, n=2000, seed=7)
    dist = iq.dist
    if len(dist) == 0 or actual_band is None:
        return None
    return {
        "crowd_mean": round(float(dist.mean()), 1),
        "p_full": round(float((dist == 100).mean()), 3),   # got exact band
        "p_close": round(float((dist == 40).mean()), 3),   # adjacent band
        "p_miss": round(float((dist == 0).mean()), 3),     # missed
    }


def build():
    df = load_and_clean(CSV_PATH)
    model, feats, scored = fit_fair_value(df)

    records = []
    for _, r in scored.iterrows():
        win_price = (None if r["winning_price_crore"] is None
                     or (isinstance(r["winning_price_crore"], float)
                         and np.isnan(r["winning_price_crore"]))
                     else round(float(r["winning_price_crore"]), 2))
        steal_truth = classify_steal(win_price, r["fair_value_crore"]) if win_price else None

        rec = {
            "lot_no": int(r["lot_no"]),
            "player_name": r["player_name"],
            "country": r["country"],
            "overseas": bool(r["overseas"]),
            "capped": bool(r["capped"]),
            "role": r["role"],
            "previous_team": r["previous_team"],
            "base_price_crore": round(float(r["base_price_crore"]), 2),
            # --- answer keys & model outputs the backend scores against ---
            "winning_team": r["winning_team"] if r["sold"] else None,
            "winning_price_crore": win_price,
            "actual_band": r["actual_band"],          # band prediction answer key
            "sold": bool(r["sold"]),
            "fair_value_crore": round(float(r["fair_value_crore"]), 2),
            "fair_value_band": r["fair_value_band"],
            "steal_truth": steal_truth,               # Steal/Fair/Overpay answer key
            "auction_iq_baseline": _iq_baseline_for_lot(
                r["fair_value_band"], r["actual_band"]),
        }
        records.append(rec)

    payload = {
        "meta": {
            "source": os.path.basename(CSV_PATH),
            "n_players": len(records),
            "bands": BAND_LABELS,
            "model": "log-linear fair value (base, capped, overseas, role)",
            "scoring": {
                "exact_band": 100, "adjacent_band": 40, "franchise_bonus": 50,
                "steal_fair_overpay": 60,
                "steal_ratio": 0.80, "overpay_ratio": 1.25,
            },
            "note": "auction_iq_baseline summarizes a 2000-fan simulated crowd per lot; "
                    "backend computes percentile from p_full/p_close/p_miss at runtime.",
        },
        "players": records,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PLAYERS_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    return payload, model, feats


if __name__ == "__main__":
    payload, model, feats = build()
    print(explain(model, feats))
    print(f"\nWrote {payload['meta']['n_players']} players -> {PLAYERS_JSON}")
    sold = [p for p in payload["players"] if p["sold"]]
    steals = [p for p in sold if p["steal_truth"] == "Steal"]
    overpays = [p for p in sold if p["steal_truth"] == "Overpay"]
    print(f"Sold {len(sold)} | Steals {len(steals)} | Overpays {len(overpays)} | "
          f"Fair {len(sold)-len(steals)-len(overpays)}")
    print("\nExample record:")
    print(json.dumps(sold[2], indent=2))
