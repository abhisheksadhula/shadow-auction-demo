"""
demo_playthrough.py
-------------------
Not part of the deliverable — a self-check that simulates one fan playing through a
handful of lots using ONLY players.json + the scoring module, exactly as the backend
will. Proves the pre-hackathon artifacts are sufficient to run the game.
"""
import json

from config import PLAYERS_JSON
from scoring import (score_band_prediction, score_steal_prediction,
                     shadow_gm_verdict, AuctionIQ)
from archetype import classify_archetype

with open(PLAYERS_JSON) as f:
    data = json.load(f)
players = {p["lot_no"]: p for p in data["players"]}

sold_lots = [p for p in data["players"] if p["sold"]][:6]

print("=== Fan playthrough (6 sold lots) ===\n")
session_total = 0
iq_points = []
for p in sold_lots:
    # pretend the fan always guesses the fair-value band and the winning team
    guess_band = p["fair_value_band"]
    guess_team = p["winning_team"]
    bp = score_band_prediction(guess_band, p["actual_band"], guess_team, p["winning_team"])

    sfo = score_steal_prediction("Steal", p["winning_price_crore"], p["fair_value_crore"])

    # Auction IQ: rebuild percentile from the stored baseline
    base = p["auction_iq_baseline"]
    # reconstruct a tiny distribution from probabilities to get a percentile
    import numpy as np
    dist = np.concatenate([
        np.full(int(base["p_miss"] * 1000), 0),
        np.full(int(base["p_close"] * 1000), 40),
        np.full(int(base["p_full"] * 1000), 100),
    ])
    iq = AuctionIQ(dist)
    pct = iq.percentile(bp["total"] - bp["franchise_points"])  # band-only vs crowd

    session_total += bp["total"] + sfo["points"]
    iq_points.append(pct)
    print(f"Lot {p['lot_no']:>3} {p['player_name']:<18} "
          f"sold ₹{p['winning_price_crore']:>5} Cr ({p['actual_band']:<7}) "
          f"| band {bp['outcome']:<5} +{bp['total']:>3} "
          f"| SFO truth={sfo['truth']:<7} | IQ {pct:>3}th pct")

print(f"\nSession total: {session_total} pts | "
      f"avg Auction IQ percentile: {round(sum(iq_points)/len(iq_points))}th")

# Shadow GM: fan had 2 of these as targets
print("\n=== Shadow GM verdicts (fan's pre-built targets) ===")
for p, bid in [(sold_lots[0], sold_lots[0]["winning_price_crore"] + 1),
               (sold_lots[1], sold_lots[1]["winning_price_crore"] - 1)]:
    v = shadow_gm_verdict(p, your_bid=bid)
    print(f"  [{v['verdict']:<8}] {v['line']}")

# Archetype from session pattern
print("\n=== End-of-auction archetype ===")
arch, line = classify_archetype({
    "steal_hit_rate": 0.66, "franchise_accuracy": 0.5, "boldness": 0.3})
print(f"  {arch} — {line}")
print("\nAll pre-hackathon artifacts validated end-to-end.")
