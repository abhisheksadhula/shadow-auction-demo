"""
archetype.py
------------
Step 6: GM archetype + Team Strength Score.

GM archetype is a transparent rule tree over a user's *prediction pattern* across the
session — not a black box. Five archetypes mapped to the deck:
  Moneyball Scout       — strong on spotting steals / undervalued players
  Aggressive Bidder     — leans toward high bands / overpay calls
  Future Talent Hunter  — backs uncapped / domestic players
  Franchise Builder     — high franchise-prediction accuracy, balanced squad
  Risk-Taking Maverick  — lots of bold (non-obvious) calls, high variance

Team Strength Score grades a completed shadow squad on role balance, budget
efficiency, and overseas ratio — used live during the auction and on the recap.
"""
import numpy as np


# ---- GM archetype ------------------------------------------------------------
def classify_archetype(stats):
    """stats: dict summarizing a user's session. Expected keys:
        steal_hit_rate     (0-1)  : fraction of steals correctly called
        overpay_lean       (0-1)  : fraction of band guesses in the top two bands
        uncapped_backing   (0-1)  : fraction of squad/targets that were uncapped
        franchise_accuracy (0-1)  : fraction of winning-team predictions correct
        boldness           (0-1)  : fraction of guesses that were NOT the fair-value band
       Returns (archetype_name, one_line_descriptor).
    """
    s = {k: float(stats.get(k, 0.0)) for k in
         ("steal_hit_rate", "overpay_lean", "uncapped_backing",
          "franchise_accuracy", "boldness")}

    # priority-ordered rules; first strong signal wins
    if s["steal_hit_rate"] >= 0.6:
        return "Moneyball Scout", "You find value others miss."
    if s["uncapped_backing"] >= 0.5:
        return "Future Talent Hunter", "You back tomorrow's stars early."
    if s["franchise_accuracy"] >= 0.55:
        return "Franchise Builder", "You read the room better than most."
    if s["overpay_lean"] >= 0.5:
        return "Aggressive Bidder", "You go big and chase the marquee names."
    if s["boldness"] >= 0.6:
        return "Risk-Taking Maverick", "You trust your gut over the obvious call."
    # fallback: balanced
    return "Franchise Builder", "A steady, balanced auction mind."


# ---- Team Strength Score -----------------------------------------------------
IDEAL_ROLE_MIX = {"Batter": 0.40, "Bowler": 0.40, "All-Rounder": 0.15, "Wicketkeeper": 0.05}
MAX_OVERSEAS = 8


def team_strength(squad_rows, purse=100.0):
    """squad_rows: list of dict-like players the user drafted (with role, overseas,
                   fair_value_crore, winning_price_crore).
       purse:      total virtual budget (crore).
       Returns dict with 0-100 score and component breakdown.
    """
    if not squad_rows:
        return {"score": 0, "role_balance": 0, "budget_efficiency": 0,
                "overseas_ok": True, "n": 0}

    n = len(squad_rows)
    # role balance: 1 - normalized distance from ideal mix
    counts = {r: 0 for r in IDEAL_ROLE_MIX}
    for p in squad_rows:
        r = p.get("role", "Unknown")
        if r in counts:
            counts[r] += 1
    actual_mix = {r: counts[r] / n for r in counts}
    dist = sum(abs(actual_mix[r] - IDEAL_ROLE_MIX[r]) for r in IDEAL_ROLE_MIX) / 2
    role_balance = max(0.0, 1.0 - dist)

    # budget efficiency: total fair value acquired per crore notionally spent
    spent = sum((p.get("winning_price_crore") or p.get("fair_value_crore") or 0)
                for p in squad_rows)
    value = sum((p.get("fair_value_crore") or 0) for p in squad_rows)
    budget_efficiency = float(np.clip(value / spent, 0, 1.5) / 1.5) if spent else 0.0

    overseas_n = sum(1 for p in squad_rows if p.get("overseas"))
    overseas_ok = overseas_n <= MAX_OVERSEAS
    overseas_penalty = 0.0 if overseas_ok else 0.15

    score = (0.5 * role_balance + 0.5 * budget_efficiency) - overseas_penalty
    score = int(round(float(np.clip(score, 0, 1)) * 100))

    return {"score": score,
            "role_balance": round(role_balance, 2),
            "budget_efficiency": round(budget_efficiency, 2),
            "overseas_count": overseas_n,
            "overseas_ok": overseas_ok,
            "n": n}


if __name__ == "__main__":
    print("Archetypes:")
    print(" ", classify_archetype({"steal_hit_rate": 0.7}))
    print(" ", classify_archetype({"uncapped_backing": 0.6}))
    print(" ", classify_archetype({"franchise_accuracy": 0.6}))
    print(" ", classify_archetype({"overpay_lean": 0.6}))
    print(" ", classify_archetype({"boldness": 0.7}))
    print(" ", classify_archetype({}))

    print("\nTeam strength (balanced 4-player squad):")
    squad = [
        {"role":"Batter","overseas":False,"fair_value_crore":8,"winning_price_crore":7},
        {"role":"Bowler","overseas":True,"fair_value_crore":6,"winning_price_crore":6},
        {"role":"All-Rounder","overseas":False,"fair_value_crore":10,"winning_price_crore":9},
        {"role":"Wicketkeeper","overseas":False,"fair_value_crore":5,"winning_price_crore":6},
    ]
    print(" ", team_strength(squad))
