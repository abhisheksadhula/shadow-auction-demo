"""
wrapped.py — post-auction Wrapped stats.
Pure function: compute_wrapped(session, players) -> dict
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring import AuctionIQ
from archetype import classify_archetype, team_strength


def _iq_from_baseline(baseline, score):
    if not baseline:
        return 50
    n = 1000
    dist = np.concatenate([
        np.full(int(baseline["p_miss"] * n), 0),
        np.full(int(baseline["p_close"] * n), 40),
        np.full(int(baseline["p_full"] * n), 100),
    ])
    return AuctionIQ(dist).percentile(score) if len(dist) else 50


def _iq_label(pct):
    if pct >= 90: return "Elite Analyst"
    if pct >= 75: return "Sharp Scout"
    if pct >= 50: return "Solid GM"
    if pct >= 25: return "Learning the Ropes"
    return "Keep Playing"


def _archetype_stats(predictions, players, shadow_squad):
    revealed = [p for p in predictions.values() if p.revealed]
    if not revealed:
        return {}
    steal_calls = [p for p in revealed if p.steal_choice == "Steal"]
    steal_hits  = [p for p in steal_calls if p.steal_truth == "Steal"]
    high_band   = [p for p in revealed if p.predicted_band in ("10-15","15+")]
    fr_correct  = [p for p in revealed if p.franchise_points > 0]
    bold = sum(1 for p in revealed
               if players.get(p.lot_no,{}).get("fair_value_band") and
               p.predicted_band != players[p.lot_no]["fair_value_band"])
    n = len(revealed)
    uncapped = (sum(1 for pick in shadow_squad
                    if not players.get(pick.lot_no,{}).get("capped",True))
                / len(shadow_squad)) if shadow_squad else 0.0
    return {
        "steal_hit_rate":     len(steal_hits)/len(steal_calls) if steal_calls else 0.0,
        "overpay_lean":       len(high_band)/n,
        "franchise_accuracy": len(fr_correct)/n,
        "boldness":           bold/n,
        "uncapped_backing":   uncapped,
    }


def _best_worst(predictions, players):
    revealed = [p for p in predictions.values() if p.revealed]
    if not revealed:
        return None, None
    best  = max(revealed, key=lambda p: p.total_points)
    worst = min(revealed, key=lambda p: p.total_points)
    def enrich(p):
        pl = players.get(p.lot_no, {})
        return {"lot_no": p.lot_no,
                "player_name": pl.get("player_name",""),
                "predicted_band": p.predicted_band,
                "actual_band": pl.get("actual_band"),
                "predicted_team": p.predicted_team,
                "actual_team": pl.get("winning_team"),
                "points": p.total_points,
                "iq_percentile": p.iq_percentile,
                "steal_truth": p.steal_truth,
                "winning_price_crore": pl.get("winning_price_crore"),
                "fair_value_crore": pl.get("fair_value_crore")}
    return enrich(best), enrich(worst)


def _narrative(best, worst):
    bl = wl = ""
    if best:
        n = best["player_name"]
        if best.get("steal_truth")=="Steal" and best.get("predicted_band")==best.get("actual_band"):
            bl = f"You spotted {n} as a steal and nailed the price band."
        elif best.get("predicted_band")==best.get("actual_band"):
            bl = f"You called {n}'s price band perfectly."
        else:
            bl = f"Your best call was on {n} ({best['points']} pts)."
    if worst:
        n = worst["player_name"]
        if worst.get("actual_band")=="15+" and worst.get("predicted_band")=="Under 5":
            wl = f"You didn't see {n}'s ₹{worst.get('winning_price_crore')} Cr bid coming."
        else:
            wl = f"Your toughest lot was {n} — the crowd read it better."
    return bl, wl


def compute_wrapped(session, players: dict) -> dict:
    preds    = session.predictions
    squad    = session.shadow_squad
    revealed = [p for p in preds.values() if p.revealed]
    if not revealed:
        return {"user_id": session.user_id, "display_name": session.display_name,
                "message": "No predictions revealed yet.", "lots_played": 0}

    n = len(revealed)
    exact  = sum(1 for p in revealed if p.band_outcome=="exact")
    close  = sum(1 for p in revealed if p.band_outcome=="close")
    miss   = sum(1 for p in revealed if p.band_outcome=="miss")
    sfo_ok = sum(1 for p in revealed if p.steal_points>0)

    iq_vals = [p.iq_percentile for p in revealed if p.iq_percentile is not None]
    overall_iq = int(round(np.mean(iq_vals))) if iq_vals else 50

    best, worst   = _best_worst(preds, players)
    best_line, worst_line = _narrative(best, worst)
    arch_stats    = _archetype_stats(preds, players, squad)
    archetype, arch_line = classify_archetype(arch_stats)

    squad_rows = [{"role": players.get(pick.lot_no,{}).get("role","Unknown"),
                   "overseas": players.get(pick.lot_no,{}).get("overseas",False),
                   "fair_value_crore": players.get(pick.lot_no,{}).get("fair_value_crore",0),
                   "winning_price_crore": pick.your_bid}
                  for pick in squad if players.get(pick.lot_no)]

    return {
        "user_id":        session.user_id,
        "display_name":   session.display_name,
        "total_score":    session.total_score,
        "lots_played":    n,
        "auction_iq":     overall_iq,
        "auction_iq_label": _iq_label(overall_iq),
        "accuracy": {"exact_band": exact, "close_band": close, "missed_band": miss,
                     "band_accuracy_pct": round(exact/n*100), "sfo_correct": sfo_ok},
        "best_call":      best,
        "best_call_line": best_line,
        "worst_call":     worst,
        "worst_call_line":worst_line,
        "archetype":      archetype,
        "archetype_line": arch_line,
        "archetype_stats":arch_stats,
        "team_strength":  team_strength(squad_rows) if squad_rows else None,
        "shadow_squad":   [{"player_name": pick.player_name, "lot_no": pick.lot_no,
                            "your_bid": pick.your_bid,
                            "actual_price": players.get(pick.lot_no,{}).get("winning_price_crore"),
                            "verdict": pick.verdict,
                            "verdict_line": pick.verdict_line}
                           for pick in squad],
    }
