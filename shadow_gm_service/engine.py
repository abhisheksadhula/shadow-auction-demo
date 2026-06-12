"""
engine.py — game logic layer.

Sits between the API (routes.py) and the data layer (store.py + scoring.py).
All mutation of session/room state happens here.

Functions
---------
join_session()          — register a fan
submit_prediction()     — fan submits band + team + SFO for current lot
reveal_lot()            — host advances to reveal; scores all predictions for that lot
update_shadow_squad()   — fan locks in pre-auction squad picks
get_lot_status()        — current lot info for the predict card
get_reveal_result()     — full reveal payload (scores + leaderboard delta)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring import (score_band_prediction, score_steal_prediction,
                     shadow_gm_verdict, AuctionIQ)
import store
from store import (Prediction, ShadowSquadPick,
                   get_or_create_session, get_session,
                   get_player, rebuild_leaderboard,
                   get_leaderboard, advance_lot, ROOM)
from wrapped import compute_wrapped


# ---- Join --------------------------------------------------------------------

def join_session(user_id: str, display_name: str) -> dict:
    session = get_or_create_session(user_id, display_name)
    rebuild_leaderboard()
    return {"user_id": session.user_id,
            "display_name": session.display_name,
            "joined_at": session.joined_at,
            "total_players": len(store.SESSION)}


# ---- Pre-auction: Shadow Squad -----------------------------------------------

def update_shadow_squad(user_id: str, picks: list) -> dict:
    """
    picks: list of {"lot_no": int, "your_bid": float}
    Validates lot exists, stores picks. Verdicts are 'pending' until lots reveal.
    """
    session = get_session(user_id)
    if not session:
        return {"error": "User not found. Call /join first."}

    squad = []
    errors = []
    for pick in picks:
        lot_no = int(pick["lot_no"])
        player = get_player(lot_no)
        if not player:
            errors.append(f"lot_no {lot_no} not found")
            continue
        squad.append(ShadowSquadPick(
            lot_no=lot_no,
            player_name=player["player_name"],
            your_bid=float(pick.get("your_bid", player["base_price_crore"])),
        ))

    session.shadow_squad = squad
    return {"squad_size": len(squad),
            "squad": [{"lot_no": p.lot_no, "player_name": p.player_name,
                        "your_bid": p.your_bid} for p in squad],
            "errors": errors}


# ---- During auction: predict -------------------------------------------------

def get_lot_status() -> dict:
    """What the predict card needs: current lot info, no answer keys."""
    lot_no = store.ROOM.current_lot
    player = get_player(lot_no)
    if not player:
        return {"auction_finished": True}
    return {
        "lot_no": lot_no,
        "player_name": player["player_name"],
        "country": player["country"],
        "overseas": player["overseas"],
        "capped": player["capped"],
        "previous_team": player["previous_team"],
        "base_price_crore": player["base_price_crore"],
        "bands": ["Under 5", "5-10", "10-15", "15+"],
        "auction_finished": store.ROOM.auction_finished,
    }


def submit_prediction(user_id: str, lot_no: int,
                       predicted_band: str, predicted_team: str,
                       steal_choice: str = None) -> dict:
    session = get_session(user_id)
    if not session:
        return {"error": "User not found. Call /join first."}
    if lot_no != store.ROOM.current_lot:
        return {"error": f"Lot {lot_no} is not the current lot ({store.ROOM.current_lot})."}
    if lot_no in session.predictions and session.predictions[lot_no].revealed:
        return {"error": "Lot already revealed — too late to predict."}

    session.predictions[lot_no] = Prediction(
        lot_no=lot_no,
        predicted_band=predicted_band,
        predicted_team=predicted_team,
        steal_choice=steal_choice,
    )
    return {"status": "prediction_received", "lot_no": lot_no,
            "predicted_band": predicted_band, "predicted_team": predicted_team}


# ---- Reveal ------------------------------------------------------------------

def _score_prediction(pred: Prediction, player: dict) -> Prediction:
    """Fill in score fields on a Prediction using the player's answer keys."""
    bp = score_band_prediction(
        pred.predicted_band, player.get("actual_band"),
        pred.predicted_team, player.get("winning_team"))

    sfo = score_steal_prediction(
        pred.steal_choice,
        player.get("winning_price_crore"),
        player.get("fair_value_crore"))

    # Auction IQ percentile from the pre-baked baseline
    baseline = player.get("auction_iq_baseline")
    band_only_score = bp["band_points"]
    if baseline:
        import numpy as np
        n = 1000
        dist = np.concatenate([
            np.full(int(baseline["p_miss"] * n), 0),
            np.full(int(baseline["p_close"] * n), 40),
            np.full(int(baseline["p_full"] * n), 100),
        ])
        iq_pct = AuctionIQ(dist).percentile(band_only_score) if len(dist) else 50
    else:
        iq_pct = 50

    pred.band_points      = bp["band_points"]
    pred.franchise_points = bp["franchise_points"]
    pred.steal_points     = sfo["points"]
    pred.total_points     = bp["total"] + sfo["points"]
    pred.band_outcome     = bp["outcome"]
    pred.steal_truth      = sfo["truth"]
    pred.iq_percentile    = iq_pct
    pred.revealed         = True
    return pred


def _update_squad_verdict(session, player: dict):
    """After a lot reveals, update Shadow GM verdict for any squad pick on this lot."""
    lot_no = player["lot_no"]
    for pick in session.shadow_squad:
        if pick.lot_no == lot_no:
            result = shadow_gm_verdict(player, your_bid=pick.your_bid)
            pick.verdict      = result["verdict"]
            pick.verdict_line = result["line"]


def reveal_lot() -> dict:
    """
    Advance the room to the next lot, scoring all pending predictions for the
    just-revealed lot. Returns the reveal payload for all clients.
    Called by the host/admin (or automatically in replay mode).
    """
    lot_no  = store.ROOM.current_lot
    player  = get_player(lot_no)
    if not player:
        return {"error": "No current lot."}

    results = {}
    for session in store.all_sessions():
        pred = session.predictions.get(lot_no)
        if pred and not pred.revealed:
            pred = _score_prediction(pred, player)
            session.predictions[lot_no] = pred
            session.total_score += pred.total_points
            results[session.user_id] = {
                "display_name":    session.display_name,
                "band_points":     pred.band_points,
                "franchise_points":pred.franchise_points,
                "steal_points":    pred.steal_points,
                "total_points":    pred.total_points,
                "band_outcome":    pred.band_outcome,
                "iq_percentile":   pred.iq_percentile,
                "steal_truth":     pred.steal_truth,
                "running_total":   session.total_score,
            }
        _update_squad_verdict(session, player)

    leaderboard = rebuild_leaderboard()

    # Answer key reveal (safe to expose now)
    reveal = {
        "lot_no":              lot_no,
        "player_name":         player["player_name"],
        "sold":                player["sold"],
        "winning_team":        player.get("winning_team"),
        "winning_price_crore": player.get("winning_price_crore"),
        "actual_band":         player.get("actual_band"),
        "fair_value_crore":    player.get("fair_value_crore"),
        "steal_truth":         player.get("steal_truth"),
        "user_scores":         results,
        "leaderboard":         [{"rank": e.rank, "display_name": e.display_name,
                                  "total_score": e.total_score, "lots_played": e.lots_played}
                                 for e in leaderboard[:10]],
    }

    # Advance room to next lot
    advance_lot()
    return reveal


# ---- Leaderboard & Wrapped ---------------------------------------------------

def get_leaderboard_snapshot(limit: int = 20) -> list:
    return [{"rank": e.rank, "display_name": e.display_name,
             "total_score": e.total_score, "lots_played": e.lots_played,
             "user_id": e.user_id}
            for e in get_leaderboard(limit)]


def get_wrapped(user_id: str) -> dict:
    session = get_session(user_id)
    if not session:
        return {"error": "User not found."}
    return compute_wrapped(session, store.PLAYERS)
