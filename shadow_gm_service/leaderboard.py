"""
leaderboard.py
--------------
Leaderboard aggregation for Shadow GM.

Two scopes:
  - Global  : all users in the session
  - League  : users in one private group (by league_id)

Each entry carries:
  rank, user_name, total_score, auction_iq_percentile, lots_played,
  exact_hits, last_lot_points (delta since previous lot)

The Auction IQ shown on the leaderboard is the user's average per-lot
percentile across all scored lots — same formula as Wrapped, kept
consistent so the live leaderboard and the final card agree.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scoring import AuctionIQ


def _iq_for_user(predictions: list, player_map: dict) -> int:
    """Average per-lot Auction IQ percentile across all scored predictions."""
    pct_sum, counted = 0, 0
    for p in predictions:
        if not p.scored:
            continue
        pl = player_map.get(p.lot_no, {})
        base = pl.get("auction_iq_baseline")
        if not base:
            continue
        dist = np.concatenate([
            np.full(int(base["p_miss"]  * 1000), 0),
            np.full(int(base["p_close"] * 1000), 40),
            np.full(int(base["p_full"]  * 1000), 100),
        ])
        pct_sum += AuctionIQ(dist).percentile(p.band_points)
        counted += 1
    return int(round(pct_sum / counted)) if counted else 0


def _build_entry(rank: int, user, predictions: list,
                 player_map: dict, prev_scores: dict) -> dict:
    scored = [p for p in predictions if p.scored]
    total = sum(p.total_points for p in scored)
    prev  = prev_scores.get(user.user_id, 0)
    return {
        "rank":                rank,
        "user_id":             user.user_id,
        "user_name":           user.name,
        "total_score":         total,
        "auction_iq_percentile": _iq_for_user(scored, player_map),
        "lots_played":         len(scored),
        "exact_hits":          sum(1 for p in scored if p.band_points == 100),
        "last_lot_delta":      total - prev,   # points earned on the most recent lot
    }


def get_global_leaderboard(store, player_map: dict,
                            prev_scores: dict | None = None) -> list[dict]:
    """
    Returns ranked list of all users sorted by total_score desc.
    prev_scores: snapshot of scores from the previous lot (for delta display).
    """
    prev = prev_scores or {}
    entries = []
    for user in store.all_users():
        preds = store.get_all_predictions_for_user(user.user_id)
        entries.append(_build_entry(0, user, preds, player_map, prev))

    entries.sort(key=lambda e: (-e["total_score"], e["user_name"]))
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return entries


def get_league_leaderboard(store, league_id: str, player_map: dict,
                            prev_scores: dict | None = None) -> list[dict]:
    """Leaderboard scoped to one private league."""
    lg = store.get_league(league_id)
    if not lg:
        return []
    prev = prev_scores or {}
    entries = []
    for uid in lg.member_ids:
        user = store.get_user(uid)
        if not user:
            continue
        preds = store.get_all_predictions_for_user(uid)
        entries.append(_build_entry(0, user, preds, player_map, prev))

    entries.sort(key=lambda e: (-e["total_score"], e["user_name"]))
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return entries


def get_lot_crowd_stats(lot_no: int, store, player_map: dict) -> dict:
    """
    After a lot closes: how did the crowd vote on band and Steal/Fair/Overpay?
    Shown to users immediately after reveal — the 'you vs crowd' moment.
    """
    preds = store.all_predictions_for_lot(lot_no)
    if not preds:
        return {"lot_no": lot_no, "total_voters": 0}

    pl = player_map.get(lot_no, {})
    bands = ["Under 5", "5-10", "10-15", "15+"]
    band_votes = {b: 0 for b in bands}
    sfo_votes  = {"Steal": 0, "Fair": 0, "Overpay": 0}
    team_votes: dict[str, int] = {}

    for p in preds:
        if p.predicted_band in band_votes:
            band_votes[p.predicted_band] += 1
        if p.steal_vote in sfo_votes:
            sfo_votes[p.steal_vote] += 1
        if p.predicted_team:
            team_votes[p.predicted_team] = team_votes.get(p.predicted_team, 0) + 1

    n = len(preds)
    top_team = max(team_votes, key=team_votes.get) if team_votes else None

    # majority Steal/Fair/Overpay verdict
    crowd_sfo = max(sfo_votes, key=sfo_votes.get) if any(sfo_votes.values()) else None

    return {
        "lot_no":          lot_no,
        "player_name":     pl.get("player_name"),
        "actual_band":     pl.get("actual_band"),
        "steal_truth":     pl.get("steal_truth"),
        "total_voters":    n,
        "band_votes":      {b: round(v/n*100, 1) for b, v in band_votes.items()},
        "sfo_votes":       {k: round(v/n*100, 1) for k, v in sfo_votes.items()},
        "crowd_sfo":       crowd_sfo,
        "top_predicted_team": top_team,
        "actual_team":     pl.get("winning_team"),
    }
