"""
api.py
------
Shadow GM Scoring Service — FastAPI application.

All endpoints the backend engineer (and iOS) will call.

Run:
    uvicorn shadow_gm.service.api:app --reload --port 8000

Base URL (local): http://localhost:8000
Interactive docs:  http://localhost:8000/docs

Endpoints
---------
POST /users                         create a user, get user_id
POST /leagues                       create a private league, get invite_code
POST /leagues/join                  join via invite_code
GET  /leagues/{league_id}           league info + member list

POST /squad/{user_id}               save pre-auction shadow squad
GET  /squad/{user_id}               fetch squad with player details

GET  /lots                          full lot list (for squad builder & game)
GET  /lots/{lot_no}                 single lot details

POST /predict/{user_id}/{lot_no}    submit band + team + SFO prediction
POST /reveal/{lot_no}               reveal result, score ALL predictions for lot
GET  /crowd/{lot_no}                crowd band/SFO breakdown after reveal

GET  /leaderboard                   global leaderboard
GET  /leaderboard/{league_id}       league-scoped leaderboard

GET  /wrapped/{user_id}             full Auction Wrapped payload
GET  /health                        service health
"""

import json
import math
import os
import sys
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _nan_safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so JSON serialization never fails."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_nan_safe(content), ensure_ascii=False).encode("utf-8")

# resolve imports whether run from root or service/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from scoring import (score_band_prediction, score_steal_prediction,
                     shadow_gm_verdict)
from service.store import store
from service.wrapped import compute_wrapped
from service.leaderboard import (get_global_leaderboard,
                                  get_league_leaderboard,
                                  get_lot_crowd_stats)

# ── load players.json once at startup ─────────────────────────────────────────
PLAYERS_JSON = os.path.join(HERE, "..", "output", "players.json")
with open(PLAYERS_JSON) as f:
    _data = json.load(f)

PLAYER_MAP: dict[int, dict] = {p["lot_no"]: p for p in _data["players"]}
PLAYERS_LIST: list[dict]    = _data["players"]
META: dict                  = _data["meta"]

# snapshot of scores before each reveal (for leaderboard delta)
_prev_scores: dict[str, int] = {}

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Shadow GM Scoring Service",
    description="Data layer for the IPL Shadow GM auction game",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────────

class CreateUserReq(BaseModel):
    name: str

class CreateLeagueReq(BaseModel):
    name: str
    creator_id: str

class JoinLeagueReq(BaseModel):
    invite_code: str
    user_id: str

class SquadPickReq(BaseModel):
    lot_no: int
    allocated_bid: float

class SquadReq(BaseModel):
    picks: list[SquadPickReq]

class PredictReq(BaseModel):
    predicted_band: str               # "Under 5" | "5-10" | "10-15" | "15+"
    predicted_team: Optional[str] = None
    steal_vote: Optional[str]    = None  # "Steal" | "Fair" | "Overpay"


# ── health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "players": len(PLAYERS_LIST),
        "sold": sum(1 for p in PLAYERS_LIST if p["sold"]),
        "users": len(store.all_users()),
        "model": META.get("model"),
        "bands": META.get("bands"),
    }


# ── users ──────────────────────────────────────────────────────────────────────

@app.post("/users", status_code=201)
def create_user(req: CreateUserReq):
    """Register a fan. Returns user_id — store this client-side."""
    if not req.name.strip():
        raise HTTPException(400, "name cannot be empty")
    user = store.create_user(req.name.strip())
    return {"user_id": user.user_id, "name": user.name}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    u = store.get_user(user_id)
    if not u:
        raise HTTPException(404, "user not found")
    return {"user_id": u.user_id, "name": u.name,
            "league_id": u.league_id,
            "score": store.get_score(user_id)}


# ── leagues ────────────────────────────────────────────────────────────────────

@app.post("/leagues", status_code=201)
def create_league(req: CreateLeagueReq):
    """Create a private WhatsApp-style league. Share invite_code with friends."""
    if not store.get_user(req.creator_id):
        raise HTTPException(404, "creator user not found")
    lg = store.create_league(req.name, req.creator_id)
    return {
        "league_id":   lg.league_id,
        "name":        lg.name,
        "invite_code": lg.invite_code,
        "members":     lg.member_ids,
    }


@app.post("/leagues/join")
def join_league(req: JoinLeagueReq):
    """Join an existing league via invite_code."""
    if not store.get_user(req.user_id):
        raise HTTPException(404, "user not found")
    lg = store.join_league(req.invite_code, req.user_id)
    if not lg:
        raise HTTPException(404, "invalid invite code")
    return {"league_id": lg.league_id, "name": lg.name, "members": lg.member_ids}


@app.get("/leagues/{league_id}")
def get_league(league_id: str):
    lg = store.get_league(league_id)
    if not lg:
        raise HTTPException(404, "league not found")
    members = [
        {"user_id": uid, "name": store.get_user(uid).name if store.get_user(uid) else uid}
        for uid in lg.member_ids
    ]
    return {"league_id": lg.league_id, "name": lg.name,
            "invite_code": lg.invite_code, "members": members}


# ── lots ───────────────────────────────────────────────────────────────────────

@app.get("/lots")
def get_lots(sold_only: bool = False, limit: int = 577, offset: int = 0):
    """
    Return the lot list for the squad builder and game.
    sold_only=true returns only sold players (for replay mode).
    """
    lots = PLAYERS_LIST
    if sold_only:
        lots = [p for p in lots if p["sold"]]
    lots = lots[offset: offset + limit]
    # strip answer keys — don't send actual_band/steal_truth to client
    return [_safe_lot(p) for p in lots]


@app.get("/lots/{lot_no}")
def get_lot(lot_no: int):
    pl = PLAYER_MAP.get(lot_no)
    if not pl:
        raise HTTPException(404, "lot not found")
    return _safe_lot(pl)


def _safe_lot(p: dict) -> dict:
    """Return player data safe to send to client — no answer keys."""
    return {
        "lot_no":           p["lot_no"],
        "player_name":      p["player_name"],
        "country":          p["country"],
        "overseas":         p["overseas"],
        "capped":           p["capped"],
        "role":             p["role"],
        "previous_team":    p["previous_team"],
        "base_price_crore": p["base_price_crore"],
        "fair_value_band":  p["fair_value_band"],  # hint for the UI
        "fair_value_crore": p["fair_value_crore"],
        "sold":             p["sold"],
        # winning_team / actual_band withheld until /reveal is called
    }


# ── squad ──────────────────────────────────────────────────────────────────────

@app.post("/squad/{user_id}", status_code=201)
def save_squad(user_id: str, req: SquadReq):
    """Save a fan's pre-auction shadow squad."""
    if not store.get_user(user_id):
        raise HTTPException(404, "user not found")

    # validate: lot_nos must exist, total bid <= purse
    purse = META.get("scoring", {}).get("virtual_purse", 100.0)
    total_bid = sum(p.allocated_bid for p in req.picks)
    if total_bid > purse:
        raise HTTPException(400, f"total allocated bid {total_bid} exceeds purse {purse}")
    for pick in req.picks:
        if pick.lot_no not in PLAYER_MAP:
            raise HTTPException(400, f"lot_no {pick.lot_no} not found")

    picks = store.set_squad(user_id, [p.dict() for p in req.picks])
    return {
        "user_id":   user_id,
        "picks":     len(picks),
        "total_bid": round(total_bid, 2),
    }


@app.get("/squad/{user_id}")
def get_squad(user_id: str):
    """Return a user's squad with full player details + fair_value."""
    if not store.get_user(user_id):
        raise HTTPException(404, "user not found")
    picks = store.get_squad(user_id)
    result = []
    for pick in picks:
        pl = PLAYER_MAP.get(pick.lot_no, {})
        result.append({
            "lot_no":        pick.lot_no,
            "allocated_bid": pick.allocated_bid,
            "player_name":   pl.get("player_name"),
            "fair_value":    pl.get("fair_value_crore"),
            "capped":        pl.get("capped"),
            "overseas":      pl.get("overseas"),
        })
    total = sum(p.allocated_bid for p in picks)
    return {"user_id": user_id, "squad": result,
            "total_allocated": round(total, 2), "n": len(result)}


# ── predict ────────────────────────────────────────────────────────────────────

@app.post("/predict/{user_id}/{lot_no}", status_code=201)
def submit_prediction(user_id: str, lot_no: int, req: PredictReq):
    """
    Fan submits their prediction for a lot before reveal.
    Can be called multiple times (overwrites previous submission).
    """
    if not store.get_user(user_id):
        raise HTTPException(404, "user not found")
    if lot_no not in PLAYER_MAP:
        raise HTTPException(404, "lot not found")

    valid_bands = META["bands"]
    if req.predicted_band not in valid_bands:
        raise HTTPException(400, f"predicted_band must be one of {valid_bands}")
    if req.steal_vote and req.steal_vote not in ("Steal", "Fair", "Overpay"):
        raise HTTPException(400, "steal_vote must be Steal, Fair, or Overpay")

    pred = store.submit_prediction(
        user_id, lot_no,
        predicted_band=req.predicted_band,
        predicted_team=req.predicted_team,
        steal_vote=req.steal_vote,
    )
    return {
        "user_id":        user_id,
        "lot_no":         lot_no,
        "predicted_band": pred.predicted_band,
        "predicted_team": pred.predicted_team,
        "steal_vote":     pred.steal_vote,
        "status":         "submitted",
    }


# ── reveal ─────────────────────────────────────────────────────────────────────

@app.post("/reveal/{lot_no}")
def reveal_lot(lot_no: int):
    """
    Called by the game host (or automatically) after a lot closes.
    Scores ALL predictions submitted for this lot.
    Returns the result + per-user score breakdown.
    """
    pl = PLAYER_MAP.get(lot_no)
    if not pl:
        raise HTTPException(404, "lot not found")

    # snapshot scores before scoring (for leaderboard delta)
    global _prev_scores
    _prev_scores = store.all_scores().copy()

    preds = store.all_predictions_for_lot(lot_no)
    results = []

    for pred in preds:
        # band + franchise
        bp = score_band_prediction(
            pred.predicted_band, pl.get("actual_band"),
            pred.predicted_team,  pl.get("winning_team"),
        )
        # steal / fair / overpay
        sp = score_steal_prediction(
            pred.steal_vote,
            pl.get("winning_price_crore"),
            pl.get("fair_value_crore"),
        ) if pred.steal_vote else {"points": 0}

        store.score_prediction(
            pred,
            band_pts=bp["band_points"],
            franchise_pts=bp["franchise_points"],
            steal_pts=sp["points"],
        )

        results.append({
            "user_id":         pred.user_id,
            "user_name":       store.get_user(pred.user_id).name if store.get_user(pred.user_id) else pred.user_id,
            "predicted_band":  pred.predicted_band,
            "predicted_team":  pred.predicted_team,
            "steal_vote":      pred.steal_vote,
            "band_outcome":    bp["outcome"],
            "band_points":     bp["band_points"],
            "franchise_points":bp["franchise_points"],
            "steal_points":    sp["points"],
            "total_points":    pred.total_points,
            "running_score":   store.get_score(pred.user_id),
        })

    # shadow GM verdict for any user whose squad included this lot
    squad_verdicts = []
    for user in store.all_users():
        for pick in store.get_squad(user.user_id):
            if pick.lot_no == lot_no:
                v = shadow_gm_verdict(pl, your_bid=pick.allocated_bid)
                squad_verdicts.append({
                    "user_id":  user.user_id,
                    "user_name":user.name,
                    "verdict":  v["verdict"],
                    "line":     v["line"],
                })

    return {
        "lot_no":          lot_no,
        "player_name":     pl["player_name"],
        "actual_band":     pl.get("actual_band"),
        "actual_price":    pl.get("winning_price_crore"),
        "winning_team":    pl.get("winning_team"),
        "fair_value":      pl.get("fair_value_crore"),
        "steal_truth":     pl.get("steal_truth"),
        "sold":            pl.get("sold"),
        "predictions_scored": len(results),
        "results":         results,
        "squad_verdicts":  squad_verdicts,
    }


# ── crowd stats ────────────────────────────────────────────────────────────────

@app.get("/crowd/{lot_no}")
def crowd_stats(lot_no: int):
    """Crowd vote breakdown after reveal — the 'you vs crowd' screen."""
    if lot_no not in PLAYER_MAP:
        raise HTTPException(404, "lot not found")
    return get_lot_crowd_stats(lot_no, store, PLAYER_MAP)


# ── leaderboard ────────────────────────────────────────────────────────────────

@app.get("/leaderboard")
def global_leaderboard():
    """Live global leaderboard — call after every reveal."""
    return {
        "scope": "global",
        "entries": get_global_leaderboard(store, PLAYER_MAP, _prev_scores),
    }


@app.get("/leaderboard/{league_id}")
def league_leaderboard(league_id: str):
    """Leaderboard scoped to a private league."""
    lg = store.get_league(league_id)
    if not lg:
        raise HTTPException(404, "league not found")
    return {
        "scope":       "league",
        "league_name": lg.name,
        "invite_code": lg.invite_code,
        "entries":     get_league_leaderboard(store, league_id, PLAYER_MAP, _prev_scores),
    }


# ── wrapped ────────────────────────────────────────────────────────────────────

@app.get("/wrapped/{user_id}")
def get_wrapped(user_id: str):
    """Full Auction Wrapped payload — call at end of session."""
    user = store.get_user(user_id)
    if not user:
        raise HTTPException(404, "user not found")
    preds  = store.get_all_predictions_for_user(user_id)
    squad  = store.get_squad(user_id)
    return compute_wrapped(user, preds, PLAYER_MAP, squad)
