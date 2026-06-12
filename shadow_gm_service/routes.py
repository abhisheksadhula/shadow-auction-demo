"""
routes.py — FastAPI router.

All endpoints the iOS app and backend call.
The backend engineer wires this into their server.
Alternatively, run standalone via main.py.

Endpoints
---------
POST /join                      — register a fan, get session
POST /squad                     — submit pre-auction shadow squad
GET  /lot/current               — current lot info for predict card
POST /lot/predict               — submit prediction for current lot
POST /lot/reveal                — reveal current lot (host only in prod)
GET  /leaderboard               — live ranked leaderboard
GET  /wrapped/{user_id}         — post-auction Wrapped stats
GET  /player/{lot_no}           — lookup a player (for squad builder)
GET  /players                   — all players (for squad builder, paginated)
POST /admin/reset               — wipe state between demo runs
GET  /health                    — ping
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import store, engine

router = APIRouter()


# ---- Schemas -----------------------------------------------------------------

class JoinRequest(BaseModel):
    user_id: str
    display_name: str = ""

class SquadPick(BaseModel):
    lot_no: int
    your_bid: float

class SquadRequest(BaseModel):
    user_id: str
    picks: List[SquadPick]

class PredictRequest(BaseModel):
    user_id: str
    lot_no: int
    predicted_band: str       # "Under 5" | "5-10" | "10-15" | "15+"
    predicted_team: str
    steal_choice: Optional[str] = None  # "Steal" | "Fair" | "Overpay"


# ---- Endpoints ---------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok", "players_loaded": len(store.PLAYERS),
            "sessions": len(store.SESSION), "current_lot": store.ROOM.current_lot}


@router.post("/join")
def join(req: JoinRequest):
    return engine.join_session(req.user_id, req.display_name)


@router.post("/squad")
def set_squad(req: SquadRequest):
    result = engine.update_shadow_squad(
        req.user_id, [p.model_dump() for p in req.picks])
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/lot/current")
def current_lot():
    return engine.get_lot_status()


@router.post("/lot/predict")
def predict(req: PredictRequest):
    result = engine.submit_prediction(
        req.user_id, req.lot_no,
        req.predicted_band, req.predicted_team, req.steal_choice)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/lot/reveal")
def reveal():
    """In production: protect with an admin key. In demo: open."""
    result = engine.reveal_lot()
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/leaderboard")
def leaderboard(limit: int = Query(20, le=100)):
    return {"leaderboard": engine.get_leaderboard_snapshot(limit),
            "total_players": len(store.SESSION)}


@router.get("/wrapped/{user_id}")
def wrapped(user_id: str):
    result = engine.get_wrapped(user_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.get("/player/{lot_no}")
def player(lot_no: int):
    p = store.get_player(lot_no)
    if not p:
        raise HTTPException(404, f"lot_no {lot_no} not found")
    # Return public fields only (no answer keys)
    return {k: p[k] for k in
            ("lot_no","player_name","country","overseas","capped",
             "role","previous_team","base_price_crore","fair_value_crore",
             "fair_value_band")}


@router.get("/players")
def players(page: int = Query(1, ge=1), per_page: int = Query(50, le=200),
            sold_only: bool = False):
    all_p = list(store.PLAYERS.values())
    if sold_only:
        all_p = [p for p in all_p if p.get("sold")]
    start = (page-1)*per_page
    chunk = all_p[start:start+per_page]
    return {
        "total": len(all_p), "page": page, "per_page": per_page,
        "players": [{k: p[k] for k in
                     ("lot_no","player_name","country","overseas","capped",
                      "role","previous_team","base_price_crore",
                      "fair_value_crore","fair_value_band")}
                    for p in chunk]
    }

@router.post("/admin/set-lot/{lot_no}")
def admin_set_lot(lot_no: int):
    if lot_no not in store.PLAYERS:
        raise HTTPException(404, f"lot_no {lot_no} not found")
    store.ROOM.current_lot = lot_no
    store.ROOM.auction_started = True
    store.ROOM.auction_finished = False
    return {
        "current_lot": lot_no,
        "player": store.PLAYERS[lot_no]["player_name"]
    }

@router.post("/admin/reset")
def admin_reset():
    store.reset_room()
    return {"status": "reset", "message": "All sessions and scores wiped."}
