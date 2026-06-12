"""
store.py
--------
In-memory data store. Single source of truth for all runtime state.

On hackathon day this is enough — one shared room, no persistence needed.
On the 6-month roadmap: swap this out for Redis or Postgres; everything
above it stays the same because nothing outside this file touches state directly.

Structure
---------
PLAYERS      : dict[lot_no -> player record] — loaded once from players.json
SESSION      : dict[user_id -> SessionState]
LEADERBOARD  : sorted list[LeaderboardEntry]  (rebuilt from SESSION on each write)
ROOM         : shared auction state (current lot, revealed lots)
"""
import json
import threading
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

# ---- Types -------------------------------------------------------------------

@dataclass
class Prediction:
    lot_no: int
    predicted_band: str
    predicted_team: str
    steal_choice: Optional[str] = None
    band_points: int = 0
    franchise_points: int = 0
    steal_points: int = 0
    total_points: int = 0
    iq_percentile: Optional[int] = None
    band_outcome: str = ""
    steal_truth: Optional[str] = None
    revealed: bool = False


@dataclass
class ShadowSquadPick:
    lot_no: int
    player_name: str
    your_bid: float
    verdict: str = "pending"
    verdict_line: str = ""


@dataclass
class SessionState:
    user_id: str
    display_name: str
    predictions: dict = field(default_factory=dict)
    shadow_squad: list = field(default_factory=list)
    total_score: int = 0
    joined_at: str = ""


@dataclass
class LeaderboardEntry:
    user_id: str
    display_name: str
    total_score: int
    lots_played: int
    rank: int = 0


@dataclass
class RoomState:
    current_lot: int = 1
    revealed_lots: set = field(default_factory=set)
    auction_started: bool = False
    auction_finished: bool = False


# ---- Module-level state ------------------------------------------------------

_lock = threading.Lock()
PLAYERS: dict = {}
SESSION: dict = {}
LEADERBOARD: list = []
ROOM: RoomState = RoomState()
META: dict = {}


# ---- Init --------------------------------------------------------------------

def load_players(json_path: str):
    global PLAYERS, META
    with open(json_path) as f:
        data = json.load(f)
    with _lock:
        PLAYERS = {p["lot_no"]: p for p in data["players"]}
        META = data.get("meta", {})
    return META


def reset_room():
    global SESSION, LEADERBOARD, ROOM
    with _lock:
        SESSION.clear()
        LEADERBOARD.clear()
        ROOM = RoomState()


# ---- Session -----------------------------------------------------------------

def get_or_create_session(user_id: str, display_name: str = "") -> SessionState:
    with _lock:
        if user_id not in SESSION:
            SESSION[user_id] = SessionState(
                user_id=user_id,
                display_name=display_name or user_id,
                joined_at=datetime.now(timezone.utc).isoformat(),
            )
        return SESSION[user_id]


def get_session(user_id: str) -> Optional[SessionState]:
    return SESSION.get(user_id)


def all_sessions() -> list:
    return list(SESSION.values())


# ---- Leaderboard -------------------------------------------------------------

def rebuild_leaderboard():
    global LEADERBOARD
    entries = [
        LeaderboardEntry(
            user_id=s.user_id,
            display_name=s.display_name,
            total_score=s.total_score,
            lots_played=sum(1 for p in s.predictions.values() if p.revealed),
        )
        for s in SESSION.values()
    ]
    entries.sort(key=lambda e: (-e.total_score, e.display_name))
    for i, e in enumerate(entries):
        e.rank = i + 1
    with _lock:
        LEADERBOARD = entries
    return entries


def get_leaderboard(limit: int = 50) -> list:
    return LEADERBOARD[:limit]


# ---- Room --------------------------------------------------------------------

def get_player(lot_no: int) -> Optional[dict]:
    return PLAYERS.get(lot_no)


def total_lots() -> int:
    return len(PLAYERS)


def advance_lot():
    """Mark current lot revealed, step to next lot. Returns next player or None if done."""
    with _lock:
        ROOM.revealed_lots.add(ROOM.current_lot)
        nxt = ROOM.current_lot + 1
        while nxt <= max(PLAYERS.keys(), default=0):
            if nxt in PLAYERS:
                ROOM.current_lot = nxt
                return PLAYERS[nxt]
            nxt += 1
        ROOM.auction_finished = True
        return None
