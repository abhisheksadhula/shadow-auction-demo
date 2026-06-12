"""
main.py — run the Shadow GM scoring service.

Usage:
    python3 shadow_gm/service/main.py
    # or:
    uvicorn shadow_gm.service.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import store
from routes import router

# ---- Players JSON path -------------------------------------------------------
# Points to the real IPL 2025 data built in the pre-hackathon pipeline.
PLAYERS_JSON = os.path.join(
    os.path.dirname(__file__), "..", "players.json")

# ---- App setup ---------------------------------------------------------------
app = FastAPI(
    title="Shadow GM — Scoring Service",
    description="Data engineering backend for the IPL Auction Game hackathon.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(router)


@app.on_event("startup")
def startup():
    meta = store.load_players(PLAYERS_JSON)
    print(f"Loaded {len(store.PLAYERS)} players from {meta.get('source','players.json')}")
    print(f"Model: {meta.get('model')}")
    print(f"Fit: {meta.get('model_fit')}")
    print("Shadow GM service ready. Swagger: http://localhost:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
