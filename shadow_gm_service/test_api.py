"""
test_api.py
-----------
End-to-end test of the Shadow GM Scoring Service.

Simulates a complete demo session:
  - 3 fans register
  - 2 form a private league, 1 joins via invite code
  - All 3 build pre-auction shadow squads
  - 6 lots are predicted and revealed (mix of sold/unsold, steals, overpays)
  - Crowd stats checked after each reveal
  - Global + league leaderboard validated
  - Auction Wrapped generated for all 3 fans

Runs against the live FastAPI app in-process using TestClient (no server needed).
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from shadow_gm.service.api import app, PLAYER_MAP, PLAYERS_LIST

client = TestClient(app)

SEP = "─" * 60

def p(label, data):
    print(f"\n{'▸'} {label}")
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                print(f"    {k}: [{len(v)} items]")
            else:
                print(f"    {k}: {v}")
    elif isinstance(data, list):
        for item in data:
            print(f"    {item}")
    else:
        print(f"    {data}")

def check(resp, label=""):
    assert resp.status_code in (200, 201), \
        f"FAIL {label}: {resp.status_code} — {resp.text}"
    return resp.json()

# ── pick 6 interesting lots from the real data ─────────────────────────────────
sold_lots   = [pl for pl in PLAYERS_LIST if pl["sold"]]
unsold_lots = [pl for pl in PLAYERS_LIST if not pl["sold"]]

# Pick a spread: a star player, a big surprise, a base-price sell, an unsold, a steal, an overpay
star    = next(p for p in sold_lots if p["winning_price_crore"] >= 20)          # Rishabh Pant
surprise= next(p for p in sold_lots if p["winning_price_crore"]/p["base_price_crore"] >= 10
               and p["winning_price_crore"] < 10)                                # e.g. Priyansh Arya
base    = next(p for p in sold_lots if p["winning_price_crore"] == p["base_price_crore"]
               and p["base_price_crore"] >= 1.5)                                 # sold at base
unsold  = unsold_lots[5]                                                          # an unsold player
steal   = next(p for p in sold_lots if p["steal_truth"] == "Steal"
               and p["fair_value_crore"] >= 3
               and p["lot_no"] != star["lot_no"])
overpay = next(p for p in sold_lots if p["steal_truth"] == "Overpay"
               and p["winning_price_crore"] >= 8
               and p["lot_no"] not in (star["lot_no"], steal["lot_no"]))

DEMO_LOTS = [star, surprise, base, unsold, steal, overpay]

print(SEP)
print("DEMO LOTS")
print(SEP)
for pl in DEMO_LOTS:
    tag = pl.get("steal_truth") or ("unsold" if not pl["sold"] else "fair")
    print(f"  Lot {pl['lot_no']:>3}  {pl['player_name']:<30}  "
          f"₹{pl.get('winning_price_crore') or 0:>5} Cr  [{tag}]")

# ── 1. health check ────────────────────────────────────────────────────────────
print(f"\n{SEP}\n1. HEALTH\n{SEP}")
r = check(client.get("/health"), "health")
p("health", r)

# ── 2. create 3 users ──────────────────────────────────────────────────────────
print(f"\n{SEP}\n2. CREATE USERS\n{SEP}")
u1 = check(client.post("/users", json={"name": "Abhishek"}), "create u1")
u2 = check(client.post("/users", json={"name": "Rohan"}),    "create u2")
u3 = check(client.post("/users", json={"name": "Priya"}),    "create u3")
uid1, uid2, uid3 = u1["user_id"], u2["user_id"], u3["user_id"]
print(f"  {u1['name']} ({uid1})  {u2['name']} ({uid2})  {u3['name']} ({uid3})")

# ── 3. league: u1 creates, u2 + u3 join ───────────────────────────────────────
print(f"\n{SEP}\n3. LEAGUES\n{SEP}")
lg = check(client.post("/leagues",
    json={"name": "Office League", "creator_id": uid1}), "create league")
invite = lg["invite_code"]
p("league created", lg)

j2 = check(client.post("/leagues/join",
    json={"invite_code": invite, "user_id": uid2}), "u2 join")
j3 = check(client.post("/leagues/join",
    json={"invite_code": invite, "user_id": uid3}), "u3 join")
print(f"  Members after joins: {j3['members']}")

league_id = lg["league_id"]

# ── 4. browse lots ─────────────────────────────────────────────────────────────
print(f"\n{SEP}\n4. LOTS\n{SEP}")
lots_resp = check(client.get("/lots?sold_only=true&limit=5"), "get lots")
print(f"  First 5 sold lots: {[l['player_name'] for l in lots_resp]}")
single = check(client.get(f"/lots/{star['lot_no']}"), "single lot")
print(f"  Single lot: {single['player_name']} — fair value band: {single['fair_value_band']}")
assert "actual_band" not in single, "FAIL: answer key leaked to client"
print("  ✓ actual_band not exposed to client")

# ── 5. shadow squads ────────────────────────────────────────────────────────────
print(f"\n{SEP}\n5. SHADOW SQUADS\n{SEP}")
# Each fan picks 3 of the 6 demo lots for their squad
squad_lots = [star, steal, overpay]

for uid, name in [(uid1,"Abhishek"),(uid2,"Rohan"),(uid3,"Priya")]:
    picks = [{"lot_no": pl["lot_no"],
              "allocated_bid": round(pl["fair_value_crore"] * 1.1, 1)}  # bid 10% over fair
             for pl in squad_lots]
    r = check(client.post(f"/squad/{uid}", json={"picks": picks}), f"squad {name}")
    print(f"  {name}: {r['picks']} picks, ₹{r['total_bid']} Cr allocated")

sq = check(client.get(f"/squad/{uid1}"), "get squad u1")
print(f"  Abhishek's squad: {[s['player_name'] for s in sq['squad']]}")

# ── 6. predictions + reveals ───────────────────────────────────────────────────
print(f"\n{SEP}\n6. PREDICTIONS & REVEALS\n{SEP}")

# Prediction strategies:
#   u1 = smart fan: always predicts fair_value_band, always picks correct team
#   u2 = casual fan: one band off, sometimes skips team
#   u3 = bold fan: often guesses wrong band (one above), aggressive SFO

bands_order = ["Under 5", "5-10", "10-15", "15+"]

def adjacent_band(band, offset=1):
    i = bands_order.index(band) if band in bands_order else 0
    return bands_order[min(max(i + offset, 0), 3)]

for pl in DEMO_LOTS:
    lot_no = pl["lot_no"]
    fv_band = pl["fair_value_band"]
    actual_team = pl.get("winning_team") or "MI"
    label = pl["player_name"]

    # u1: smart — predicts fair_value_band and correct team
    check(client.post(f"/predict/{uid1}/{lot_no}", json={
        "predicted_band": fv_band,
        "predicted_team": actual_team,
        "steal_vote": pl.get("steal_truth") or "Fair",
    }), f"u1 predict {label}")

    # u2: casual — one band higher, no team, no SFO vote
    check(client.post(f"/predict/{uid2}/{lot_no}", json={
        "predicted_band": adjacent_band(fv_band, +1),
        "predicted_team": None,
        "steal_vote": None,
    }), f"u2 predict {label}")

    # u3: bold — one band lower, wrong team, opposite SFO
    sfo_map = {"Steal":"Overpay","Overpay":"Steal","Fair":"Overpay",None:"Steal"}
    check(client.post(f"/predict/{uid3}/{lot_no}", json={
        "predicted_band": adjacent_band(fv_band, -1),
        "predicted_team": "RCB",   # bold (wrong) guess
        "steal_vote": sfo_map.get(pl.get("steal_truth"), "Fair"),
    }), f"u3 predict {label}")

    # reveal
    rev = check(client.post(f"/reveal/{lot_no}"), f"reveal {label}")
    print(f"\n  Lot {lot_no} {label}")
    print(f"    Result: {rev.get('actual_band')} | "
          f"₹{rev.get('actual_price') or 'unsold'} | "
          f"{rev.get('winning_team') or '—'} | "
          f"truth: {rev.get('steal_truth')}")
    for res in rev["results"]:
        print(f"    {res['user_name']:<12} band={res['band_outcome']:<8} "
              f"+{res['band_points']:>3} +fr{res['franchise_points']:>3} "
              f"+sfo{res['steal_points']:>3} = {res['total_points']:>4} pts "
              f"| running: {res['running_score']}")
    if rev["squad_verdicts"]:
        for v in rev["squad_verdicts"]:
            print(f"    🎯 {v['user_name']}: {v['line']}")

    # crowd stats
    crowd = check(client.get(f"/crowd/{lot_no}"), f"crowd {label}")
    print(f"    Crowd band votes: {crowd['band_votes']}")
    print(f"    Crowd SFO: {crowd['sfo_votes']} | truth: {crowd.get('steal_truth')}")

# ── 7. leaderboards ────────────────────────────────────────────────────────────
print(f"\n{SEP}\n7. LEADERBOARDS\n{SEP}")

lb_global = check(client.get("/leaderboard"), "global lb")
print("  Global:")
for e in lb_global["entries"]:
    print(f"    #{e['rank']} {e['user_name']:<12} "
          f"{e['total_score']:>5} pts | IQ: {e['auction_iq_percentile']}th pct | "
          f"lots: {e['lots_played']} | exact: {e['exact_hits']}")

lb_league = check(client.get(f"/leaderboard/{league_id}"), "league lb")
print(f"  {lb_league['league_name']} (code: {lb_league['invite_code']}):")
for e in lb_league["entries"]:
    print(f"    #{e['rank']} {e['user_name']:<12} "
          f"{e['total_score']:>5} pts | IQ: {e['auction_iq_percentile']}th pct")

# ── 8. Auction Wrapped ─────────────────────────────────────────────────────────
print(f"\n{SEP}\n8. AUCTION WRAPPED\n{SEP}")
for uid, name in [(uid1,"Abhishek"),(uid2,"Rohan"),(uid3,"Priya")]:
    w = check(client.get(f"/wrapped/{uid}"), f"wrapped {name}")
    print(f"\n  ── {w['user_name']} ──")
    print(f"    Score: {w['total_score']} pts | Auction IQ: {w['auction_iq_percentile']}th pct")
    print(f"    Lots: {w['lots_played']} | Exact hits: {w['exact_hits']} | "
          f"Band acc: {w['band_accuracy_pct']}%")
    print(f"    Steal acc: {w['steal_accuracy_pct']}% | "
          f"Shadow squad score: {w['shadow_squad_score']}/100")
    if w["best_call"]:
        bc = w["best_call"]
        print(f"    Best call: {bc['player']} — predicted {bc['predicted_band']}, "
              f"actual {bc['actual_band']} (+{bc['points']}pts)")
    if w["best_steal_spotted"]:
        st = w["best_steal_spotted"]
        print(f"    Steal spotted: {st['player']} ₹{st['winning_price']} Cr "
              f"(fair: ₹{st['fair_value']} Cr)")
    if w["squad_verdicts"]:
        for v in w["squad_verdicts"][:2]:
            print(f"    Squad: {v['line']}")
    print(f"    Archetype: {w['gm_archetype']} — {w['gm_line']}")
    print(f"    📱 Share: \"{w['shareable_headline']}\"")

print(f"\n{SEP}")
print("ALL TESTS PASSED ✓")
print(SEP)
