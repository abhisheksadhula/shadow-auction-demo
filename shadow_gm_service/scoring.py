"""
scoring.py
----------
Steps 3-5 of the pipeline, and the module the backend will call on hackathon day.

Pure functions only -- no I/O, no global state. Backend passes in a prediction and
the relevant player record; gets back points / verdicts / percentile. Keeping these
pure is what lets the backend engineer port or call them trivially.

Contains:
  score_band_prediction()   -- Step 3: points for a price-band guess (+ franchise bonus)
  classify_steal()          -- Step 3: the correct Steal/Fair/Overpay answer + scoring
  shadow_gm_verdict()       -- Step 4: how a player in your squad played out
  AuctionIQ                 -- Step 5: percentile engine (built from a baseline crowd)
"""
import numpy as np

from clean_data import BAND_LABELS

# ---- Step 3: band prediction scoring -----------------------------------------
EXACT_BAND_POINTS = 100
ADJACENT_BAND_POINTS = 40
FRANCHISE_BONUS = 50


def _band_index(band):
    return BAND_LABELS.index(band) if band in BAND_LABELS else None


def score_band_prediction(predicted_band, actual_band,
                          predicted_team=None, actual_team=None):
    """Points for one lot's prediction.

    Exact band = 100, one band off = 40, further = 0.
    Correct winning franchise adds a 50-point bonus (only meaningful if sold).
    Returns a dict so the UI can show the breakdown.
    """
    if actual_band is None:  # unsold -> no band to score against
        return {"band_points": 0, "franchise_points": 0, "total": 0,
                "outcome": "unsold", "exact": False}

    pi, ai = _band_index(predicted_band), _band_index(actual_band)
    if pi is None:
        band_pts, exact = 0, False
    else:
        dist = abs(pi - ai)
        if dist == 0:
            band_pts, exact = EXACT_BAND_POINTS, True
        elif dist == 1:
            band_pts, exact = ADJACENT_BAND_POINTS, False
        else:
            band_pts, exact = 0, False

    fr_pts = 0
    if predicted_team and actual_team and predicted_team == actual_team:
        fr_pts = FRANCHISE_BONUS

    return {"band_points": band_pts, "franchise_points": fr_pts,
            "total": band_pts + fr_pts,
            "outcome": "exact" if exact else ("close" if band_pts else "miss"),
            "exact": exact}


# ---- Step 3: Steal / Fair / Overpay ------------------------------------------
# The "correct" verdict is derived from actual winning price vs model fair value.
STEAL_RATIO = 0.80    # sold for <= 80% of fair value -> a steal
OVERPAY_RATIO = 1.25  # sold for >= 125% of fair value -> an overpay
SFO_POINTS = 60


def classify_steal(winning_price, fair_value):
    """Return the ground-truth Steal/Fair/Overpay label for a sold lot."""
    if winning_price is None or fair_value is None or np.isnan(winning_price):
        return None
    ratio = winning_price / fair_value
    if ratio <= STEAL_RATIO:
        return "Steal"
    if ratio >= OVERPAY_RATIO:
        return "Overpay"
    return "Fair"


def score_steal_prediction(user_choice, winning_price, fair_value):
    truth = classify_steal(winning_price, fair_value)
    if truth is None:
        return {"correct": False, "truth": None, "points": 0}
    correct = (user_choice == truth)
    return {"correct": correct, "truth": truth, "points": SFO_POINTS if correct else 0}


# ---- Step 4: Shadow GM verdict -----------------------------------------------
# A player on your pre-built shadow squad. After the real auction we tell you how
# that pick played out -- this is the connective tissue that makes it one game.
def shadow_gm_verdict(player_row, your_bid=None):
    """player_row: dict-like with sold, winning_team, winning_price_crore,
                   fair_value_crore, player_name.
       your_bid:   optional crore you allocated for this target.

    Returns a verdict label + a short narrative line for the UI.
    """
    name = player_row.get("player_name", "Player")
    if not player_row.get("sold", False):
        return {"verdict": "unsold",
                "line": f"{name} went unsold — no contest."}

    win_price = player_row.get("winning_price_crore")
    fair = player_row.get("fair_value_crore")

    # If you set a bid and it was at/above the actual price, you'd have landed them.
    if your_bid is not None and win_price is not None and your_bid >= win_price:
        if win_price <= fair * STEAL_RATIO:
            return {"verdict": "bargain",
                    "line": f"You landed {name} at a bargain — under fair value."}
        return {"verdict": "won",
                "line": f"You landed {name}, paying around the going rate."}

    # Otherwise a franchise took your target.
    team = player_row.get("winning_team", "a franchise")
    if win_price is not None and fair is not None and win_price >= fair * OVERPAY_RATIO:
        return {"verdict": "dodged",
                "line": f"{team} overpaid for {name} — you dodged that one."}
    return {"verdict": "missed",
            "line": f"{team} grabbed {name} before you could. Tough miss."}


# ---- Step 5: Auction IQ percentile -------------------------------------------
class AuctionIQ:
    """Turns a raw per-lot or session score into a percentile vs a 'crowd'.

    For the hackathon we build the baseline crowd distribution from the dataset
    itself, so we need no live users on day one:
      - For each sold lot, the 'expected' band is the fair-value band. A simulated
        crowd of fans guessing with realistic accuracy gives a points distribution
        per lot. A user's percentile = where their score falls in that distribution.

    On the 6-month roadmap this same object is fed real user scores instead.
    """
    def __init__(self, per_lot_scores):
        """per_lot_scores: 1D array of simulated crowd points for a lot (or overall)."""
        self.dist = np.sort(np.asarray(per_lot_scores, dtype=float))

    @classmethod
    def from_simulated_crowd(cls, fair_band, actual_band, n=2000, skill=0.55, seed=0):
        """Simulate n fans predicting one lot, return an AuctionIQ for it.

        skill = probability a simulated fan lands the fair-value (expected) band.
        Models the crowd as mostly guessing toward the obvious band, sometimes right,
        sometimes adjacent, occasionally far off.
        """
        rng = np.random.default_rng(seed)
        ai = _band_index(actual_band)
        if ai is None:  # unsold or unrecognized band -> no scorable lot
            return cls(np.zeros(n))
        fi = _band_index(fair_band)
        if fi is None:
            fi = ai
        scores = np.empty(n)
        for k in range(n):
            r = rng.random()
            if r < skill:
                guess = fi                      # guesses the expected band
            elif r < skill + 0.25:
                guess = fi + rng.choice([-1, 1])  # adjacent to expected
            else:
                guess = rng.integers(0, len(BAND_LABELS))  # wild guess
            guess = int(np.clip(guess, 0, len(BAND_LABELS) - 1))
            dist = abs(guess - ai)
            scores[k] = (EXACT_BAND_POINTS if dist == 0
                         else ADJACENT_BAND_POINTS if dist == 1 else 0)
        return cls(scores)

    def percentile(self, user_score):
        """What % of the crowd this user beat or tied (0-100, rounded)."""
        if len(self.dist) == 0:
            return 50
        pct = np.searchsorted(self.dist, user_score, side="right") / len(self.dist)
        return int(round(pct * 100))


if __name__ == "__main__":
    # smoke test the scoring functions
    print("Band scoring:")
    print("  exact + team :", score_band_prediction("5-10", "5-10", "CSK", "CSK"))
    print("  adjacent     :", score_band_prediction("Under 5", "5-10"))
    print("  miss         :", score_band_prediction("Under 5", "15+"))
    print("  unsold       :", score_band_prediction("5-10", None))

    print("\nSteal/Fair/Overpay:")
    print("  truth @ ratio 0.7 :", classify_steal(7.0, 10.0))
    print("  truth @ ratio 1.0 :", classify_steal(10.0, 10.0))
    print("  truth @ ratio 1.4 :", classify_steal(14.0, 10.0))
    print("  user correct      :", score_steal_prediction("Steal", 7.0, 10.0))

    print("\nShadow GM verdict:")
    row = {"player_name":"KL Rahul","sold":True,"winning_team":"DC",
           "winning_price_crore":14.0,"fair_value_crore":12.0}
    print("  you bid 15 (land) :", shadow_gm_verdict(row, your_bid=15.0)["line"])
    print("  you bid 10 (miss) :", shadow_gm_verdict(row, your_bid=10.0)["line"])

    print("\nAuction IQ:")
    iq = AuctionIQ.from_simulated_crowd(fair_band="5-10", actual_band="5-10")
    print("  score 100 ->", iq.percentile(100), "percentile")
    print("  score 40  ->", iq.percentile(40), "percentile")
    print("  score 0   ->", iq.percentile(0), "percentile")
