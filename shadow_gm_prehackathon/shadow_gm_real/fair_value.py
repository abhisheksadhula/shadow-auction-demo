"""
fair_value.py
-------------
Step 2 of the pipeline. The heart of the "Use of Data / AI" story.

Builds an EXPLAINABLE fair-value model. We deliberately avoid a black box: a judge
should be able to ask "why is this player's fair value 8 crore?" and get an answer.

Approach: log-linear regression of winning price on a handful of interpretable
features (base price, capped, overseas, role). Log space because auction prices are
heavily right-skewed (a few huge buys, many small ones). The model is fit only on
SOLD players (unsold have no winning price), then applied to everyone to produce a
fair value and a fair-value band.

If role is "Unknown" for the whole dataset (real file lacks it), the role terms
simply contribute nothing and the model still works on base/capped/overseas.

Output columns added:
  fair_value_crore : model's estimate of a reasonable price
  fair_value_band  : that estimate mapped to a UI band
  value_ratio      : winning_price / fair_value (for sold players) -- >1 overpay, <1 bargain
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from clean_data import price_to_band, BAND_LABELS

ROLE_DUMMIES = ["Batter", "Bowler", "All-Rounder", "Wicketkeeper"]


def _features(df):
    """Build the interpretable design matrix."""
    X = pd.DataFrame(index=df.index)
    X["log_base"] = np.log(df["base_price_crore"].clip(lower=0.1))
    X["capped"] = df["capped"].astype(float)
    X["overseas"] = df["overseas"].astype(float)
    for r in ROLE_DUMMIES:
        X[f"role_{r}"] = (df["role"] == r).astype(float)
    return X


def fit_fair_value(df):
    """Fit on sold players, return (model, feature_names, fitted_df_with_fair_value)."""
    sold = df[df["sold"] & df["winning_price_crore"].notna()].copy()
    X = _features(sold)
    y = np.log(sold["winning_price_crore"].clip(lower=0.1))

    model = LinearRegression()
    model.fit(X, y)

    # apply to everyone
    X_all = _features(df)
    pred_log = model.predict(X_all)
    fair = np.exp(pred_log)
    fair = np.clip(fair, df["base_price_crore"], 27.0)

    out = df.copy()
    out["fair_value_crore"] = np.round(fair, 2)
    out["fair_value_band"] = out["fair_value_crore"].apply(price_to_band)
    out["value_ratio"] = np.where(
        out["winning_price_crore"].notna(),
        np.round(out["winning_price_crore"] / out["fair_value_crore"], 3),
        np.nan,
    )
    return model, list(X.columns), out


def explain(model, feature_names):
    """Return a human-readable description of the fitted coefficients."""
    lines = ["Fair-value model — log-linear coefficients (effect on price):"]
    coefs = dict(zip(feature_names, model.coef_))
    # base-price elasticity
    lines.append(f"  base price elasticity : {coefs['log_base']:.2f} "
                 f"(1% higher base -> ~{coefs['log_base']:.2f}% higher value)")
    lines.append(f"  capped premium        : x{np.exp(coefs['capped']):.2f}")
    lines.append(f"  overseas premium      : x{np.exp(coefs['overseas']):.2f}")
    for r in ROLE_DUMMIES:
        key = f"role_{r}"
        if key in coefs and abs(coefs[key]) > 1e-9:
            lines.append(f"  role: {r:<12}     : x{np.exp(coefs[key]):.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from clean_data import load_and_clean
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "shadow_gm/data/ipl_2025_auction_players_SAMPLE.csv"
    df = load_and_clean(path)
    model, feats, scored = fit_fair_value(df)

    print(explain(model, feats))

    # quick fit quality on sold players (in crore, not log)
    sold = scored[scored["sold"] & scored["winning_price_crore"].notna()]
    mae = (sold["winning_price_crore"] - sold["fair_value_crore"]).abs().mean()
    band_hit = (sold["actual_band"] == sold["fair_value_band"]).mean()
    print(f"\nFit on sold players: MAE = {mae:.2f} Cr | "
          f"fair-band matches actual-band {band_hit:.0%} of the time")

    print("\nBiggest bargains (lowest value_ratio, i.e. sold well below fair value):")
    cols = ["player_name","winning_price_crore","fair_value_crore","value_ratio","actual_band"]
    print(sold.nsmallest(5, "value_ratio")[cols].to_string(index=False))
    print("\nBiggest overpays (highest value_ratio):")
    print(sold.nlargest(5, "value_ratio")[cols].to_string(index=False))
