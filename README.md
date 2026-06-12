# Shadow GM — IPL Auction Game

A gamified IPL auction experience that turns fans into active participants across all three phases of the auction.

## What is Shadow GM?

Shadow GM lets IPL fans play the auction rather than just watch it.

- **Pre-Auction** — Build a shadow squad of target players within a ₹100 Cr virtual budget
- **During Auction** — Predict price band, winning franchise, and Steal/Fair/Overpay for each player in real time
- **Post-Auction** — Get a personalised Auction Wrapped card with your IQ percentile, GM archetype, and squad verdicts

## Live Demo

👉 [shadow-auction-demo link here](https://abhisheksadhula.github.io/shadow-auction-demo/)

## How to play

1. Join as one of 3 fans — enter your name
2. Pick 2 target players for your shadow squad
3. For each player coming up at auction — predict the price band, which team buys them, and whether it was a steal or overpay
4. After the reveal — see your score, Auction IQ percentile, and leaderboard position
5. After all players — view your Auction Wrapped card

## Data & Scoring

- **Price bands** — Under ₹5 Cr / ₹5–10 Cr / ₹10–15 Cr / ₹15+ Cr
- **Exact band** → 100 pts | **Adjacent band** → 40 pts | **Correct franchise** → +50 pts | **SFO correct** → +60 pts
- **Auction IQ** — your band score ranked against a simulated crowd of 2000 fans per player
- **Fair value model** — hybrid log-linear regression trained on real IPL 2025 auction data (577 players, 174 sold)

## Tech

- **Data pipeline** — Python (pandas, scikit-learn) — clean, model, score, export
- **Scoring service** — FastAPI, in-memory store, 12 REST endpoints
- **Demo** — vanilla HTML/CSS/JS, no dependencies, runs offline

## Team

Built for the BCCI/IPL Hackathon 2026 by a team of 3 — Data Engineer, Backend Engineer, iOS Developer.
