# Changelog
All notable changes to BakerT MatchDashboard. Versioning: MAJOR.MINOR.PATCH (e.g. 1.1.0 = new feature, 1.0.1 = bug fix).

## [1.1.0] – 2026-11-01
### Added
- **ML engine** (`matchdash/ml.py`, `matchdash/engine.py`): gradient-boosted trees on leakage-free features (Elo, form, venue goal rates, rest days, head-to-head, league averages, optional bookmaker-implied probabilities). Goal-rate regressors plus calibrated 1X2 / Over 2.5 / BTTS classifiers.
- ML predictions are fitted back into the scoreline grid (iterative proportional fitting), so all 41 markets, HT/FT and correct scores stay consistent.
- Engine selector in the sidebar: Poisson, ML, or Blend with an adjustable weight.
- **ML Lab** tab: train, walk-forward comparison vs Poisson and bookmaker odds, paired significance test, feature importance.
- Tests: no-leakage invariant, grid fitting, engine modes, walk-forward comparison.
### Changed
- Requires scikit-learn. Poisson model refactored into reusable grid/market helpers (results unchanged and still match the Excel workbook).

## [1.0.0] – 2026-10-31
### Added
- Streamlit app: Dashboard, All fixtures, Backtest and Data tabs.
- Poisson model matching the original Excel workbook (verified by tests), with optional recency weighting and Dixon-Coles correction.
- 41 markets: 1X2, double chance, DNB, HT result, goal lines, BTTS, team goals, clean sheets, HT/FT, correct scores.
- Colour-coded signal tiers, strongest / least likely scenarios, value check vs bookmaker odds.
- Head-to-head across all history; no row limit; football-data.co.uk files read directly.
- Walk-forward backtest with calibration and parameter sweeps; CSV export of predictions.
\n## 1.2.0 — Community open-core foundation\n\n- Added central Community feature policy and edition label.\n- Kept gradient-boosted ML training, prediction and basic walk-forward comparison in Community.\n- Hid Pro-roadmap feature importance and parameter-sweep controls from the Community UI.\n- Replaced the proprietary notice with the MIT licence for this Community repository.\n- Updated README and monetization guidance for a separate commercial Pro extension.\n